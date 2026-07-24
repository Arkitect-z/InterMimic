#!/usr/bin/env python3
"""Pure-CPU tests for formal Theia policy result processing."""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_theia_policy_ab import aggregate_experiment
from prepare_theia_policy_ab import (
    parse_reference_fields,
    validate_formal_s1_fields,
)
from summarize_theia_eval import ValidationError, summarize_evaluation


EPISODE_FIELDS = [
    "env_id",
    "sequence_id",
    "sequence",
    "trial_id",
    "steps",
    "completed",
    "mean_human_error_m",
    "mean_object_surface_error_m",
]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path, reference_count=3):
    heights = ("L1", "L2", "L3")
    variations = ("V1", "V2", "V3")
    references = []
    for index in range(reference_count):
        references.append({
            "reference_id": "ref_{:02d}".format(index),
            "raw_filename": "raw_ref_{:02d}.pt".format(index),
            "full_filename": "full_ref_{:02d}.pt".format(index),
            "raw_sha256": hashlib.sha256(
                "raw_ref_{:02d}".format(index).encode()
            ).hexdigest(),
            "full_sha256": hashlib.sha256(
                "full_ref_{:02d}".format(index).encode()
            ).hexdigest(),
            "height": heights[index % len(heights)],
            "variation": variations[index % len(variations)],
            "included": True,
        })
    path.write_text(json.dumps({"references": references}, indent=2) + "\n")
    return references


def condition_data_sha256(references, condition):
    records = sorted(
        (
            reference[condition + "_filename"],
            reference[condition + "_sha256"],
        )
        for reference in references
    )
    payload = "".join(
        "{}  {}\n".format(digest, filename)
        for filename, digest in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CanonicalS1Test(unittest.TestCase):
    def test_same_height_and_variation_are_canonical(self):
        fields = parse_reference_fields("S1L33P01T0508V01")
        validate_formal_s1_fields(fields)
        self.assertEqual(fields["height"], "L3")
        self.assertEqual(fields["variation"], "V1")

    def test_cross_height_s1_is_rejected(self):
        fields = parse_reference_fields("S1L13P01T0508V01")
        with self.assertRaises(ValueError):
            validate_formal_s1_fields(fields)

    def test_nonprotocol_variation_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_reference_fields("S1L33P01T0508V04")


class SummarizeEvaluationTest(unittest.TestCase):
    def test_exact_k_any_completed_best_trial_and_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "pairs.json"
            references = write_manifest(manifest, reference_count=2)
            episodes = root / "episodes.csv"
            rows = [
                {
                    "env_id": 0,
                    "sequence_id": 0,
                    "sequence": references[0]["raw_filename"],
                    "trial_id": 0,
                    "steps": 90,
                    "completed": 0,
                    "mean_human_error_m": 0.02,
                    "mean_object_surface_error_m": 0.03,
                },
                {
                    "env_id": 1,
                    "sequence_id": 0,
                    "sequence": references[0]["raw_filename"],
                    "trial_id": 1,
                    "steps": 100,
                    "completed": 1,
                    "mean_human_error_m": 0.03,
                    "mean_object_surface_error_m": 0.02,
                },
                {
                    "env_id": 2,
                    "sequence_id": 0,
                    "sequence": references[0]["raw_filename"],
                    "trial_id": 2,
                    "steps": 100,
                    "completed": 0,
                    "mean_human_error_m": 0.01,
                    "mean_object_surface_error_m": 0.01,
                },
                {
                    "env_id": 3,
                    "sequence_id": 1,
                    "sequence": references[1]["raw_filename"],
                    "trial_id": 0,
                    "steps": 70,
                    "completed": 0,
                    "mean_human_error_m": 0.04,
                    "mean_object_surface_error_m": 0.05,
                },
                {
                    "env_id": 4,
                    "sequence_id": 1,
                    "sequence": references[1]["raw_filename"],
                    "trial_id": 1,
                    "steps": 60,
                    "completed": 0,
                    "mean_human_error_m": 0.02,
                    "mean_object_surface_error_m": 0.02,
                },
                {
                    "env_id": 5,
                    "sequence_id": 1,
                    "sequence": references[1]["raw_filename"],
                    "trial_id": 2,
                    "steps": 50,
                    "completed": 0,
                    "mean_human_error_m": 0.01,
                    "mean_object_surface_error_m": 0.01,
                },
            ]
            write_csv(episodes, EPISODE_FIELDS, rows)
            output = root / "formal"
            summarized, validation, summary = summarize_evaluation(
                episodes_path=episodes,
                manifest_path=manifest,
                condition="raw",
                training_seed=2,
                evaluation_seed=42,
                k_trials=3,
                fps=10.0,
                output_dir=output,
            )

            by_id = {row["reference_id"]: row for row in summarized}
            first = by_id["ref_00"]
            self.assertEqual(first["success"], 1)
            self.assertEqual(first["completed_trials"], 1)
            self.assertEqual(first["best_trial_id"], 2)
            self.assertEqual(first["duration_s"], 10.0)
            self.assertEqual(first["human_error_cm"], 1.0)
            self.assertEqual(first["object_error_cm"], 1.0)
            self.assertEqual(by_id["ref_01"]["success"], 0)
            self.assertEqual(validation["actual_episodes"], 6)
            self.assertEqual(summary["metrics"]["success_rate_percent"], 50.0)
            self.assertTrue((output / "per_reference.csv").is_file())
            self.assertTrue((output / "validation.json").is_file())
            self.assertTrue((output / "summary.json").is_file())

    def test_wrong_trial_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "pairs.json"
            references = write_manifest(manifest, reference_count=1)
            episodes = root / "episodes.csv"
            write_csv(episodes, EPISODE_FIELDS, [{
                "env_id": 0,
                "sequence_id": 0,
                "sequence": references[0]["raw_filename"],
                "trial_id": 0,
                "steps": 10,
                "completed": 1,
                "mean_human_error_m": 0.01,
                "mean_object_surface_error_m": 0.01,
            }])
            with self.assertRaises(ValidationError):
                summarize_evaluation(
                    episodes_path=episodes,
                    manifest_path=manifest,
                    condition="raw",
                    training_seed=0,
                    evaluation_seed=42,
                    k_trials=2,
                    fps=30.0,
                    output_dir=root / "formal",
                )


class AggregateExperimentTest(unittest.TestCase):
    def _build_formal_experiment(self, root):
        manifest = root / "pairs.json"
        references = write_manifest(manifest, reference_count=3)
        for condition in ("raw", "full"):
            for training_seed in range(4):
                rows = []
                for reference_index, reference in enumerate(references):
                    for trial_id in range(10):
                        if condition == "raw":
                            completed = int(
                                reference_index == 0 and trial_id == 0
                            )
                            steps = 300 + training_seed
                            human_error = 0.02
                            object_error = 0.03
                        else:
                            completed = int(
                                reference_index < 2 and trial_id == 0
                            )
                            steps = 330 + training_seed
                            human_error = 0.01
                            object_error = 0.02
                        rows.append({
                            "env_id": reference_index + trial_id * 3,
                            "sequence_id": reference_index,
                            "sequence": reference[
                                condition + "_filename"
                            ],
                            "steps": steps,
                            "completed": completed,
                            "mean_human_error_m": human_error,
                            "mean_object_surface_error_m": object_error,
                        })
                # Omit trial_id to exercise stable env-id derivation for the
                # current simulator CSV schema.
                episode_fields = [
                    field for field in EPISODE_FIELDS if field != "trial_id"
                ]
                formal_dir = (
                    root
                    / condition
                    / "seed_{}".format(training_seed)
                    / "evaluation"
                    / "final"
                )
                episodes = formal_dir / "episodes.csv"
                write_csv(episodes, episode_fields, rows)
                run_root = formal_dir.parents[1]
                data_sha = condition_data_sha256(references, condition)
                evaluation_manifest = formal_dir / "manifest.txt"
                evaluation_manifest.write_text(
                    "\n".join([
                        "checkpoint_sha256={}".format(
                            hashlib.sha256(
                                "{}-{}".format(
                                    condition, training_seed
                                ).encode()
                            ).hexdigest()
                        ),
                        "eval_seed={}".format(10000 + training_seed),
                        "training_seed={}".format(training_seed),
                        "condition={}".format(condition),
                        "condition_data_sha256={}".format(data_sha),
                        "evaluation_pipeline_sha256={}".format("c" * 64),
                        "evaluation_config=isaacgym/src/intermimic/data/cfg/"
                        "theia_policy_eval.yaml",
                        "",
                    ])
                )
                summarize_evaluation(
                    episodes_path=episodes,
                    manifest_path=manifest,
                    condition=condition,
                    training_seed=training_seed,
                    evaluation_seed=10000 + training_seed,
                    k_trials=10,
                    fps=30.0,
                    output_dir=formal_dir,
                    evaluation_manifest_path=evaluation_manifest,
                )
                run_root.mkdir(parents=True, exist_ok=True)
                (run_root / "run_spec.txt").write_text(
                    "\n".join([
                        "condition={}".format(condition),
                        "training_seed={}".format(training_seed),
                        "data_dir=/frozen/{}".format(condition),
                        "condition_data_sha256={}".format(data_sha),
                        "pair_manifest={}".format(manifest.resolve()),
                        "reference_count=3",
                        "target_envs=2048",
                        "num_envs=2046",
                        "minibatch_size=16368",
                        "bootstrap_epochs=20000",
                        "finetune_epochs=2000",
                        "eval_k=10",
                        "eval_seed={}".format(10000 + training_seed),
                        "evaluation_fps=30",
                        "allow_nonformal_protocol=0",
                        "git_commit=test-commit",
                        "git_diff_sha256={}".format("d" * 64),
                        "{}  {}".format(
                            file_sha256(manifest), manifest.resolve()
                        ),
                        "{}  isaacgym/scripts/formal_source.py".format(
                            "e" * 64
                        ),
                        "",
                    ])
                )
        return manifest

    def test_four_seed_pairing_bootstrap_and_tables_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._build_formal_experiment(root)
            first_output = root / "results_a"
            second_output = root / "results_b"
            first = aggregate_experiment(
                experiment_root=root,
                manifest_path=manifest,
                output_dir=first_output,
                bootstrap_samples=200,
                bootstrap_seed=7,
            )
            second = aggregate_experiment(
                experiment_root=root,
                manifest_path=manifest,
                output_dir=second_output,
                bootstrap_samples=200,
                bootstrap_seed=7,
            )

            raw_success = first["metrics"]["raw"][
                "success_rate_percent"
            ]["mean"]
            full_success = first["metrics"]["full"][
                "success_rate_percent"
            ]["mean"]
            self.assertAlmostEqual(raw_success, 100.0 / 3.0)
            self.assertAlmostEqual(full_success, 200.0 / 3.0)
            self.assertAlmostEqual(
                first["paired_differences"]["success_rate_percent"][
                    "delta_full_minus_raw"
                ],
                100.0 / 3.0,
            )
            self.assertEqual(first["num_references"], 3)
            self.assertEqual(first["training_seeds"], [0, 1, 2, 3])
            self.assertEqual(first["k_trials"], 10)
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(
                (first_output / "bootstrap_results.json").read_bytes(),
                (second_output / "bootstrap_results.json").read_bytes(),
            )
            self.assertEqual(
                (first_output / "summary.json").read_bytes(),
                (second_output / "summary.json").read_bytes(),
            )
            self.assertEqual(
                (first_output / "results.md").read_bytes(),
                (second_output / "results.md").read_bytes(),
            )
            main_table = (first_output / "main_table.tex").read_text()
            spatial_table = (
                first_output / "spatial_table.tex"
            ).read_text()
            self.assertIn("33.33", main_table)
            self.assertIn("66.67", main_table)
            self.assertNotIn("3333.33", spatial_table)
            for filename in (
                "per_reference_all.csv",
                "per_seed.csv",
                "paired_results.csv",
                "bootstrap_results.json",
                "summary.json",
                "main_table.tex",
                "spatial_table.tex",
                "results.md",
            ):
                self.assertTrue((first_output / filename).is_file())

    def test_reference_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._build_formal_experiment(root)
            path = (
                root
                / "full"
                / "seed_0"
                / "evaluation"
                / "final"
                / "per_reference.csv"
            )
            with path.open(newline="") as source:
                reader = csv.DictReader(source)
                fieldnames = reader.fieldnames
                rows = list(reader)
            write_csv(path, fieldnames, rows[:-1])
            with self.assertRaises(ValidationError):
                aggregate_experiment(
                    experiment_root=root,
                    manifest_path=manifest,
                    output_dir=root / "results",
                    bootstrap_samples=10,
                    bootstrap_seed=1,
                )

    def test_cross_run_spec_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._build_formal_experiment(root)
            run_spec = root / "full" / "seed_2" / "run_spec.txt"
            run_spec.write_text(
                run_spec.read_text().replace(
                    "minibatch_size=16368",
                    "minibatch_size=8184",
                )
            )
            with self.assertRaises(ValidationError):
                aggregate_experiment(
                    experiment_root=root,
                    manifest_path=manifest,
                    output_dir=root / "results",
                    bootstrap_samples=10,
                    bootstrap_seed=1,
                )


if __name__ == "__main__":
    unittest.main()
