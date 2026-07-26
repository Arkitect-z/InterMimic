#!/usr/bin/env python3
"""Regression tests for the one-run-per-reference rebuttal pipeline."""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EMPTY_GIT_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_theia_policy_references import aggregate  # noqa: E402
from check_theia_server_versions import (  # noqa: E402
    VERSION_MANIFEST,
    github_slug,
)
from check_theia_protomotions import (  # noqa: E402
    EXPECTED_COMMIT,
    check_protomotions,
)
from prepare_theia_reference_view import create_view  # noqa: E402
from validate_theia_reference_lists import validate_partition  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def reference(reference_id, raw_hash, full_hash):
    return {
        "reference_id": reference_id,
        "raw_filename": "sub1_A+B_{}.pt".format(reference_id),
        "full_filename": "sub1_A+B_{}.pt".format(reference_id),
        "raw_sha256": raw_hash,
        "full_sha256": full_hash,
        "height": "L1",
        "variation": "V1",
        "action": "pour",
        "task": "T0001",
        "included": True,
    }


class ProtoMotionsPinTest(unittest.TestCase):
    def test_local_checkout_matches_conversion_pin(self):
        proto = REPO_ROOT.parents[1] / "thirdparty" / "ProtoMotions"
        report = check_protomotions(proto)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["actual_commit"], EXPECTED_COMMIT)


class ServerVersionManifestTest(unittest.TestCase):
    def test_formal_release_manifest_matches_protocol(self):
        specification = json.loads(VERSION_MANIFEST.read_text())
        protocol = specification["protocol"]
        self.assertEqual(
            specification["status"],
            "only_supported_formal_policy_method",
        )
        self.assertEqual(
            protocol["id"], "single_reference_raw_vs_refined_v2"
        )
        self.assertEqual(protocol["training_epochs"], 2000)
        self.assertEqual(
            specification["repositories"]["ProtoMotions"]["commit"],
            EXPECTED_COMMIT,
        )

    def test_github_remote_normalization_accepts_https_and_ssh(self):
        expected = "Arkitect-z/InterMimic"
        self.assertEqual(
            github_slug(
                "https://github.com/Arkitect-z/InterMimic.git"
            ),
            expected,
        )
        self.assertEqual(
            github_slug("git@github.com:Arkitect-z/InterMimic.git"),
            expected,
        )


class ReferenceViewTest(unittest.TestCase):
    def test_one_reference_view_is_hash_checked_and_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            raw_dir = prepared / "eligible" / "raw"
            full_dir = prepared / "eligible" / "full"
            raw_dir.mkdir(parents=True)
            full_dir.mkdir(parents=True)
            filename = "sub1_A+B_ref1.pt"
            (raw_dir / filename).write_bytes(b"raw")
            (full_dir / filename).write_bytes(b"refined")
            entry = reference(
                "ref1",
                digest(raw_dir / filename),
                digest(full_dir / filename),
            )
            (prepared / "policy_ab_manifest.json").write_text(
                json.dumps({
                    "eligible_count": 1,
                    "excluded_count": 0,
                    "references": [entry],
                })
            )
            view = root / "view"
            first = create_view(prepared, "ref1", view)
            second = create_view(prepared, "ref1", view)
            self.assertEqual(first, second)
            self.assertEqual(
                (view / "data" / "raw" / filename).resolve(),
                (raw_dir / filename).resolve(),
            )
            manifest = json.loads((view / "pair_manifest.json").read_text())
            self.assertEqual(manifest["candidate_reference_ids"], ["ref1"])
            self.assertEqual(len(manifest["references"]), 1)


class ReferenceListPartitionTest(unittest.TestCase):
    def test_cluster_lists_must_be_an_exact_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "pairs.json"
            refs = [
                reference("ref1", "a" * 64, "b" * 64),
                reference("ref2", "c" * 64, "d" * 64),
            ]
            manifest.write_text(json.dumps({"references": refs}))
            first = root / "cluster_a.txt"
            second = root / "cluster_b.txt"
            first.write_text("/data/ref1\n")
            second.write_text("ref2 # assigned to GPU 1\n")
            report = validate_partition(manifest, [first, second])
            self.assertTrue(report["valid"])
            self.assertEqual(report["assigned_reference_count"], 2)
            second.write_text("ref1\n")
            report = validate_partition(manifest, [first, second])
            self.assertFalse(report["valid"])
            self.assertIn("ref2", report["missing"])
            self.assertIn("ref1", report["duplicates"])


class FormalReferenceProtocolTest(unittest.TestCase):
    def test_training_contract_is_one_run_per_condition(self):
        source = (
            SCRIPT_DIR / "run_theia_policy_reference.sh"
        ).read_text()
        self.assertIn('TRAIN_EPOCHS="${TRAIN_EPOCHS:-2000}"', source)
        self.assertIn('TRAINING_SEED="${TRAINING_SEED:-0}"', source)
        self.assertIn("FINETUNE_EPOCHS=0", source)
        self.assertIn("PROTOCOL_MODE=single_reference_rebuttal", source)
        self.assertIn(
            'ALLOW_NONFORMAL_PROTOCOL="$ALLOW_PROTOCOL_OVERRIDE"', source
        )
        self.assertIn("run_condition raw\nrun_condition full", source)
        self.assertNotIn("for seed in", source)

    def test_legacy_universal_entry_is_blocked_by_default(self):
        server = (SCRIPT_DIR / "run_theia_server.sh").read_text()
        seed_runner = (SCRIPT_DIR / "run_theia_policy_seed.sh").read_text()
        self.assertIn('ALLOW_LEGACY_UNIVERSAL:-0', server)
        self.assertIn(
            "Formal jobs must use run_theia_policy_reference_list.sh",
            server,
        )
        self.assertIn('PROTOCOL_MODE="${PROTOCOL_MODE:-}"', seed_runner)
        self.assertIn(
            "legacy_universal is historical and cannot produce formal results",
            seed_runner,
        )

    def test_reference_config_keeps_successful_recipe_without_hard_gt_reset(self):
        path = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "data"
            / "cfg"
            / "theia_reference_train.yaml"
        )
        env = yaml.safe_load(path.read_text())["env"]
        self.assertEqual(env["stateInit"], "Hybrid")
        self.assertEqual(env["rolloutLength"], 100)
        self.assertTrue(env["adaptiveRolloutFromLatestContact"])
        self.assertEqual(env["contactRewardMode"], "legacy_multiplicative")
        self.assertFalse(env["enableContactFailureTermination"])
        self.assertFalse(env["validateReferenceFK"])
        self.assertEqual(env["wrongContactPenalty"], 0.0)
        self.assertFalse(env["enableTrainingDiagnostics"])
        self.assertFalse(env["enableStepDiagnostics"])
        self.assertEqual(env["physicalBufferSize"], 3)

    def test_ppo_core_matches_successful_from_scratch_recipe(self):
        path = (
            REPO_ROOT
            / "isaacgym"
            / "src"
            / "intermimic"
            / "data"
            / "cfg"
            / "train"
            / "rlg"
            / "theia.yaml"
        )
        config = yaml.safe_load(path.read_text())["params"]["config"]
        self.assertEqual(float(config["learning_rate"]), 2e-5)
        self.assertEqual(config["e_clip"], 0.2)
        self.assertEqual(config["mini_epochs"], 6)
        self.assertEqual(config["horizon_length"], 32)
        self.assertEqual(config["save_frequency"], 100)
        self.assertFalse(config["save_intermediate"])
        self.assertEqual(
            config["checkpoint_milestones"],
            list(range(100, 2001, 100)),
        )
        self.assertFalse(config["save_reward_best"])
        self.assertIn(config["resume_from"], (None, "None"))


class ReferenceAggregationTest(unittest.TestCase):
    def _write_repository_receipt(self, experiment):
        specification = json.loads(VERSION_MANIFEST.read_text())
        commit = "f" * 40
        repositories = {
            name: {"valid": True}
            for name in ("Theia", "InterMimic", "ProtoMotions")
        }
        repositories["InterMimic"].update({
            "head": commit,
            "release_tag": specification["repositories"]["InterMimic"]["tag"],
            "release_tag_commit": commit,
        })
        experiment.mkdir(parents=True, exist_ok=True)
        (experiment / "repository_versions.json").write_text(json.dumps({
            "valid": True,
            "release_name": specification["release_name"],
            "formal_method_status": specification["status"],
            "protocol": specification["protocol"],
            "repositories": repositories,
        }))
        return commit

    def _write_result(
        self,
        experiment,
        entry,
        condition,
        commit,
        completed,
        duration,
        human,
        obj,
    ):
        reference_id = entry["reference_id"]
        pair = experiment / "references" / reference_id
        pair.mkdir(parents=True, exist_ok=True)
        (pair / "pair_spec.txt").write_text(
            "\n".join([
                "protocol=single_reference_raw_vs_refined_v2",
                "reference_id={}".format(reference_id),
                "train_epochs=2000",
                "training_seed=0",
                "evaluation_seed=10000",
                "k_trials=10",
                "torch_deterministic=0",
                (
                    "train_env_config=isaacgym/src/intermimic/data/cfg/"
                    "theia_reference_train.yaml"
                ),
                "git_commit={}".format(commit),
                "git_diff_sha256={}".format(EMPTY_GIT_DIFF_SHA256),
                "",
            ])
        )
        (pair / "PAIR_READY.json").write_text(json.dumps({
            "protocol": "single_reference_raw_vs_refined_v2",
            "reference_id": reference_id,
            "training_runs_per_condition": 1,
            "training_seed": 0,
            "k_trials": 10,
        }))
        run = pair / "runs" / condition / "seed_0"
        run.mkdir(parents=True, exist_ok=True)
        (run / "run_spec.txt").write_text(
            "\n".join([
                "condition={}".format(condition),
                "training_seed=0",
                "reference_count=1",
                "bootstrap_epochs=2000",
                "finetune_epochs=0",
                (
                    "train_env_config=isaacgym/src/intermimic/data/cfg/"
                    "theia_reference_train.yaml"
                ),
                "eval_k=10",
                "eval_seed=10000",
                "protocol_mode=single_reference_rebuttal",
                "allow_nonformal_protocol=0",
                "git_commit={}".format(commit),
                "git_diff_sha256={}".format(EMPTY_GIT_DIFF_SHA256),
                "",
            ])
        )
        evaluation = run / "evaluation" / "final"
        per_reference = evaluation / "per_reference.csv"
        fields = [
            "condition",
            "training_seed",
            "evaluation_seed",
            "reference_id",
            "k_trials",
            "completed_trials",
            "success",
            "duration_s",
            "human_error_cm",
            "object_error_cm",
        ]
        write_csv(per_reference, fields, [{
            "condition": condition,
            "training_seed": 0,
            "evaluation_seed": 10000,
            "reference_id": reference_id,
            "k_trials": 10,
            "completed_trials": completed,
            "success": int(completed > 0),
            "duration_s": duration,
            "human_error_cm": human,
            "object_error_cm": obj,
        }])
        (evaluation / "validation.json").write_text(json.dumps({
            "valid": True,
            "condition": condition,
            "training_seed": 0,
            "evaluation_seed": 10000,
            "k_trials": 10,
            "actual_episodes": 10,
            "expected_episodes": 10,
            "per_reference_sha256": digest(per_reference),
        }))

    def test_paired_reference_bootstrap_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            commit = self._write_repository_receipt(experiment)
            manifest = root / "pairs.json"
            refs = [
                reference("ref1", "a" * 64, "b" * 64),
                reference("ref2", "c" * 64, "d" * 64),
            ]
            manifest.write_text(json.dumps({"references": refs}))
            for index, entry in enumerate(refs):
                self._write_result(
                    experiment,
                    entry,
                    "raw",
                    commit,
                    completed=index,
                    duration=5.0 + index,
                    human=3.0,
                    obj=4.0,
                )
                self._write_result(
                    experiment,
                    entry,
                    "full",
                    commit,
                    completed=2 + index,
                    duration=8.0 + index,
                    human=2.0,
                    obj=1.0,
                )
            output = root / "results"
            summary = aggregate(
                experiment,
                manifest,
                output,
                bootstrap_samples=1000,
                bootstrap_seed=7,
            )
            self.assertEqual(summary["num_references"], 2)
            self.assertEqual(
                summary["repository_release"],
                "theia-policy-rebuttal-v3-2000",
            )
            success = summary["metrics"]["ref_success_at_k_percent"]
            self.assertEqual(success["raw_mean"], 50.0)
            self.assertEqual(success["refined_mean"], 100.0)
            completion = summary["metrics"]["episode_completion_percent"]
            self.assertEqual(completion["raw_mean"], 5.0)
            self.assertEqual(completion["refined_mean"], 25.0)
            for filename in (
                "per_reference_paired.csv",
                "paired_results.csv",
                "by_pose.csv",
                "summary.json",
                "results.md",
                "main_table.tex",
            ):
                self.assertTrue((output / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
