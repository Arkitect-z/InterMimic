#!/usr/bin/env python3
"""Aggregate one Raw and one Refined policy trained for each reference.

The statistical unit is the reference.  There is exactly one training run per
condition/reference; K evaluation rollouts are reduced to RefSucc@K and are
not treated as independent training runs.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from summarize_theia_eval import ValidationError, load_paired_manifest


CONDITIONS = ("raw", "full")
REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_VERSION_MANIFEST = REPO_ROOT / "THEIA_POLICY_SERVER_VERSION.json"
SERVER_RELEASE = json.loads(SERVER_VERSION_MANIFEST.read_text())
EMPTY_GIT_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
LABELS = {
    "raw": "Raw MoCap",
    "full": "Measured-tactile refinement",
}
FORMAL_PROTOCOL = {
    "protocol": "single_reference_raw_vs_refined_v2",
    "train_epochs": "3000",
    "training_seed": "0",
    "evaluation_seed": "10000",
    "k_trials": "10",
    "torch_deterministic": "0",
    "train_env_config": (
        "isaacgym/src/intermimic/data/cfg/theia_reference_train.yaml"
    ),
}
METRICS = (
    {
        "key": "ref_success_at_k_percent",
        "label": "RefSucc@10",
        "unit": "%",
        "direction": "higher",
        "digits": 2,
        "primary": True,
    },
    {
        "key": "episode_completion_percent",
        "label": "Episode completion",
        "unit": "%",
        "direction": "higher",
        "digits": 2,
        "primary": False,
    },
    {
        "key": "duration_s",
        "label": "Duration",
        "unit": "s",
        "direction": "higher",
        "digits": 2,
        "primary": True,
    },
    {
        "key": "human_error_cm",
        "label": "E_h",
        "unit": "cm",
        "direction": "lower",
        "digits": 2,
        "primary": True,
    },
    {
        "key": "object_error_cm",
        "label": "E_o",
        "unit": "cm",
        "direction": "lower",
        "digits": 2,
        "primary": True,
    },
)
PAIR_FIELDS = [
    "reference_id",
    "height",
    "variation",
    "action",
    "task",
    "raw_completed_trials",
    "refined_completed_trials",
    "raw_ref_success_at_k",
    "refined_ref_success_at_k",
    "raw_episode_completion_percent",
    "refined_episode_completion_percent",
    "raw_duration_s",
    "refined_duration_s",
    "raw_human_error_cm",
    "refined_human_error_cm",
    "raw_object_error_cm",
    "refined_object_error_cm",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_key_values(path):
    values = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key in values:
            raise ValidationError(
                "{} duplicates key {!r} on line {}".format(
                    path, key, line_number
                )
            )
        values[key] = value
    return values


def parse_reference_list(path):
    references = []
    seen = set()
    for raw_line in Path(path).read_text().splitlines():
        token = raw_line.split("#", 1)[0].strip()
        if not token:
            continue
        reference_id = Path(token.rstrip("/")).name
        if reference_id in seen:
            raise ValidationError(
                "Reference list duplicates {!r}".format(reference_id)
            )
        seen.add(reference_id)
        references.append(reference_id)
    if not references:
        raise ValidationError("Reference list is empty: {}".format(path))
    return references


def parse_int(value, label, minimum=0):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "{} must be an integer, got {!r}".format(label, value)
        ) from exc
    if parsed < minimum:
        raise ValidationError(
            "{} must be >= {}, got {}".format(label, minimum, parsed)
        )
    return parsed


def parse_float(value, label):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "{} must be numeric, got {!r}".format(label, value)
        ) from exc
    if not math.isfinite(parsed):
        raise ValidationError("{} must be finite".format(label))
    return parsed


def validate_repository_receipt(experiment_root):
    receipt_path = Path(experiment_root) / "repository_versions.json"
    if not receipt_path.is_file():
        raise ValidationError(
            "Missing formal repository version receipt: {}".format(
                receipt_path
            )
        )
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("valid") is not True
        or receipt.get("release_name") != SERVER_RELEASE["release_name"]
        or receipt.get("formal_method_status")
        != "only_supported_formal_policy_method"
        or receipt.get("protocol", {}).get("id")
        != SERVER_RELEASE["protocol"]["id"]
    ):
        raise ValidationError(
            "Invalid formal repository version receipt: {}".format(
                receipt_path
            )
        )
    repositories = receipt.get("repositories", {})
    if set(repositories) != {"Theia", "InterMimic", "ProtoMotions"}:
        raise ValidationError(
            "Repository receipt does not cover exactly Theia, InterMimic, "
            "and ProtoMotions"
        )
    for name, report in repositories.items():
        if report.get("valid") is not True:
            raise ValidationError(
                "{} is invalid in {}".format(name, receipt_path)
            )
    intermimic = repositories["InterMimic"]
    expected_tag = SERVER_RELEASE["repositories"]["InterMimic"]["tag"]
    release_commit = intermimic.get("release_tag_commit")
    if (
        intermimic.get("release_tag") != expected_tag
        or not release_commit
        or intermimic.get("head") != release_commit
    ):
        raise ValidationError(
            "InterMimic release tag/commit mismatch in {}".format(
                receipt_path
            )
        )
    return receipt_path, receipt, release_commit


def load_one_result(
    experiment_root,
    reference,
    condition,
    expected_intermimic_commit,
):
    reference_id = reference["reference_id"]
    pair_root = Path(experiment_root) / "references" / reference_id
    pair_spec_path = pair_root / "pair_spec.txt"
    pair_ready_path = pair_root / "PAIR_READY.json"
    if not pair_spec_path.is_file() or not pair_ready_path.is_file():
        raise ValidationError(
            "Incomplete paired result for {}: {}".format(
                reference_id, pair_root
            )
        )

    pair_spec = parse_key_values(pair_spec_path)
    for key, expected in FORMAL_PROTOCOL.items():
        if pair_spec.get(key) != expected:
            raise ValidationError(
                "{} {} must be {!r}, got {!r}".format(
                    pair_spec_path, key, expected, pair_spec.get(key)
                )
            )
    if pair_spec.get("reference_id") != reference_id:
        raise ValidationError("{} reference_id mismatch".format(pair_spec_path))
    if pair_spec.get("git_commit") != expected_intermimic_commit:
        raise ValidationError(
            "{} was not produced by InterMimic commit {}".format(
                pair_spec_path, expected_intermimic_commit
            )
        )
    if pair_spec.get("git_diff_sha256") != EMPTY_GIT_DIFF_SHA256:
        raise ValidationError(
            "{} was produced from a dirty InterMimic worktree".format(
                pair_spec_path
            )
        )

    pair_ready = json.loads(pair_ready_path.read_text())
    if (
        pair_ready.get("protocol") != FORMAL_PROTOCOL["protocol"]
        or pair_ready.get("reference_id") != reference_id
        or int(pair_ready.get("training_runs_per_condition", -1)) != 1
        or int(pair_ready.get("training_seed", -1)) != 0
        or int(pair_ready.get("k_trials", -1)) != 10
    ):
        raise ValidationError(
            "Invalid paired completion receipt: {}".format(pair_ready_path)
        )

    run_root = pair_root / "runs" / condition / "seed_0"
    run_spec = parse_key_values(run_root / "run_spec.txt")
    expected_run = {
        "condition": condition,
        "training_seed": "0",
        "reference_count": "1",
        "bootstrap_epochs": "3000",
        "finetune_epochs": "0",
        "train_env_config": FORMAL_PROTOCOL["train_env_config"],
        "eval_k": "10",
        "eval_seed": "10000",
        "protocol_mode": "single_reference_rebuttal",
        "allow_nonformal_protocol": "0",
    }
    for key, expected in expected_run.items():
        if run_spec.get(key) != expected:
            raise ValidationError(
                "{} {} must be {!r}, got {!r}".format(
                    run_root / "run_spec.txt",
                    key,
                    expected,
                    run_spec.get(key),
                )
            )
    if (
        run_spec.get("git_commit") != expected_intermimic_commit
        or run_spec.get("git_diff_sha256") != EMPTY_GIT_DIFF_SHA256
    ):
        raise ValidationError(
            "{} does not match the clean formal InterMimic release".format(
                run_root / "run_spec.txt"
            )
        )

    evaluation = run_root / "evaluation" / "final"
    validation_path = evaluation / "validation.json"
    reference_path = evaluation / "per_reference.csv"
    validation = json.loads(validation_path.read_text())
    if (
        validation.get("valid") is not True
        or validation.get("condition") != condition
        or int(validation.get("training_seed", -1)) != 0
        or int(validation.get("evaluation_seed", -1)) != 10000
        or int(validation.get("k_trials", -1)) != 10
        or int(validation.get("actual_episodes", -1)) != 10
        or int(validation.get("expected_episodes", -2)) != 10
    ):
        raise ValidationError(
            "Invalid formal evaluation: {}".format(validation_path)
        )
    if validation.get("per_reference_sha256") != sha256(reference_path):
        raise ValidationError(
            "per_reference.csv hash mismatch: {}".format(reference_path)
        )
    with reference_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValidationError(
            "{} must contain exactly one reference row".format(reference_path)
        )
    row = rows[0]
    if (
        row.get("condition") != condition
        or row.get("reference_id") != reference_id
        or parse_int(row.get("training_seed"), "training_seed") != 0
        or parse_int(row.get("evaluation_seed"), "evaluation_seed") != 10000
        or parse_int(row.get("k_trials"), "k_trials", minimum=1) != 10
    ):
        raise ValidationError(
            "Reference metadata mismatch in {}".format(reference_path)
        )
    completed = parse_int(
        row.get("completed_trials"), "completed_trials"
    )
    success = parse_int(row.get("success"), "success")
    if completed > 10 or success not in (0, 1):
        raise ValidationError("Invalid completion values in {}".format(reference_path))
    if success != int(completed > 0):
        raise ValidationError(
            "success disagrees with completed_trials in {}".format(
                reference_path
            )
        )
    return {
        "completed_trials": completed,
        "ref_success_at_k_percent": 100.0 * success,
        "episode_completion_percent": 10.0 * completed,
        "duration_s": parse_float(row.get("duration_s"), "duration_s"),
        "human_error_cm": parse_float(
            row.get("human_error_cm"), "human_error_cm"
        ),
        "object_error_cm": parse_float(
            row.get("object_error_cm"), "object_error_cm"
        ),
    }


def mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def sample_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def bootstrap_ci(raw, refined, samples, seed):
    raw = np.asarray(raw, dtype=np.float64)
    refined = np.asarray(refined, dtype=np.float64)
    if raw.shape != refined.shape or raw.ndim != 1 or not len(raw):
        raise ValidationError("Invalid paired arrays for bootstrap")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(raw), size=(samples, len(raw)))
    raw_means = raw[indices].mean(axis=1)
    refined_means = refined[indices].mean(axis=1)
    delta = refined_means - raw_means
    return {
        "raw_ci95_low": float(np.percentile(raw_means, 2.5)),
        "raw_ci95_high": float(np.percentile(raw_means, 97.5)),
        "refined_ci95_low": float(np.percentile(refined_means, 2.5)),
        "refined_ci95_high": float(np.percentile(refined_means, 97.5)),
        "delta_ci95_low": float(np.percentile(delta, 2.5)),
        "delta_ci95_high": float(np.percentile(delta, 97.5)),
    }


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def format_value(value, digits):
    return ("{:." + str(digits) + "f}").format(value)


def aggregate(
    experiment_root,
    manifest_path,
    output_dir,
    reference_list=None,
    bootstrap_samples=10000,
    bootstrap_seed=20260725,
):
    (
        repository_receipt_path,
        repository_receipt,
        intermimic_commit,
    ) = validate_repository_receipt(experiment_root)
    all_references = load_paired_manifest(manifest_path)
    by_id = {entry["reference_id"]: entry for entry in all_references}
    if reference_list is None:
        selected_ids = sorted(by_id)
    else:
        selected_ids = parse_reference_list(reference_list)
        unknown = sorted(set(selected_ids) - set(by_id))
        if unknown:
            raise ValidationError(
                "Reference list contains IDs absent from the paired manifest: "
                + ", ".join(unknown[:10])
            )
    references = [by_id[reference_id] for reference_id in selected_ids]
    if bootstrap_samples < 1000:
        raise ValidationError("bootstrap_samples must be at least 1000")

    paired_rows = []
    values = {
        condition: {metric["key"]: [] for metric in METRICS}
        for condition in CONDITIONS
    }
    per_reference_values = {}
    for reference in references:
        reference_id = reference["reference_id"]
        condition_results = {
            condition: load_one_result(
                experiment_root,
                reference,
                condition,
                intermimic_commit,
            )
            for condition in CONDITIONS
        }
        per_reference_values[reference_id] = condition_results
        row = {
            "reference_id": reference_id,
            "height": reference.get("height", ""),
            "variation": reference.get("variation", ""),
            "action": reference.get("action") or "",
            "task": reference.get("task") or "",
            "raw_completed_trials": condition_results["raw"][
                "completed_trials"
            ],
            "refined_completed_trials": condition_results["full"][
                "completed_trials"
            ],
            "raw_ref_success_at_k": int(
                condition_results["raw"]["ref_success_at_k_percent"] > 0
            ),
            "refined_ref_success_at_k": int(
                condition_results["full"]["ref_success_at_k_percent"] > 0
            ),
        }
        for condition in CONDITIONS:
            prefix = "raw" if condition == "raw" else "refined"
            for metric in METRICS:
                key = metric["key"]
                values[condition][key].append(
                    condition_results[condition][key]
                )
                if key != "ref_success_at_k_percent":
                    row[prefix + "_" + key] = condition_results[
                        condition
                    ][key]
        paired_rows.append(row)

    metric_rows = []
    for metric_index, metric in enumerate(METRICS):
        key = metric["key"]
        raw = values["raw"][key]
        refined = values["full"][key]
        intervals = bootstrap_ci(
            raw,
            refined,
            bootstrap_samples,
            bootstrap_seed + metric_index,
        )
        delta = mean(refined) - mean(raw)
        orientation = 1.0 if metric["direction"] == "higher" else -1.0
        metric_rows.append({
            "metric": key,
            "label": metric["label"],
            "unit": metric["unit"],
            "direction": metric["direction"],
            "num_references": len(references),
            "raw_mean": mean(raw),
            "raw_std_across_references": sample_std(raw),
            "raw_ci95_low": intervals["raw_ci95_low"],
            "raw_ci95_high": intervals["raw_ci95_high"],
            "refined_mean": mean(refined),
            "refined_std_across_references": sample_std(refined),
            "refined_ci95_low": intervals["refined_ci95_low"],
            "refined_ci95_high": intervals["refined_ci95_high"],
            "delta_refined_minus_raw": delta,
            "delta_ci95_low": intervals["delta_ci95_low"],
            "delta_ci95_high": intervals["delta_ci95_high"],
            "effect_refined_better": orientation * delta,
        })

    pose_rows = []
    for group_key in ("height", "variation"):
        groups = sorted(
            {
                str(reference.get(group_key, "")).strip()
                for reference in references
                if str(reference.get(group_key, "")).strip()
            }
        )
        for group in groups:
            ids = [
                reference["reference_id"]
                for reference in references
                if str(reference.get(group_key, "")).strip() == group
            ]
            for metric in METRICS:
                key = metric["key"]
                raw = [
                    per_reference_values[reference_id]["raw"][key]
                    for reference_id in ids
                ]
                refined = [
                    per_reference_values[reference_id]["full"][key]
                    for reference_id in ids
                ]
                pose_rows.append({
                    "group_type": group_key,
                    "group_value": group,
                    "num_references": len(ids),
                    "metric": key,
                    "unit": metric["unit"],
                    "raw_mean": mean(raw),
                    "refined_mean": mean(refined),
                    "delta_refined_minus_raw": mean(refined) - mean(raw),
                })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_reference_paired.csv", PAIR_FIELDS, paired_rows)
    metric_fields = list(metric_rows[0])
    write_csv(output_dir / "paired_results.csv", metric_fields, metric_rows)
    pose_fields = list(pose_rows[0]) if pose_rows else [
        "group_type",
        "group_value",
        "num_references",
        "metric",
        "unit",
        "raw_mean",
        "refined_mean",
        "delta_refined_minus_raw",
    ]
    write_csv(output_dir / "by_pose.csv", pose_fields, pose_rows)

    summary = {
        "schema_version": 1,
        "protocol": FORMAL_PROTOCOL["protocol"],
        "conditions": {
            "raw": LABELS["raw"],
            "full": LABELS["full"],
        },
        "training_runs_per_reference_per_condition": 1,
        "training_seed": 0,
        "evaluation_seed": 10000,
        "k_trials": 10,
        "num_references": len(references),
        "reference_ids": selected_ids,
        "paired_manifest": str(Path(manifest_path).resolve()),
        "paired_manifest_sha256": sha256(manifest_path),
        "repository_release": repository_receipt["release_name"],
        "repository_versions": {
            "path": str(repository_receipt_path.resolve()),
            "sha256": sha256(repository_receipt_path),
            "intermimic_commit": intermimic_commit,
        },
        "confidence_interval": {
            "method": "paired_nonparametric_bootstrap_over_references",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "does_not_estimate_training_seed_variance": True,
        },
        "metrics": {
            row["metric"]: row for row in metric_rows
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    primary = [
        (metric, row)
        for metric, row in zip(METRICS, metric_rows)
        if metric["primary"]
    ]
    latex_header = (
        "Demonstrations & RefSucc@10 (\\%) $\\uparrow$ "
        "& Duration (s) $\\uparrow$ "
        "& $E_h$ (cm) $\\downarrow$ & $E_o$ (cm) $\\downarrow$ \\\\\n"
    )
    latex_rows = []
    for condition in CONDITIONS:
        label = LABELS[condition]
        values_text = []
        for metric, _ in primary:
            values_text.append(
                format_value(
                    mean(values[condition][metric["key"]]),
                    metric["digits"],
                )
            )
        latex_rows.append(
            "{} & {} \\\\\n".format(label, " & ".join(values_text))
        )
    (output_dir / "main_table.tex").write_text(
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        + latex_header
        + "\\midrule\n"
        + "".join(latex_rows)
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    md_lines = [
        "# Theia S1 per-reference Raw vs. Refined policy results",
        "",
        (
            "Each reference was trained exactly once per condition with "
            "training seed 0. K=10 denotes parallel evaluation rollouts, "
            "not ten training runs."
        ),
        "",
        "| Metric | Raw | Refined | Refined − Raw | Paired 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, row in zip(METRICS, metric_rows):
        digits = metric["digits"]
        md_lines.append(
            "| {label} ({unit}) | {raw} | {refined} | {delta} | "
            "[{low}, {high}] |".format(
                label=metric["label"],
                unit=metric["unit"],
                raw=format_value(row["raw_mean"], digits),
                refined=format_value(row["refined_mean"], digits),
                delta=format_value(
                    row["delta_refined_minus_raw"], digits
                ),
                low=format_value(row["delta_ci95_low"], digits),
                high=format_value(row["delta_ci95_high"], digits),
            )
        )
    md_lines.extend([
        "",
        (
            "The confidence interval is a paired non-parametric bootstrap "
            "over references. With one training run per condition/reference, "
            "it measures sequence-level uncertainty, not training-seed "
            "variance."
        ),
        "",
        (
            "Primary InterMimic-style metrics are RefSucc@10, Duration, "
            "$E_h$, and $E_o$. Episode completion is retained as a secondary "
            "rollout-reliability diagnostic."
        ),
        "",
    ])
    (output_dir / "results.md").write_text("\n".join(md_lines))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--reference-list")
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        Path(args.experiment_root) / "results"
    )
    summary = aggregate(
        experiment_root=Path(args.experiment_root),
        manifest_path=Path(args.pair_manifest),
        output_dir=Path(output_dir),
        reference_list=(
            Path(args.reference_list) if args.reference_list else None
        ),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        "Aggregated {} paired references: {}".format(
            summary["num_references"],
            Path(output_dir).resolve(),
        )
    )


if __name__ == "__main__":
    main()
