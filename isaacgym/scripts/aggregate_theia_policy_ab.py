#!/usr/bin/env python3
"""Aggregate the preregistered Raw-vs-Full Theia policy experiment.

Expected layout::

    EXPERIMENT_ROOT/
      raw/seed_0/evaluation/final/per_reference.csv
      ...
      full/seed_3/evaluation/final/per_reference.csv

Every input must have been produced by ``summarize_theia_eval.py``.  Raw and
Full are paired by training seed and canonical reference id.  The confidence
interval resamples training seeds and references as crossed paired clusters;
the K technical trials have already been reduced to one reference-level
observation and are never treated as independent samples.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from summarize_theia_eval import ValidationError, load_paired_manifest


CONDITIONS = ("raw", "full")
CONDITION_LABELS = {
    "raw": "Raw MoCap (pre-refinement)",
    "full": "Measured-tactile full refinement",
}
DEFAULT_SEEDS = (0, 1, 2, 3)

METRICS = (
    {
        "key": "success",
        "output": "success_rate_percent",
        "label": "Succ.",
        "unit": "%",
        "direction": "higher",
        "scale": 100.0,
        "digits": 2,
    },
    {
        "key": "duration_s",
        "output": "duration_s",
        "label": "Duration",
        "unit": "s",
        "direction": "higher",
        "scale": 1.0,
        "digits": 2,
    },
    {
        "key": "human_error_cm",
        "output": "human_error_cm",
        "label": "E_h",
        "unit": "cm",
        "direction": "lower",
        "scale": 1.0,
        "digits": 2,
    },
    {
        "key": "object_error_cm",
        "output": "object_error_cm",
        "label": "E_o",
        "unit": "cm",
        "direction": "lower",
        "scale": 1.0,
        "digits": 2,
    },
)

REQUIRED_REFERENCE_COLUMNS = {
    "condition",
    "training_seed",
    "evaluation_seed",
    "reference_id",
    "filename",
    "height",
    "variation",
    "k_trials",
    "completed_trials",
    "success",
    "best_trial_id",
    "best_env_id",
    "best_steps",
    "duration_s",
    "human_error_cm",
    "object_error_cm",
}

PER_REFERENCE_ALL_FIELDS = [
    "condition",
    "training_seed",
    "evaluation_seed",
    "reference_id",
    "filename",
    "height",
    "variation",
    "k_trials",
    "completed_trials",
    "success",
    "best_trial_id",
    "best_env_id",
    "best_steps",
    "duration_s",
    "human_error_cm",
    "object_error_cm",
]

PER_SEED_FIELDS = [
    "condition",
    "training_seed",
    "evaluation_seed",
    "num_references",
    "k_trials",
    "success_rate_percent",
    "duration_s",
    "human_error_cm",
    "object_error_cm",
]

PAIRED_FIELDS = [
    "metric",
    "unit",
    "direction",
    "raw_mean",
    "raw_std",
    "full_mean",
    "full_std",
    "delta_full_minus_raw",
    "ci95_low",
    "ci95_high",
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_int(value, label, minimum=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("{} must be integer, got {!r}".format(
            label, value
        )) from exc
    if minimum is not None and parsed < minimum:
        raise ValidationError(
            "{} must be >= {}, got {}".format(label, minimum, parsed)
        )
    return parsed


def _as_float(value, label, minimum=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("{} must be numeric, got {!r}".format(
            label, value
        )) from exc
    if not math.isfinite(parsed):
        raise ValidationError("{} must be finite, got {!r}".format(
            label, value
        ))
    if minimum is not None and parsed < minimum:
        raise ValidationError(
            "{} must be >= {}, got {}".format(label, minimum, parsed)
        )
    return parsed


def _canonical_height(value):
    value = str(value).strip().upper()
    match = re.fullmatch(r"L0?([1-5])", value)
    if match:
        return "L" + match.group(1)
    match = re.fullmatch(r"L([1-5])\1", value)
    if match:
        return "L" + match.group(1)
    raise ValidationError(
        "S1 height must be L1-L5 (L11-L55 also accepted), got {!r}".format(
            value
        )
    )


def _canonical_variation(value):
    value = str(value).strip().upper()
    match = re.fullmatch(r"V0?([1-3])", value)
    if not match:
        raise ValidationError(
            "S1 variation must be V1-V3, got {!r}".format(value)
        )
    return "V" + match.group(1)


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(
            "Cannot read {} {}: {}".format(label, path, exc)
        ) from exc


def _read_record_file(path, label, allow_other=False):
    """Read key=value metadata plus trailing sha256sum records."""
    path = Path(path)
    metadata = {}
    file_hashes = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValidationError(
            "Cannot read {} {}: {}".format(label, path, exc)
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if separator and re.fullmatch(r"[A-Za-z0-9_]+", key):
            if key in metadata:
                raise ValidationError(
                    "{} {} duplicates key {!r}".format(
                        label, path, key
                    )
                )
            metadata[key] = value
            continue
        fields = line.split(maxsplit=1)
        if (
            len(fields) == 2
            and re.fullmatch(r"[0-9a-fA-F]{64}", fields[0])
        ):
            filename = fields[1].lstrip("*").strip()
            if filename in file_hashes:
                raise ValidationError(
                    "{} {} duplicates hash path {!r}".format(
                        label, path, filename
                    )
                )
            file_hashes[filename] = fields[0].lower()
            continue
        if not allow_other:
            raise ValidationError(
                "{} {} has an unrecognized line {}: {!r}".format(
                    label, path, line_number, line
                )
            )
    return metadata, file_hashes


def _condition_data_sha256(manifest_references, condition):
    records = sorted(
        (
            entry[condition + "_filename"],
            entry[condition + "_sha256"],
        )
        for entry in manifest_references
    )
    payload = "".join(
        "{}  {}\n".format(digest, filename)
        for filename, digest in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_run_spec(
    run_root,
    condition,
    training_seed,
    expected_manifest_path,
    expected_references,
    expected_data_sha,
):
    path = Path(run_root) / "run_spec.txt"
    metadata, file_hashes = _read_record_file(path, "run specification")
    required = {
        "condition",
        "training_seed",
        "condition_data_sha256",
        "pair_manifest",
        "reference_count",
        "target_envs",
        "num_envs",
        "minibatch_size",
        "bootstrap_epochs",
        "finetune_epochs",
        "eval_k",
        "eval_seed",
        "evaluation_fps",
        "allow_nonformal_protocol",
        "git_commit",
        "git_diff_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValidationError(
            "{} is missing keys {}".format(path, missing)
        )
    if metadata["condition"] != condition:
        raise ValidationError(
            "{} condition is {!r}, expected {!r}".format(
                path, metadata["condition"], condition
            )
        )
    if _as_int(
        metadata["training_seed"], str(path) + " training_seed", minimum=0
    ) != training_seed:
        raise ValidationError("{} has the wrong training seed".format(path))
    fixed_integer_fields = {
        "bootstrap_epochs": 20000,
        "finetune_epochs": 2000,
        "eval_k": 10,
        "eval_seed": 10000 + training_seed,
        "allow_nonformal_protocol": 0,
    }
    for key, expected in fixed_integer_fields.items():
        if _as_int(metadata[key], "{} {}".format(path, key)) != expected:
            raise ValidationError(
                "{} must have {}={}, got {}".format(
                    path, key, expected, metadata[key]
                )
            )
    if not math.isclose(
        _as_float(metadata["evaluation_fps"], str(path) + " evaluation_fps"),
        30.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValidationError("{} evaluation_fps must be 30".format(path))
    if Path(metadata["pair_manifest"]).resolve() != Path(
        expected_manifest_path
    ).resolve():
        raise ValidationError(
            "{} points to a different paired manifest".format(path)
        )
    reference_count = _as_int(
        metadata["reference_count"],
        str(path) + " reference_count",
        minimum=1,
    )
    if reference_count != expected_references:
        raise ValidationError(
            "{} reference_count is {}, expected {}".format(
                path, reference_count, expected_references
            )
        )
    num_envs = _as_int(
        metadata["num_envs"], str(path) + " num_envs", minimum=1
    )
    minibatch = _as_int(
        metadata["minibatch_size"],
        str(path) + " minibatch_size",
        minimum=1,
    )
    _as_int(metadata["target_envs"], str(path) + " target_envs", minimum=1)
    if num_envs % reference_count:
        raise ValidationError(
            "{} num_envs is not reference-balanced".format(path)
        )
    if (num_envs * 32) % minibatch or minibatch % 4:
        raise ValidationError(
            "{} has an invalid PPO batch/minibatch shape".format(path)
        )
    for key in ("condition_data_sha256", "git_diff_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", metadata[key]):
            raise ValidationError("{} has invalid {}".format(path, key))
    if metadata["condition_data_sha256"] != expected_data_sha:
        raise ValidationError(
            "{} condition data fingerprint differs from the frozen manifest".format(
                path
            )
        )
    if not metadata["git_commit"].strip():
        raise ValidationError("{} has empty git_commit".format(path))
    if not file_hashes:
        raise ValidationError("{} contains no source file hashes".format(path))
    return {
        "path": str(path.resolve()),
        "metadata": metadata,
        "file_hashes": file_hashes,
    }


def _load_formal_run(
    path,
    condition,
    training_seed,
    expected_manifest_sha,
    expected_data_sha,
):
    path = Path(path)
    validation_path = path.with_name("validation.json")
    formal_summary_path = path.with_name("summary.json")
    episodes_path = path.with_name("episodes.csv")
    evaluation_manifest_path = path.with_name("manifest.txt")
    validation = _read_json(validation_path, "validation JSON")
    formal_summary = _read_json(formal_summary_path, "formal summary JSON")

    if validation.get("valid") is not True:
        raise ValidationError("{} does not declare valid=true".format(
            validation_path
        ))
    if _as_int(
        validation.get("actual_episodes"),
        str(validation_path) + " actual_episodes",
        minimum=1,
    ) != _as_int(
        validation.get("expected_episodes"),
        str(validation_path) + " expected_episodes",
        minimum=1,
    ):
        raise ValidationError(
            "{} does not contain a complete evaluation cohort".format(
                validation_path
            )
        )
    for document, document_path in (
        (validation, validation_path),
        (formal_summary, formal_summary_path),
    ):
        if document.get("condition") != condition:
            raise ValidationError(
                "{} condition is {!r}, expected {!r}".format(
                    document_path, document.get("condition"), condition
                )
            )
        if _as_int(
            document.get("training_seed"),
            str(document_path) + " training_seed",
        ) != training_seed:
            raise ValidationError(
                "{} has the wrong training seed".format(document_path)
            )
    if validation.get("paired_manifest_sha256") != expected_manifest_sha:
        raise ValidationError(
            "{} was produced from a different paired manifest".format(
                validation_path
            )
        )
    artifact_hashes = (
        (episodes_path, "episodes_sha256"),
        (path, "per_reference_sha256"),
        (formal_summary_path, "summary_sha256"),
        (evaluation_manifest_path, "evaluation_manifest_sha256"),
    )
    for artifact_path, hash_key in artifact_hashes:
        expected_hash = validation.get(hash_key)
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)):
            raise ValidationError(
                "{} lacks a valid {}".format(validation_path, hash_key)
            )
        try:
            actual_hash = _sha256(artifact_path)
        except OSError as exc:
            raise ValidationError(
                "Cannot hash formal artifact {}: {}".format(
                    artifact_path, exc
                )
            ) from exc
        if actual_hash != expected_hash:
            raise ValidationError(
                "{} hash does not match {} in {}".format(
                    artifact_path, hash_key, validation_path
                )
            )

    eval_metadata, _ = _read_record_file(
        evaluation_manifest_path,
        "evaluation manifest",
        allow_other=True,
    )
    required_eval_metadata = {
        "checkpoint_sha256",
        "eval_seed",
        "training_seed",
        "condition",
        "condition_data_sha256",
        "evaluation_pipeline_sha256",
        "evaluation_config",
    }
    missing_eval = sorted(required_eval_metadata - set(eval_metadata))
    if missing_eval:
        raise ValidationError(
            "{} is missing keys {}".format(
                evaluation_manifest_path, missing_eval
            )
        )
    if eval_metadata["condition"] != condition:
        raise ValidationError(
            "{} has the wrong condition".format(evaluation_manifest_path)
        )
    if _as_int(
        eval_metadata["training_seed"],
        str(evaluation_manifest_path) + " training_seed",
    ) != training_seed:
        raise ValidationError(
            "{} has the wrong training seed".format(evaluation_manifest_path)
        )
    if _as_int(
        eval_metadata["eval_seed"],
        str(evaluation_manifest_path) + " eval_seed",
    ) != 10000 + training_seed:
        raise ValidationError(
            "{} has the wrong evaluation seed".format(
                evaluation_manifest_path
            )
        )
    if eval_metadata["condition_data_sha256"] != expected_data_sha:
        raise ValidationError(
            "{} has the wrong condition data fingerprint".format(
                evaluation_manifest_path
            )
        )
    if (
        eval_metadata["evaluation_config"]
        != "isaacgym/src/intermimic/data/cfg/theia_policy_eval.yaml"
    ):
        raise ValidationError(
            "{} has the wrong evaluation config".format(
                evaluation_manifest_path
            )
        )
    for key in ("checkpoint_sha256", "evaluation_pipeline_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", eval_metadata[key]):
            raise ValidationError(
                "{} has invalid {}".format(evaluation_manifest_path, key)
            )

    try:
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValidationError("{} has no CSV header".format(path))
            missing = REQUIRED_REFERENCE_COLUMNS.difference(reader.fieldnames)
            if missing:
                raise ValidationError(
                    "{} is missing columns {}".format(path, sorted(missing))
                )
            source_rows = list(reader)
    except OSError as exc:
        raise ValidationError("Cannot read {}: {}".format(path, exc)) from exc
    if not source_rows:
        raise ValidationError("{} contains no references".format(path))

    rows = {}
    evaluation_seeds = set()
    k_values = set()
    for line_number, row in enumerate(source_rows, start=2):
        label = "{} row {}".format(path, line_number)
        if row["condition"] != condition:
            raise ValidationError(
                "{} condition is {!r}, expected {!r}".format(
                    label, row["condition"], condition
                )
            )
        if _as_int(
            row["training_seed"], label + " training_seed"
        ) != training_seed:
            raise ValidationError("{} has the wrong training seed".format(
                label
            ))
        reference_id = row["reference_id"].strip()
        if not reference_id:
            raise ValidationError("{} has empty reference_id".format(label))
        if reference_id in rows:
            raise ValidationError(
                "{} contains duplicate reference_id {}".format(
                    path, reference_id
                )
            )
        evaluation_seed = _as_int(
            row["evaluation_seed"], label + " evaluation_seed", minimum=0
        )
        k_trials = _as_int(
            row["k_trials"], label + " k_trials", minimum=1
        )
        completed_trials = _as_int(
            row["completed_trials"],
            label + " completed_trials",
            minimum=0,
        )
        if completed_trials > k_trials:
            raise ValidationError(
                "{} completed_trials exceeds k_trials".format(label)
            )
        success = _as_int(row["success"], label + " success")
        if success not in (0, 1):
            raise ValidationError("{} success must be 0 or 1".format(label))
        if success != int(completed_trials > 0):
            raise ValidationError(
                "{} success is inconsistent with completed_trials".format(
                    label
                )
            )

        parsed = {
            "condition": condition,
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
            "reference_id": reference_id,
            "filename": Path(row["filename"]).name,
            "height": row["height"].strip(),
            "variation": row["variation"].strip(),
            "k_trials": k_trials,
            "completed_trials": completed_trials,
            "success": success,
            "best_trial_id": _as_int(
                row["best_trial_id"],
                label + " best_trial_id",
                minimum=0,
            ),
            "best_env_id": _as_int(
                row["best_env_id"], label + " best_env_id", minimum=0
            ),
            "best_steps": _as_int(
                row["best_steps"], label + " best_steps", minimum=0
            ),
            "duration_s": _as_float(
                row["duration_s"], label + " duration_s", minimum=0.0
            ),
            "human_error_cm": _as_float(
                row["human_error_cm"],
                label + " human_error_cm",
                minimum=0.0,
            ),
            "object_error_cm": _as_float(
                row["object_error_cm"],
                label + " object_error_cm",
                minimum=0.0,
            ),
        }
        if parsed["best_trial_id"] >= k_trials:
            raise ValidationError(
                "{} best_trial_id must be below k_trials".format(label)
            )
        rows[reference_id] = parsed
        evaluation_seeds.add(evaluation_seed)
        k_values.add(k_trials)

    if len(evaluation_seeds) != 1:
        raise ValidationError(
            "{} mixes evaluation seeds {}".format(path, evaluation_seeds)
        )
    if len(k_values) != 1:
        raise ValidationError("{} mixes K values {}".format(path, k_values))
    if _as_int(
        validation.get("k_trials"),
        str(validation_path) + " k_trials",
    ) != next(iter(k_values)):
        raise ValidationError(
            "{} K disagrees with per_reference.csv".format(validation_path)
        )
    return rows, validation, eval_metadata


def _format_number(value, digits):
    return ("{:.%df}" % digits).format(value)


def _write_main_table(path, condition_stats):
    lines = [
        "% Generated by aggregate_theia_policy_ab.py; do not hand-edit.",
        "\\begin{table}[t]",
        "  \\caption{\\textbf{Downstream policy learnability on paired S1 references.} "
        "We report the four InterMimic metrics using exactly $K=10$ trials per "
        "reference. Entries are mean $\\pm$ standard deviation over four "
        "independent policy-training seeds.}",
        "  \\label{tab:policy_main}",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    Demonstrations & Succ. (\\%)$\\uparrow$ & Duration (s)$\\uparrow$ & "
        "$E_h$ (cm)$\\downarrow$ & $E_o$ (cm)$\\downarrow$ \\\\",
        "    \\midrule",
    ]
    for condition in CONDITIONS:
        cells = []
        for metric in METRICS:
            stats = condition_stats[condition][metric["output"]]
            cells.append(
                "{} $\\pm$ {}".format(
                    _format_number(stats["mean"], metric["digits"]),
                    _format_number(stats["std"], metric["digits"]),
                )
            )
        lines.append(
            "    {} & {} \\\\".format(
                CONDITION_LABELS[condition], " & ".join(cells)
            )
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    Path(path).write_text("\n".join(lines))


def _spatial_cell(values, reference_indices, metric_index):
    if not reference_indices:
        return None
    subset = values[:, reference_indices, :]
    seed_metrics = subset.mean(axis=1)
    return seed_metrics.mean(axis=0)


def _write_spatial_table(
    path,
    raw_values,
    full_values,
    reference_records,
    metric_index,
):
    height_labels = {
        "L1": "L1 (Floor)",
        "L2": "L2 (Knee)",
        "L3": "L3 (Waist)",
        "L4": "L4 (Chest)",
        "L5": "L5 (Neck)",
    }
    canonical = [
        (
            _canonical_height(record["height"]),
            _canonical_variation(record["variation"]),
        )
        for record in reference_records
    ]
    lines = [
        "% Generated by aggregate_theia_policy_ab.py; do not hand-edit.",
        "\\begin{table*}[t]",
        "  \\caption{\\textbf{InterMimic policy performance across S1 spatial "
        "configurations.} Each cell reports Raw / Full using the same four "
        "InterMimic metrics and the same paired references. Values are averaged "
        "over independent policy-training seeds.}",
        "  \\label{tab:policy_s1_spatial}",
        "  \\centering",
        "  \\footnotesize",
        "  \\begin{tabular}{@{}llrcccc@{}}",
        "    \\toprule",
        "    Height & Var. & $N$ & Succ. (\\%)$\\uparrow$ & "
        "Duration (s)$\\uparrow$ & $E_h$ (cm)$\\downarrow$ & "
        "$E_o$ (cm)$\\downarrow$ \\\\",
        "    & & & Raw / Full & Raw / Full & Raw / Full & Raw / Full \\\\",
        "    \\midrule",
    ]
    for height in ("L1", "L2", "L3", "L4", "L5"):
        for variation in ("V1", "V2", "V3"):
            indices = [
                index
                for index, group in enumerate(canonical)
                if group == (height, variation)
            ]
            raw_cell = _spatial_cell(
                raw_values, indices, metric_index
            )
            full_cell = _spatial_cell(
                full_values, indices, metric_index
            )
            if raw_cell is None:
                cells = ["-- / --"] * len(METRICS)
            else:
                cells = [
                    "{} / {}".format(
                        _format_number(
                            raw_cell[index], metric["digits"]
                        ),
                        _format_number(
                            full_cell[index], metric["digits"]
                        ),
                    )
                    for index, metric in enumerate(METRICS)
                ]
            lines.append(
                "    {} & {} & {} & {} \\\\".format(
                    height_labels[height],
                    variation,
                    len(indices),
                    " & ".join(cells),
                )
            )
        lines.append("    \\addlinespace")

    all_indices = list(range(len(reference_records)))
    raw_all = _spatial_cell(raw_values, all_indices, metric_index)
    full_all = _spatial_cell(full_values, all_indices, metric_index)
    all_cells = [
        "{} / {}".format(
            _format_number(raw_all[index], metric["digits"]),
            _format_number(full_all[index], metric["digits"]),
        )
        for index, metric in enumerate(METRICS)
    ]
    lines.extend([
        "    \\midrule",
        "    \\multicolumn{{2}}{{l}}{{All S1}} & {} & {} \\\\".format(
            len(reference_records), " & ".join(all_cells)
        ),
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
        "",
    ])
    Path(path).write_text("\n".join(lines))


def _write_results_markdown(
    path,
    condition_stats,
    paired_summary,
    num_references,
    k_trials,
    seeds,
    bootstrap_samples,
    bootstrap_seed,
):
    lines = [
        "# Theia Raw-vs-Full policy results",
        "",
        "Machine-generated formal InterMimic evaluation summary.",
        "",
        "- References: **{}**".format(num_references),
        "- Trials per reference: **K={}**".format(k_trials),
        "- Independent policy-training seeds: **{}**".format(
            ", ".join(str(seed) for seed in seeds)
        ),
        "- Confidence interval: crossed paired percentile bootstrap, "
        "{} samples, RNG seed {}".format(
            bootstrap_samples, bootstrap_seed
        ),
        "- Pairing unit: training seed and reference ID; technical trials "
        "are not treated as independent samples.",
        "",
        "| Metric | Direction | Raw mean ± std | Full mean ± std | "
        "Full − Raw | Paired 95% CI |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        output_key = metric["output"]
        digits = metric["digits"]
        raw_stats = condition_stats["raw"][output_key]
        full_stats = condition_stats["full"][output_key]
        paired = paired_summary[output_key]
        unit = metric["unit"]
        label = "{} ({})".format(metric["label"], unit)
        direction = "↑" if metric["direction"] == "higher" else "↓"
        lines.append(
            "| {label} | {direction} | {raw_mean} ± {raw_std} | "
            "{full_mean} ± {full_std} | {delta} | [{low}, {high}] |".format(
                label=label,
                direction=direction,
                raw_mean=_format_number(raw_stats["mean"], digits),
                raw_std=_format_number(raw_stats["std"], digits),
                full_mean=_format_number(full_stats["mean"], digits),
                full_std=_format_number(full_stats["std"], digits),
                delta=_format_number(
                    paired["delta_full_minus_raw"], digits
                ),
                low=_format_number(paired["ci95"][0], digits),
                high=_format_number(paired["ci95"][1], digits),
            )
        )
    lines.extend([
        "",
        "The delta is always reported as Full minus Raw. Therefore, positive "
        "is favorable for Succ./Duration and negative is favorable for "
        "tracking errors.",
        "",
        "## Artifacts",
        "",
        "- [All reference-level results](per_reference_all.csv)",
        "- [Per-seed results](per_seed.csv)",
        "- [Paired numerical comparison](paired_results.csv)",
        "- [Bootstrap metadata and intervals](bootstrap_results.json)",
        "- [Complete machine-readable summary](summary.json)",
        "- [Main-paper LaTeX table](main_table.tex)",
        "- [S1 spatial LaTeX table](spatial_table.tex)",
        "",
    ])
    Path(path).write_text("\n".join(lines))


def aggregate_experiment(
    experiment_root,
    manifest_path,
    output_dir,
    seeds=DEFAULT_SEEDS,
    bootstrap_samples=10000,
    bootstrap_seed=20260724,
):
    """Validate all formal runs and generate statistics and paper tables."""
    experiment_root = Path(experiment_root)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    seeds = tuple(_as_int(seed, "seed", minimum=0) for seed in seeds)
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValidationError(
            "Formal experiment requires four distinct training seeds"
        )
    bootstrap_samples = _as_int(
        bootstrap_samples, "bootstrap_samples", minimum=1
    )
    bootstrap_seed = _as_int(
        bootstrap_seed, "bootstrap_seed", minimum=0
    )

    manifest_references = load_paired_manifest(manifest_path)
    manifest_sha = _sha256(manifest_path)
    manifest_by_id = {
        record["reference_id"]: record for record in manifest_references
    }
    reference_ids = sorted(manifest_by_id)
    expected_set = set(reference_ids)
    expected_data_sha = {
        condition: _condition_data_sha256(
            manifest_references, condition
        )
        for condition in CONDITIONS
    }

    runs = {}
    validations = {}
    run_specs = {}
    evaluation_metadata = {}
    all_rows = []
    k_values = set()
    for condition in CONDITIONS:
        for seed in seeds:
            path = (
                experiment_root
                / condition
                / "seed_{}".format(seed)
                / "evaluation"
                / "final"
                / "per_reference.csv"
            )
            run_root = path.parents[2]
            run_spec = _load_run_spec(
                run_root=run_root,
                condition=condition,
                training_seed=seed,
                expected_manifest_path=manifest_path,
                expected_references=len(reference_ids),
                expected_data_sha=expected_data_sha[condition],
            )
            rows, validation, eval_metadata = _load_formal_run(
                path,
                condition,
                seed,
                manifest_sha,
                expected_data_sha[condition],
            )
            observed_set = set(rows)
            if observed_set != expected_set:
                raise ValidationError(
                    "{} reference set differs from the paired manifest; "
                    "missing={}, extra={}".format(
                        path,
                        sorted(expected_set - observed_set),
                        sorted(observed_set - expected_set),
                    )
                )
            for reference_id in reference_ids:
                row = rows[reference_id]
                manifest_record = manifest_by_id[reference_id]
                expected_filename = manifest_record[
                    condition + "_filename"
                ]
                if row["filename"] != expected_filename:
                    raise ValidationError(
                        "{} {} filename is {!r}, expected {!r}".format(
                            condition,
                            reference_id,
                            row["filename"],
                            expected_filename,
                        )
                    )
                if (
                    row["height"] != str(manifest_record["height"]).strip()
                    or row["variation"]
                    != str(manifest_record["variation"]).strip()
                ):
                    raise ValidationError(
                        "{} {} spatial metadata disagrees with manifest".format(
                            condition, reference_id
                        )
                    )
                k_values.add(row["k_trials"])
                all_rows.append(row)
            runs[(condition, seed)] = rows
            validations[(condition, seed)] = validation
            run_specs[(condition, seed)] = run_spec
            evaluation_metadata[(condition, seed)] = eval_metadata

    if len(k_values) != 1:
        raise ValidationError(
            "Formal runs use inconsistent K values: {}".format(k_values)
        )
    k_trials = next(iter(k_values))
    if k_trials != 10:
        raise ValidationError(
            "Formal paper protocol requires K=10, got K={}".format(k_trials)
        )

    comparable_spec_keys = (
        "pair_manifest",
        "reference_count",
        "target_envs",
        "num_envs",
        "minibatch_size",
        "bootstrap_epochs",
        "finetune_epochs",
        "eval_k",
        "evaluation_fps",
        "allow_nonformal_protocol",
        "git_commit",
        "git_diff_sha256",
    )
    baseline_key = ("raw", seeds[0])
    baseline_spec = run_specs[baseline_key]
    for key, spec in run_specs.items():
        for field in comparable_spec_keys:
            if (
                spec["metadata"][field]
                != baseline_spec["metadata"][field]
            ):
                raise ValidationError(
                    "Run specifications disagree on {}: {} has {!r}, "
                    "{} has {!r}".format(
                        field,
                        baseline_key,
                        baseline_spec["metadata"][field],
                        key,
                        spec["metadata"][field],
                    )
                )
        if spec["file_hashes"] != baseline_spec["file_hashes"]:
            raise ValidationError(
                "Run specifications use different source-file hashes: "
                "{} versus {}".format(baseline_key, key)
            )

    evaluation_pipelines = {
        metadata["evaluation_pipeline_sha256"]
        for metadata in evaluation_metadata.values()
    }
    if len(evaluation_pipelines) != 1:
        raise ValidationError(
            "Formal runs use different evaluation pipeline fingerprints"
        )

    for seed in seeds:
        raw_eval_seed = runs[("raw", seed)][reference_ids[0]][
            "evaluation_seed"
        ]
        full_eval_seed = runs[("full", seed)][reference_ids[0]][
            "evaluation_seed"
        ]
        if raw_eval_seed != full_eval_seed:
            raise ValidationError(
                "Raw/Full evaluation seeds differ for training seed {}: "
                "{} vs {}".format(seed, raw_eval_seed, full_eval_seed)
            )
        for condition in CONDITIONS:
            spec_eval_seed = _as_int(
                run_specs[(condition, seed)]["metadata"]["eval_seed"],
                "{} seed {} run-spec eval_seed".format(condition, seed),
            )
            if spec_eval_seed != raw_eval_seed:
                raise ValidationError(
                    "{} seed {} evaluation seed differs from its run spec".format(
                        condition, seed
                    )
                )

    metric_index = {
        metric["key"]: index for index, metric in enumerate(METRICS)
    }
    values = {}
    for condition in CONDITIONS:
        values[condition] = np.asarray([
            [
                [
                    runs[(condition, seed)][reference_id][metric["key"]]
                    * metric["scale"]
                    for metric in METRICS
                ]
                for reference_id in reference_ids
            ]
            for seed in seeds
        ], dtype=np.float64)
        if not np.isfinite(values[condition]).all():
            raise ValidationError(
                "{} contains non-finite metric values".format(condition)
            )

    seed_metrics = {
        condition: values[condition].mean(axis=1)
        for condition in CONDITIONS
    }
    condition_stats = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for metric_id, metric in enumerate(METRICS):
            per_seed = seed_metrics[condition][:, metric_id]
            condition_stats[condition][metric["output"]] = {
                "mean": float(per_seed.mean()),
                "std": float(per_seed.std(ddof=1)),
                "per_seed": [float(value) for value in per_seed],
            }

    point_delta = (
        values["full"].mean(axis=(0, 1))
        - values["raw"].mean(axis=(0, 1))
    )
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_delta = np.empty(
        (bootstrap_samples, len(METRICS)), dtype=np.float64
    )
    num_seeds = len(seeds)
    num_references = len(reference_ids)
    for sample_id in range(bootstrap_samples):
        sampled_seeds = rng.integers(
            0, num_seeds, size=num_seeds
        )
        sampled_references = rng.integers(
            0, num_references, size=num_references
        )
        raw_sample = values["raw"][sampled_seeds][
            :, sampled_references, :
        ].mean(axis=(0, 1))
        full_sample = values["full"][sampled_seeds][
            :, sampled_references, :
        ].mean(axis=(0, 1))
        bootstrap_delta[sample_id] = full_sample - raw_sample
    ci_low, ci_high = np.percentile(
        bootstrap_delta, [2.5, 97.5], axis=0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows.sort(
        key=lambda row: (
            CONDITIONS.index(row["condition"]),
            row["training_seed"],
            row["reference_id"],
        )
    )
    _write_csv(
        output_dir / "per_reference_all.csv",
        PER_REFERENCE_ALL_FIELDS,
        all_rows,
    )

    per_seed_rows = []
    for condition in CONDITIONS:
        for seed_index, seed in enumerate(seeds):
            first = runs[(condition, seed)][reference_ids[0]]
            per_seed_rows.append({
                "condition": condition,
                "training_seed": seed,
                "evaluation_seed": first["evaluation_seed"],
                "num_references": num_references,
                "k_trials": k_trials,
                "success_rate_percent": seed_metrics[condition][
                    seed_index, metric_index["success"]
                ],
                "duration_s": seed_metrics[condition][
                    seed_index, metric_index["duration_s"]
                ],
                "human_error_cm": seed_metrics[condition][
                    seed_index, metric_index["human_error_cm"]
                ],
                "object_error_cm": seed_metrics[condition][
                    seed_index, metric_index["object_error_cm"]
                ],
            })
    _write_csv(
        output_dir / "per_seed.csv", PER_SEED_FIELDS, per_seed_rows
    )

    paired_rows = []
    paired_summary = {}
    for metric_id, metric in enumerate(METRICS):
        raw_stats = condition_stats["raw"][metric["output"]]
        full_stats = condition_stats["full"][metric["output"]]
        paired_rows.append({
            "metric": metric["output"],
            "unit": metric["unit"],
            "direction": metric["direction"],
            "raw_mean": raw_stats["mean"],
            "raw_std": raw_stats["std"],
            "full_mean": full_stats["mean"],
            "full_std": full_stats["std"],
            "delta_full_minus_raw": float(point_delta[metric_id]),
            "ci95_low": float(ci_low[metric_id]),
            "ci95_high": float(ci_high[metric_id]),
        })
        paired_summary[metric["output"]] = {
            "unit": metric["unit"],
            "direction": metric["direction"],
            "delta_full_minus_raw": float(point_delta[metric_id]),
            "ci95": [
                float(ci_low[metric_id]),
                float(ci_high[metric_id]),
            ],
        }
    _write_csv(
        output_dir / "paired_results.csv", PAIRED_FIELDS, paired_rows
    )

    bootstrap_results = {
        "schema_version": 1,
        "method": "crossed_paired_percentile_bootstrap",
        "pairing": ["training_seed", "reference_id"],
        "technical_trials_resampled": False,
        "samples": bootstrap_samples,
        "random_seed": bootstrap_seed,
        "confidence_level": 0.95,
        "metrics": paired_summary,
    }
    (output_dir / "bootstrap_results.json").write_text(
        json.dumps(bootstrap_results, indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "schema_version": 2,
        "experiment_root": str(experiment_root.resolve()),
        "paired_manifest": str(manifest_path.resolve()),
        "paired_manifest_sha256": manifest_sha,
        "conditions": list(CONDITIONS),
        "training_seeds": list(seeds),
        "num_references": num_references,
        "k_trials": k_trials,
        "metrics": condition_stats,
        "paired_differences": paired_summary,
        "protocol": {
            key: baseline_spec["metadata"][key]
            for key in comparable_spec_keys
        },
        "evaluation_pipeline_sha256": next(iter(evaluation_pipelines)),
        "bootstrap": {
            "method": bootstrap_results["method"],
            "samples": bootstrap_samples,
            "random_seed": bootstrap_seed,
        },
        "input_validations": {
            "{}_seed_{}".format(condition, seed): {
                "episodes_sha256": validations[(condition, seed)][
                    "episodes_sha256"
                ],
                "actual_episodes": validations[(condition, seed)][
                    "actual_episodes"
                ],
                "expected_episodes": validations[(condition, seed)][
                    "expected_episodes"
                ],
                "run_spec": run_specs[(condition, seed)]["path"],
                "checkpoint_sha256": evaluation_metadata[
                    (condition, seed)
                ]["checkpoint_sha256"],
            }
            for condition in CONDITIONS
            for seed in seeds
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    _write_main_table(output_dir / "main_table.tex", condition_stats)
    reference_records = [manifest_by_id[key] for key in reference_ids]
    _write_spatial_table(
        output_dir / "spatial_table.tex",
        values["raw"],
        values["full"],
        reference_records,
        metric_index,
    )
    _write_results_markdown(
        output_dir / "results.md",
        condition_stats=condition_stats,
        paired_summary=paired_summary,
        num_references=num_references,
        k_trials=k_trials,
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Strictly aggregate four-seed paired Raw-vs-Full Theia policy "
            "results and generate paper tables."
        )
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.experiment_root) / "results"
    )
    try:
        summary = aggregate_experiment(
            experiment_root=args.experiment_root,
            manifest_path=args.pair_manifest,
            output_dir=output_dir,
            seeds=args.seeds,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except ValidationError as exc:
        raise SystemExit(
            "Formal Raw-vs-Full aggregation failed: {}".format(exc)
        )
    success = summary["metrics"]
    print(
        "Aggregated {references} references x {seeds} seeds: "
        "Raw Succ.={raw:.2f}%, Full Succ.={full:.2f}%; results={output}".format(
            references=summary["num_references"],
            seeds=len(summary["training_seeds"]),
            raw=success["raw"]["success_rate_percent"]["mean"],
            full=success["full"]["success_rate_percent"]["mean"],
            output=output_dir,
        )
    )


if __name__ == "__main__":
    main()
