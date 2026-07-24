#!/usr/bin/env python3
"""Validate and summarize one formal Theia policy evaluation.

The simulator writes one row per environment to ``episodes.csv``.  This
script converts those rows into the reference-level quantities used by the
paper:

* success: at least one completed trial out of exactly K trials;
* duration/error: the longest-duration trial for the reference, breaking
  ties by lower human-plus-object error and then by trial id.

The paired manifest is deliberately authoritative.  Evaluation output that
contains an unknown reference, omits an included reference, or has a trial
count other than K is rejected instead of being silently averaged.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


REQUIRED_EPISODE_COLUMNS = {
    "env_id",
    "sequence",
    "steps",
    "completed",
    "mean_human_error_m",
    "mean_object_surface_error_m",
}

PER_REFERENCE_FIELDS = [
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


class ValidationError(ValueError):
    """Raised when an evaluation cannot support the formal result."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_int(value, label, minimum=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("{} must be an integer, got {!r}".format(
            label, value
        )) from exc
    if minimum is not None and parsed < minimum:
        raise ValidationError(
            "{} must be >= {}, got {}".format(label, minimum, parsed)
        )
    return parsed


def _parse_float(value, label, minimum=None):
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


def _parse_binary(value, label):
    parsed = _parse_int(value, label)
    if parsed not in (0, 1):
        raise ValidationError("{} must be 0 or 1, got {}".format(
            label, parsed
        ))
    return parsed


def load_paired_manifest(manifest_path):
    """Return validated included-reference records from a paired manifest."""
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(
            "Cannot read paired manifest {}: {}".format(manifest_path, exc)
        ) from exc

    references = manifest.get("references")
    if not isinstance(references, list):
        raise ValidationError(
            "Paired manifest must contain a top-level 'references' list"
        )

    required = {
        "reference_id",
        "raw_filename",
        "full_filename",
        "raw_sha256",
        "full_sha256",
        "height",
        "variation",
        "included",
    }
    included = []
    seen_ids = set()
    filenames = {"raw": set(), "full": set()}
    for index, entry in enumerate(references):
        if not isinstance(entry, dict):
            raise ValidationError(
                "references[{}] must be an object".format(index)
            )
        missing = required.difference(entry)
        if missing:
            raise ValidationError(
                "references[{}] is missing {}".format(
                    index, sorted(missing)
                )
            )
        reference_id = str(entry["reference_id"]).strip()
        if not reference_id:
            raise ValidationError(
                "references[{}].reference_id is empty".format(index)
            )
        if reference_id in seen_ids:
            raise ValidationError(
                "Duplicate reference_id in manifest: {}".format(reference_id)
            )
        seen_ids.add(reference_id)

        if not isinstance(entry["included"], bool):
            raise ValidationError(
                "references[{}].included must be boolean".format(index)
            )
        if not entry["included"]:
            continue

        normalized = dict(entry)
        normalized["reference_id"] = reference_id
        normalized["height"] = str(entry["height"]).strip()
        normalized["variation"] = str(entry["variation"]).strip()
        if not normalized["height"] or not normalized["variation"]:
            raise ValidationError(
                "{} has empty height or variation".format(reference_id)
            )
        for condition in ("raw", "full"):
            key = condition + "_filename"
            hash_key = condition + "_sha256"
            filename = Path(str(entry[key])).name
            if not filename:
                raise ValidationError(
                    "{} has an empty {}".format(reference_id, key)
                )
            if filename in filenames[condition]:
                raise ValidationError(
                    "Duplicate {} filename in manifest: {}".format(
                        condition, filename
                    )
                )
            filenames[condition].add(filename)
            normalized[key] = filename
            digest = str(entry[hash_key]).strip().lower()
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValidationError(
                    "{} has invalid {}".format(reference_id, hash_key)
                )
            normalized[hash_key] = digest
        included.append(normalized)

    if not included:
        raise ValidationError("Paired manifest has no included references")
    return included


def _load_episode_rows(episodes_path):
    episodes_path = Path(episodes_path)
    try:
        with episodes_path.open(newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValidationError(
                    "{} has no CSV header".format(episodes_path)
                )
            missing = REQUIRED_EPISODE_COLUMNS.difference(reader.fieldnames)
            if missing:
                raise ValidationError(
                    "{} is missing columns {}".format(
                        episodes_path, sorted(missing)
                    )
                )
            rows = list(reader)
            fieldnames = list(reader.fieldnames)
    except OSError as exc:
        raise ValidationError(
            "Cannot read episodes CSV {}: {}".format(episodes_path, exc)
        ) from exc
    if not rows:
        raise ValidationError("{} contains no episode rows".format(
            episodes_path
        ))
    return rows, fieldnames


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_evaluation(
    episodes_path,
    manifest_path,
    condition,
    training_seed,
    evaluation_seed,
    k_trials,
    fps,
    output_dir,
    published_output_dir=None,
    evaluation_manifest_path=None,
):
    """Validate one evaluation and write reference-level results."""
    if condition not in ("raw", "full"):
        raise ValidationError(
            "condition must be 'raw' or 'full', got {!r}".format(condition)
        )
    training_seed = _parse_int(training_seed, "training_seed", minimum=0)
    evaluation_seed = _parse_int(
        evaluation_seed, "evaluation_seed", minimum=0
    )
    k_trials = _parse_int(k_trials, "k_trials", minimum=1)
    fps = _parse_float(fps, "fps", minimum=1e-12)
    episodes_path = Path(episodes_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    published_output_dir = (
        Path(published_output_dir)
        if published_output_dir is not None
        else output_dir
    )
    evaluation_manifest_path = (
        Path(evaluation_manifest_path)
        if evaluation_manifest_path is not None
        else None
    )

    manifest_references = load_paired_manifest(manifest_path)
    filename_key = condition + "_filename"
    reference_by_filename = {
        entry[filename_key]: entry for entry in manifest_references
    }
    expected_ids = {entry["reference_id"] for entry in manifest_references}

    raw_rows, episode_fields = _load_episode_rows(episodes_path)
    has_trial_id = "trial_id" in episode_fields
    grouped = {reference_id: [] for reference_id in expected_ids}
    seen_env_ids = set()
    for row_index, row in enumerate(raw_rows, start=2):
        prefix = "{} row {}".format(episodes_path, row_index)
        if "condition" in episode_fields and row["condition"] != condition:
            raise ValidationError(
                "{} condition is {!r}, expected {!r}".format(
                    prefix, row["condition"], condition
                )
            )
        if (
            "training_seed" in episode_fields
            and _parse_int(
                row["training_seed"], prefix + " training_seed", minimum=0
            )
            != training_seed
        ):
            raise ValidationError(
                "{} training_seed does not match CLI metadata".format(prefix)
            )
        if (
            "eval_seed" in episode_fields
            and _parse_int(
                row["eval_seed"], prefix + " eval_seed", minimum=0
            )
            != evaluation_seed
        ):
            raise ValidationError(
                "{} eval_seed does not match CLI metadata".format(prefix)
            )
        if "fps" in episode_fields:
            row_fps = _parse_float(
                row["fps"], prefix + " fps", minimum=1e-12
            )
            if not math.isclose(row_fps, fps, rel_tol=0.0, abs_tol=1e-9):
                raise ValidationError(
                    "{} fps {} does not match CLI fps {}".format(
                        prefix, row_fps, fps
                    )
                )
        env_id = _parse_int(row["env_id"], prefix + " env_id", minimum=0)
        if env_id in seen_env_ids:
            raise ValidationError(
                "{} has duplicate env_id {}".format(episodes_path, env_id)
            )
        seen_env_ids.add(env_id)

        filename = Path(str(row["sequence"])).name
        if filename not in reference_by_filename:
            raise ValidationError(
                "{} contains unknown or excluded {} reference {!r}".format(
                    episodes_path, condition, filename
                )
            )
        reference = reference_by_filename[filename]
        parsed = {
            "env_id": env_id,
            "reference_id": reference["reference_id"],
            "filename": filename,
            "steps": _parse_int(
                row["steps"], prefix + " steps", minimum=0
            ),
            "completed": _parse_binary(
                row["completed"], prefix + " completed"
            ),
            "human_error_m": _parse_float(
                row["mean_human_error_m"],
                prefix + " mean_human_error_m",
                minimum=0.0,
            ),
            "object_error_m": _parse_float(
                row["mean_object_surface_error_m"],
                prefix + " mean_object_surface_error_m",
                minimum=0.0,
            ),
        }
        if "reference_frames" in episode_fields:
            reference_frames = _parse_int(
                row["reference_frames"],
                prefix + " reference_frames",
                minimum=2,
            )
            if parsed["steps"] > reference_frames - 1:
                raise ValidationError(
                    "{} steps exceeds reference_frames - 1".format(prefix)
                )
            if (
                parsed["completed"]
                and parsed["steps"] != reference_frames - 1
            ):
                raise ValidationError(
                    "{} completed trial must execute reference_frames - 1 "
                    "steps".format(prefix)
                )
            parsed["reference_frames"] = reference_frames
        if has_trial_id:
            parsed["trial_id"] = _parse_int(
                row["trial_id"], prefix + " trial_id", minimum=0
            )
        grouped[reference["reference_id"]].append(parsed)

    per_reference = []
    trial_counts = {}
    for reference in manifest_references:
        reference_id = reference["reference_id"]
        trials = grouped[reference_id]
        trial_counts[reference_id] = len(trials)
        if len(trials) != k_trials:
            raise ValidationError(
                "{} requires exactly {} trials, found {}".format(
                    reference_id, k_trials, len(trials)
                )
            )

        if has_trial_id:
            observed = sorted(trial["trial_id"] for trial in trials)
            expected = list(range(k_trials))
            if observed != expected:
                raise ValidationError(
                    "{} trial_id set must be {}, found {}".format(
                        reference_id, expected, observed
                    )
                )
        else:
            # Current InterMimic CSVs predate an explicit trial_id.  Environment
            # ids are unique and each environment contributes exactly one row,
            # so their stable order defines trial ids without changing results.
            trials.sort(key=lambda trial: trial["env_id"])
            for trial_id, trial in enumerate(trials):
                trial["trial_id"] = trial_id

        best = min(
            trials,
            key=lambda trial: (
                -trial["steps"],
                trial["human_error_m"] + trial["object_error_m"],
                trial["trial_id"],
            ),
        )
        completed_trials = sum(trial["completed"] for trial in trials)
        per_reference.append({
            "condition": condition,
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
            "reference_id": reference_id,
            "filename": reference[filename_key],
            "height": reference["height"],
            "variation": reference["variation"],
            "k_trials": k_trials,
            "completed_trials": completed_trials,
            "success": int(completed_trials > 0),
            "best_trial_id": best["trial_id"],
            "best_env_id": best["env_id"],
            "best_steps": best["steps"],
            "duration_s": best["steps"] / fps,
            "human_error_cm": 100.0 * best["human_error_m"],
            "object_error_cm": 100.0 * best["object_error_m"],
        })

    per_reference.sort(key=lambda row: row["reference_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    per_reference_path = output_dir / "per_reference.csv"
    validation_path = output_dir / "validation.json"
    summary_path = output_dir / "summary.json"
    _write_csv(per_reference_path, PER_REFERENCE_FIELDS, per_reference)

    best_trial_order = [
        "steps_desc",
        "human_plus_object_error_asc",
        "trial_id_asc",
    ]
    validation = {
        "schema_version": 2,
        "valid": True,
        "condition": condition,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "k_trials": k_trials,
        "fps": fps,
        "num_references": len(manifest_references),
        "expected_episodes": len(manifest_references) * k_trials,
        "actual_episodes": len(raw_rows),
        "trial_id_source": (
            "episodes.csv" if has_trial_id else "ranked_unique_env_id"
        ),
        "success_definition": "any_completed_over_exactly_k_trials",
        "best_trial_order": best_trial_order,
        "unit_conversions": {
            "duration_s": "steps / fps",
            "human_error_cm": "mean_human_error_m * 100",
            "object_error_cm": "mean_object_surface_error_m * 100",
        },
        "episodes_csv": str(
            (published_output_dir / "episodes.csv").resolve()
        ),
        "episodes_sha256": _sha256(episodes_path),
        "paired_manifest": str(manifest_path.resolve()),
        "paired_manifest_sha256": _sha256(manifest_path),
        "per_reference_csv": str(
            (published_output_dir / "per_reference.csv").resolve()
        ),
        "trial_counts": {
            key: trial_counts[key] for key in sorted(trial_counts)
        },
    }
    count = len(per_reference)
    summary = {
        "schema_version": 1,
        "condition": condition,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "k_trials": k_trials,
        "fps": fps,
        "num_references": count,
        "metrics": {
            "success_rate_percent": (
                100.0
                * sum(row["success"] for row in per_reference)
                / count
            ),
            "duration_s": (
                sum(row["duration_s"] for row in per_reference) / count
            ),
            "human_error_cm": (
                sum(row["human_error_cm"] for row in per_reference) / count
            ),
            "object_error_cm": (
                sum(row["object_error_cm"] for row in per_reference) / count
            ),
        },
        "aggregation": {
            "reference_weighting": "equal",
            "success": "any_completed_over_exactly_k_trials",
            "duration_and_errors": "best_trial",
            "best_trial_order": best_trial_order,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    validation.update({
        "per_reference_sha256": _sha256(per_reference_path),
        "summary_json": str(
            (published_output_dir / "summary.json").resolve()
        ),
        "summary_sha256": _sha256(summary_path),
    })
    if evaluation_manifest_path is not None:
        if not evaluation_manifest_path.is_file():
            raise ValidationError(
                "Evaluation manifest not found: {}".format(
                    evaluation_manifest_path
                )
            )
        validation.update({
            "evaluation_manifest": str(
                (published_output_dir / "manifest.txt").resolve()
            ),
            "evaluation_manifest_sha256": _sha256(
                evaluation_manifest_path
            ),
        })
    # The validation receipt is deliberately written last.  Its presence
    # therefore certifies a complete, hash-linked output bundle.
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    return per_reference, validation, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate one Theia episodes.csv and aggregate exactly K trials "
            "per reference."
        )
    )
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument(
        "--condition", required=True, choices=("raw", "full")
    )
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--expected-k", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--published-output-dir",
        help="Final artifact path when --output-dir is a transactional staging directory",
    )
    parser.add_argument("--evaluation-manifest")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        per_reference, validation, _ = summarize_evaluation(
            episodes_path=args.episodes,
            manifest_path=args.pair_manifest,
            condition=args.condition,
            training_seed=args.training_seed,
            evaluation_seed=args.eval_seed,
            k_trials=args.expected_k,
            fps=args.fps,
            output_dir=args.output_dir,
            published_output_dir=args.published_output_dir,
            evaluation_manifest_path=args.evaluation_manifest,
        )
    except ValidationError as exc:
        raise SystemExit("Formal evaluation validation failed: {}".format(exc))
    successes = sum(row["success"] for row in per_reference)
    print(
        "Validated {actual}/{expected} episodes: {success}/{total} "
        "references successful; results={output}".format(
            actual=validation["actual_episodes"],
            expected=validation["expected_episodes"],
            success=successes,
            total=len(per_reference),
            output=Path(args.output_dir) / "per_reference.csv",
        )
    )


if __name__ == "__main__":
    main()
