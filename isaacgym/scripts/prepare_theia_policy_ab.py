#!/usr/bin/env python3
"""Prepare paired Raw/Full Theia S1 references for the policy A/B experiment.

This script deliberately creates immutable ``eligible/raw`` and
``eligible/full`` directories.  Training should use those directories, not the
larger ``converted`` staging directories, so an excluded or partially
converted sequence can never enter a run silently.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from check_theia_server_versions import check_server_versions


REPO_ROOT = Path(__file__).resolve().parents[2]
THEIA_ROOT = REPO_ROOT.parents[1]
DEFAULT_CONVERTER = THEIA_ROOT / "toolkit" / "scripts" / "theia2intermimic.py"
PAIR_COLUMNS = slice(318, 386)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_bucket(value, prefix: str, minimum: int, maximum: int):
    if value is None:
        return None
    match = re.fullmatch(
        rf"{prefix}0*(\d+)", str(value).strip(), flags=re.IGNORECASE
    )
    if match is None:
        raise ValueError(f"Invalid {prefix} bucket {value!r}")
    index = int(match.group(1))
    if not minimum <= index <= maximum:
        raise ValueError(
            f"{prefix} bucket {value!r} is outside {prefix}{minimum}..{prefix}{maximum}"
        )
    return f"{prefix}{index}"


def parse_reference_fields(reference_id: str, override=None) -> dict:
    override = override or {}
    identity = re.match(
        r"^S(?P<stage>\d+)(?P<level>L.*?)P(?P<subject>\d+)",
        reference_id,
        flags=re.IGNORECASE,
    )
    variation = re.search(r"V(\d+)", reference_id, flags=re.IGNORECASE)
    task = re.search(r"T(\d+)", reference_id, flags=re.IGNORECASE)
    level = identity.group("level") if identity else None
    level_numbers = re.findall(r"\d+", level or "")
    canonical_level = re.fullmatch(r"L([1-5])([1-5])", level or "")
    inferred_height = (
        f"L{level_numbers[0][0]}" if level_numbers and level_numbers[0] else None
    )
    height = normalize_bucket(
        override.get("height", inferred_height), "L", 1, 5
    )
    normalized_variation = normalize_bucket(
        override.get(
            "variation",
            f"V{variation.group(1)}" if variation else None,
        ),
        "V",
        1,
        3,
    )
    return {
        "stage": f"S{identity.group('stage')}" if identity else None,
        "level": level,
        "left_height": (
            int(canonical_level.group(1)) if canonical_level else None
        ),
        "right_height": (
            int(canonical_level.group(2)) if canonical_level else None
        ),
        "height": height,
        "variation": normalized_variation,
        "subject_from_reference": (
            int(identity.group("subject")) if identity else None
        ),
        "task": override.get(
            "task", f"T{task.group(1)}" if task else None
        ),
        "action": override.get("action"),
    }


def validate_formal_s1_fields(fields):
    if fields["stage"] != "S1":
        raise ValueError(
            f"formal policy A/B only accepts S1; parsed {fields['stage']!r}"
        )
    if (
        fields["left_height"] is None
        or fields["right_height"] is None
        or fields["left_height"] != fields["right_height"]
    ):
        raise ValueError(
            "formal S1 requires a same-height L11/L22/L33/L44/L55 "
            f"level, got {fields['level']!r}"
        )
    canonical_height = f"L{fields['left_height']}"
    if fields["height"] != canonical_height:
        raise ValueError(
            "height metadata conflicts with the canonical S1 level: "
            f"{fields['height']!r} vs {canonical_height!r}"
        )
    if fields["variation"] is None:
        raise ValueError(
            "cannot freeze canonical variation V1..V3; "
            "provide --reference-metadata"
        )


def load_reference_metadata(path) -> dict:
    if not path:
        return {}
    metadata_path = Path(path).resolve()
    if metadata_path.suffix.lower() == ".csv":
        with metadata_path.open(newline="") as source:
            rows = list(csv.DictReader(source))
    else:
        payload = json.loads(metadata_path.read_text())
        if isinstance(payload, dict) and "references" in payload:
            rows = payload["references"]
        elif isinstance(payload, dict):
            rows = [
                {"reference_id": reference_id, **values}
                for reference_id, values in payload.items()
            ]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(
                f"Unsupported reference metadata schema: {metadata_path}"
            )
    result = {}
    for row in rows:
        reference_id = row["reference_id"]
        if reference_id in result:
            raise ValueError(
                f"Reference metadata duplicates {reference_id!r}"
            )
        result[reference_id] = row
    return result


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path}: expected torch.Tensor, got {type(value).__name__}")
    if value.ndim != 2 or value.shape[1] != 594:
        raise ValueError(f"{path}: expected [T, 594], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return value


def read_sequence_list(path: Path, source_root: Path) -> list:
    sequences = []
    for line in path.read_text().splitlines():
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = source_root / candidate
        sequences.append(candidate.resolve())
    return sequences


def discover_sequences(args) -> list:
    source_root = Path(args.source_root).resolve()
    if args.sequence_list:
        sequences = read_sequence_list(
            Path(args.sequence_list).resolve(), source_root
        )
    else:
        iterator = (
            source_root.rglob(args.sequence_glob)
            if args.recursive
            else source_root.glob(args.sequence_glob)
        )
        sequences = sorted(path.resolve() for path in iterator if path.is_dir())
    by_reference = {}
    for sequence in sequences:
        if not sequence.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {sequence}")
        if sequence.name in by_reference:
            raise ValueError(
                f"Duplicate reference ID {sequence.name!r}: "
                f"{by_reference[sequence.name]} and {sequence}"
            )
        by_reference[sequence.name] = sequence
    return [by_reference[key] for key in sorted(by_reference)]


def run_converter(
    *,
    converter: Path,
    sequence: Path,
    objects_dir: Path,
    output_dir: Path,
    metadata_path: Path,
    log_path: Path,
    variant: str,
    target_fps: float,
    ground_clearance: float,
    ground_shift=None,
    subject_id=None,
) -> dict:
    command = [
        sys.executable,
        str(converter),
        "--data_dir",
        str(sequence),
        "--objects_dir",
        str(objects_dir),
        "--output_dir",
        str(output_dir),
        "--target_fps",
        str(target_fps),
        "--ground_clearance",
        str(ground_clearance),
        "--motion-variant",
        variant,
        "--metadata-json",
        str(metadata_path),
    ]
    if ground_shift is not None:
        command.extend(["--ground-shift", repr(float(ground_shift))])
    if subject_id is not None:
        command.extend(["--subject-id", str(subject_id)])
    result = subprocess.run(
        command,
        cwd=str(THEIA_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout)
    if result.returncode:
        tail = "\n".join(result.stdout.splitlines()[-20:])
        raise RuntimeError(
            f"{variant} converter exited {result.returncode}; "
            f"log={log_path}\n{tail}"
        )
    if not metadata_path.is_file():
        raise RuntimeError(
            f"{variant} converter succeeded without metadata: {metadata_path}"
        )
    return json.loads(metadata_path.read_text())


def copy_to_frozen_set(source: Path, destination: Path):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite eligible file: {destination}")
    # Do not hard-link staging and eligible data: rewriting a staging file
    # would otherwise mutate the supposedly frozen training input in place.
    shutil.copy2(source, destination)


def write_csv(path: Path, rows: list, fieldnames: list):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--objects-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--converter", default=str(DEFAULT_CONVERTER))
    parser.add_argument("--sequence-list")
    parser.add_argument("--sequence-glob", default="S1*")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--reference-metadata",
        help="Optional JSON/CSV mapping with reference_id,height,variation "
             "(and optional action/task) when IDs do not encode them",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--ground-clearance", type=float, default=0.002)
    parser.add_argument(
        "--subject-id",
        help="Global subject override; omit for per-reference P<number> inference",
    )
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--allow-exclusions",
        action="store_true",
        help="Finish successfully after recording technical exclusions",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    converter = Path(args.converter).resolve()
    source_root = Path(args.source_root).resolve()
    objects_dir = Path(args.objects_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if not converter.is_file():
        raise SystemExit(f"Converter not found: {converter}")
    repository_versions = check_server_versions()
    if not repository_versions["valid"]:
        raise SystemExit(
            "Formal repository version check failed before conversion:\n"
            + "\n".join(repository_versions["errors"])
        )
    if not source_root.is_dir():
        raise SystemExit(f"Source root not found: {source_root}")
    if not objects_dir.is_dir():
        raise SystemExit(f"Objects directory not found: {objects_dir}")

    sequences = discover_sequences(args)
    reference_metadata = load_reference_metadata(args.reference_metadata)
    if not sequences:
        raise SystemExit(
            f"No sequence directories selected from {source_root} "
            f"with glob {args.sequence_glob!r}"
        )
    if args.expected_count is not None and len(sequences) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} candidate references, found "
            f"{len(sequences)}"
        )

    converted_raw = output_root / "converted" / "raw"
    converted_full = output_root / "converted" / "full"
    eligible_raw = output_root / "eligible" / "raw"
    eligible_full = output_root / "eligible" / "full"
    metadata_raw = output_root / "metadata" / "raw"
    metadata_full = output_root / "metadata" / "full"
    probe_metadata_raw = output_root / "metadata" / "probe_raw"
    probe_metadata_full = output_root / "metadata" / "probe_full"
    logs_dir = output_root / "conversion_logs"
    for directory in (
        converted_raw,
        converted_full,
        eligible_raw,
        eligible_full,
        metadata_raw,
        metadata_full,
        probe_metadata_raw,
        probe_metadata_full,
        logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    existing_pt = [
        path
        for directory in (converted_raw, converted_full, eligible_raw, eligible_full)
        for path in directory.glob("*.pt")
    ]
    if existing_pt:
        raise SystemExit(
            "Output root already contains converted PT files. Use a fresh "
            f"--output-root to avoid stale references: {existing_pt[0]}"
        )

    eligible = []
    excluded = []
    assets = {}
    for index, sequence in enumerate(sequences, start=1):
        reference_id = sequence.name
        print(f"[{index}/{len(sequences)}] {reference_id}", flush=True)
        try:
            fields = parse_reference_fields(
                reference_id, reference_metadata.get(reference_id)
            )
            validate_formal_s1_fields(fields)
            full_metadata_path = metadata_full / f"{reference_id}.json"
            raw_metadata_path = metadata_raw / f"{reference_id}.json"
            full_probe = run_converter(
                converter=converter,
                sequence=sequence,
                objects_dir=objects_dir,
                output_dir=converted_full,
                metadata_path=probe_metadata_full / f"{reference_id}.json",
                log_path=logs_dir / f"{reference_id}.full_probe.log",
                variant="refined",
                target_fps=args.target_fps,
                ground_clearance=args.ground_clearance,
                subject_id=args.subject_id,
            )
            raw_probe = run_converter(
                converter=converter,
                sequence=sequence,
                objects_dir=objects_dir,
                output_dir=converted_raw,
                metadata_path=probe_metadata_raw / f"{reference_id}.json",
                log_path=logs_dir / f"{reference_id}.raw_probe.log",
                variant="raw",
                target_fps=args.target_fps,
                ground_clearance=args.ground_clearance,
                subject_id=args.subject_id,
            )
            raw_required_shift = float(raw_probe["ground_shift_m"])
            full_required_shift = float(full_probe["ground_shift_m"])
            shared_ground_shift = max(
                raw_required_shift, full_required_shift
            )
            # The final pass applies one preregistered scene transform to both
            # variants.  Taking the larger required shift keeps both sets of
            # collision geometry at or above the requested clearance.
            full_meta = run_converter(
                converter=converter,
                sequence=sequence,
                objects_dir=objects_dir,
                output_dir=converted_full,
                metadata_path=full_metadata_path,
                log_path=logs_dir / f"{reference_id}.full.log",
                variant="refined",
                target_fps=args.target_fps,
                ground_clearance=args.ground_clearance,
                ground_shift=shared_ground_shift,
                subject_id=args.subject_id,
            )
            raw_meta = run_converter(
                converter=converter,
                sequence=sequence,
                objects_dir=objects_dir,
                output_dir=converted_raw,
                metadata_path=raw_metadata_path,
                log_path=logs_dir / f"{reference_id}.raw.log",
                variant="raw",
                target_fps=args.target_fps,
                ground_clearance=args.ground_clearance,
                ground_shift=shared_ground_shift,
                subject_id=args.subject_id,
            )
            paired_alignment = {
                "raw_required_shift_m": raw_required_shift,
                "full_required_shift_m": full_required_shift,
                "shared_shift_m": shared_ground_shift,
                "rule": "max(raw_required_shift_m, full_required_shift_m)",
            }
            raw_meta["paired_ground_alignment"] = paired_alignment
            full_meta["paired_ground_alignment"] = paired_alignment
            raw_metadata_path.write_text(
                json.dumps(raw_meta, indent=2, sort_keys=True) + "\n"
            )
            full_metadata_path.write_text(
                json.dumps(full_meta, indent=2, sort_keys=True) + "\n"
            )

            raw_path = Path(raw_meta["output"]["path"])
            full_path = Path(full_meta["output"]["path"])
            if raw_path.name != full_path.name:
                raise ValueError(
                    f"paired filenames differ: {raw_path.name} vs {full_path.name}"
                )
            if raw_meta["reference_id"] != full_meta["reference_id"]:
                raise ValueError("converter metadata reference IDs differ")
            if raw_meta["subject_id"] != full_meta["subject_id"]:
                raise ValueError("converter metadata subject IDs differ")
            if raw_meta["frame_indices_sha256"] != full_meta["frame_indices_sha256"]:
                raise ValueError("Raw/Full frame-index sets differ")
            if raw_meta["objects"] != full_meta["objects"]:
                raise ValueError("Raw/Full object order or hand assignment differs")
            for source_key in ("object_motion", "contact_intervals"):
                if (
                    raw_meta["sources"][source_key]["sha256"]
                    != full_meta["sources"][source_key]["sha256"]
                ):
                    raise ValueError(f"Raw/Full {source_key} sources differ")

            raw_tensor = load_tensor(raw_path)
            full_tensor = load_tensor(full_path)
            if raw_tensor.shape != full_tensor.shape:
                raise ValueError(
                    f"paired tensor shapes differ: {tuple(raw_tensor.shape)} "
                    f"vs {tuple(full_tensor.shape)}"
                )
            if not torch.equal(
                raw_tensor[:, PAIR_COLUMNS], full_tensor[:, PAIR_COLUMNS]
            ):
                difference = (
                    raw_tensor[:, PAIR_COLUMNS]
                    - full_tensor[:, PAIR_COLUMNS]
                ).abs()
                raise ValueError(
                    "object/contact columns 318:386 differ; "
                    f"max_abs={float(difference.max()):.9g}"
                )

            subject_id = int(raw_meta["subject_id"])
            if (
                fields["subject_from_reference"] is not None
                and args.subject_id is None
                and fields["subject_from_reference"] != subject_id
            ):
                raise ValueError("subject inference disagrees with reference ID")

            pending_assets = {}
            for asset in full_meta["assets"]:
                previous = assets.get(asset["name"])
                signature = (
                    asset["installed_mesh_sha256"],
                    asset["installed_urdf_sha256"],
                )
                if previous is not None and previous["signature"] != signature:
                    raise ValueError(
                        f"object asset {asset['name']} changed across references"
                    )
                pending_assets[asset["name"]] = {
                    "name": asset["name"],
                    "mesh": asset["installed_mesh"],
                    "mesh_sha256": asset["installed_mesh_sha256"],
                    "urdf": asset["installed_urdf"],
                    "urdf_sha256": asset["installed_urdf_sha256"],
                    "source_mesh": asset["source_mesh"],
                    "source_mesh_sha256": asset["source_mesh_sha256"],
                    "signature": signature,
                }

            raw_hash = sha256(raw_path)
            full_hash = sha256(full_path)
            entry = {
                "reference_id": reference_id,
                "raw_filename": raw_path.name,
                "full_filename": full_path.name,
                "included": True,
                "stage": fields["stage"],
                "level": fields["level"],
                "height": fields["height"],
                "variation": fields["variation"],
                "action": fields["action"],
                "task": fields["task"],
                "subject": subject_id,
                "frames": int(raw_tensor.shape[0]),
                "raw_sha256": raw_hash,
                "full_sha256": full_hash,
                "raw_required_ground_shift_m": raw_required_shift,
                "full_required_ground_shift_m": full_required_shift,
                "shared_ground_shift_m": shared_ground_shift,
                "objects": [item["name"] for item in raw_meta["objects"]],
                "source_dir": str(sequence),
                "raw_metadata": str(raw_metadata_path),
                "full_metadata": str(full_metadata_path),
            }
            copy_to_frozen_set(raw_path, eligible_raw / raw_path.name)
            copy_to_frozen_set(full_path, eligible_full / full_path.name)
            eligible.append(entry)
            assets.update(pending_assets)
        except Exception as error:
            print(f"  EXCLUDED: {error}", file=sys.stderr, flush=True)
            excluded.append(
                {
                    "reference_id": reference_id,
                    "source_dir": str(sequence),
                    "reason": str(error),
                }
            )

    raw_names = {path.name for path in eligible_raw.glob("*.pt")}
    full_names = {path.name for path in eligible_full.glob("*.pt")}
    frozen_names = {entry["raw_filename"] for entry in eligible}
    if raw_names != full_names or raw_names != frozen_names:
        raise RuntimeError(
            "Eligible directory contents do not match the frozen paired set"
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "objects_dir": str(objects_dir),
        "converter": str(converter),
        "converter_sha256": sha256(converter),
        "repository_versions": repository_versions,
        "protomotions": repository_versions["repositories"]["ProtoMotions"],
        "target_fps": float(args.target_fps),
        "ground_clearance_m": float(args.ground_clearance),
        "ground_alignment": (
            "per-reference max(Raw-required, Full-required) shared global Z shift"
        ),
        "candidate_count": len(sequences),
        "candidate_reference_ids": [path.name for path in sequences],
        "eligible_count": len(eligible),
        "excluded_count": len(excluded),
        "eligible_raw_dir": str(eligible_raw),
        "eligible_full_dir": str(eligible_full),
        "references": eligible,
        "excluded_pairs": excluded,
        "assets": [
            {key: value for key, value in entry.items() if key != "signature"}
            for entry in sorted(assets.values(), key=lambda item: item["name"])
        ],
    }
    manifest_path = output_root / "policy_ab_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_csv(
        output_root / "eligible_pairs.csv",
        eligible,
        [
            "reference_id",
            "raw_filename",
            "full_filename",
            "included",
            "stage",
            "level",
            "height",
            "variation",
            "action",
            "task",
            "subject",
            "frames",
            "raw_sha256",
            "full_sha256",
            "raw_required_ground_shift_m",
            "full_required_ground_shift_m",
            "shared_ground_shift_m",
            "source_dir",
        ],
    )
    write_csv(
        output_root / "excluded_pairs.csv",
        excluded,
        ["reference_id", "source_dir", "reason"],
    )
    (output_root / "data_hashes_raw.txt").write_text(
        "".join(
            f"{entry['raw_sha256']}  {entry['raw_filename']}\n"
            for entry in eligible
        )
    )
    (output_root / "data_hashes_full.txt").write_text(
        "".join(
            f"{entry['full_sha256']}  {entry['full_filename']}\n"
            for entry in eligible
        )
    )
    (output_root / "asset_hashes.txt").write_text(
        "".join(
            f"{entry['mesh_sha256']}  {entry['mesh']}\n"
            f"{entry['urdf_sha256']}  {entry['urdf']}\n"
            for entry in manifest["assets"]
        )
    )
    print(
        f"Prepared {len(eligible)}/{len(sequences)} paired references; "
        f"manifest={manifest_path}"
    )
    if not eligible:
        raise SystemExit("No eligible Raw/Full references were produced")
    if excluded and not args.allow_exclusions:
        raise SystemExit(
            f"{len(excluded)} reference(s) were excluded. Inspect "
            f"{output_root / 'excluded_pairs.csv'}. The prepared eligible "
            "set is preserved; acknowledge accepted technical exclusions "
            "with ACCEPT_EXCLUSIONS=1 at preflight. Use a new empty output "
            "root if conversion itself must be rerun."
        )


if __name__ == "__main__":
    main()
