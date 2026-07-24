#!/usr/bin/env python3
"""Fail-fast paired validator for Theia Raw/Full policy references."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = (
    REPO_ROOT
    / "isaacgym"
    / "src"
    / "intermimic"
    / "data"
    / "assets"
    / "objects"
)
PAIR_COLUMNS = slice(318, 386)


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
            f"{prefix} bucket {value!r} is outside "
            f"{prefix}{minimum}..{prefix}{maximum}"
        )
    return f"{prefix}{index}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_motion_filename(path: Path) -> dict:
    pieces = path.stem.split("_", 2)
    if len(pieces) != 3:
        raise ValueError(
            f"{path.name}: expected sub<number>_<left>+<right>_<reference>.pt"
        )
    subject_token, pair_token, reference_id = pieces
    subject_match = re.fullmatch(r"sub(\d+)", subject_token)
    if subject_match is None or pair_token.count("+") != 1:
        raise ValueError(
            f"{path.name}: invalid subject or object-pair filename token"
        )
    obj1, obj2 = pair_token.split("+", 1)
    if not obj1 or not obj2:
        raise ValueError(f"{path.name}: empty object name")
    identity = re.match(
        r"^S(?P<stage>\d+)(?P<level>L.*?)P(?P<subject>\d+)",
        reference_id,
        flags=re.IGNORECASE,
    )
    variation = re.search(r"V(\d+)", reference_id, flags=re.IGNORECASE)
    level = identity.group("level") if identity else None
    level_numbers = re.findall(r"\d+", level or "")
    height = normalize_bucket(
        f"L{level_numbers[0][0]}"
        if level_numbers and level_numbers[0]
        else None,
        "L",
        1,
        5,
    )
    return {
        "reference_id": reference_id,
        "subject": int(subject_match.group(1)),
        "objects": [obj1, obj2],
        "stage": f"S{identity.group('stage')}" if identity else None,
        "level": level,
        "height": height,
        "variation": normalize_bucket(
            f"V{variation.group(1)}" if variation else None,
            "V",
            1,
            3,
        ),
        "reference_subject": int(identity.group("subject")) if identity else None,
    }


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    if value.ndim != 2 or value.shape[1] != 594:
        raise ValueError(f"expected [T, 594], got {tuple(value.shape)}")
    if value.shape[0] < 2:
        raise ValueError("fewer than two frames")
    if not torch.isfinite(value).all():
        raise ValueError("contains NaN or Inf")
    return value


def index_directory(directory: Path, errors: list, label: str) -> dict:
    by_reference = {}
    for path in sorted(directory.glob("*.pt")):
        try:
            fields = parse_motion_filename(path)
        except Exception as error:
            errors.append(f"{label}/{path.name}: {error}")
            continue
        reference_id = fields["reference_id"]
        if reference_id in by_reference:
            errors.append(
                f"{label}: duplicate reference ID {reference_id!r}: "
                f"{by_reference[reference_id]['path'].name}, {path.name}"
            )
            continue
        by_reference[reference_id] = {"path": path, **fields}
    return by_reference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--full-dir", required=True)
    parser.add_argument(
        "--manifest",
        help="Frozen policy_ab_manifest.json produced by the preparation script",
    )
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--output",
        required=True,
        help="Machine-readable validation JSON (written even when invalid)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    full_dir = Path(args.full_dir).resolve()
    asset_root = Path(args.asset_root).resolve()
    output_path = Path(args.output).resolve()
    errors = []
    for label, directory in (("raw", raw_dir), ("full", full_dir)):
        if not directory.is_dir():
            errors.append(f"{label} directory not found: {directory}")
    if errors:
        raw_index, full_index = {}, {}
    else:
        raw_index = index_directory(raw_dir, errors, "raw")
        full_index = index_directory(full_dir, errors, "full")

    raw_ids = set(raw_index)
    full_ids = set(full_index)
    if raw_ids != full_ids:
        errors.append(
            "Raw/Full reference sets differ: "
            f"raw_only={sorted(raw_ids - full_ids)}, "
            f"full_only={sorted(full_ids - raw_ids)}"
        )
    if not raw_ids:
        errors.append("No paired PT references found")
    if args.expected_count is not None and len(raw_ids & full_ids) != args.expected_count:
        errors.append(
            f"Expected {args.expected_count} paired references, found "
            f"{len(raw_ids & full_ids)}"
        )

    frozen_manifest = None
    frozen_by_reference = {}
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        try:
            frozen_manifest = json.loads(manifest_path.read_text())
            for entry in frozen_manifest["references"]:
                if entry.get("included") is not True:
                    errors.append(
                        f"Manifest reference {entry.get('reference_id')!r} "
                        "must have included=true"
                    )
                    continue
                reference_id = entry["reference_id"]
                if reference_id in frozen_by_reference:
                    errors.append(
                        f"Manifest duplicates reference {reference_id!r}"
                    )
                frozen_by_reference[reference_id] = entry
            if set(frozen_by_reference) != raw_ids or set(frozen_by_reference) != full_ids:
                errors.append(
                    "Manifest eligible reference set differs from PT directories"
                )
        except Exception as error:
            errors.append(f"Cannot read frozen manifest {manifest_path}: {error}")

    pairs = []
    asset_names = set()
    for reference_id in sorted(raw_ids & full_ids):
        raw_info = raw_index[reference_id]
        full_info = full_index[reference_id]
        raw_path = raw_info["path"]
        full_path = full_info["path"]
        pair_errors = []
        if raw_path.name != full_path.name:
            pair_errors.append(
                f"filenames differ: {raw_path.name} vs {full_path.name}"
            )
        for key in (
            "subject", "objects", "stage", "level", "height", "variation"
        ):
            if raw_info[key] != full_info[key]:
                pair_errors.append(f"{key} differs")
        if (
            raw_info["reference_subject"] is not None
            and raw_info["reference_subject"] != raw_info["subject"]
        ):
            pair_errors.append(
                "filename sub<number> disagrees with reference P<number>"
            )
        try:
            raw_tensor = load_tensor(raw_path)
            full_tensor = load_tensor(full_path)
            if raw_tensor.shape != full_tensor.shape:
                pair_errors.append(
                    f"frame/shape differs: {tuple(raw_tensor.shape)} vs "
                    f"{tuple(full_tensor.shape)}"
                )
            elif not torch.equal(
                raw_tensor[:, PAIR_COLUMNS], full_tensor[:, PAIR_COLUMNS]
            ):
                delta = (
                    raw_tensor[:, PAIR_COLUMNS]
                    - full_tensor[:, PAIR_COLUMNS]
                ).abs()
                pair_errors.append(
                    "object/contact columns 318:386 are not bit-identical "
                    f"(max_abs={float(delta.max()):.9g})"
                )
            contacts = raw_tensor[:, 332:386]
            if torch.any((contacts - contacts.round()).abs() > 1e-4):
                pair_errors.append("contact columns are not discrete")
            if torch.any((contacts < -1) | (contacts > 1)):
                pair_errors.append("contact columns fall outside [-1, 1]")
            frames = int(raw_tensor.shape[0])
        except Exception as error:
            pair_errors.append(str(error))
            frames = None
        raw_hash = sha256(raw_path)
        full_hash = sha256(full_path)

        frozen = frozen_by_reference.get(reference_id)
        result_height = (
            frozen.get("height") if frozen is not None else raw_info["height"]
        )
        result_variation = (
            frozen.get("variation")
            if frozen is not None
            else raw_info["variation"]
        )
        try:
            result_height = normalize_bucket(result_height, "L", 1, 5)
            result_variation = normalize_bucket(
                result_variation, "V", 1, 3
            )
        except Exception as error:
            pair_errors.append(str(error))
        if result_height is None or result_variation is None:
            pair_errors.append(
                "canonical height L1..L5 and variation V1..V3 are required"
            )
        if frozen is not None:
            expected = {
                "raw_filename": raw_path.name,
                "full_filename": full_path.name,
                "level": raw_info["level"],
                "subject": raw_info["subject"],
                "frames": frames,
                "raw_sha256": raw_hash,
                "full_sha256": full_hash,
            }
            if raw_info["height"] is not None:
                expected["height"] = raw_info["height"]
            if raw_info["variation"] is not None:
                expected["variation"] = raw_info["variation"]
            for key, actual in expected.items():
                if frozen.get(key) != actual:
                    pair_errors.append(
                        f"manifest {key}={frozen.get(key)!r}, actual={actual!r}"
                    )

        if pair_errors:
            errors.extend(
                f"{reference_id}: {message}" for message in pair_errors
            )
        asset_names.update(raw_info["objects"])
        pairs.append(
            {
                "reference_id": reference_id,
                "raw_filename": raw_path.name,
                "full_filename": full_path.name,
                "stage": raw_info["stage"],
                "level": raw_info["level"],
                "height": result_height,
                "variation": result_variation,
                "subject": raw_info["subject"],
                "objects": raw_info["objects"],
                "frames": frames,
                "raw_sha256": raw_hash,
                "full_sha256": full_hash,
                "pair_columns_equal": not any(
                    "object/contact columns" in message
                    for message in pair_errors
                ),
                "valid": not pair_errors,
            }
        )

    asset_entries = []
    for object_name in sorted(asset_names):
        mesh = asset_root / "objects" / object_name / f"{object_name}.obj"
        urdf = asset_root / f"{object_name}.urdf"
        entry = {
            "name": object_name,
            "mesh": str(mesh),
            "urdf": str(urdf),
            "mesh_sha256": sha256(mesh) if mesh.is_file() else None,
            "urdf_sha256": sha256(urdf) if urdf.is_file() else None,
        }
        if not mesh.is_file():
            errors.append(f"Missing object mesh: {mesh}")
        if not urdf.is_file():
            errors.append(f"Missing object URDF: {urdf}")
        asset_entries.append(entry)

    if frozen_manifest is not None:
        frozen_assets = {
            entry["name"]: entry for entry in frozen_manifest.get("assets", [])
        }
        if set(frozen_assets) != asset_names:
            errors.append("Manifest object asset set differs from paired PT files")
        for entry in asset_entries:
            frozen = frozen_assets.get(entry["name"])
            if frozen is None:
                continue
            for key in ("mesh_sha256", "urdf_sha256"):
                if frozen.get(key) != entry[key]:
                    errors.append(
                        f"Asset {entry['name']} {key} differs from manifest"
                    )

    result = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "full_dir": str(full_dir),
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "num_pairs": len(pairs),
        "reference_ids": [entry["reference_id"] for entry in pairs],
        "pairs": pairs,
        "assets": asset_entries,
        "valid": not errors,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(
            f"Policy A/B validation failed with {len(errors)} error(s); "
            f"report={output_path}"
        )


if __name__ == "__main__":
    main()
