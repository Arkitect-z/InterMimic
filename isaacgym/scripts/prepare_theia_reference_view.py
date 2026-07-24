#!/usr/bin/env python3
"""Create an immutable one-reference Raw/Refined view of prepared A/B data."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Existing immutable file differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content)
    temporary.replace(path)


def ensure_link(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(
                f"Existing link has the wrong target: {link} -> {link.resolve()}"
            )
        return
    if link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target)


def create_view(
    prepared_root: Path,
    reference_id: str,
    output_root: Path,
) -> dict:
    prepared_root = prepared_root.resolve()
    manifest_path = prepared_root / "policy_ab_manifest.json"
    with manifest_path.open() as source:
        parent = json.load(source)

    matches = [
        entry
        for entry in parent.get("references", [])
        if entry.get("included", True)
        and str(entry.get("reference_id")) == reference_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one included reference {reference_id!r}, "
            f"found {len(matches)}"
        )
    entry = dict(matches[0])
    output_root.mkdir(parents=True, exist_ok=True)

    for condition in ("raw", "full"):
        filename = entry[f"{condition}_filename"]
        expected_hash = entry[f"{condition}_sha256"]
        source = prepared_root / "eligible" / condition / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{condition} hash mismatch for {reference_id}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        directory = output_root / "data" / condition
        directory.mkdir(parents=True, exist_ok=True)
        ensure_link(directory / filename, source)

    view_manifest = {
        key: value
        for key, value in parent.items()
        if key not in {
            "references",
            "candidate_reference_ids",
            "candidate_count",
            "eligible_count",
            "excluded_count",
            "excluded_pairs",
            "eligible_raw_dir",
            "eligible_full_dir",
        }
    }
    view_manifest.update(
        {
            "schema_version": 1,
            "view_kind": "single_reference_paired_policy",
            "parent_manifest": str(manifest_path),
            "parent_manifest_sha256": sha256(manifest_path),
            "candidate_count": 1,
            "candidate_reference_ids": [reference_id],
            "eligible_count": 1,
            "excluded_count": 0,
            "excluded_pairs": [],
            "eligible_raw_dir": str((output_root / "data" / "raw").resolve()),
            "eligible_full_dir": str((output_root / "data" / "full").resolve()),
            "references": [entry],
        }
    )
    rendered = json.dumps(view_manifest, indent=2, sort_keys=True) + "\n"
    write_once(output_root / "pair_manifest.json", rendered)
    receipt = {
        "schema_version": 1,
        "reference_id": reference_id,
        "view_root": str(output_root.resolve()),
        "pair_manifest": str((output_root / "pair_manifest.json").resolve()),
        "pair_manifest_sha256": sha256(output_root / "pair_manifest.json"),
        "raw_data_dir": str((output_root / "data" / "raw").resolve()),
        "full_data_dir": str((output_root / "data" / "full").resolve()),
    }
    write_once(
        output_root / "VIEW_READY.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = create_view(
        Path(args.prepared_root),
        args.reference_id.strip(),
        Path(args.output_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
