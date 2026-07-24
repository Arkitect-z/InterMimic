#!/usr/bin/env python3
"""Verify that cluster/GPU lists partition the frozen eligible references."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_list(path):
    values = []
    seen = set()
    for raw_line in Path(path).read_text().splitlines():
        token = raw_line.split("#", 1)[0].strip()
        if not token:
            continue
        reference_id = Path(token.rstrip("/")).name
        if reference_id in seen:
            raise ValueError(
                "{} duplicates reference {!r}".format(path, reference_id)
            )
        seen.add(reference_id)
        values.append(reference_id)
    if not values:
        raise ValueError("{} contains no references".format(path))
    return values


def validate_partition(manifest_path, list_paths):
    manifest_path = Path(manifest_path)
    with manifest_path.open() as source:
        manifest = json.load(source)
    expected = {
        str(entry["reference_id"])
        for entry in manifest.get("references", [])
        if entry.get("included") is True
    }
    if not expected:
        raise ValueError("Paired manifest has no eligible references")

    owners = {}
    lists = []
    duplicates = {}
    for list_path in list_paths:
        list_path = Path(list_path).resolve()
        reference_ids = parse_list(list_path)
        lists.append({
            "path": str(list_path),
            "sha256": sha256(list_path),
            "count": len(reference_ids),
            "reference_ids": reference_ids,
        })
        for reference_id in reference_ids:
            if reference_id in owners:
                duplicates.setdefault(reference_id, [owners[reference_id]])
                duplicates[reference_id].append(str(list_path))
            else:
                owners[reference_id] = str(list_path)

    actual = set(owners)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    report = {
        "schema_version": 1,
        "valid": not missing and not extra and not duplicates,
        "paired_manifest": str(manifest_path.resolve()),
        "paired_manifest_sha256": sha256(manifest_path),
        "eligible_reference_count": len(expected),
        "assigned_reference_count": len(actual),
        "list_count": len(lists),
        "missing": missing,
        "extra": extra,
        "duplicates": duplicates,
        "lists": lists,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("lists", nargs="+")
    args = parser.parse_args()
    report = validate_partition(
        Path(args.pair_manifest), [Path(path) for path in args.lists]
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
