#!/usr/bin/env python3
"""Fail fast unless Theia uses the ProtoMotions revision used for conversion."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "4a905b998101333a2fb91f2de8e2cab4bd0db68e"
EXPECTED_REMOTE = "https://github.com/NVlabs/ProtoMotions.git"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_protomotions(proto: Path) -> dict:
    """Return a machine-readable report for the pinned conversion dependency."""
    proto = proto.resolve()
    skeleton = (
        proto
        / "poselib"
        / "poselib"
        / "skeleton"
        / "skeleton3d.py"
    )
    errors = []
    try:
        commit = git(proto, "rev-parse", "HEAD")
        tracked_status = git(
            proto, "status", "--short", "--untracked-files=no"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        commit = None
        tracked_status = None
        errors.append(f"not a readable Git repository: {exc}")
    try:
        remote = git(proto, "remote", "get-url", "origin")
    except (OSError, subprocess.CalledProcessError):
        remote = None
        errors.append("origin remote is missing")

    if commit != EXPECTED_COMMIT:
        errors.append(
            f"commit mismatch: expected {EXPECTED_COMMIT}, got {commit}"
        )
    if tracked_status:
        errors.append(
            "tracked ProtoMotions files are modified; conversion would no "
            f"longer match the pinned revision: {tracked_status!r}"
        )
    if not skeleton.is_file():
        errors.append(f"SkeletonMotion source is missing: {skeleton}")

    return {
        "schema_version": 1,
        "valid": not errors,
        "protomotions_dir": str(proto),
        "expected_commit": EXPECTED_COMMIT,
        "actual_commit": commit,
        "expected_remote": EXPECTED_REMOTE,
        "actual_remote": remote,
        "tracked_status": tracked_status,
        "skeleton_motion_source": str(skeleton),
        "skeleton_motion_sha256": (
            sha256(skeleton) if skeleton.is_file() else None
        ),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protomotions-dir")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Diagnostic escape hatch; forbidden for the paper conversion.",
    )
    args = parser.parse_args()

    intermimic_root = Path(__file__).resolve().parents[2]
    theia_root = intermimic_root.parents[1]
    proto = Path(
        args.protomotions_dir
        or theia_root / "thirdparty" / "ProtoMotions"
    ).resolve()
    report = check_protomotions(proto)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    if report["errors"] and not args.allow_version_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
