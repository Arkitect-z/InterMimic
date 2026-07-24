#!/usr/bin/env python3
"""Fail fast unless all repositories match the formal server release."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


INTERMIMIC_ROOT = Path(__file__).resolve().parents[2]
THEIA_ROOT = INTERMIMIC_ROOT.parents[1]
VERSION_MANIFEST = INTERMIMIC_ROOT / "THEIA_POLICY_SERVER_VERSION.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def github_slug(remote):
    """Normalize HTTPS and common SSH GitHub remotes to owner/repository."""
    if not remote:
        return None
    match = re.search(
        r"(?:github\.com[:/]|ssh\.github\.com(?::\d+)?/)"
        r"([^/]+/[^/]+?)(?:\.git)?$",
        remote.strip(),
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def repository_report(name, path, specification):
    path = Path(path).resolve()
    errors = []
    try:
        head = git(path, "rev-parse", "HEAD")
        remote = git(path, "remote", "get-url", "origin")
        tracked_status = git(
            path, "status", "--short", "--untracked-files=no"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "valid": False,
            "path": str(path),
            "errors": ["not a readable Git repository: {}".format(exc)],
        }

    expected_slug = specification["github_slug"]
    actual_slug = github_slug(remote)
    if actual_slug is None or actual_slug.lower() != expected_slug.lower():
        errors.append(
            "origin mismatch: expected GitHub {}, got {!r}".format(
                expected_slug, remote
            )
        )
    if tracked_status:
        errors.append(
            "tracked files are modified: {!r}".format(tracked_status)
        )

    expected_commit = specification.get("commit")
    tag = specification.get("tag")
    tag_commit = None
    if expected_commit and head != expected_commit:
        errors.append(
            "commit mismatch: expected {}, got {}".format(
                expected_commit, head
            )
        )
    if tag:
        try:
            tag_commit = git(
                path, "rev-parse", "refs/tags/{}^{{commit}}".format(tag)
            )
        except (OSError, subprocess.CalledProcessError):
            errors.append(
                "required tag {!r} is missing; fetch tags first".format(tag)
            )
        if tag_commit is not None and head != tag_commit:
            errors.append(
                "HEAD {} is not the release tag {} ({})".format(
                    head, tag, tag_commit
                )
            )

    required_file = specification.get("required_file")
    expected_file_hash = specification.get("required_file_sha256")
    actual_file_hash = None
    if required_file:
        required_path = path / required_file
        if not required_path.is_file():
            errors.append("required file is missing: {}".format(required_path))
        else:
            actual_file_hash = sha256(required_path)
            if actual_file_hash != expected_file_hash:
                errors.append(
                    "{} hash mismatch: expected {}, got {}".format(
                        required_file,
                        expected_file_hash,
                        actual_file_hash,
                    )
                )

    return {
        "valid": not errors,
        "path": str(path),
        "head": head,
        "origin": remote,
        "github_slug": actual_slug,
        "tracked_status": tracked_status,
        "expected_commit": expected_commit,
        "release_tag": tag,
        "release_tag_commit": tag_commit,
        "required_file": required_file,
        "required_file_sha256": actual_file_hash,
        "errors": errors,
    }


def check_server_versions(version_manifest=VERSION_MANIFEST):
    version_manifest = Path(version_manifest).resolve()
    specification = json.loads(version_manifest.read_text())
    repositories = specification["repositories"]
    paths = {
        "Theia": THEIA_ROOT,
        "InterMimic": INTERMIMIC_ROOT,
        "ProtoMotions": THEIA_ROOT / "thirdparty" / "ProtoMotions",
    }
    reports = {
        name: repository_report(name, paths[name], repositories[name])
        for name in ("Theia", "InterMimic", "ProtoMotions")
    }
    errors = [
        "{}: {}".format(name, error)
        for name, report in reports.items()
        for error in report["errors"]
    ]
    return {
        "schema_version": 1,
        "valid": not errors,
        "release_name": specification["release_name"],
        "formal_method_status": specification["status"],
        "protocol": specification["protocol"],
        "version_manifest": str(version_manifest),
        "version_manifest_sha256": sha256(version_manifest),
        "repositories": reports,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version-manifest", default=str(VERSION_MANIFEST)
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()
    report = check_server_versions(args.version_manifest)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(
            ".{}.tmp.{}".format(output.name, os.getpid())
        )
        temporary.write_text(rendered)
        temporary.replace(output)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
