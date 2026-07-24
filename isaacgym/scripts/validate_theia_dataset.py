#!/usr/bin/env python3
"""Fast CPU preflight for Theia multi-sequence training data."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
OBJECT_ROOT = (
    REPO_ROOT
    / "isaacgym"
    / "src"
    / "intermimic"
    / "data"
    / "assets"
    / "objects"
)
THEIA_XML = (
    REPO_ROOT
    / "isaacgym"
    / "src"
    / "intermimic"
    / "data"
    / "assets"
    / "smplx"
    / "theia.xml"
)
JOINT_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]
LEFT_HAND_IDS = list(range(17, 33))
RIGHT_HAND_IDS = list(range(36, 52))
HAND_IDS = set(LEFT_HAND_IDS + RIGHT_HAND_IDS)
NON_HAND_IDS = [idx for idx in range(52) if idx not in HAND_IDS]
FOOT_NAMES = ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_for_manifest(path):
    """Prefer repository-relative paths, while supporting external datasets."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def foot_collision_min_z(data):
    """Compute minimum world-space foot-box corner height from PT body poses."""
    body_elements = {
        elem.attrib["name"]: elem
        for elem in ET.parse(THEIA_XML).getroot().iter("body")
        if "name" in elem.attrib
    }
    body_pos = data[:, 162:318].reshape(-1, 52, 3).numpy()
    body_rot = data[:, 386:594].reshape(-1, 52, 4).numpy()
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1],
        [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1],
        [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    min_z = np.inf

    for body_name in FOOT_NAMES:
        body_idx = JOINT_NAMES.index(body_name)
        geom = body_elements[body_name].find("geom")
        if geom is None or geom.attrib.get("type") != "box":
            raise ValueError(
                f"expected box collision geometry for {body_name}"
            )
        center = np.fromstring(
            geom.attrib.get("pos", "0 0 0"), sep=" "
        )
        half_size = np.fromstring(geom.attrib["size"], sep=" ")
        quat_wxyz = np.fromstring(
            geom.attrib.get("quat", "1 0 0 0"), sep=" "
        )
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        local_points = (
            Rotation.from_quat(quat_xyzw).apply(signs * half_size)
            + center
        )
        rotations = Rotation.from_quat(body_rot[:, body_idx])
        for local_point in local_points:
            world_point = rotations.apply(
                np.broadcast_to(
                    local_point, body_pos[:, body_idx].shape
                )
            )
            world_point += body_pos[:, body_idx]
            min_z = min(min_z, float(world_point[:, 2].min()))
    return min_z


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="isaacgym/src/intermimic/data/cfg/theia_full_train.yaml",
    )
    parser.add_argument(
        "--motion-file",
        help="Override env.motion_file with a server data directory",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--manifest")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    with config_path.open() as source:
        env = yaml.safe_load(source)["env"]

    data_dir = Path(args.motion_file or env["motion_file"])
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    subsets = env.get("dataSub", ["*"])
    include_all = not subsets or any(
        str(value).lower() in {"*", "all"} for value in subsets
    )
    files = sorted(
        path
        for path in data_dir.iterdir()
        if path.suffix == ".pt"
        and (include_all or path.name.split("_", 1)[0] in subsets)
    )
    if not files:
        raise SystemExit(
            f"No .pt files selected from {data_dir} by dataSub={subsets}"
        )

    num_envs = args.num_envs or int(env["numEnvs"])
    if len(files) > num_envs:
        raise SystemExit(
            f"{len(files)} sequences require at least that many environments; "
            f"got numEnvs={num_envs}"
        )

    density = env["objectDensity"]
    require_density = bool(env.get("requireObjectDensity", False))
    require_bimanual = bool(env.get("requireBimanualContact", False))
    minimum_foot_z = float(
        env.get("minimumFootCollisionZ", -float("inf"))
    )
    sequences = []
    objects = set()
    errors = []
    if require_density and not isinstance(density, dict):
        errors.append(
            "requireObjectDensity needs an objectDensity mapping"
        )
    for path in files:
        stem_tokens = path.stem.split("_")
        subset = stem_tokens[0]
        pair_tokens = [token for token in stem_tokens if "+" in token]
        if not subset.startswith("sub") or not subset[3:].isdigit():
            errors.append(f"{path.name}: filename must start with sub<number>_")
            continue
        if len(pair_tokens) != 1:
            errors.append(
                f"{path.name}: expected exactly one ObjA+ObjB filename token"
            )
            continue
        obj1, obj2 = pair_tokens[0].split("+", 1)
        objects.update((obj1, obj2))

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            errors.append(f"{path.name}: torch.load failed: {exc}")
            continue
        if data.ndim != 2 or data.shape[1] != 594:
            errors.append(
                f"{path.name}: expected [T, 594], got {tuple(data.shape)}"
            )
            continue
        if data.shape[0] < 2:
            errors.append(f"{path.name}: fewer than two frames")
        if not torch.isfinite(data).all():
            errors.append(f"{path.name}: contains NaN or Inf")

        for label, start, end in [
            ("root", 3, 7),
            ("obj1", 321, 325),
            ("obj2", 328, 332),
        ]:
            norms = data[:, start:end].norm(dim=-1)
            if torch.any((norms < 0.99) | (norms > 1.01)):
                errors.append(f"{path.name}: non-unit {label} quaternion")
        body_norms = data[:, 386:594].view(-1, 52, 4).norm(dim=-1)
        if torch.any((body_norms < 0.99) | (body_norms > 1.01)):
            errors.append(f"{path.name}: non-unit body quaternion")
        contacts = data[:, 332:386]
        if torch.any((contacts - contacts.round()).abs() > 1e-4):
            errors.append(f"{path.name}: non-discrete contact labels")
        if torch.any((contacts < -1.0) | (contacts > 1.0)):
            errors.append(f"{path.name}: contact labels outside [-1, 1]")
        human_contact = data[:, 334:386]
        if torch.any(human_contact[:, NON_HAND_IDS].abs() > 1e-4):
            errors.append(
                f"{path.name}: non-hand contact labels must be neutral 0"
            )
        left_contact = (
            human_contact[:, LEFT_HAND_IDS] > 0.5
        ).any(dim=-1)
        right_contact = (
            human_contact[:, RIGHT_HAND_IDS] > 0.5
        ).any(dim=-1)
        obj1_contact = data[:, 332] > 0.5
        obj2_contact = data[:, 333] > 0.5
        if not torch.equal(left_contact, obj1_contact):
            errors.append(
                f"{path.name}: obj1 contact is inconsistent with left hand"
            )
        if not torch.equal(right_contact, obj2_contact):
            errors.append(
                f"{path.name}: obj2 contact is inconsistent with right hand"
            )
        if require_bimanual and not left_contact.any():
            errors.append(f"{path.name}: no positive left/obj1 contact frames")
        if require_bimanual and not right_contact.any():
            errors.append(f"{path.name}: no positive right/obj2 contact frames")

        try:
            foot_min_z = foot_collision_min_z(data)
        except Exception as exc:
            errors.append(
                f"{path.name}: foot collision check failed: {exc}"
            )
            foot_min_z = None
        if foot_min_z is not None and foot_min_z < minimum_foot_z:
            errors.append(
                f"{path.name}: foot collision min z {foot_min_z:.6f}m "
                f"is below {minimum_foot_z:.6f}m"
            )

        sequences.append({
            "path": path_for_manifest(path),
            "frames": int(data.shape[0]),
            "objects": [obj1, obj2],
            "left_contact_frames": int(left_contact.sum()),
            "right_contact_frames": int(right_contact.sum()),
            "simultaneous_contact_frames": int(
                (left_contact & right_contact).sum()
            ),
            "foot_collision_min_z_m": foot_min_z,
            "sha256": sha256(path),
        })

    for object_name in sorted(objects):
        for suffix in (".obj",):
            asset = (
                OBJECT_ROOT
                / "objects"
                / object_name
                / f"{object_name}{suffix}"
            )
            if not asset.is_file():
                errors.append(f"missing object mesh: {asset}")
        urdf = OBJECT_ROOT / f"{object_name}.urdf"
        if not urdf.is_file():
            errors.append(f"missing object URDF: {urdf}")
        if (
            require_density
            and isinstance(density, dict)
            and object_name not in density
        ):
            errors.append(f"missing explicit objectDensity for {object_name}")

    manifest = {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "data_dir": str(data_dir),
        "num_sequences": len(files),
        "num_envs": num_envs,
        "envs_per_sequence_min": num_envs // len(files),
        "envs_per_sequence_max": (
            num_envs + len(files) - 1
        ) // len(files),
        "frame_count_min": min(
            (sequence["frames"] for sequence in sequences),
            default=None,
        ),
        "frame_count_max": max(
            (sequence["frames"] for sequence in sequences),
            default=None,
        ),
        "objects": sorted(objects),
        "sequences": sequences,
        "valid": not errors,
        "errors": errors,
    }
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(f"Theia dataset preflight failed with {len(errors)} error(s)")


if __name__ == "__main__":
    main()
