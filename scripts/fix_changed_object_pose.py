#!/usr/bin/env python3
"""Align changed-object PKLs to a reference USD and table metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from pxr import Usd, UsdGeom


DEFAULT_OBJECT_QUAT_WXYZ = np.array(
    [0.70396221, 0.70212492, 0.08075711, 0.07025730], dtype=np.float32
)
TABLE_THICKNESS = 0.01
TABLE_CLEARANCE = 0.01


def quaternion_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def usd_origin_to_bottom(usd_path: Path, quat_wxyz: np.ndarray) -> float:
    """Return origin-to-bottom distance after applying the requested root rotation."""
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Could not open USD: {usd_path}")

    points = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        points.extend(
            np.asarray(transform.Transform(point), dtype=np.float64)
            for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()
        )

    if not points:
        raise ValueError(f"USD contains no mesh vertices: {usd_path}")

    rotated_points = np.asarray(points) @ quaternion_rotation_matrix(quat_wxyz).T
    return -float(rotated_points[:, 2].min())


def unwrap_single_entry(data: object, path: Path) -> dict:
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Expected one wrapped object motion in {path}")
    motion = next(iter(data.values()))
    if not isinstance(motion, dict):
        raise ValueError(f"Expected a motion dictionary in {path}")
    return motion


def refine_file(
    object_path: Path,
    meta_path: Path,
    origin_to_bottom: float,
    dry_run: bool,
) -> float:
    wrapped_motion = joblib.load(object_path)
    motion = unwrap_single_entry(wrapped_motion, object_path)
    meta = joblib.load(meta_path)

    root_pos = np.asarray(motion.get("root_pos"))
    root_quat = np.asarray(motion.get("root_quat"))
    if root_pos.ndim != 3 or root_pos.shape[1:] != (1, 3):
        raise ValueError(
            f"Expected root_pos shape (T, 1, 3), got {root_pos.shape}: "
            f"{object_path}"
        )
    if root_quat.shape != (root_pos.shape[0], 1, 4):
        raise ValueError(
            f"Expected root_quat shape {(root_pos.shape[0], 1, 4)}, "
            f"got {root_quat.shape}: {object_path}"
        )
    if root_pos.shape[0] < 20:
        raise ValueError(f"Expected at least 20 frames: {object_path}")
    if not isinstance(meta, dict) or "table_pos" not in meta:
        raise KeyError(f"Meta PKL has no table_pos: {meta_path}")

    table_pos = np.asarray(meta["table_pos"], dtype=np.float64)
    if table_pos.shape != (3,):
        raise ValueError(
            f"Expected table_pos shape (3,), got {table_pos.shape}: {meta_path}"
        )

    h1 = float(root_pos[:20, 0, 2].mean())
    target_z = (
        TABLE_THICKNESS
        + origin_to_bottom
        + float(table_pos[2])
        + TABLE_CLEARANCE
    )
    z_offset = target_z - h1

    root_pos[:, 0, 2] += np.asarray(z_offset, dtype=root_pos.dtype)
    root_quat[:, 0, :] = DEFAULT_OBJECT_QUAT_WXYZ.astype(root_quat.dtype, copy=False)

    if not dry_run:
        temporary_path = object_path.with_suffix(object_path.suffix + ".tmp")
        try:
            joblib.dump(wrapped_motion, temporary_path)
            os.replace(temporary_path, object_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    return z_offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix object root orientation and table-aligned z using a reference USD."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw_grail_chaged_obj/pickup_table"),
    )
    parser.add_argument(
        "--reference-usd",
        type=Path,
        default=Path(
            "data/raw_grail_chaged_obj/pickup_table/object_usd/"
            "pickup_table__apple_0__000.usd"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    object_dir = args.dataset_dir / "objects"
    meta_dir = args.dataset_dir / "meta"
    object_paths = sorted(object_dir.glob("*.pkl"))
    if not object_paths:
        raise ValueError(f"No object PKLs found in {object_dir}")

    origin_to_bottom = usd_origin_to_bottom(args.reference_usd, DEFAULT_OBJECT_QUAT_WXYZ)
    offsets = []
    for object_path in object_paths:
        meta_path = meta_dir / object_path.name
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing matching metadata: {meta_path}")
        offsets.append(
            refine_file(object_path, meta_path, origin_to_bottom, args.dry_run)
        )

    mode = "Checked" if args.dry_run else "Updated"
    print(f"{mode} {len(object_paths)} object PKLs")
    print(f"reference USD: {args.reference_usd}")
    print(f"origin-to-bottom after fixed rotation: {origin_to_bottom:.9f} m")
    print(f"z offset range: [{min(offsets):.9f}, {max(offsets):.9f}] m")


if __name__ == "__main__":
    main()
