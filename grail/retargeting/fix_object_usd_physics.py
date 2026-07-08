#!/usr/bin/env python3
"""
Repair externally generated object USDs so SONIC/IsaacLab can spawn them as rigid objects.

This is mainly for motion libraries that did not go through GRAIL's retargeting
pipeline and therefore may be missing:
  - PhysicsRigidBodyAPI on the default prim
  - PhysicsMassAPI on the default prim
  - PhysicsCollisionAPI / PhysicsMeshCollisionAPI on mesh prims
  - SDF collision approximation on collision-enabled meshes

Example:
  conda activate sonic
  python -m grail.retargeting.fix_object_usd_physics \
      --dir data/SBTO/pickBottle/2026_05_30__19_01_53/object_usd \
      --recursive
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdPhysics

from grail.retargeting.convert_collision_to_sdf import convert_to_sdf


def ensure_physics_schemas(usd_path: Path, mass: float = 1.0, dry_run: bool = False) -> bool:
    """Apply missing rigid-body related schemas in-place."""
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError(f"USD has no defaultPrim: {usd_path}")

    modified = False
    root_applied = set(root.GetAppliedSchemas())

    if "PhysicsRigidBodyAPI" not in root_applied:
        print(f"[fix] {usd_path}: applying PhysicsRigidBodyAPI to {root.GetPath()}")
        if not dry_run:
            UsdPhysics.RigidBodyAPI.Apply(root)
        modified = True

    if "PhysicsMassAPI" not in root_applied:
        print(f"[fix] {usd_path}: applying PhysicsMassAPI(mass={mass}) to {root.GetPath()}")
        if not dry_run:
            mass_api = UsdPhysics.MassAPI.Apply(root)
            mass_api.CreateMassAttr().Set(mass)
        modified = True

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh_applied = set(prim.GetAppliedSchemas())
        if "PhysicsCollisionAPI" not in mesh_applied:
            print(f"[fix] {usd_path}: applying PhysicsCollisionAPI to {prim.GetPath()}")
            if not dry_run:
                UsdPhysics.CollisionAPI.Apply(prim)
            modified = True
        if "PhysicsMeshCollisionAPI" not in mesh_applied:
            print(f"[fix] {usd_path}: applying PhysicsMeshCollisionAPI to {prim.GetPath()}")
            if not dry_run:
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            modified = True

    if modified and not dry_run:
        stage.GetRootLayer().Save()

    return modified


def iter_usd_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*.usd*" if recursive else "*.usd*"
    return sorted(
        p for p in path.glob(pattern) if p.is_file() and p.suffix in {".usd", ".usda", ".usdc"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        help="Repair a single USD file.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Repair all USD files in a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when used with --dir.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="Default mass to assign when PhysicsMassAPI is missing.",
    )
    parser.add_argument(
        "--skip-sdf",
        action="store_true",
        help="Do not convert collision approximation to SDF after schema repair.",
    )
    parser.add_argument(
        "--sdf-resolution",
        type=int,
        default=256,
        help="SDF resolution passed to convert_collision_to_sdf.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report intended changes without modifying files.",
    )
    args = parser.parse_args()

    if bool(args.file) == bool(args.dir):
        raise SystemExit("Pass exactly one of --file or --dir.")

    target = args.file or args.dir
    files = iter_usd_files(target, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No USD files found under: {target}")

    repaired = 0
    processed = 0
    for usd_path in files:
        processed += 1
        modified = ensure_physics_schemas(usd_path, mass=args.mass, dry_run=args.dry_run)
        if not args.skip_sdf:
            convert_to_sdf(usd_path, dry_run=args.dry_run, sdf_resolution=args.sdf_resolution)
        if modified:
            repaired += 1

    print(
        f"[summary] processed={processed} repaired={repaired} "
        f"sdf={'off' if args.skip_sdf else 'on'} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
