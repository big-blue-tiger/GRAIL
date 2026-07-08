#!/usr/bin/env python3
"""Convert one GRAIL object OBJ/mesh_data directory into an IsaacLab-ready USD.

This is the mesh-only version of the object conversion tail in
``grail.retargeting.retarget``:

    mesh_data/model.obj
      -> prepare_unique_mesh()
      -> convert_mesh.py / IsaacLab MeshConverter
      -> convert_to_sdf()
      -> fixup_texture_paths()
      -> ensure_physics_schemas()
      -> object_usd/<seq_name>.usd

Example:
    python -m grail.retargeting.convert_mesh2usd /home/robot/sbto/sbto/models/mesh/small_cylinder.obj /home/robot/sbto/sbto/models/usd/cylinder.usd
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _resolve_obj_path(input_path: Path) -> Path:
    """Accept either an OBJ file or a directory containing model.obj."""
    if input_path.is_dir():
        obj_path = input_path / "model.obj"
    else:
        obj_path = input_path

    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    if obj_path.suffix.lower() != ".obj":
        raise ValueError(f"Expected an .obj file or directory containing model.obj, got: {input_path}")
    return obj_path


def _default_seq_name(obj_path: Path, output_path: Path | None) -> str:
    if output_path is not None and output_path.stem:
        return output_path.stem
    if obj_path.parent.name == "mesh_data" and obj_path.parent.parent.name:
        return obj_path.parent.parent.name
    return obj_path.stem


def prepare_unique_mesh(obj_path: Path, seq_name: str, tmp_dir: Path) -> Path:
    """Copy an OBJ asset to tmp_dir with uniquely named texture files.

    This mirrors retarget.py's mesh preparation, but accepts any OBJ filename.
    It copies sibling MTL files and texture images from the OBJ directory and
    its images/ subdirectory, then rewrites MTL texture references to avoid
    filename collisions across batch conversions.
    """
    mesh_dir = obj_path.parent
    work_dir = tmp_dir / seq_name
    work_dir.mkdir(parents=True, exist_ok=True)

    patched_obj = work_dir / obj_path.name
    shutil.copy2(obj_path, patched_obj)

    rename_map = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for tex in mesh_dir.glob(ext):
            new_name = f"{seq_name}_{tex.name}"
            rename_map[tex.name] = new_name
            shutil.copy2(tex, work_dir / new_name)
        for tex in mesh_dir.glob(f"images/{ext}"):
            new_name = f"{seq_name}_{tex.name}"
            rename_map[f"images/{tex.name}"] = new_name
            shutil.copy2(tex, work_dir / new_name)

    for mtl_path in mesh_dir.glob("*.mtl"):
        mtl_text = mtl_path.read_text()
        for old_ref, new_name in rename_map.items():
            mtl_text = mtl_text.replace(old_ref, new_name)
        (work_dir / mtl_path.name).write_text(mtl_text)

    return patched_obj


def convert_to_sdf(file_path: Path, sdf_resolution: int = 256) -> bool:
    """Convert USD collision approximation attributes to PhysX SDF collision."""
    from pxr import Sdf, Usd

    try:
        from pxr import PhysxSchema

        has_physx = True
    except ImportError:
        PhysxSchema = None
        has_physx = False

    stage = Usd.Stage.Open(str(file_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD: {file_path}")

    modified_count = 0
    for prim in stage.Traverse():
        approx_attr = prim.GetAttribute("physics:approximation")
        if not approx_attr or not approx_attr.IsValid():
            continue

        current = approx_attr.Get()
        if current == "sdf":
            continue

        print(f"  {prim.GetPath()}: {current} -> sdf")
        if has_physx:
            approx_attr.Set(PhysxSchema.Tokens.sdf)
            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            sdf_api.CreateSdfResolutionAttr().Set(sdf_resolution)
        else:
            approx_attr.Set("sdf")
            current_schemas = list(prim.GetAppliedSchemas())
            if "PhysxSDFMeshCollisionAPI" not in current_schemas:
                current_schemas.append("PhysxSDFMeshCollisionAPI")
                new_listop = Sdf.TokenListOp()
                new_listop.explicitItems = current_schemas
                prim.SetMetadata("apiSchemas", new_listop)

            sdf_res_attr = prim.CreateAttribute(
                "physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int, custom=False
            )
            sdf_res_attr.Set(sdf_resolution)

        modified_count += 1

    if modified_count > 0:
        stage.GetRootLayer().Save()
        print(f"Saved {file_path} ({modified_count} SDF collision prims)")

    return True


def fixup_texture_paths(usd_path: Path, seq_name: str) -> None:
    """Move USD texture references to textures/<seq_name>/ relative paths."""
    from pxr import Sdf, Usd

    usd_dir = usd_path.parent
    seq_tex_dir = usd_dir / "textures" / seq_name

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    modified = False
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            val = attr.Get()
            if not isinstance(val, Sdf.AssetPath):
                continue

            path_str = val.resolvedPath or val.path
            if not path_str:
                continue
            if not any(ext in path_str.lower() for ext in [".jpg", ".png", ".jpeg"]):
                continue

            tex_file = Path(path_str)
            if not tex_file.is_absolute():
                tex_file = (usd_dir / path_str).resolve()

            if not tex_file.exists():
                continue

            fname = tex_file.name
            clean_name = fname[len(seq_name) + 1 :] if fname.startswith(f"{seq_name}_") else fname

            seq_tex_dir.mkdir(parents=True, exist_ok=True)
            dest = seq_tex_dir / clean_name
            if not dest.exists():
                shutil.copy2(tex_file, dest)

            attr.Set(Sdf.AssetPath(f"textures/{seq_name}/{clean_name}"))
            modified = True

    if modified:
        stage.GetRootLayer().Save()


def ensure_physics_schemas(usd_path: Path, mass: float = 1.0) -> None:
    """Apply missing rigid-body, mass, and collision schemas in-place."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    root = stage.GetDefaultPrim()
    if not root:
        return

    modified = False
    applied = set(root.GetAppliedSchemas())
    if "PhysicsRigidBodyAPI" not in applied:
        UsdPhysics.RigidBodyAPI.Apply(root)
        modified = True
    if "PhysicsMassAPI" not in applied:
        mass_api = UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr().Set(mass)
        modified = True

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh_applied = set(prim.GetAppliedSchemas())
        if "PhysicsCollisionAPI" not in mesh_applied:
            UsdPhysics.CollisionAPI.Apply(prim)
            modified = True
        if "PhysicsMeshCollisionAPI" not in mesh_applied:
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            modified = True

    if modified:
        stage.GetRootLayer().Save()


def convert_mesh2usd(
    input_path: Path,
    output_path: Path,
    *,
    seq_name: str | None = None,
    mass: float = 1.0,
    mesh_scale: float = 1.0,
    collision_approximation: str = "meshSimplification",
    sdf_resolution: int = 256,
    skip_sdf: bool = False,
    skip_texture_fix: bool = False,
    skip_physics_fix: bool = False,
) -> Path:
    """Run the same OBJ->USD conversion pipeline used by retarget.py."""
    obj_path = _resolve_obj_path(input_path)
    seq_name = seq_name or _default_seq_name(obj_path, output_path)

    output_path = output_path.with_suffix(".usd")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        patched_obj = prepare_unique_mesh(obj_path, seq_name, Path(tmp_dir))
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("convert_mesh.py")),
            str(patched_obj),
            str(output_path),
            "--headless",
            "--mass",
            str(mass),
            "--scale",
            str(mesh_scale),
            "--collision-approximation",
            collision_approximation,
        ]
        subprocess.run(cmd, check=True)

    if not skip_sdf:
        convert_to_sdf(output_path, sdf_resolution=sdf_resolution)
    if not skip_texture_fix:
        fixup_texture_paths(output_path, seq_name)
    if not skip_physics_fix:
        ensure_physics_schemas(output_path, mass=mass)

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an OBJ file or a directory containing model.obj.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output USD path, e.g. data/.../object_usd/<seq_name>.usd.",
    )
    parser.add_argument(
        "--seq-name",
        type=str,
        default=None,
        help="Sequence/object stem used for texture names. Defaults to output filename stem.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="Mass passed to MeshConverter and physics schema repair.",
    )
    parser.add_argument(
        "--mesh-scale",
        "--scale",
        dest="mesh_scale",
        type=float,
        default=1.0,
        help="Uniform mesh scale passed to MeshConverter.",
    )
    parser.add_argument(
        "--collision-approximation",
        type=str,
        default="meshSimplification",
        choices=[
            "convexDecomposition",
            "convexHull",
            "boundingCube",
            "boundingSphere",
            "meshSimplification",
            "none",
        ],
        help="Initial collision approximation passed to convert_mesh.py.",
    )
    parser.add_argument(
        "--sdf-resolution",
        type=int,
        default=256,
        help="SDF resolution used by convert_to_sdf().",
    )
    parser.add_argument(
        "--skip-sdf",
        action="store_true",
        help="Skip converting physics:approximation to SDF.",
    )
    parser.add_argument(
        "--skip-texture-fix",
        action="store_true",
        help="Skip moving USD texture references to textures/<seq_name>/.",
    )
    parser.add_argument(
        "--skip-physics-fix",
        action="store_true",
        help="Skip the RigidBody/Mass/Collision schema repair pass.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    usd_path = convert_mesh2usd(
        args.input,
        args.output,
        seq_name=args.seq_name,
        mass=args.mass,
        mesh_scale=args.mesh_scale,
        collision_approximation=args.collision_approximation,
        sdf_resolution=args.sdf_resolution,
        skip_sdf=args.skip_sdf,
        skip_texture_fix=args.skip_texture_fix,
        skip_physics_fix=args.skip_physics_fix,
    )
    print(f"Converted USD: {usd_path}")


if __name__ == "__main__":
    main()
