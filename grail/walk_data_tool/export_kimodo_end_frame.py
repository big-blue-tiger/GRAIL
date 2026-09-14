#!/usr/bin/env python3
"""Export robot PKL frame 0 as a Kimodo G1 fullbody JSON constraint.

Run in the GRAIL environment; uses local Kimodo kinematics without loading a
model, CUDA, IsaacLab, or the language encoder. See kimodo_end_frame.md.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import shlex
from pathlib import Path
import sys
import types

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/hf_dataset/data_update/data/pickup_table_cleaned_succeeded/robot"


def load_kinematics(repo: Path):
    """Expose the local package without its eager LLM/model imports.

    Only skip kimodo/__init__.py; all kinematics/conversion code and assets are
    imported unchanged from the selected checkout. This is a standalone CLI.
    """
    package = repo.resolve() / "kimodo"
    if not (package / "exports/mujoco.py").is_file():
        raise ValueError(f"Not a Kimodo checkout: {repo}")
    if "kimodo" not in sys.modules:
        module = types.ModuleType("kimodo")
        module.__path__ = [str(package)]
        module.__file__ = str(package / "__init__.py")
        sys.modules["kimodo"] = module
    elif Path(sys.modules["kimodo"].__file__).resolve().parent != package:
        raise ValueError("A different Kimodo checkout is already imported")
    from kimodo.constraints import FullBodyConstraintSet
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.geometry import matrix_to_axis_angle
    from kimodo.skeleton import G1Skeleton34

    skeleton = G1Skeleton34()
    return skeleton, MujocoQposConverter(skeleton), matrix_to_axis_angle, FullBodyConstraintSet


def first_qpos(motion):
    """PKL stores absolute Z-up xyz, xyzw and MuJoCo-ordered body angles."""
    parts = []
    lengths = []
    for key, width in (("root_trans_offset", 3), ("root_rot", 4), ("dof", 29)):
        value = np.asarray(motion[key])
        if value.ndim != 2 or value.shape[1] != width or len(value) == 0:
            raise ValueError(f"{key}: expected nonempty (T,{width}), got {value.shape}")
        if not np.isfinite(value[0]).all():
            raise ValueError(f"{key}: nonfinite first frame")
        parts.append(value[:1].astype(np.float32))
        lengths.append(len(value))
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent frame counts: {lengths}")
    qpos = np.concatenate(parts, axis=1)
    norm = np.linalg.norm(qpos[:, 3:7])
    if abs(float(norm) - 1.0) > 0.01:
        raise ValueError(f"Invalid xyzw quaternion norm: {norm}")
    qpos[:, 3:7] /= norm
    return qpos


def sample_start(qpos, meta, rng):
    """Uniform in area of the 1--2 m, 150-degree annular sector.

    IsaacLab consumes table_quat as wxyz. Choose the outward normal to the
    long table edge by the robot position, never by its heading.
    """
    center = np.asarray(meta["table_pos"], dtype=float)
    size = np.asarray(meta["table_size"], dtype=float)
    quat = np.asarray(meta["table_quat"], dtype=float)
    if (center.shape != (3,) or size.shape != (3,) or quat.shape != (4,)
            or not all(np.isfinite(v).all() for v in (center, size, quat))
            or np.any(size <= 0) or abs(np.linalg.norm(quat) - 1) > 0.01):
        raise ValueError("Invalid table metadata")
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    if abs(rot[2, 2]) < 0.999:
        raise ValueError("Sampling requires a horizontal tabletop")
    short_axis = int(np.argmin(size[:2]))
    normal = rot[:2, short_axis]
    side = float((qpos[0, :2] - center[:2]) @ normal)
    if abs(side) <= size[short_axis] / 2:
        raise ValueError("Robot root is not outside the long table edge")
    normal = normal * np.sign(side)
    radius = float(np.sqrt(rng.uniform(1.0, 4.0)))
    angle = float(rng.uniform(-np.deg2rad(75), np.deg2rad(75)))
    tangent = np.array([-normal[1], normal[0]])
    start = qpos[0, :2] + radius * (np.cos(angle) * normal + np.sin(angle) * tangent)
    return start, radius, angle, normal


def export(args):
    if not all(np.isfinite(v) for v in (args.fps, args.speed, args.preparation_time)):
        raise ValueError("fps, speed and preparation-time must be finite")
    if args.fps <= 0 or args.speed <= 0 or args.preparation_time < 0:
        raise ValueError("fps/speed must be positive; preparation-time must be nonnegative")
    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise ValueError(f"No PKL files matched in {args.input_dir}")
    output = args.output_dir or args.input_dir.parent / "kimodo_end_frame"
    skeleton, converter, to_axis_angle, _ = load_kinematics(args.kimodo_root)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    server_root = args.server_kimodo_root
    server_data = args.server_data_dir
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        shlex.join(["cd", str(server_root)]),
        shlex.join(["source", str(server_root / "deployment/env.sh")]),
        "export HF_HUB_OFFLINE=1",
        "export TEXT_ENCODER_MODE=local",
        "export TEXT_ENCODER_DTYPE=float32",
        shlex.join(["mkdir", "-p", str(server_data / "motions")]),
        "",
    ]
    destinations = set()
    for path in files:
        motions = joblib.load(path)
        if not isinstance(motions, dict) or not motions:
            raise ValueError(f"{path}: expected a nonempty motion-key dictionary")
        for index, (key, motion) in enumerate(motions.items()):
            name = path.stem if len(motions) == 1 else f"{path.stem}__motion_{index:03d}"
            destination = output / f"{name}.json"
            if destination in destinations:
                raise ValueError(f"Output name collision: {destination}")
            destinations.add(destination)
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"{destination} exists; use --overwrite to replace")
            try:
                qpos = first_qpos(motion)
                meta_path = args.input_dir.parent / "meta" / path.name
                meta = joblib.load(meta_path)
                # Stable per-motion randomness even when --pattern changes.
                digest = hashlib.sha256(f"{args.seed}:{path.name}:{key}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                start, distance, angle, normal = sample_start(qpos, meta, rng)
                duration = distance / args.speed + args.preparation_time
                num_frames = int(duration * args.fps)  # Kimodo CLI truncates.
                if num_frames < 2:
                    raise ValueError("Generated motion must contain at least two frames")
                frame = num_frames - 1
                # Official complete_motion_dict needs a trajectory for smoothing
                # and velocities. Eight identical frames avoid its single-frame
                # singular system; only frame 0 is exported, without smoothing.
                converted = converter.qpos_to_motion_dict(
                    np.repeat(qpos, 8, axis=0), source_fps=args.fps,
                    root_quat_w_first=False, mujoco_rest_zero=False,
                )
                local = converted["local_rot_mats"][:1].clone()
                world_positions = converted["root_positions"][:1]
                # Kimodo is trained with the initial smooth root at XZ=(0,0).
                # Face the target along canonical +Z so the CLI's default
                # first_heading_angle=0 agrees with the frame-0 constraint.
                world_offset = world_positions.new_tensor([start[1], 0.0, start[0]])
                displacement = world_positions - world_offset
                heading = float(np.arctan2(displacement[0, 0].item(), displacement[0, 2].item()))
                canonical_to_world = world_positions.new_tensor(
                    Rotation.from_euler("y", heading).as_matrix()
                )
                positions = displacement @ canonical_to_world
                local[:, skeleton.root_idx] = canonical_to_world.T @ local[:, skeleton.root_idx]
                constraint = {
                    "type": "fullbody",
                    "frame_indices": [frame],
                    "local_joints_rot": to_axis_angle(local).tolist(),
                    "root_positions": positions.tolist(),
                }
                # Verify the actual JSON parser and its FK, not just array shape.
                start_constraint = {
                    "type": "root2d", "frame_indices": [0],
                    "smooth_root_2d": [[0.0, 0.0]],
                    "global_root_heading": [[1.0, 0.0]],  # [cos(0), sin(0)]: +Z
                }
                payload = json.dumps([start_constraint, constraint], indent=2, allow_nan=False) + "\n"
                from kimodo.constraints import load_constraints_lst
                parsed = load_constraints_lst(json.loads(payload), skeleton)
                loaded = parsed[1]
                if parsed[0].smooth_root_2d.abs().max().item() != 0:
                    raise ValueError("Initial root constraint must be at the canonical origin")
                facing_xz = parsed[0].global_root_heading[:, [1, 0]]
                direction_xz = positions[:, [0, 2]] / positions[:, [0, 2]].norm(dim=-1, keepdim=True)
                if float((facing_xz - direction_xz).abs().max()) > 1e-5:
                    raise ValueError("Initial heading must face the final root position")
                relative_distance = float(positions[0, [0, 2]].norm())
                if abs(relative_distance - distance) > 1e-5:
                    raise ValueError("Constraint displacement differs from sampled walking distance")
                fk_error = float((loaded.global_joints_positions @ canonical_to_world.T + world_offset
                                  - converted["posed_joints"][:1]).abs().max())
                if fk_error > 1e-5:
                    raise ValueError(f"JSON FK error {fk_error} exceeds 1e-5 m")
                roundtrip = converter.dict_to_qpos(converted, root_quat_w_first=False)
                dof_error = float(np.max(np.abs(roundtrip[0, 7:] - qpos[0, 7:])))
                destination.write_text(payload)
                command = shlex.join([
                    "python", "-m", "kimodo.scripts.generate",
                    "A person walks forward slowly.",
                    "--model", "Kimodo-G1-RP-v1", "--duration", repr(duration),
                    "--constraints", str(server_data / destination.name),
                    "--num_samples", "1", "--seed", str(args.seed),
                    "--output", str(server_data / "motions" / name),
                ])
                commands.append(command)
                records.append({
                    "start_xy_isaaclab": start.tolist(),
                    "end_xy_isaaclab": qpos[0, :2].tolist(),
                    "sector_normal_xy_isaaclab": normal.tolist(),
                    "table_meta": str(meta_path.resolve()),
                    "distance_m": distance, "sector_angle_deg": float(np.rad2deg(angle)),
                    "generation_duration": duration, "num_frames": num_frames,
                    "target_frame": frame, "generation_command": command,
                    "source": str(path.resolve()), "motion_key": str(key),
                    "source_frame": 0, "source_fps": float(motion["fps"]),
                    "constraint": destination.name, "root_positions": positions.tolist(),
                    "world_root_positions": world_positions.tolist(),
                    "first_heading_angle": 0.0,
                    "start_heading_angle_world_kimodo": heading,
                    "canonical_to_world_rotation_kimodo": canonical_to_world.tolist(),
                    "canonical_to_world_rotation_isaaclab": canonical_to_world[[2, 0, 1]][:, [2, 0, 1]].tolist(),
                    "canonical_to_world_translation_kimodo": world_offset.tolist(),
                    "canonical_to_world_translation_isaaclab": [float(start[0]), float(start[1]), 0.0],
                    "json_fk_max_error_m": fk_error,
                    "official_csv_roundtrip_max_dof_error_rad": dof_error,
                })
            except Exception as exc:
                raise ValueError(f"{path} / {key}: {exc}") from exc
    manifest = {
        "kimodo_root": str(args.kimodo_root.resolve()), "skeleton": "g1skel34",
        "generation_fps": args.fps, "speed_m_s": args.speed,
        "preparation_time_s": args.preparation_time, "seed": args.seed,
        "sampling": "uniform area; radii 1..2 m; angle -75..75 deg; one start per motion",
        "coordinate_policy": "Y-up; initial smooth root XZ zero; face target along +Z; world = canonical @ rotation.T + translation",
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(records)} poses from {len(files)} PKLs to {output}")
    (output / "generate_commands.sh").write_text("\n".join(commands) + "\n")
    durations = [r["generation_duration"] for r in records]
    print(f"Duration range: {min(durations):.3f}..{max(durations):.3f} s at {args.fps:g} Hz")
    print(f"Max JSON FK error: {max(r['json_fk_max_error_m'] for r in records):.3g} m")
    print(f"Max official CSV roundtrip DOF error: {max(r['official_csv_roundtrip_max_dof_error_rad'] for r in records):.3g} rad")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, help="Default: input-dir/../kimodo_end_frame")
    parser.add_argument("--kimodo-root", type=Path, default=ROOT.parent / "kimodo")
    parser.add_argument(
        "--server-kimodo-root", type=Path, default=Path("/workspace/kimodo"),
        help="Kimodo checkout on the server, used in generate_commands.sh",
    )
    parser.add_argument(
        "--server-data-dir", type=Path, default=Path("/workspace/kimodo/data/local_pickup"),
        help="Server directory containing uploaded constraint JSONs; motions are written below it",
    )
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--speed", type=float, default=0.5, help="Nominal walking speed, m/s")
    parser.add_argument("--preparation-time", type=float, default=0.1,
                        help="Extra duration in seconds (default: 0); does not impose a stationary hold")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible per-motion sampling seed")
    parser.add_argument("--fps", type=float, default=30.0, help="Kimodo model FPS, not PKL FPS")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        export(args)
    except (ValueError, KeyError, FileExistsError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
