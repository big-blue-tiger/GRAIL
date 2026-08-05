#!/usr/bin/env python3
"""Run a SONIC checkpoint and save successful ego videos and policy tensors.

Each rollout starts with an independently sampled robot-root XY offset in
``[-0.05, 0.05]`` meters. Alongside every successful
``<motion_key>_x<dx>_y<dy>.mp4``, this launcher writes a matching
``.object_aware.pkl`` containing the generator observation terms, the executed
binary hand primitive, the 64-D latent residual, the pre-FSQ
``z + lambda * delta_z`` latent, and world-frame robot/object root poses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib


def _trim_robot_motion(source: Path, destination: Path, start_time_s: float) -> int:
    """Write a source motion suffix and return its first retained frame."""
    payload = joblib.load(source)
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(f"Expected one motion in {source}, got {type(payload).__name__}")
    motion_key, motion = next(iter(payload.items()))
    if not isinstance(motion, dict) or "root_trans_offset" not in motion:
        raise ValueError(f"Motion {source} has no root_trans_offset array")
    frame_count = len(motion["root_trans_offset"])
    fps = float(motion.get("fps", 30.0))
    start_frame = math.ceil(start_time_s * fps)
    if start_frame >= frame_count:
        raise ValueError(
            f"Trim point {start_frame} is outside {source.name} ({frame_count} frames)"
        )

    trimmed = {}
    for name, value in motion.items():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0 and shape[0] == frame_count:
            trimmed[name] = value[start_frame:].copy()
        elif isinstance(value, list) and len(value) == frame_count:
            trimmed[name] = value[start_frame:]
        else:
            trimmed[name] = value

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}_", suffix=".pkl", dir=destination.parent, delete=False
    ) as file:
        temporary_path = Path(file.name)
    try:
        joblib.dump({motion_key: trimmed}, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return start_frame


def _save_selected_reference_data(
    manifest_path: Path,
    batch_keys: list[str],
    robot_dir: Path,
    selected_dir: Path,
    trim_penetration: bool,
    post_penetration_seconds: float = 0.3,
) -> None:
    """Save successful reference motions without modifying the source dataset."""
    allowed_keys = set(batch_keys)
    records: dict[str, dict] = {}
    with manifest_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            motion_key = record["motion_key"]
            if motion_key not in allowed_keys:
                raise RuntimeError(
                    f"Refusing to select unexpected motion key from manifest: "
                    f"{motion_key!r}"
                )
            records[motion_key] = record

    missing_records = allowed_keys - records.keys()
    if missing_records:
        raise RuntimeError(
            "Selection manifest is missing motion(s): " + ", ".join(sorted(missing_records))
        )

    selected_dir.mkdir(parents=True, exist_ok=True)
    for motion_key in batch_keys:
        record = records[motion_key]
        source = robot_dir / f"{motion_key}.pkl"
        destination = selected_dir / source.name
        reasons = record.get("termination_reasons", [])
        timed_out = record.get("timed_out", False)
        grasp_success = record.get("grasp_success", False)
        if not timed_out or not grasp_success:
            destination.unlink(missing_ok=True)
            if not timed_out:
                reason_text = ", ".join(reasons) or "rollout terminated early"
            else:
                reason_text = "grasp not achieved"
            print(f"Skipped {motion_key}: {reason_text}")
            continue

        detected = record.get("penetration_detected_in_initial_window", False)
        if trim_penetration and detected:
            clear_time = record.get("penetration_clear_time_seconds")
            if clear_time is None:
                destination.unlink(missing_ok=True)
                print(f"Skipped {motion_key}: penetration did not clear before rollout ended")
                continue
            trim_time = float(clear_time) + post_penetration_seconds
            try:
                start_frame = _trim_robot_motion(source, destination, trim_time)
            except ValueError as error:
                destination.unlink(missing_ok=True)
                print(f"Skipped {motion_key}: {error}")
                continue
            print(
                f"Saved trimmed motion to {destination} "
                f"(source frame {start_frame}, {trim_time:.3f}s)"
            )
        else:
            shutil.copy2(source, destination)
            print(f"Saved selected motion to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A dataset root, robot motion directory, or single robot motion .pkl",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="Number of motions/rendering environments per subprocess (default: 16)",
    )
    parser.add_argument(
        "--dataset-root", type=Path, help="Directory containing robot/objects/object_usd/bps"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, help="Base seed; defaults to a random seed")
    parser.add_argument(
        "--delete-failed-reference-data",
        action="store_true",
        help=(
            "Save successfully grasped robot motions under sibling robot_select; "
            "source dataset files are never deleted"
        ),
    )
    parser.add_argument(
        "--delete-reference-penetration-frames",
        "--delete-penetrated-reference-frames",
        dest="delete_reference_penetration_frames",
        action="store_true",
        help=(
            "For successful grasps, detect hand/table penetration in the first 2s "
            "and save the motion starting 0.3s after penetration clears"
        ),
    )
    args, extra = parser.parse_known_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    input_path = args.input.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.exists():
        parser.error(f"input does not exist: {input_path}")
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    if input_path.is_file():
        if input_path.suffix != ".pkl":
            parser.error("single-motion input must be a .pkl file")
        robot_dir = input_path.parent
        motion_keys = [input_path.stem]
    else:
        robot_dir = input_path / "robot" if (input_path / "robot").is_dir() else input_path
        motion_keys = sorted(path.stem for path in robot_dir.glob("*.pkl"))
    if not motion_keys:
        parser.error(f"no .pkl motions found in {robot_dir}")

    if args.dataset_root is not None:
        dataset_root = args.dataset_root.expanduser().resolve()
    elif robot_dir.parent == input_path and robot_dir.name == "robot":
        dataset_root = input_path
    else:
        dataset_root = robot_dir.parent
    required = {name: dataset_root / name for name in ("objects", "object_usd", "bps")}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        parser.error("missing paired dataset paths: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = dataset_root / "robot_select"

    sonic_root = Path(__file__).resolve().parents[2]
    atm_dir = sonic_root / "models" / "sonic_manipulation_base"
    atm_config = atm_dir / "model_config.yaml"
    atm_checkpoint = atm_dir / "last.pt"
    missing_atm = [str(path) for path in (atm_config, atm_checkpoint) if not path.is_file()]
    if missing_atm:
        parser.error("missing action-transform model files: " + ", ".join(missing_atm))

    # Match grail/visualization/scripts/visualize.sh:
    # eye = env_origin + [1.5, -1.5, 1.0], target = env_origin + [0, 0, 0.8].
    camera_offset = "[1.5,-1.5,1.0]"
    camera_target = "[0.0,0.0,0.8]"
    common_cmd = [
        sys.executable,
        "-u",
        "gear_sonic/eval_agent_trl.py",
        f"+checkpoint={checkpoint}",
        "+headless=True",
        "++run_once=True",
        "++manager_env.config.render_results=True",
        "++manager_env.commands.motion.start_from_first_frame=false",
        "++manager_env.commands.motion.fixed_start_frame=50",
        "++manager_env.commands.motion.sample_from_n_initial_frames=null",
        "++manager_env.commands.motion.sample_before_contact=false",
        "++manager_env.config.enable_cameras=False",
        "++manager_env.config.gpu_collision_stack_size_exp=28",
        "++manager_env.config.render_camera=eval_camera",
        "++manager_env.config.eval_camera_use_env_origin=True",
        f"++manager_env.config.eval_camera_offset={camera_offset}",
        f"++manager_env.config.eval_camera_target_offset={camera_target}",
        "++manager_env.config.eval_camera_focal_length=5.0",
        "++manager_env.config.eval_camera_focus_distance=100.0",
        "++manager_env.config.eval_camera_horizontal_aperture=10.0",
        "++manager_env.config.eval_camera_clipping_range=[0.1,500.0]",
        "++manager_env.config.render_width=640",
        "++manager_env.config.render_height=480",
        "++manager_env.config.render_frame_skip=1",
        f"++manager_env.config.save_rendering_dir={output_dir}",
        "++manager_env.recorders.render_envs._target_="
        "gear_sonic.envs.manager_env.mdp.recorders.RenderEnvsRecorderCfg",
        f"++manager_env.recorders.render_envs.video_save_path={output_dir}",
        "++manager_env.recorders.render_envs.video_quality=5",
        "++manager_env.recorders.render_envs.save_only_timeouts=False",
        "++manager_env.recorders.render_envs.append_initial_xy_offset=False",
        "++manager_env.recorders.trajectory._target_="
        "gear_sonic.envs.manager_env.mdp.recorders.ObjectAwareStateRecorderCfg",
        f"++manager_env.recorders.trajectory.save_path={output_dir}",
        "++manager_env.recorders.trajectory.save_only_timeouts=False",
        "++manager_env.recorders.trajectory.append_initial_xy_offset=False",
        "++manager_env.commands.motion.randomize_initial_pose_during_evaluation=False",
        "++manager_env.commands.motion.init_z_offset=0.05",
        "++manager_env.commands.motion.pose_range.x=[-0.0,0.0]",
        "++manager_env.commands.motion.pose_range.y=[-0.0,0.0]",
        # Released pnp configs still reference an internal ATM bundle that is not
        # shipped with this repository. Use the compatible public bundle instead.
        f"++manager_env.config.action_transform_module_cfg={atm_config}",
        f"++manager_env.config.action_transform_module_checkpoint={atm_checkpoint}",
        # Training checkpoints may persist the distributed training shard size
        # (for example 64). This launcher starts one evaluation subprocess, so
        # retaining that value can give rank 0 an empty motion/USD slice.
        "++motion_shard_by_rank=False",
        "++manager_env.commands.motion.motion_lib_cfg.motion_shard_rank=0",
        "++manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size=1",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={robot_dir}",
        f"++manager_env.commands.motion.motion_lib_cfg.object_motion_file={required['objects']}",
        f"++manager_env.config.object_usd_path={required['object_usd']}",
        f"++manager_env.commands.motion.motion_lib_cfg.bps_dir={required['bps']}",
    ]
    selection_enabled = (
        args.delete_failed_reference_data or args.delete_reference_penetration_frames
    )
    if args.delete_reference_penetration_frames:
        # Observe the entire penetration interval instead of ending at first contact.
        common_cmd.append("++manager_env.terminations.hand_table_contact_termination=null")
    batches = [
        motion_keys[start : start + args.batch_size]
        for start in range(0, len(motion_keys), args.batch_size)
    ]
    base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    print(
        f"Rendering {len(motion_keys)} motion(s) to {output_dir} "
        f"in {len(batches)} batch(es), up to {args.batch_size} environment(s) each"
    )
    for batch_index, batch_keys in enumerate(batches, start=1):
        cmd = common_cmd.copy()
        selection_manifest = None
        if selection_enabled:
            manifest_file = tempfile.NamedTemporaryFile(
                prefix=f".selection_batch_{batch_index}_",
                suffix=".jsonl",
                dir=output_dir,
                delete=False,
            )
            manifest_file.close()
            selection_manifest = Path(manifest_file.name)
            cmd.append(
                "++manager_env.recorders.trajectory.selection_manifest_path="
                f"{selection_manifest}"
            )
        batch_seed = (base_seed + batch_index - 1) % (2**32)
        cmd.append(f"++seed={batch_seed}")
        cmd.append(f"+num_envs={len(batch_keys)}")
        keys_override = ",".join(batch_keys)
        cmd.append(
            "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="
            f"[{keys_override}]"
        )
        cmd.extend(extra)
        print(
            f"Batch {batch_index}/{len(batches)}: "
            f"{', '.join(batch_keys)} (seed={batch_seed})"
        )
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=sonic_root,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
            )
            if selection_manifest is not None:
                _save_selected_reference_data(
                    selection_manifest,
                    batch_keys,
                    robot_dir,
                    selected_dir,
                    args.delete_reference_penetration_frames,
                )
        finally:
            if selection_manifest is not None:
                selection_manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
