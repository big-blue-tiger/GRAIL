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
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _delete_failed_reference_data(
    manifest_path: Path,
    batch_keys: list[str],
    robot_dir: Path,
    paired_dirs: dict[str, Path],
) -> None:
    """Delete the four explicitly paired reference files for failed motions."""
    allowed_keys = set(batch_keys)
    suffixes = {
        "robot": ".pkl",
        "objects": ".pkl",
        "object_usd": ".usd",
        "bps": ".npy",
    }
    failed: dict[str, list[str]] = {}
    with manifest_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            motion_key = record["motion_key"]
            if motion_key not in allowed_keys:
                raise RuntimeError(
                    f"Refusing to delete unexpected motion key from failure manifest: "
                    f"{motion_key!r}"
                )
            failed[motion_key] = record.get("termination_reasons", [])

    roots = {"robot": robot_dir, **paired_dirs}
    for motion_key, reasons in failed.items():
        deleted = []
        missing = []
        for name, suffix in suffixes.items():
            path = roots[name] / f"{motion_key}{suffix}"
            if path.is_file():
                path.unlink()
                deleted.append(str(path))
            else:
                missing.append(str(path))
        reason_text = ", ".join(reasons) or "unknown"
        print(
            f"Deleted failed reference set for {motion_key} "
            f"(termination={reason_text}): {', '.join(deleted)}"
        )
        if missing:
            print(f"Reference files already missing for {motion_key}: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="A robot motion .pkl or directory")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--batch-size",
        default=16,
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
            "Permanently delete matching robot/objects/object_usd/bps files when "
            "a rollout terminates before timeout"
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
        robot_dir = input_path
        motion_keys = sorted(path.stem for path in robot_dir.glob("*.pkl"))
    if not motion_keys:
        parser.error(f"no .pkl motions found in {robot_dir}")

    dataset_root = (args.dataset_root or robot_dir.parent).expanduser().resolve()
    required = {name: dataset_root / name for name in ("objects", "object_usd", "bps")}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        parser.error("missing paired dataset paths: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "++manager_env.commands.motion.start_from_first_frame=true",
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
        "++manager_env.recorders.render_envs.save_only_timeouts=True",
        "++manager_env.recorders.render_envs.append_initial_xy_offset=True",
        "++manager_env.recorders.trajectory._target_="
        "gear_sonic.envs.manager_env.mdp.recorders.ObjectAwareStateRecorderCfg",
        f"++manager_env.recorders.trajectory.save_path={output_dir}",
        "++manager_env.recorders.trajectory.save_only_timeouts=True",
        "++manager_env.recorders.trajectory.append_initial_xy_offset=True",
        "++manager_env.commands.motion.randomize_initial_pose_during_evaluation=True",
        "++manager_env.commands.motion.start_from_first_frame=true",
        "++manager_env.commands.motion.init_z_offset=0.05",
        "++manager_env.commands.motion.pose_range.x=[-0.1,0.1]",
        "++manager_env.commands.motion.pose_range.y=[-0.1,0.1]",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={robot_dir}",
        f"++manager_env.commands.motion.motion_lib_cfg.object_motion_file={required['objects']}",
        f"++manager_env.config.object_usd_path={required['object_usd']}",
        f"++manager_env.commands.motion.motion_lib_cfg.bps_dir={required['bps']}",
    ]
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
        failure_manifest = None
        if args.delete_failed_reference_data:
            manifest_file = tempfile.NamedTemporaryFile(
                prefix=f".failed_batch_{batch_index}_",
                suffix=".jsonl",
                dir=output_dir,
                delete=False,
            )
            manifest_file.close()
            failure_manifest = Path(manifest_file.name)
            cmd.append(
                "++manager_env.recorders.trajectory.failure_manifest_path="
                f"{failure_manifest}"
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
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
            )
            if failure_manifest is not None:
                _delete_failed_reference_data(
                    failure_manifest,
                    batch_keys,
                    robot_dir,
                    required,
                )
        finally:
            if failure_manifest is not None:
                failure_manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
