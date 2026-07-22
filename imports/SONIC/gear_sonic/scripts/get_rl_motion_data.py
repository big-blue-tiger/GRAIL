#!/usr/bin/env python3
"""Run a SONIC checkpoint and save successful ego videos and policy tensors.

Each rollout starts with an independently sampled robot-root XY offset in
``[-0.05, 0.05]`` meters. Alongside every successful
``<motion_key>_x<dx>_y<dy>.mp4``, this launcher writes a matching
``.object_aware.pkl`` containing per-frame proprioception, object
reference observations, the 64-D latent residual, the two raw/executed hand
primitives, the pre-FSQ ``z + lambda * delta_z`` latent, and world-frame
robot/object root poses. Episodes that terminate before motion timeout are discarded.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="A robot motion .pkl or directory")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Number of motions/rendering environments per subprocess (default: 16)",
    )
    parser.add_argument(
        "--dataset-root", type=Path, help="Directory containing robot/objects/object_usd/bps"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, help="Base seed; defaults to a random seed")
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

    # G1's head_link origin is below its visible head mesh; z=0.44 is near eye level.
    # Isaac Lab's "world" camera convention uses +X forward and +Z up.
    # A +30 degree rotation about +Y points +X forward and 30 degrees down.
    camera_pos = "[0.09,0.0,0.44]"
    camera_rot = "[0.965926,0.0,0.258819,0.0]"
    camera_focal_length = 9
    camera_horizontal_aperture = 20.955
    common_cmd = [
        sys.executable,
        "-u",
        "gear_sonic/eval_agent_trl.py",
        f"+checkpoint={checkpoint}",
        "+headless=True",
        "++run_once=True",
        "++warmup_rollout_steps=8",
        # The recorder reads ego_camera directly; do not create the extra eval_camera.
        "++manager_env.config.render_results=False",
        "++manager_env.config.enable_cameras=True",
        "++manager_env.config.gpu_collision_stack_size_exp=28",
        "++manager_env.config.render_camera=ego_camera",
        "++manager_env.config.render_width=640",
        "++manager_env.config.render_height=480",
        "++manager_env.config.render_frame_skip=1",
        f"++manager_env.config.cameras.camera_focal_length={camera_focal_length}",
        f"++manager_env.config.cameras.camera_horizontal_aperture={camera_horizontal_aperture}",
        "++manager_env.config.cameras.camera_attached_link=head_link",
        f"++manager_env.config.cameras.camera_pos_offset={camera_pos}",
        f"++manager_env.config.cameras.camera_rot_offset={camera_rot}",
        "++manager_env.config.cameras.camera_resolution=[480,640]",
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
        "++manager_env.commands.motion.pose_range.x=[-0.06,0.06]",
        "++manager_env.commands.motion.pose_range.y=[-0.06,0.06]",
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
        subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
        )


if __name__ == "__main__":
    main()
