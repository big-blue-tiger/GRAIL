#!/usr/bin/env python3
"""Run a SONIC checkpoint and save ego videos plus object-aware policy tensors.

Alongside every ``<motion_key>.mp4``, this launcher writes an
``<motion_key>.object_aware.pkl`` containing per-frame proprioception, object
reference observations, the 64-D latent residual, the two raw/executed hand
primitives, the pre-FSQ ``z + lambda * delta_z`` latent, and world-frame
robot/object root poses.
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
        "--dataset-root", type=Path, help="Directory containing robot/objects/object_usd/bps"
    )
    parser.add_argument("--gpu", default="0")
    args, extra = parser.parse_known_args()

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
    cmd = [
        sys.executable,
        "-u",
        "gear_sonic/eval_agent_trl.py",
        f"+checkpoint={checkpoint}",
        f"+num_envs={len(motion_keys)}",
        "+headless=True",
        "++run_once=True",
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
        "++manager_env.recorders.trajectory._target_="
        "gear_sonic.envs.manager_env.mdp.recorders.ObjectAwareStateRecorderCfg",
        f"++manager_env.recorders.trajectory.save_path={output_dir}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={robot_dir}",
        f"++manager_env.commands.motion.motion_lib_cfg.object_motion_file={required['objects']}",
        f"++manager_env.config.object_usd_path={required['object_usd']}",
        f"++manager_env.commands.motion.motion_lib_cfg.bps_dir={required['bps']}",
    ]
    if input_path.is_file():
        cmd.append(
            "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="
            f"[{motion_keys[0]}]"
        )
    cmd.extend(extra)

    print(f"Rendering {len(motion_keys)} motion(s) to {output_dir}")
    subprocess.run(
        cmd,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
    )


if __name__ == "__main__":
    main()
