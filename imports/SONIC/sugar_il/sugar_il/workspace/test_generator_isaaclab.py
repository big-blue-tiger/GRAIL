#!/usr/bin/env python3
"""Parallel flow-matching evaluation with a hard-coded frame-5 robot state.

The robot orientation and all 43 joint positions below come from array index 4
(`frame_idx=4`, `motion_step=54`) of::

    outputs/grail/all/pickup_table__alcohol_11__005.object_aware.pkl

Robot and object XY positions are sampled in table-centred coordinates.  Robot
root Z is deliberately fixed at 0.827 m for every episode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import run_generator_isaaclab as base


DEFAULT_MOTION_FILE = (
    Path(__file__).resolve().parents[5]
    / "data/hf_dataset/data/pickup_table_update/robot"
    / "pickup_table__alcohol_11__005.pkl"
)

# Source values retained explicitly so this test has no runtime dependency on
# the object-aware recording.  The world position contains the recording
# environment origin (-15, 5, 0); randomized test positions are table-centred.
SOURCE_FRAME_INDEX = 4
SOURCE_MOTION_STEP = 54
SOURCE_ROOT_POSITION_W = (-15.0318574905, 5.5359063148, 0.7884901762)
INITIAL_ROOT_QUATERNION_WXYZ = (
    -0.6644826531,
    -0.0066797943,
    0.0213959832,
    0.7469673753,
)
INITIAL_JOINT_POSITIONS = (
    0.1238551140,
    0.0506065190,
    -0.0543037094,
    0.0454488397,
    -0.0600250997,
    0.0003553614,
    0.1308768094,
    0.1639951468,
    -0.0031797322,
    -0.2164152265,
    -0.1513829827,
    -0.1658617109,
    0.0177452564,
    0.2488028854,
    0.2077847570,
    0.0783909708,
    -0.1863341779,
    -0.0246327482,
    0.0447884314,
    -0.3680039644,
    0.2729246616,
    0.7192628384,
    0.2920869589,
    0.2356615365,
    -0.2401672900,
    0.0809113234,
    0.0577144511,
    0.0307373852,
    -0.0190879013,
    -0.0545705706,
    -0.0561467558,
    -0.0017813881,
    0.0583781525,
    0.0598403625,
    -0.0031964066,
    -0.0606557280,
    -0.0609071068,
    0.0090803728,
    0.0612510964,
    0.0614758320,
    -0.0336270221,
    0.0652864724,
    -0.0783640221,
)

ROBOT_ROOT_Z = 0.827
DEFAULT_ROBOT_X_RANGE = (-1.0, 1.0)
DEFAULT_ROBOT_Y_RANGE = (0.65, 1.65)
DEFAULT_OBJECT_X_RANGE = (-0.85, 0.85)
DEFAULT_OBJECT_Y_RANGE = (-0.18, 0.18)


def _parse_args(app_launcher_cls) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-checkpoint", required=True, type=Path)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=16,
        help="Number of robots evaluated concurrently in each batch.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--lift-height", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--render-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record one MP4 for every tested robot; disabled by default.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/generator_test"))
    parser.add_argument(
        "--robot-x-range", type=float, nargs=2, default=DEFAULT_ROBOT_X_RANGE
    )
    parser.add_argument(
        "--robot-y-range", type=float, nargs=2, default=DEFAULT_ROBOT_Y_RANGE
    )
    parser.add_argument(
        "--object-x-range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE
    )
    parser.add_argument(
        "--object-y-range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE
    )
    app_launcher_cls.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.episodes < 1 or args.parallel_envs < 1:
        parser.error("--episodes and --parallel-envs must be positive")
    if args.max_episode_steps < 1 or args.lift_height <= 0:
        parser.error("--max-episode-steps and --lift-height must be positive")
    for name in ("robot_x_range", "robot_y_range", "object_x_range", "object_y_range"):
        bounds = getattr(args, name)
        if bounds[0] > bounds[1]:
            parser.error(f"--{name.replace('_', '-')} minimum exceeds maximum")
    return args


def _compose_config(sonic_root, assets, reference, args):
    """Reuse the production generator config, changing only parallelism."""
    cfg = base._compose_sonic_config(sonic_root, assets, reference, args)
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.num_envs = args.parallel_envs
        cfg.manager_env.config.num_envs = args.parallel_envs
    return cfg


def _sample_scenes(reference, args, first_episode: int, count: int):
    scenes = []
    for episode in range(first_episode, first_episode + count):
        rng = random.Random(args.seed + episode)
        scenes.append(
            base.EpisodeScene(
                robot_position=(
                    rng.uniform(*args.robot_x_range),
                    rng.uniform(*args.robot_y_range),
                    ROBOT_ROOT_Z,
                ),
                object_position=(
                    rng.uniform(*args.object_x_range),
                    rng.uniform(*args.object_y_range),
                    reference.object_position[2],
                ),
                table_position=reference.table_position,
            )
        )
    return scenes


def _reset_batch(env, scenes, reference, torch):
    """Reset all parallel environments and inject the hard-coded robot state."""
    env.env.sim.reset()
    env.reset()
    num_envs = env.env.num_envs
    if len(scenes) != num_envs:
        raise ValueError(f"Expected {num_envs} scenes, got {len(scenes)}")

    device = env.device
    origins = env.env.scene.env_origins

    def poses(positions, quaternion):
        positions_t = torch.tensor(positions, dtype=torch.float32, device=device)
        quaternions_t = torch.tensor(
            quaternion, dtype=torch.float32, device=device
        ).repeat(num_envs, 1)
        return torch.cat((positions_t + origins, quaternions_t), dim=-1)

    robot = env.env.scene["robot"]
    obj = env.env.scene["object"]
    table = env.env.scene["table"]
    robot.write_root_pose_to_sim(
        poses([scene.robot_position for scene in scenes], INITIAL_ROOT_QUATERNION_WXYZ)
    )
    obj.write_root_pose_to_sim(
        poses([scene.object_position for scene in scenes], reference.object_quaternion_wxyz)
    )
    table.write_root_pose_to_sim(
        poses([scene.table_position for scene in scenes], reference.table_quaternion_wxyz)
    )

    if robot.num_joints != len(INITIAL_JOINT_POSITIONS):
        raise ValueError(
            f"Hard-coded state has {len(INITIAL_JOINT_POSITIONS)} joints, "
            f"but the robot has {robot.num_joints}"
        )
    joint_pos = torch.tensor(
        INITIAL_JOINT_POSITIONS, dtype=torch.float32, device=device
    ).repeat(num_envs, 1)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # Prevent stale actuator targets from overwriting the injected state before
    # the first direct-latent action is decoded.
    robot.set_joint_position_target(joint_pos)
    robot.set_joint_velocity_target(joint_vel)
    robot.set_joint_effort_target(torch.zeros_like(joint_pos))

    zero_root_velocity = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)
    robot.write_root_velocity_to_sim(zero_root_velocity)
    obj.write_root_velocity_to_sim(zero_root_velocity)

    env.env.sim.forward()
    env.env.scene.update(dt=0.0)
    raw_obs = env.env.observation_manager.compute()
    obs_dict = env.process_raw_obs(raw_obs, flatten_dict_obs=True)
    env._last_obs_dict = obs_dict  # noqa: SLF001
    return obs_dict


def _render_batch(base_env, torch):
    camera = base_env.scene["eval_camera"]
    origins = base_env.scene.env_origins
    eye = origins + torch.tensor((2.8, 2.8, 2.0), device=base_env.device)
    target = origins + torch.tensor((0.0, 0.0, 0.9), device=base_env.device)
    camera._view._sync_usd_on_fabric_write = True  # noqa: SLF001
    camera.set_world_poses_from_view(eye, target)
    base_env.sim.render()
    base_env.sim.render()
    camera._is_outdated[:] = True  # noqa: SLF001
    camera.update(dt=0.0, force_recompute=True)
    return camera.data.output["rgb"].detach().cpu().numpy()


def _video_writers(args, episode_ids):
    if not args.render_video:
        return []
    import imageio.v2 as imageio

    video_dir = args.output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    return [
        imageio.get_writer(
            video_dir / f"episode_{episode:04d}.mp4",
            fps=base.CONTROL_FPS,
            codec="libx264",
            quality=5,
            pixelformat="yuv420p",
        )
        for episode in episode_ids
    ]


def _append_video_frames(writers, frames):
    for env_id, writer in enumerate(writers):
        writer.append_data(frames[env_id])


def _run_batch(
    env,
    generator,
    scenes,
    episode_ids,
    active_count,
    reference,
    args,
    hand_frame_cfg,
    hand_transform_fn,
    simulation_app,
    torch,
):
    obs_dict = _reset_batch(env, scenes, reference, torch)
    initial_z = env.env.scene["object"].data.root_pos_w[:, 2].clone()
    max_lift = torch.zeros(env.env.num_envs, device=env.device)
    successes = torch.zeros(env.env.num_envs, dtype=torch.bool, device=env.device)
    finished = torch.zeros_like(successes)
    writers = _video_writers(args, episode_ids[:active_count])
    steps = 0
    episode_steps = torch.zeros(
        env.env.num_envs, dtype=torch.int64, device=env.device
    )

    try:
        if writers:
            _append_video_frames(writers, _render_batch(env.env, torch))
        while simulation_app.is_running() and steps < args.max_episode_steps:
            generator_obs = base._generator_observation(
                env,
                generator,
                obs_dict,
                hand_frame_cfg,
                hand_transform_fn,
                torch,
            )
            prediction = generator.policy.predict_action(generator_obs)
            latents = prediction["latent"]
            hands = prediction["hand_primitive"]
            expected_latent = (env.env.num_envs, base.PREDICTION_HORIZON, 64)
            expected_hand = (env.env.num_envs, base.PREDICTION_HORIZON, 2)
            if tuple(latents.shape) != expected_latent:
                raise ValueError(f"Unexpected latent shape: {tuple(latents.shape)}")
            if tuple(hands.shape) != expected_hand:
                raise ValueError(f"Unexpected hand shape: {tuple(hands.shape)}")

            for index in range(base.EXECUTION_HORIZON):
                action = torch.cat(
                    (
                        latents[:, index],
                        base.binary_hand_to_sonic(hands[:, index]),
                    ),
                    dim=-1,
                )
                # Finished environments may have been automatically reset by
                # Isaac Lab.  Their subsequent state is intentionally ignored.
                action[finished] = 0.0
                obs_dict, _, dones, _ = env.step(
                    {
                        "actions": action,
                        "obs_dict": obs_dict,
                        "action_mode": "direct_latent",
                    }
                )
                steps += 1
                episode_steps += (~finished).to(episode_steps.dtype)
                current_lift = (
                    env.env.scene["object"].data.root_pos_w[:, 2] - initial_z
                )
                max_lift = torch.where(
                    finished, max_lift, torch.maximum(max_lift, current_lift)
                )
                successes |= max_lift >= args.lift_height
                finished |= dones.bool()
                if writers:
                    _append_video_frames(writers, _render_batch(env.env, torch))
                if bool(finished.all().item()) or steps >= args.max_episode_steps:
                    break
            if bool(finished.all().item()):
                break
    finally:
        for writer in writers:
            writer.close()

    return [
        base.EpisodeResult(
            episode=episode,
            seed=args.seed + episode,
            success=bool(successes[env_id].item()),
            steps=int(episode_steps[env_id].item()),
            lift=float(max_lift[env_id].item()),
            scene=scenes[env_id],
        )
        for env_id, episode in enumerate(episode_ids)
    ]


def _save_results(args, results):
    successes = sum(result.success for result in results)
    payload = {
        "episodes": len(results),
        "parallel_envs": args.parallel_envs,
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "lift_height": args.lift_height,
        "render_video": args.render_video,
        "flow_matching": {
            "prediction_horizon": base.PREDICTION_HORIZON,
            "execution_horizon": base.EXECUTION_HORIZON,
            "inference_steps": base.FLOW_INFERENCE_STEPS,
        },
        "initial_state": {
            "source_frame_index": SOURCE_FRAME_INDEX,
            "source_motion_step": SOURCE_MOTION_STEP,
            "source_root_position_w": SOURCE_ROOT_POSITION_W,
            "root_quaternion_wxyz": INITIAL_ROOT_QUATERNION_WXYZ,
            "robot_root_z": ROBOT_ROOT_Z,
            "joint_positions": INITIAL_JOINT_POSITIONS,
            "joint_velocities": "zeros",
        },
        "ranges": {
            "robot_x": args.robot_x_range,
            "robot_y": args.robot_y_range,
            "object_x": args.object_x_range,
            "object_y": args.object_y_range,
        },
        "results": [asdict(result) for result in results],
    }
    path = args.output_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\nOverall result: {successes}/{len(results)} successful "
        f"({payload['success_rate']:.2%})\nSaved: {path}"
    )


def _run(args, simulation_app, sonic_root, assets):
    # Isaac Lab and SONIC imports must happen after AppLauncher starts.
    import dill
    import hydra
    import torch
    from isaaclab.managers import SceneEntityCfg

    from gear_sonic import train_agent_trl
    from gear_sonic.envs.manager_env.mdp.observations import hand_object_transform_6d
    from gear_sonic.utils.common import seeding
    from sugar_il.wrapper.sugar_il_wrapper import (
        GeneratorWrapper,
        load_generator_policy_state,
    )

    reference = base.load_reference_scene(assets)
    cfg = _compose_config(sonic_root, assets, reference, args)
    seeding(args.seed)
    env = train_agent_trl.create_manager_env(cfg, args.device, args)

    payload = torch.load(
        args.generator_checkpoint,
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
    )
    policy = hydra.utils.instantiate(payload["cfg"].policy)
    load_generator_policy_state(policy, payload)
    generator = GeneratorWrapper(policy, args.device)
    base._validate_generator(generator.policy, payload["cfg"], env)

    hand_frame_cfg = SceneEntityCfg("object_to_hand_frame_transformer")
    results = []
    with torch.inference_mode():
        for first_episode in range(0, args.episodes, args.parallel_envs):
            if not simulation_app.is_running():
                break
            active_count = min(args.parallel_envs, args.episodes - first_episode)
            episode_ids = list(range(first_episode, first_episode + active_count))
            active_scenes = _sample_scenes(
                reference, args, first_episode, active_count
            )
            # The simulator has a fixed number of environments.  On the final
            # partial batch, fill unused slots deterministically and discard them.
            scenes = active_scenes + [active_scenes[-1]] * (
                args.parallel_envs - active_count
            )
            batch_episode_ids = episode_ids + [episode_ids[-1]] * (
                args.parallel_envs - active_count
            )
            batch_results = _run_batch(
                env,
                generator,
                scenes,
                batch_episode_ids,
                active_count,
                reference,
                args,
                hand_frame_cfg,
                hand_object_transform_6d,
                simulation_app,
                torch,
            )[:active_count]
            results.extend(batch_results)
            batch_successes = sum(result.success for result in batch_results)
            print(
                f"Batch {first_episode // args.parallel_envs + 1}: "
                f"episodes {episode_ids[0]}-{episode_ids[-1]}, "
                f"success={batch_successes}/{active_count}"
            )

    _save_results(args, results)
    env.env.close()


def _resolve(path: Path, launch_cwd: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (launch_cwd / path).resolve()


def main() -> None:
    launch_cwd = Path.cwd()
    sonic_root = Path(__file__).resolve().parents[3]
    os.chdir(sonic_root)
    for path in (sonic_root, sonic_root / "sugar_il"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise RuntimeError(
            "Run this script inside the SONIC Isaac Lab environment."
        ) from exc

    args = _parse_args(AppLauncher)
    args.generator_checkpoint = _resolve(args.generator_checkpoint, launch_cwd)
    args.motion_file = _resolve(args.motion_file, launch_cwd)
    args.output_dir = _resolve(args.output_dir, launch_cwd)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.enable_cameras = args.render_video
    if not args.generator_checkpoint.is_file():
        raise FileNotFoundError(args.generator_checkpoint)
    assets = base.resolve_motion_assets(args.motion_file)

    launcher = AppLauncher(args)
    try:
        _run(args, launcher.app, sonic_root, assets)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
