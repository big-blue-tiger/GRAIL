#!/usr/bin/env python3
"""Evaluate a flow-matching generator in the SONIC Isaac Lab environment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PREDICTION_HORIZON = 40
EXECUTION_HORIZON = 20
CONTROL_FPS = 50
FLOW_INFERENCE_STEPS = 4
FLOW_TIMESTEP_BUCKETS = 1000
INITIAL_MOTION_FRAME_INDEX = 50

DEFAULT_MOTION = Path(
    "/home/tide/robot/GRAIL/data/hf_dataset/data/pickup_table_update/robot/"
    "pickup_table__alcohol_5__001.pkl"
)


@dataclass(frozen=True)
class RandomizationRanges:
    """Batch-test ranges. Edit these values to change the test distribution."""

    table_height: tuple[float, float] = (0.60, 0.61)
    object_x_offset: tuple[float, float] = (-0.0, 0.0)
    object_y_offset: tuple[float, float] = (-0.0, 0.0)
    robot_x_offset: tuple[float, float] = (-0.3, 0.3)
    robot_y_offset: tuple[float, float] = (-0.01, 0.4)
    robot_z_offset: tuple[float, float] = (0.04, 0.06)


RANDOMIZATION = RandomizationRanges()


@dataclass(frozen=True)
class MotionAssets:
    robot: Path
    objects: Path
    object_usd: Path
    bps: Path
    meta: Path

    @property
    def stem(self) -> str:
        return self.robot.stem


@dataclass(frozen=True)
class ReferenceScene:
    robot_position: tuple[float, float, float]
    robot_quaternion_wxyz: tuple[float, float, float, float]
    robot_body_joint_positions_mujoco: tuple[float, ...]
    robot_hand_joint_positions: tuple[float, ...]
    object_position: tuple[float, float, float]
    object_quaternion_wxyz: tuple[float, float, float, float]
    table_position: tuple[float, float, float]
    table_quaternion_wxyz: tuple[float, float, float, float]
    table_size: tuple[float, float, float]


@dataclass(frozen=True)
class EpisodeScene:
    robot_position: tuple[float, float, float]
    object_position: tuple[float, float, float]
    table_position: tuple[float, float, float]


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    seed: int
    success: bool
    steps: int
    lift: float
    scene: EpisodeScene


def resolve_motion_assets(motion_file: str | Path) -> MotionAssets:
    """Resolve runtime assets without reading the reference robot trajectory."""
    robot = Path(motion_file).expanduser().resolve()
    if robot.suffix != ".pkl" or robot.parent.name != "robot":
        raise ValueError("--motion-file must point to robot/<name>.pkl")
    root = robot.parent.parent
    stem = robot.stem
    assets = MotionAssets(
        robot=robot,
        objects=root / "objects" / f"{stem}.pkl",
        object_usd=root / "object_usd" / f"{stem}.usd",
        bps=root / "bps" / f"{stem}.npy",
        meta=root / "meta" / f"{stem}.pkl",
    )
    missing = [str(path) for path in asdict(assets).values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing SONIC runtime assets: " + ", ".join(missing))
    return assets


def _first_motion(payload: dict, path: Path) -> dict:
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path}: expected a non-empty motion dictionary")
    motion = next(iter(payload.values()))
    if not isinstance(motion, dict):
        raise ValueError(f"{path}: invalid motion entry")
    return motion


def load_reference_scene(assets: MotionAssets) -> ReferenceScene:
    """Load robot frame index 50, frame-zero object pose, and table metadata."""
    import joblib

    robot_motion = _first_motion(joblib.load(assets.robot), assets.robot)
    object_motion = _first_motion(joblib.load(assets.objects), assets.objects)
    meta = joblib.load(assets.meta)
    try:
        robot_position = robot_motion["root_trans_offset"][
            INITIAL_MOTION_FRAME_INDEX
        ]
        # Motion files store quaternions as xyzw; Isaac Lab expects wxyz.
        robot_quaternion_xyzw = robot_motion["root_rot"][
            INITIAL_MOTION_FRAME_INDEX
        ]
        robot_body_joint_positions_mujoco = robot_motion["dof"][
            INITIAL_MOTION_FRAME_INDEX
        ]
        robot_hand_joint_positions = robot_motion["hand_dof_pos"][
            INITIAL_MOTION_FRAME_INDEX
        ]
        object_position = object_motion["root_pos"][0, 0]
        object_quaternion = object_motion["root_quat"][0, 0]
        table_position = meta["table_pos"]
        table_quaternion = meta["table_quat"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Invalid robot/object/table initialization data: {exc}") from exc

    table_size = meta.get("table_size", (1.0, 0.6, 0.04))
    return ReferenceScene(
        robot_position=tuple(float(x) for x in robot_position),
        robot_quaternion_wxyz=tuple(
            float(robot_quaternion_xyzw[index]) for index in (3, 0, 1, 2)
        ),
        robot_body_joint_positions_mujoco=tuple(
            float(x) for x in robot_body_joint_positions_mujoco
        ),
        robot_hand_joint_positions=tuple(
            float(x) for x in robot_hand_joint_positions
        ),
        object_position=tuple(float(x) for x in object_position),
        object_quaternion_wxyz=tuple(float(x) for x in object_quaternion),
        table_position=tuple(float(x) for x in table_position),
        table_quaternion_wxyz=tuple(float(x) for x in table_quaternion),
        table_size=tuple(float(x) for x in table_size),
    )


def _range(value: list[float] | None, default: tuple[float, float]):
    return tuple(value) if value is not None else default


def _parse_args(app_launcher_cls) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-checkpoint", required=True, type=Path)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--mode", choices=("single", "batch"), default="single")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--lift-height", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--render-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save one MP4 per episode. Use --no-render-video for fast evaluation.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/generator_eval"))

    parser.add_argument("--table-height-range", type=float, nargs=2)
    parser.add_argument("--object-x-offset-range", type=float, nargs=2)
    parser.add_argument("--object-y-offset-range", type=float, nargs=2)
    parser.add_argument("--robot-x-offset-range", type=float, nargs=2)
    parser.add_argument("--robot-y-offset-range", type=float, nargs=2)
    parser.add_argument("--robot-z-offset-range", type=float, nargs=2)

    app_launcher_cls.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_episode_steps < 1:
        parser.error("--episodes and --max-episode-steps must be positive")
    if args.lift_height <= 0:
        parser.error("--lift-height must be positive")
    args.episodes = 1 if args.mode == "single" else args.episodes
    args.randomization = RandomizationRanges(
        table_height=_range(args.table_height_range, RANDOMIZATION.table_height),
        object_x_offset=_range(
            args.object_x_offset_range, RANDOMIZATION.object_x_offset
        ),
        object_y_offset=_range(
            args.object_y_offset_range, RANDOMIZATION.object_y_offset
        ),
        robot_x_offset=_range(args.robot_x_offset_range, RANDOMIZATION.robot_x_offset),
        robot_y_offset=_range(args.robot_y_offset_range, RANDOMIZATION.robot_y_offset),
        robot_z_offset=_range(args.robot_z_offset_range, RANDOMIZATION.robot_z_offset),
    )
    for name, bounds in asdict(args.randomization).items():
        if bounds[0] > bounds[1]:
            parser.error(f"--{name.replace('_', '-')} minimum exceeds maximum")
    return args


def _sample_scene(reference: ReferenceScene, args, episode_seed: int) -> EpisodeScene:
    if args.mode == "single":
        return EpisodeScene(
            robot_position=reference.robot_position,
            object_position=reference.object_position,
            table_position=reference.table_position,
        )

    import random

    rng = random.Random(episode_seed)
    ranges = args.randomization
    table_z = rng.uniform(*ranges.table_height)
    table_delta_z = table_z - reference.table_position[2]
    return EpisodeScene(
        robot_position=(
            reference.robot_position[0] + rng.uniform(*ranges.robot_x_offset),
            reference.robot_position[1] + rng.uniform(*ranges.robot_y_offset),
            reference.robot_position[2] + rng.uniform(*ranges.robot_z_offset),
        ),
        object_position=(
            reference.object_position[0] + rng.uniform(*ranges.object_x_offset),
            reference.object_position[1] + rng.uniform(*ranges.object_y_offset),
            reference.object_position[2] + table_delta_z,
        ),
        table_position=(
            reference.table_position[0],
            reference.table_position[1],
            table_z,
        ),
    )


def binary_hand_to_sonic(hand_binary):
    """Map generator hands from 0/1 to SONIC's -1/+1 convention."""
    return hand_binary * 2.0 - 1.0


def _compose_sonic_config(sonic_root: Path, assets: MotionAssets, reference, args):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    with initialize_config_dir(
        config_dir=str(sonic_root / "gear_sonic" / "config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="base",
            overrides=["+exp=manager/universal_token/hoi/pnp_table"],
        )

    model_dir = sonic_root / "models" / "sonic_manipulation_base"
    with open_dict(cfg):
        cfg.seed = args.seed
        cfg.num_envs = 1
        cfg.headless = args.headless
        cfg.exp_base = "generator_eval"
        cfg.experiment_name = "generator_eval"
        cfg.experiment_dir = str(args.output_dir)
        cfg.output_dir = str(args.output_dir)

        env_cfg = cfg.manager_env.config
        env_cfg.num_envs = 1
        env_cfg.headless = args.headless
        env_cfg.gpu_collision_stack_size_exp = 28
        env_cfg.render_results = args.render_video
        env_cfg.enable_cameras = False
        env_cfg.render_width = 640
        env_cfg.render_height = 480
        env_cfg.use_motion_hand_actions = False
        env_cfg.use_latent_residual = False
        env_cfg.use_student_direct_latent = True
        env_cfg.action_transform_module_cfg = str(model_dir / "model_config.yaml")
        env_cfg.action_transform_module_checkpoint = str(model_dir / "last.pt")
        env_cfg.object_usd_path = str(assets.object_usd)
        env_cfg.motion_meta_info_path = str(assets.meta)
        env_cfg.table_size = list(reference.table_size)

        termination_cfg = cfg.manager_env.terminations
        for name in list(termination_cfg):
            if name not in {"_target_", "time_out"}:
                del termination_cfg[name]

        motion_cfg = cfg.manager_env.commands.motion
        motion_cfg.sample_before_contact = False
        motion_cfg.sample_from_n_initial_frames = None
        motion_cfg.start_from_first_frame = True
        motion_cfg.use_paired_motions = True
        motion_cfg.motion_lib_cfg.motion_file = str(assets.robot.parent)
        motion_cfg.motion_lib_cfg.object_motion_file = str(assets.objects.parent)
        motion_cfg.motion_lib_cfg.bps_dir = str(assets.bps.parent)
        motion_cfg.motion_lib_cfg.filter_motion_keys = [assets.stem]
        motion_cfg.motion_lib_cfg.target_fps = CONTROL_FPS
    return cfg


def _validate_generator(policy, cfg, env) -> None:
    action = cfg.shape_meta.action
    actual = (int(action.horizon), int(action.latent_dim), int(action.hand_dim))
    expected = (PREDICTION_HORIZON, 64, 2)
    if actual != expected:
        raise ValueError(f"Generator action shape must be {expected}, got {actual}")
    if (policy.action_horizon, policy.latent_dim) != expected[:2]:
        raise ValueError("Generator checkpoint dimensions disagree with its config")
    if (policy.num_inference_steps, policy.num_timestep_buckets) != (
        FLOW_INFERENCE_STEPS,
        FLOW_TIMESTEP_BUCKETS,
    ):
        raise ValueError("Unexpected flow-matching sampler configuration")
    if abs(float(env.env.step_dt) - 1.0 / CONTROL_FPS) > 1e-8:
        raise ValueError(f"Isaac Lab must run at {CONTROL_FPS} Hz")


def _generator_observation(
    env,
    generator,
    obs_dict,
    hand_frame_cfg,
    hand_transform_fn,
    torch,
):
    """Build generator input exclusively from the current simulator state."""
    actor_obs = obs_dict["actor_obs"]
    actor_obs = actor_obs[:, -1] if actor_obs.ndim == 3 else actor_obs
    manager = env.env.observation_manager
    names = manager._group_obs_term_names["policy"]  # noqa: SLF001
    dimensions = manager._group_obs_term_dim["policy"]  # noqa: SLF001
    terms = {}
    offset = 0
    for name, shape in zip(names, dimensions):
        width = math.prod(shape)
        terms[name] = actor_obs[:, offset : offset + width].reshape(
            actor_obs.shape[0], *shape
        )
        offset += width

    forces = terms["finger_tips_force"].reshape(actor_obs.shape[0], -1, 3)
    table_size = env.config["table_size"]
    table_height = (
        env.env.scene["table"].data.root_pos_w[:, 2]
        - env.env.scene.env_origins[:, 2]
    )
    table_geometry = torch.stack(
        (
            table_height,
            torch.full_like(table_height, float(table_size[2])),
            torch.full_like(table_height, float(table_size[0])),
            torch.full_like(table_height, float(table_size[1])),
        ),
        dim=-1,
    )
    robot = env.env.scene["robot"]
    obj = env.env.scene["object"]
    return generator.observation_from_world(
        object_bps=env.motion_command.object_bps.detach(),
        table_geometry=table_geometry,
        robot_position_w=robot.data.root_pos_w.detach(),
        robot_quaternion_w=robot.data.root_quat_w.detach(),
        object_position_w=obj.data.root_pos_w.detach(),
        object_quaternion_w=obj.data.root_quat_w.detach(),
        hand_object_transform_6d=hand_transform_fn(
            env.env, hand_frame_cfg
        ).detach(),
        hand_object_contact_force_magnitude=torch.linalg.vector_norm(
            forces, dim=-1
        ),
        **{
            name: terms[name].detach()
            for name in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")
        },
    )


def _reset_episode(env, scene: EpisodeScene, reference: ReferenceScene, torch):
    env.env.sim.reset()
    obs_dict = env.reset()
    origin = env.env.scene.env_origins[0]
    device = env.device

    def pose(position, quaternion):
        value = torch.tensor(
            (*position, *quaternion), dtype=torch.float32, device=device
        ).unsqueeze(0)
        value[:, :3] += origin
        return value

    robot = env.env.scene["robot"]
    obj = env.env.scene["object"]
    table = env.env.scene["table"]
    robot.write_root_pose_to_sim(
        pose(scene.robot_position, reference.robot_quaternion_wxyz)
    )
    obj.write_root_pose_to_sim(
        pose(scene.object_position, reference.object_quaternion_wxyz)
    )
    table.write_root_pose_to_sim(
        pose(scene.table_position, reference.table_quaternion_wxyz)
    )
    zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)

    from gear_sonic.envs.env_utils.joint_utils import get_hand_joint_indices
    from gear_sonic.envs.manager_env.mdp.observations import G1_MUJOCO_ORDER

    body_joint_indices = torch.tensor(
        [robot.joint_names.index(name) for name in G1_MUJOCO_ORDER],
        dtype=torch.long,
        device=device,
    )
    hand_joint_indices = get_hand_joint_indices(robot)
    if len(body_joint_indices) != len(reference.robot_body_joint_positions_mujoco):
        raise ValueError(
            "Reference motion body joint count does not match the robot: "
            f"{len(reference.robot_body_joint_positions_mujoco)} != "
            f"{len(body_joint_indices)}"
        )
    if len(hand_joint_indices) != len(reference.robot_hand_joint_positions):
        raise ValueError(
            "Reference motion hand joint count does not match the robot: "
            f"{len(reference.robot_hand_joint_positions)} != {len(hand_joint_indices)}"
        )
    joint_pos = robot.data.default_joint_pos[:1].clone()
    joint_pos[:, body_joint_indices] = torch.tensor(
        reference.robot_body_joint_positions_mujoco,
        dtype=torch.float32,
        device=device,
    )
    joint_pos[:, hand_joint_indices] = torch.tensor(
        reference.robot_hand_joint_positions,
        dtype=torch.float32,
        device=device,
    )
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # Keep reset-time actuator targets consistent with the injected frame.
    robot.set_joint_position_target(joint_pos)
    robot.set_joint_velocity_target(joint_vel)
    robot.set_joint_effort_target(torch.zeros_like(joint_pos))

    robot.write_root_velocity_to_sim(zero_velocity)
    obj.write_root_velocity_to_sim(zero_velocity)

    env.env.sim.forward()
    env.env.scene.update(dt=0.0)
    raw_obs = env.env.observation_manager.compute()
    obs_dict = env.process_raw_obs(raw_obs, flatten_dict_obs=True)
    env._last_obs_dict = obs_dict  # noqa: SLF001
    return obs_dict


def _render_frame(base_env, torch):
    camera = base_env.scene["eval_camera"]
    origin = base_env.scene.env_origins
    eye = origin + torch.tensor((2.8, 2.8, 2.0), device=base_env.device)
    target = origin + torch.tensor((0.0, 0.0, 0.9), device=base_env.device)
    camera._view._sync_usd_on_fabric_write = True  # noqa: SLF001
    camera.set_world_poses_from_view(eye, target)
    base_env.sim.render()
    base_env.sim.render()
    camera._is_outdated[:] = True  # noqa: SLF001
    camera.update(dt=0.0, force_recompute=True)
    return camera.data.output["rgb"][0].detach().cpu().numpy()


def _video_writer(args, episode: int):
    if not args.render_video:
        return None
    import imageio.v2 as imageio

    video_dir = args.output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        video_dir / f"episode_{episode:04d}.mp4",
        fps=CONTROL_FPS,
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
    )


def _run_episode(
    env,
    generator,
    scene,
    reference,
    args,
    episode,
    episode_seed,
    hand_frame_cfg,
    hand_transform_fn,
    simulation_app,
    torch,
) -> EpisodeResult:
    obs_dict = _reset_episode(env, scene, reference, torch)
    initial_object_z = float(env.env.scene["object"].data.root_pos_w[0, 2].item())
    writer = _video_writer(args, episode)
    steps = 0
    success = False
    lift = 0.0
    try:
        if writer is not None:
            writer.append_data(_render_frame(env.env, torch))
        while simulation_app.is_running() and steps < args.max_episode_steps:
            generator_obs = _generator_observation(
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
            if latents.shape != (1, PREDICTION_HORIZON, 64):
                raise ValueError(f"Unexpected latent shape: {tuple(latents.shape)}")
            if hands.shape != (1, PREDICTION_HORIZON, 2):
                raise ValueError(f"Unexpected hand shape: {tuple(hands.shape)}")

            for index in range(EXECUTION_HORIZON):
                action = torch.cat(
                    (latents[:, index], binary_hand_to_sonic(hands[:, index])),
                    dim=-1,
                )
                obs_dict, _, dones, _ = env.step(
                    {
                        "actions": action,
                        "obs_dict": obs_dict,
                        "action_mode": "direct_latent",
                    }
                )
                steps += 1
                object_z = float(env.env.scene["object"].data.root_pos_w[0, 2].item())
                lift = max(lift, object_z - initial_object_z)
                if writer is not None:
                    writer.append_data(_render_frame(env.env, torch))
                # Success is recorded, but never terminates the episode.
                success = success or lift >= args.lift_height
                if bool(dones.any().item()) or steps >= args.max_episode_steps:
                    break
            # The environment has only the time_out termination enabled.
            if bool(dones.any().item()):
                break
    finally:
        if writer is not None:
            writer.close()

    return EpisodeResult(
        episode=episode,
        seed=episode_seed,
        success=success,
        steps=steps,
        lift=lift,
        scene=scene,
    )


def _save_results(args, results: list[EpisodeResult]) -> None:
    successes = sum(result.success for result in results)
    payload = {
        "mode": args.mode,
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "lift_height": args.lift_height,
        "prediction_horizon": PREDICTION_HORIZON,
        "execution_horizon": EXECUTION_HORIZON,
        "randomization": asdict(args.randomization),
        "results": [asdict(result) for result in results],
    }
    path = args.output_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\nResult: {successes}/{len(results)} successful "
        f"({payload['success_rate']:.1%})\nSaved: {path}"
    )


def _run(args, simulation_app, sonic_root: Path, assets: MotionAssets) -> None:
    # Isaac Lab and SONIC must be imported after AppLauncher has started.
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

    reference = load_reference_scene(assets)
    cfg = _compose_sonic_config(sonic_root, assets, reference, args)
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
    _validate_generator(generator.policy, payload["cfg"], env)

    hand_frame_cfg = SceneEntityCfg("object_to_hand_frame_transformer")
    results = []
    with torch.inference_mode():
        for episode in range(args.episodes):
            if not simulation_app.is_running():
                break
            episode_seed = args.seed + episode
            seeding(episode_seed)
            scene = _sample_scene(reference, args, episode_seed)
            result = _run_episode(
                env,
                generator,
                scene,
                reference,
                args,
                episode,
                episode_seed,
                hand_frame_cfg,
                hand_object_transform_6d,
                simulation_app,
                torch,
            )
            results.append(result)
            print(
                f"Episode {episode + 1}/{args.episodes}: "
                f"{'SUCCESS' if result.success else 'FAIL'}, "
                f"steps={result.steps}, lift={result.lift:.3f} m, scene={scene}"
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
    assets = resolve_motion_assets(args.motion_file)

    launcher = AppLauncher(args)
    try:
        _run(args, launcher.app, sonic_root, assets)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
