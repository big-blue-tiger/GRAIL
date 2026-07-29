#!/usr/bin/env python3
"""Run an ObjectAware flow-matching generator in a SONIC Isaac Lab loop."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path


HORIZON = 40
EXECUTION_HORIZON = 20
FLOW_INFERENCE_STEPS = 4
FLOW_TIMESTEP_BUCKETS = 1000
CONTROL_FPS = 50
DEFAULT_MOTION = Path(
    "/home/tide/robot/sbto/datas/sbto_to_grail/pickup_table/robot/"
    "pickup_table__apple_18__001.pkl"
)
DEFAULT_INFERENCE_DATA = Path(
    "outputs/ego_view/all/"
    "pickup_table__apple_18__001_x+0.00_y+0.00.object_aware.pkl"
)


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


def resolve_motion_assets(motion_file: str | Path) -> MotionAssets:
    """Resolve the paired SBTO files for one robot motion."""
    robot = Path(motion_file).expanduser().resolve()
    if robot.suffix != ".pkl" or robot.parent.name != "robot":
        raise ValueError("--motion-file must be a robot/<motion-stem>.pkl file")
    root = robot.parent.parent
    stem = robot.stem
    assets = MotionAssets(
        robot=robot,
        objects=root / "objects" / f"{stem}.pkl",
        object_usd=root / "object_usd" / f"{stem}.usd",
        bps=root / "bps" / f"{stem}.npy",
        meta=root / "meta" / f"{stem}.pkl",
    )
    missing = [str(path) for path in assets.__dict__.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing paired motion assets: " + ", ".join(missing))
    return assets


def binary_hand_to_sonic(hand_binary):
    """Map generator hand values (0=open, 1=closed) to SONIC (-1/+1)."""
    return hand_binary * 2.0 - 1.0


def _print_termination_terms(env) -> None:
    """Print the per-term result retained by Isaac Lab across its automatic reset."""
    manager = env.env.termination_manager
    term_dones = getattr(manager, "_last_episode_dones", None)
    if term_dones is None:
        term_dones = getattr(manager, "_term_dones", None)
    if term_dones is None:
        print("Termination terms unavailable")
        return

    values = term_dones[0].detach().cpu().tolist()
    print("Termination terms:")
    for name, triggered in zip(manager.active_terms, values, strict=True):
        print(f"  {name}: {bool(triggered)}")


def _resolve_from(path: Path, base: Path) -> Path:
    path = path.expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_inference_data(path: Path, launch_cwd: Path, sonic_root: Path) -> Path:
    """Accept paths relative to either the launch directory or SONIC root."""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    launch_relative = (launch_cwd / path).resolve()
    return launch_relative if launch_relative.is_file() else (sonic_root / path).resolve()


def _parse_args(app_launcher_cls) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-checkpoint", required=True, type=Path)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument(
        "--inference-data",
        type=Path,
        default=DEFAULT_INFERENCE_DATA,
        help=(
            "Schema-v3 *.object_aware.pkl used only to initialize the scene."
        ),
    )
    parser.add_argument("--robot-position", type=float, nargs=3)
    parser.add_argument("--robot-quaternion-wxyz", type=float, nargs=4)
    parser.add_argument("--object-initial-position", type=float, nargs=3)
    parser.add_argument("--object-initial-quaternion-wxyz", type=float, nargs=4)
    parser.add_argument("--table-position", type=float, nargs=3)
    parser.add_argument("--table-quaternion-wxyz", type=float, nargs=4)
    parser.add_argument(
        "--table-size",
        type=float,
        nargs=3,
        metavar=("WIDTH", "DEPTH", "THICKNESS"),
        help="Override the table dimensions loaded from the paired meta file.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Seed increment between episodes; episode N uses seed + N * seed-step.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum total 50 Hz control steps; 0 keeps running episodes until the app closes.",
    )
    parser.add_argument(
        "--debug-reset-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for complete per-episode trajectory snapshots, including "
            "all states, generator observations, predicted latent/hand horizons, executed "
            "actions, rewards, and dones."
        ),
    )
    app_launcher_cls.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    return args


def _debug_tensor(value):
    """Detach a tensor-like debug value without retaining simulator storage."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    return value


def _capture_reset_debug_state(env, obs_dict, torch) -> dict:
    """Capture state needed to diagnose discontinuities across automatic/manual resets."""
    robot = env.env.scene["robot"]
    command = env.motion_command
    state = {
        "robot": {
            "root_state_w": _debug_tensor(robot.data.root_state_w),
            "root_pos_w": _debug_tensor(robot.data.root_pos_w),
            "root_quat_w": _debug_tensor(robot.data.root_quat_w),
            "root_lin_vel_w": _debug_tensor(robot.data.root_lin_vel_w),
            "root_ang_vel_w": _debug_tensor(robot.data.root_ang_vel_w),
            "joint_pos": _debug_tensor(robot.data.joint_pos),
            "joint_vel": _debug_tensor(robot.data.joint_vel),
            "body_pos_w": _debug_tensor(robot.data.body_pos_w),
            "body_quat_w": _debug_tensor(robot.data.body_quat_w),
        },
        "motion": {
            "motion_ids": _debug_tensor(command.motion_ids),
            "motion_start_time_steps": _debug_tensor(command.motion_start_time_steps),
            "time_steps": _debug_tensor(command.time_steps),
            "reference_frame": _debug_tensor(
                command.motion_start_time_steps + command.time_steps
            ),
            "first_contact": _debug_tensor(
                getattr(command, "_per_env_first_contact", None)
            ),
            "initial_root_pose_offset": _debug_tensor(
                getattr(command, "initial_root_pose_offset", None)
            ),
        },
        "observations": {
            key: _debug_tensor(value)
            for key, value in obs_dict.items()
            if isinstance(value, torch.Tensor)
        },
    }
    if "object" in env.env.scene.rigid_objects:
        obj = env.env.scene["object"]
        state["object"] = {
            "root_state_w": _debug_tensor(obj.data.root_state_w),
            "root_pos_w": _debug_tensor(obj.data.root_pos_w),
            "root_quat_w": _debug_tensor(obj.data.root_quat_w),
            "root_lin_vel_w": _debug_tensor(obj.data.root_lin_vel_w),
            "root_ang_vel_w": _debug_tensor(obj.data.root_ang_vel_w),
        }
    return state


def _save_reset_debug_snapshot(debug_dir: Path | None, episode: int, phase: str, payload, torch):
    if debug_dir is None:
        return
    path = debug_dir / f"episode_{episode:06d}_{phase}.pt"
    torch.save(payload, path)
    print(f"Reset debug snapshot: {path}")


def _reset_physics_and_env(env):
    """Clear PhysX state, reset managers/assets, and rebuild synchronized observations."""
    base_env = env.env

    # ManagerBasedRLEnv.reset() only rewrites asset state. SimulationContext.reset()
    # performs a real physics-scene reset first, clearing contact manifolds and
    # solver warm-start impulses retained by PhysX from the preceding episode.
    base_env.sim.reset()
    obs_dict = env.reset()

    # Synchronize the state written by the manager reset before the first policy
    # action, then recompute observations from the refreshed scene buffers.
    base_env.sim.forward()
    base_env.scene.update(dt=0.0)

    raw_obs = base_env.observation_manager.compute()
    obs_dict = env.process_raw_obs(raw_obs, flatten_dict_obs=True)
    env._last_obs_dict = obs_dict  # noqa: SLF001
    return obs_dict


def _load_scene_initialization(path: Path, assets: MotionAssets) -> dict:
    """Read only the poses needed to initialize inference from a recording."""
    import joblib

    with path.open("rb") as file:
        episode = pickle.load(file)
    if episode.get("schema_version") != 3:
        raise ValueError(f"{path}: expected schema_version=3")
    if episode.get("pose_quaternion_format") != "wxyz":
        raise ValueError(f"{path}: expected wxyz quaternions")
    required = (
        "motion_key",
        "robot_root_pos_w",
        "robot_root_quat_w",
        "object_root_pos_w",
        "object_root_quat_w",
    )
    missing = [key for key in required if key not in episode]
    if missing:
        raise KeyError(f"{path}: missing initialization fields: {', '.join(missing)}")
    if len(episode["robot_root_pos_w"]) < 1:
        raise ValueError(f"{path}: contains no recorded frames")

    robot_payload = joblib.load(assets.robot)
    robot_motion = robot_payload[episode["motion_key"]]
    reference_robot_position = robot_motion["root_trans_offset"][0]
    initial_offset = episode.get("initial_root_pose_offset_xyz", (0.0, 0.0, 0.0))
    source_env_origin = (
        episode["robot_root_pos_w"][0] - reference_robot_position - initial_offset
    )

    table_meta = joblib.load(assets.meta)
    if "table_pos" not in table_meta or "table_quat" not in table_meta:
        raise KeyError(f"{assets.meta}: missing table_pos and/or table_quat")

    def local_position(key, index):
        return episode[key][index] - source_env_origin

    return {
        "motion_key": episode["motion_key"],
        "source_env_origin": source_env_origin,
        "robot_position": local_position("robot_root_pos_w", 0),
        "robot_quaternion": episode["robot_root_quat_w"][0],
        "object_initial_position": local_position("object_root_pos_w", 0),
        "object_initial_quaternion": episode["object_root_quat_w"][0],
        "table_position": table_meta["table_pos"],
        "table_quaternion": table_meta["table_quat"],
    }


def _override(value, manual_value):
    return manual_value if manual_value is not None else value


def _apply_scene_initialization(env, initialization, args, torch):
    """Write the selected initial robot, object, and table poses."""
    device = env.device
    env_origin = env.env.scene.env_origins[0]
    robot = env.env.scene["robot"]
    obj = env.env.scene["object"]
    print(f"the robot pose is {initialization['robot_position']}, {initialization['robot_quaternion']} \n the object pose is {initialization['object_initial_position']}, {initialization['object_initial_quaternion']} \n the table pose is {initialization['table_position']}, {initialization['table_quaternion']}")
    robot_pose = torch.tensor(
        [
            *(_override(initialization["robot_position"], args.robot_position)),
            *(
                _override(
                    initialization["robot_quaternion"],
                    args.robot_quaternion_wxyz,
                )
            ),
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    object_pose = torch.tensor(
        [
            *(
                _override(
                    initialization["object_initial_position"],
                    args.object_initial_position,
                )
            ),
            *(
                _override(
                    initialization["object_initial_quaternion"],
                    args.object_initial_quaternion_wxyz,
                )
            ),
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    robot_pose[:, :3] += env_origin
    object_pose[:, :3] += env_origin
    robot.write_root_pose_to_sim(robot_pose)
    obj.write_root_pose_to_sim(object_pose)

    table = env.env.scene["table"]
    table_position = torch.tensor(
        _override(initialization["table_position"], args.table_position),
        dtype=torch.float32,
        device=device,
    ) + env_origin
    table_quaternion = torch.tensor(
        _override(
            initialization["table_quaternion"],
            args.table_quaternion_wxyz,
        ),
        dtype=torch.float32,
        device=device,
    )
    table.write_root_pose_to_sim(
        torch.cat((table_position, table_quaternion)).unsqueeze(0)
    )

    env.env.sim.forward()
    env.env.scene.update(dt=0.0)
    raw_obs = env.env.observation_manager.compute()
    obs_dict = env.process_raw_obs(raw_obs, flatten_dict_obs=True)
    env._last_obs_dict = obs_dict  # noqa: SLF001
    return obs_dict


def _compose_sonic_config(sonic_root: Path, assets: MotionAssets, args):
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
    run_dir = sonic_root / "output" / "generator_isaaclab"
    with open_dict(cfg):
        cfg.seed = args.seed
        cfg.num_envs = 1
        cfg.headless = args.headless
        cfg.exp_base = "generator_isaaclab"
        cfg.experiment_name = "generator_isaaclab_pnp_table"
        cfg.experiment_dir = str(run_dir)
        cfg.output_dir = str(run_dir)

        env_cfg = cfg.manager_env.config
        env_cfg.num_envs = 1
        env_cfg.gpu_collision_stack_size_exp = 28
        env_cfg.headless = args.headless
        env_cfg.render_results = False
        env_cfg.enable_cameras = False
        env_cfg.use_motion_hand_actions = False
        env_cfg.use_latent_residual = False
        env_cfg.use_student_direct_latent = True
        env_cfg.action_transform_module_cfg = str(model_dir / "model_config.yaml")
        env_cfg.action_transform_module_checkpoint = str(model_dir / "last.pt")
        env_cfg.object_usd_path = str(assets.object_usd)
        env_cfg.motion_meta_info_path = str(assets.meta)
        if args.table_size is not None:
            env_cfg.table_size = list(args.table_size)

        termination_cfg = cfg.manager_env.terminations
        for term_name in list(termination_cfg):
            if term_name not in {"_target_", "time_out"}:
                del termination_cfg[term_name]

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
    action_meta = cfg.shape_meta.action
    actual = (int(action_meta.horizon), int(action_meta.latent_dim), int(action_meta.hand_dim))
    expected = (HORIZON, 64, 2)
    if actual != expected:
        raise ValueError(f"Generator action shape must be {expected}, got {actual}")
    if (policy.action_horizon, policy.latent_dim) != expected[:2]:
        raise ValueError(
            "Checkpoint policy dimensions disagree with its config: "
            f"horizon={policy.action_horizon}, latent_dim={policy.latent_dim}"
        )
    flow_config = (policy.num_inference_steps, policy.num_timestep_buckets)
    expected_flow_config = (FLOW_INFERENCE_STEPS, FLOW_TIMESTEP_BUCKETS)
    if flow_config != expected_flow_config:
        raise ValueError(
            "Flow-matching sampler must use "
            f"steps={expected_flow_config[0]}, buckets={expected_flow_config[1]}; "
            f"got steps={flow_config[0]}, buckets={flow_config[1]}"
        )
    if abs(float(env.env.step_dt) - 1.0 / CONTROL_FPS) > 1e-8:
        raise ValueError(
            f"Isaac Lab control dt must be {1.0 / CONTROL_FPS}, got {env.env.step_dt}"
        )
    motion_fps = float(env.motion_command.motion_lib._sim_fps)  # noqa: SLF001
    if abs(motion_fps - CONTROL_FPS) > 1e-8:
        raise ValueError(f"MotionLib must run at {CONTROL_FPS} Hz, got {motion_fps}")


def _generator_observation(
    env,
    generator,
    obs_dict,
    hand_frame_cfg,
    hand_transform_fn,
    torch,
):
    """Build observations from the current simulator state and fixed goals."""
    robot = env.env.scene["robot"]
    obj = env.env.scene["object"]
    actor_obs = obs_dict["actor_obs"]
    actor_obs = actor_obs[:, -1] if actor_obs.ndim == 3 else actor_obs
    manager = env.env.observation_manager
    names = manager._group_obs_term_names["policy"]  # noqa: SLF001
    dims = manager._group_obs_term_dim["policy"]  # noqa: SLF001
    slices = {}
    offset = 0
    for name, shape in zip(names, dims):
        width = math.prod(shape)
        slices[name] = actor_obs[:, offset : offset + width].reshape(
            actor_obs.shape[0], *shape
        )
        offset += width
    hand_object_transform = hand_transform_fn(env.env, hand_frame_cfg).detach()
    finger_force = slices["finger_tips_force"].reshape(actor_obs.shape[0], -1, 3)
    hand_object_contact_force_magnitude = torch.linalg.vector_norm(
        finger_force, dim=-1
    )
    table = env.env.scene["table"]
    table_size = env.config.get("table_size")
    if table_size is None or len(table_size) != 3:
        raise RuntimeError(
            "Generator inference requires table_size=[length, width, thickness]."
        )
    table_height = table.data.root_pos_w[:, 2] - env.env.scene.env_origins[:, 2]
    table_geometry = torch.stack(
        (
            table_height,
            torch.full_like(table_height, float(table_size[2])),
            torch.full_like(table_height, float(table_size[0])),
            torch.full_like(table_height, float(table_size[1])),
        ),
        dim=-1,
    )
    return generator.observation_from_world(
        object_bps=env.motion_command.object_bps.detach(),
        table_geometry=table_geometry,
        robot_position_w=robot.data.root_pos_w.detach(),
        robot_quaternion_w=robot.data.root_quat_w.detach(),
        object_position_w=obj.data.root_pos_w.detach(),
        object_quaternion_w=obj.data.root_quat_w.detach(),
        hand_object_transform_6d=hand_object_transform,
        hand_object_contact_force_magnitude=hand_object_contact_force_magnitude,
        **{
            key: slices[key].detach()
            for key in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")
        },
    )


def _run(args, simulation_app, sonic_root: Path, assets: MotionAssets) -> None:
    # Isaac Lab and SONIC modules must be imported only after AppLauncher starts.
    import dill
    import hydra
    import torch
    from isaaclab.managers import SceneEntityCfg

    from gear_sonic import train_agent_trl
    from gear_sonic.envs.manager_env.mdp.observations import hand_object_transform_6d
    from gear_sonic.utils.common import seeding
    from sugar_il.wrapper.sugar_il_wrapper import GeneratorWrapper

    initialization = _load_scene_initialization(args.inference_data, assets)
    if initialization["motion_key"] != assets.stem:
        raise ValueError(
            f"Inference data motion_key={initialization['motion_key']!r} does not match "
            f"motion asset {assets.stem!r}"
        )
    cfg = _compose_sonic_config(sonic_root, assets, args)
    seeding(args.seed)
    env = train_agent_trl.create_manager_env(cfg, args.device, args)
    payload = torch.load(
        args.generator_checkpoint,
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
    )
    policy = hydra.utils.instantiate(payload["cfg"].policy)
    policy.load_state_dict(payload["state_dicts"]["model"])
    generator = GeneratorWrapper(policy, args.device)
    _validate_generator(generator.policy, payload["cfg"], env)
    hand_frame_cfg = SceneEntityCfg("object_to_hand_frame_transformer")

    total_steps = 0
    episode_index = 0
    with torch.inference_mode():
        while simulation_app.is_running() and not (
            args.max_steps and total_steps >= args.max_steps
        ):
            episode_seed = args.seed + episode_index * args.seed_step
            seeding(episode_seed)

            # ManagerBasedRLEnv automatically resets on termination. Reset once
            # more after reseeding at the recording's initial reference frame.
            obs_dict = _reset_physics_and_env(env)
            obs_dict = _apply_scene_initialization(env, initialization, args, torch)
            start = int(env.motion_command.motion_start_time_steps[0].item())
            first_contact = int(
                env.motion_command._per_env_first_contact[0].item()  # noqa: SLF001
            )
            print(
                f"Episode={episode_index}, seed={episode_seed}, motion={assets.stem}, "
                f"start={start}, first_contact={first_contact}, control_fps={CONTROL_FPS}, "
                f"flow_steps={generator.policy.num_inference_steps}"
            )
            episode_debug = None
            if args.debug_reset_dir is not None:
                episode_debug = {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "motion": assets.stem,
                    "initial_state": _capture_reset_debug_state(env, obs_dict, torch),
                    "predictions": [],
                    "steps": [],
                }

            episode_steps = 0
            done = False

            while (
                simulation_app.is_running()
                and not done
                and not (args.max_steps and total_steps >= args.max_steps)
            ):
                generator_obs = _generator_observation(
                    env,
                    generator,
                    obs_dict,
                    hand_frame_cfg,
                    hand_object_transform_6d,
                    torch,
                )
                prediction = generator.policy.predict_action(generator_obs)
                latents = prediction["latent"]
                hands = prediction["hand_primitive"]
                prediction_index = None
                if episode_debug is not None:
                    prediction_index = len(episode_debug["predictions"])
                    episode_debug["predictions"].append(
                        {
                            "episode_step": episode_steps,
                            "generator_observation": {
                                key: _debug_tensor(value)
                                for key, value in generator_obs.items()
                            },
                            "latent_horizon": _debug_tensor(latents),
                            "hand_horizon": _debug_tensor(hands),
                            "hand_logits_horizon": _debug_tensor(
                                prediction.get("hand_logits")
                            ),
                            "hand_probability_horizon": _debug_tensor(
                                prediction.get("hand_probability")
                            ),
                        }
                    )
                if latents.shape != (1, HORIZON, 64) or hands.shape != (1, HORIZON, 2):
                    raise ValueError(
                        f"Unexpected flow-matching output shapes: latent={tuple(latents.shape)}, "
                        f"hand={tuple(hands.shape)}"
                    )

                for index in range(EXECUTION_HORIZON):
                    hand_binary = hands[:, index]
                    meta_action = torch.cat(
                        (latents[:, index], binary_hand_to_sonic(hand_binary)), dim=-1
                    )
                    step_debug = None
                    if episode_debug is not None:
                        step_debug = {
                            "episode_step": episode_steps,
                            "prediction_index": prediction_index,
                            "horizon_index": index,
                            "state_before_step": _capture_reset_debug_state(
                                env, obs_dict, torch
                            ),
                            "meta_action": _debug_tensor(meta_action),
                            "latent": _debug_tensor(latents[:, index]),
                            "hand_binary": _debug_tensor(hand_binary),
                        }
                    obs_dict, reward, dones, _ = env.step(
                        {
                            "actions": meta_action,
                            "obs_dict": obs_dict,
                            "action_mode": "direct_latent",
                        }
                    )
                    total_steps += 1
                    episode_steps += 1
                    done = bool(dones.any().item())
                    if step_debug is not None:
                        step_debug["reward"] = _debug_tensor(reward)
                        step_debug["dones"] = _debug_tensor(dones)
                        step_debug["state_after_step"] = _capture_reset_debug_state(
                            env, obs_dict, torch
                        )
                        episode_debug["steps"].append(step_debug)
                    if done:
                        _print_termination_terms(env)
                    if done or (args.max_steps and total_steps >= args.max_steps):
                        break

            if episode_debug is not None:
                episode_debug["episode_steps"] = episode_steps
                episode_debug["finished_with_done"] = done
                _save_reset_debug_snapshot(
                    args.debug_reset_dir,
                    episode_index,
                    "trajectory",
                    episode_debug,
                    torch,
                )
            print(
                f"Episode {episode_index} finished after {episode_steps} control steps "
                f"(total={total_steps})"
            )
            episode_index += 1

    print(
        f"Closed-loop inference stopped after {episode_index} episodes and "
        f"{total_steps} control steps"
    )
    env.env.close()


def main() -> None:
    launch_cwd = Path.cwd()
    sonic_root = Path(__file__).resolve().parents[3]
    os.chdir(sonic_root)
    if str(sonic_root) not in sys.path:
        sys.path.insert(0, str(sonic_root))
    sugar_root = sonic_root / "sugar_il"
    if str(sugar_root) not in sys.path:
        sys.path.insert(0, str(sugar_root))

    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        try:
            from omni.isaac.lab.app import AppLauncher
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Lab is required. Run this script inside the SONIC Isaac Lab environment."
            ) from exc

    args = _parse_args(AppLauncher)
    args.generator_checkpoint = _resolve_from(args.generator_checkpoint, launch_cwd)
    args.motion_file = _resolve_from(args.motion_file, launch_cwd)
    args.inference_data = _resolve_inference_data(
        args.inference_data, launch_cwd, sonic_root
    )
    if args.debug_reset_dir is not None:
        args.debug_reset_dir = _resolve_from(args.debug_reset_dir, launch_cwd)
        args.debug_reset_dir.mkdir(parents=True, exist_ok=True)
    if not args.generator_checkpoint.is_file():
        raise FileNotFoundError(args.generator_checkpoint)
    if not args.inference_data.is_file():
        raise FileNotFoundError(args.inference_data)
    assets = resolve_motion_assets(args.motion_file)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        _run(args, simulation_app, sonic_root, assets)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
