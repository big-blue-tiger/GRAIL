#!/usr/bin/env python3
"""Run an ObjectAware generator in a receding-horizon SONIC Isaac Lab loop."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


HORIZON = 16
EXECUTION_HORIZON = 8
CONTROL_FPS = 50
CONTACT_MARGIN = 50
DEFAULT_MOTION = Path(
    "/home/tide/robot/sbto/datas/sbto_to_grail/pickup_table/robot/"
    "pickup_table__apple_3__004.pkl"
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


def valid_start_upper_bound(first_contact_frame: int, margin: int = CONTACT_MARGIN) -> int:
    """Return the exclusive upper bound used for pre-contact reset sampling."""
    if first_contact_frame <= margin:
        return 1
    return first_contact_frame - margin


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


def _parse_args(app_launcher_cls) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-checkpoint", required=True, type=Path)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum 50 Hz control steps; 0 runs until termination or motion end.",
    )
    app_launcher_cls.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    return args


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

        termination_cfg = cfg.manager_env.terminations
        for term_name in list(termination_cfg):
            if term_name not in {"_target_", "time_out"}:
                del termination_cfg[term_name]

        motion_cfg = cfg.manager_env.commands.motion
        motion_cfg.sample_before_contact = True
        motion_cfg.sample_before_contact_margin = CONTACT_MARGIN
        motion_cfg.sample_from_n_initial_frames = None
        motion_cfg.start_from_first_frame = False
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
    if abs(float(env.env.step_dt) - 1.0 / CONTROL_FPS) > 1e-8:
        raise ValueError(
            f"Isaac Lab control dt must be {1.0 / CONTROL_FPS}, got {env.env.step_dt}"
        )
    motion_fps = float(env.motion_command.motion_lib._sim_fps)  # noqa: SLF001
    if abs(motion_fps - CONTROL_FPS) > 1e-8:
        raise ValueError(f"MotionLib must run at {CONTROL_FPS} Hz, got {motion_fps}")


def _initial_atm_latent(env, obs_dict, torch):
    """Encode the reset state with zero residual without advancing physics."""
    atm_obs = env._prepare_obs_for_action_transform_module(obs_dict)  # noqa: SLF001
    zeros = torch.zeros(1, 64, device=env.device)
    env.action_transform_module(
        atm_obs,
        latent_residual=zeros,
        latent_residual_mode="pre_quantization",
    )
    atm = env.action_transform_module.actor_module
    latent = atm._last_pre_quantization_latent_flat  # noqa: SLF001
    if latent is None:
        raise RuntimeError("SONIC ATM did not expose its pre-FSQ latent")
    latent = latent[:, -1] if latent.ndim == 3 else latent
    if latent.shape != (1, 64):
        raise ValueError(f"Expected initial ATM latent [1,64], got {tuple(latent.shape)}")
    return latent.detach()


def _initial_hand_primitive(env, torch):
    command = env.motion_command
    hands = [command.get_hand_action(name) for name in ("left_hand", "right_hand")]
    if any(value is None for value in hands):
        raise RuntimeError("Motion file must contain both left and right hand actions")
    return (torch.stack(hands, dim=-1) >= 0).to(dtype=torch.float32)


def _fixed_object_goal(env, torch):
    """Read the selected object motion's final world pose through MotionLib."""
    command = env.motion_command
    motion_ids = command.motion_ids
    final_steps = command.motion_lib.get_motion_num_steps(motion_ids) - 1
    final_steps = torch.clamp(final_steps, min=0)
    position = command.motion_lib.get_object_root_pos(motion_ids, final_steps)[:, 0]
    position = position + env.env.scene.env_origins
    quaternion = command.motion_lib.get_object_root_quat(motion_ids, final_steps)[:, 0]
    return position.detach(), quaternion.detach()


def _generator_observation(
    env,
    generator,
    goal_position_w,
    goal_quaternion_w,
    last_latent,
    last_hand,
    hand_frame_cfg,
    hand_transform_fn,
):
    robot = env.env.scene["robot"]
    obj = env.env.scene["object"]
    hand_object = hand_transform_fn(env.env, hand_frame_cfg).detach()
    return generator.observation_from_world(
        object_bps=env.motion_command.object_bps.detach(),
        robot_position_w=robot.data.root_pos_w.detach(),
        robot_quaternion_w=robot.data.root_quat_w.detach(),
        object_position_w=obj.data.root_pos_w.detach(),
        object_quaternion_w=obj.data.root_quat_w.detach(),
        target_object_position_w=goal_position_w,
        target_object_quaternion_w=goal_quaternion_w,
        hand_object_transform_6d=hand_object,
        target_hand_object_transform_6d=hand_object,
        last_latent=last_latent,
        last_hand_primitive=last_hand,
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

    obs_dict = env.reset()
    start = int(env.motion_command.motion_start_time_steps[0].item())
    first_contact = int(env.motion_command._per_env_first_contact[0].item())  # noqa: SLF001
    upper = valid_start_upper_bound(first_contact)
    if not 0 <= start < upper:
        raise RuntimeError(
            f"Reset frame {start} is outside expected pre-contact range [0,{upper})"
        )
    print(
        f"Motion={assets.stem}, start={start}, first_contact={first_contact}, "
        f"control_fps={CONTROL_FPS}"
    )

    last_latent = _initial_atm_latent(env, obs_dict, torch)
    last_hand = _initial_hand_primitive(env, torch)
    goal_position_w, goal_quaternion_w = _fixed_object_goal(env, torch)
    hand_frame_cfg = SceneEntityCfg("object_to_hand_frame_transformer")

    steps = 0
    done = False
    with torch.inference_mode():
        while simulation_app.is_running() and not done:
            generator_obs = _generator_observation(
                env,
                generator,
                goal_position_w,
                goal_quaternion_w,
                last_latent,
                last_hand,
                hand_frame_cfg,
                hand_object_transform_6d,
            )
            prediction = generator.policy.predict_action(generator_obs)
            latents = prediction["latent"]
            hands = prediction["hand_primitive"]
            if latents.shape != (1, HORIZON, 64) or hands.shape != (1, HORIZON, 2):
                raise ValueError(
                    f"Unexpected DiT output shapes: latent={tuple(latents.shape)}, "
                    f"hand={tuple(hands.shape)}"
                )

            for index in range(EXECUTION_HORIZON):
                hand_binary = hands[:, index]
                meta_action = torch.cat(
                    (latents[:, index], binary_hand_to_sonic(hand_binary)), dim=-1
                )
                obs_dict, _, dones, _ = env.step(
                    {
                        "actions": meta_action,
                        "obs_dict": obs_dict,
                        "action_mode": "direct_latent",
                    }
                )
                steps += 1
                last_latent = latents[:, index].detach()
                last_hand = hand_binary.detach()
                done = bool(dones.any().item())
                if done:
                    _print_termination_terms(env)
                if done or (args.max_steps and steps >= args.max_steps):
                    done = True
                    break

    print(f"Closed-loop inference finished after {steps} control steps")
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
    if not args.generator_checkpoint.is_file():
        raise FileNotFoundError(args.generator_checkpoint)
    assets = resolve_motion_assets(args.motion_file)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        _run(args, simulation_app, sonic_root, assets)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
