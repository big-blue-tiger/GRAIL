"""SONIC environment construction and object-aware observations for DAgger."""

from __future__ import annotations

import faulthandler
import io
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict


ASSET_SUFFIXES = {
    "objects": ".pkl",
    "object_usd": ".usd",
    "bps": ".npy",
    "meta": ".pkl",
}


def resolve_motion_inputs(cfg) -> tuple[Path, Path, list[str]]:
    input_path = Path(cfg.paths.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.is_file():
        if input_path.suffix != ".pkl":
            raise ValueError("paths.input must be a motion .pkl or directory")
        robot_dir, keys = input_path.parent, [input_path.stem]
    else:
        robot_dir = input_path
        keys = sorted(path.stem for path in robot_dir.glob("*.pkl"))
    if not keys:
        raise ValueError(f"No motion files found in {robot_dir}")

    dataset_root = Path(cfg.paths.dataset_root or robot_dir.parent).expanduser().resolve()
    missing = [
        dataset_root / directory / f"{key}{suffix}"
        for key in keys
        for directory, suffix in ASSET_SUFFIXES.items()
        if not (dataset_root / directory / f"{key}{suffix}").is_file()
    ]
    if missing:
        preview = ", ".join(map(str, missing[:8]))
        raise FileNotFoundError(f"Missing paired motion assets: {preview}")
    return robot_dir, dataset_root, keys


def _checkpoint_config(checkpoint: Path) -> Path:
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        candidate = directory / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"config.yaml not found beside {checkpoint}")


def load_sonic_config(
    cfg,
    robot_dir: Path,
    dataset_root: Path,
    keys: list[str],
    num_envs: int,
):
    checkpoint = Path(cfg.paths.teacher_checkpoint)
    raw = _checkpoint_config(checkpoint).read_text()
    replacements = {
        "groot.rl.trl.": "gear_sonic.trl.",
        "groot.rl.envs.": "gear_sonic.envs.",
        "groot.rl.utils.": "gear_sonic.utils.",
        "groot.rl.agents.modules.modules.": "gear_sonic.trl.modules.base_module.",
        "groot.rl.agents.": "gear_sonic.trl.",
        "groot/rl/data/": "gear_sonic/data/",
        "assets/bm/unitree_description/": "assets/robot_description/",
        "1215_bones_seed_filtered": "bones_seed_smpl",
    }
    for old, new in replacements.items():
        raw = raw.replace(old, new)
    sonic_cfg = OmegaConf.load(io.StringIO(raw))
    if sonic_cfg.get("eval_overrides") is not None:
        sonic_cfg = OmegaConf.merge(sonic_cfg, sonic_cfg.eval_overrides)

    with open_dict(sonic_cfg):
        sonic_cfg.checkpoint = str(checkpoint)
        sonic_cfg.seed = cfg.seed
        sonic_cfg.num_envs = num_envs
        sonic_cfg.headless = True
        sonic_cfg.multi_gpu = False
        sonic_cfg.output_dir = str(cfg.paths.output_dir)
        sonic_cfg.experiment_dir = str(checkpoint.parent)
        env_cfg = sonic_cfg.manager_env.config
        env_cfg.num_envs = num_envs
        env_cfg.headless = True
        env_cfg.render_results = False
        env_cfg.enable_cameras = False
        # DAgger never consumes rendered observations.  Clear all camera
        # switches because the teacher checkpoint may contain evaluation-only
        # fields that would otherwise create RTX sensors.
        for name in (
            "render_ego",
            "overview_camera",
            "render_ego_opencv_usd",
            "overview_camera_side",
        ):
            env_cfg[name] = False
        cameras_cfg = env_cfg.get("cameras")
        if cameras_cfg is not None:
            cameras_cfg.enable_cameras = False
        env_cfg.gpu_collision_stack_size_exp = cfg.environment.collision_stack_size_exp
        env_cfg.use_motion_hand_actions = False
        env_cfg.use_latent_residual = False
        env_cfg.use_student_direct_latent = True
        # Load the ATM checkpoint on CPU before Kit and PyTorch share the
        # device.  Direct CUDA deserialization here can block the first
        # environment construction on some Isaac/driver combinations.
        env_cfg.action_transform_module_load_on_cpu = True
        env_cfg.object_usd_path = str(dataset_root / "object_usd")

        motion = sonic_cfg.manager_env.commands.motion
        motion.randomize_initial_pose_during_evaluation = cfg.environment.randomize_initial_pose
        motion.start_from_first_frame = True
        motion.init_z_offset = cfg.environment.init_z_offset
        motion.pose_range.x = list(cfg.environment.root_x_range)
        motion.pose_range.y = list(cfg.environment.root_y_range)
        motion.motion_lib_cfg.motion_file = str(robot_dir)
        motion.motion_lib_cfg.object_motion_file = str(dataset_root / "objects")
        motion.motion_lib_cfg.bps_dir = str(dataset_root / "bps")
        motion.motion_lib_cfg.filter_motion_keys = keys
        motion.motion_lib_cfg.target_fps = cfg.rollout.control_fps
        for name in env_cfg.get("train_only_events", []):
            sonic_cfg.manager_env.events.pop(name, None)
        for name in env_cfg.get("train_only_terminations", []):
            sonic_cfg.manager_env.terminations.pop(name, None)
        if "recorders" in sonic_cfg.manager_env:
            sonic_cfg.manager_env.recorders.render_envs = None
            sonic_cfg.manager_env.recorders.trajectory = None
    return sonic_cfg


def prepare_evaluation_env(env, *, stack_timeout: int = 120) -> dict[str, torch.Tensor]:
    """Load the paired motions and reset every environment in evaluation mode."""
    print("[dagger] enabling evaluation mode", flush=True)
    env.is_evaluating = True
    if env.motion_command is not None:
        env.motion_command.set_is_evaluating(True)

    env.start_idx = 0
    print("[dagger] loading evaluation motions", flush=True)
    env._motion_lib.load_motions_for_evaluation(start_idx=env.start_idx)  # noqa: SLF001
    print("[dagger] resetting all environments after motion loading", flush=True)
    faulthandler.dump_traceback_later(stack_timeout, repeat=True)
    try:
        obs = env.reset_all()
    finally:
        faulthandler.cancel_dump_traceback_later()
    print("[dagger] evaluation environments are ready", flush=True)
    return obs


def build_teacher(config, env, device, checkpoint: Path):
    from gear_sonic.trl.utils import common as trl_common
    from gear_sonic.utils import obs_utils

    dims = getattr(config.algo.config, "module_dim", {})
    for key, group in (("actor_obs", "policy"), ("critic_obs", "critic")):
        width = env.env.observation_space[group].shape[-1]
        env.config["obs"]["obs_dims"][key] = width
        env.config["robot"]["algo_obs_dim_dict"][key] = width
    example = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key in {"policy", "critic"}:
            continue
        shapes, names, total = obs_utils.get_group_term_obs_shape(example, key)
        env.config["obs"]["group_obs_dims"][key] = shapes
        env.config["obs"]["group_obs_names"][key] = names
        env.config["obs"]["obs_dims"][key] = total
        env.config["robot"]["algo_obs_dim_dict"][key] = total
    env.config["robot"]["actions_dim"] = int(env.config.get("meta_action_dim", 66))
    policy = trl_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=dims,
        backbone_kwargs={},
        _resolve=False,
    ).to("cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("actor_model_state_dict", payload.get("policy_state_dict"))
    if state is None:
        raise KeyError(f"{checkpoint} has no teacher policy state")
    state = dict(state)
    if "std" in policy.state_dict() and "log_std" in state and "std" not in state:
        state["std"] = torch.exp(state.pop("log_std"))
    elif "std" not in policy.state_dict() and "std" in state and "log_std" not in state:
        state["log_std"] = torch.log(state.pop("std"))
    policy.load_state_dict(state)
    del state, payload
    return policy.to(device).eval()


def generator_observation(env, generator, obs_dict, hand_frame_cfg, hand_transform_fn):
    actor_obs = obs_dict["actor_obs"]
    actor_obs = actor_obs[:, -1] if actor_obs.ndim == 3 else actor_obs
    manager = env.env.observation_manager
    names = manager._group_obs_term_names["policy"]  # noqa: SLF001
    dimensions = manager._group_obs_term_dim["policy"]  # noqa: SLF001
    terms, offset = {}, 0
    for name, shape in zip(names, dimensions, strict=True):
        width = math.prod(shape)
        terms[name] = actor_obs[:, offset : offset + width].reshape(actor_obs.shape[0], *shape)
        offset += width
    forces = terms["finger_tips_force"].reshape(actor_obs.shape[0], -1, 3)
    table_size = env.config["table_size"]
    height = env.env.scene["table"].data.root_pos_w[:, 2] - env.env.scene.env_origins[:, 2]
    table_geometry = torch.stack(
        (
            height,
            torch.full_like(height, float(table_size[2])),
            torch.full_like(height, float(table_size[0])),
            torch.full_like(height, float(table_size[1])),
        ),
        dim=-1,
    )
    robot, obj = env.env.scene["robot"], env.env.scene["object"]
    return generator.observation_from_world(
        object_bps=env.motion_command.object_bps.detach(),
        table_geometry=table_geometry,
        robot_position_w=robot.data.root_pos_w.detach(),
        robot_quaternion_w=robot.data.root_quat_w.detach(),
        object_position_w=obj.data.root_pos_w.detach(),
        object_quaternion_w=obj.data.root_quat_w.detach(),
        hand_object_transform_6d=hand_transform_fn(env.env, hand_frame_cfg).detach(),
        hand_object_contact_force_magnitude=torch.linalg.vector_norm(forces, dim=-1),
        **{
            name: terms[name].detach()
            for name in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")
        },
    )


def binary_hand_to_sonic(hand: torch.Tensor) -> torch.Tensor:
    return hand * 2.0 - 1.0
