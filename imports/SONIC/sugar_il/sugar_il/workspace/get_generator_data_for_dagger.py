#!/usr/bin/env python3
"""Collect schema-v3 DAgger data under a Flow Matching rollout policy."""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path


HORIZON = 40
EXECUTION_HORIZON = 20
CONTROL_FPS = 50
CAMERA_OFFSET = (1.5, -1.5, 1.0)
CAMERA_TARGET = (0.0, 0.0, 0.8)
TRAIN_FIELDS = (
    "hand_primitive_executed_binary",
    "combined_latent_pre_fsq",
    "hand_object_contact_force_magnitude",
    "table_geometry",
    "robot_root_pos_w",
    "robot_root_quat_w",
    "object_root_pos_w",
    "object_root_quat_w",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "hand_object_transform_6d",
    "object_bps",
)
AUDIT_FIELDS = (
    "grail_base_latent_pre_fsq",
    "sonic_residual_latent",
    "flow_latent_executed",
    "flow_hand_primitive_executed_binary",
)
TRAILING_SHAPES = {
    "hand_primitive_executed_binary": (2,),
    "combined_latent_pre_fsq": (64,),
    "hand_object_contact_force_magnitude": (8,),
    "table_geometry": (4,),
    "robot_root_pos_w": (3,),
    "robot_root_quat_w": (4,),
    "object_root_pos_w": (3,),
    "object_root_quat_w": (4,),
    "base_lin_vel": (3,),
    "base_ang_vel": (3,),
    "joint_pos": (43,),
    "joint_vel": (43,),
    "hand_object_transform_6d": (9,),
    "object_bps": (10,),
    "grail_base_latent_pre_fsq": (64,),
    "sonic_residual_latent": (64,),
    "flow_latent_executed": (64,),
    "flow_hand_primitive_executed_binary": (2,),
}


def _parser(app_launcher_cls=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--generator-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--no-video-rendering",
        action="store_true",
        help="disable camera rendering and MP4 encoding to speed up data collection",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-keys", help=argparse.SUPPRESS)
    if app_launcher_cls is not None:
        app_launcher_cls.add_app_launcher_args(parser)
    return parser


def _resolve_inputs(args) -> tuple[Path, Path, list[str]]:
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"input does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix != ".pkl":
            raise ValueError("single-motion input must be a .pkl file")
        robot_dir, motion_keys = input_path.parent, [input_path.stem]
    else:
        robot_dir = input_path
        motion_keys = sorted(path.stem for path in robot_dir.glob("*.pkl"))
    if not motion_keys:
        raise ValueError(f"no .pkl motions found in {robot_dir}")

    dataset_root = (args.dataset_root or robot_dir.parent).expanduser().resolve()
    suffixes = {
        "objects": ".pkl",
        "object_usd": ".usd",
        "bps": ".npy",
        "meta": ".pkl",
    }
    missing = [
        str(dataset_root / name / f"{key}{suffix}")
        for key in motion_keys
        for name, suffix in suffixes.items()
        if not (dataset_root / name / f"{key}{suffix}").is_file()
    ]
    if missing:
        preview = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise ValueError(f"missing paired motion assets: {preview}{more}")
    return robot_dir, dataset_root, motion_keys


def _parent(args, extra: list[str]) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    robot_dir, dataset_root, motion_keys = _resolve_inputs(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    generator_checkpoint = args.generator_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    if not generator_checkpoint.is_file():
        raise ValueError(f"generator checkpoint does not exist: {generator_checkpoint}")

    output_dir = args.output_dir.expanduser().resolve()
    (output_dir / "success").mkdir(parents=True, exist_ok=True)
    (output_dir / "failed").mkdir(exist_ok=True)
    batches = [
        motion_keys[start : start + args.batch_size]
        for start in range(0, len(motion_keys), args.batch_size)
    ]
    base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    print(
        f"Collecting {len(motion_keys)} motion(s) in {len(batches)} batch(es), "
        f"up to {args.batch_size} environments each"
    )
    for batch_index, keys in enumerate(batches):
        seed = (base_seed + batch_index) % (2**32)
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-keys",
            json.dumps(keys),
            "--input",
            str(args.input.expanduser().resolve()),
            "--dataset-root",
            str(dataset_root),
            "--checkpoint",
            str(checkpoint),
            "--generator-checkpoint",
            str(generator_checkpoint),
            "--output-dir",
            str(output_dir),
            "--gpu",
            args.gpu,
            "--seed",
            str(seed),
            "--headless",
            *extra,
        ]
        if args.no_video_rendering:
            cmd.append("--no-video-rendering")
        print(
            f"Batch {batch_index + 1}/{len(batches)}: {', '.join(keys)} "
            f"(seed={seed})"
        )
        subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "SONIC_GPU": str(args.gpu)},
        )


def _config_path(checkpoint: Path) -> Path:
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        path = directory / "config.yaml"
        if path.is_file():
            return path
    raise FileNotFoundError(f"config.yaml not found beside {checkpoint}")


def _load_config(args, robot_dir: Path, dataset_root: Path, keys: list[str]):
    from omegaconf import OmegaConf, open_dict

    raw = _config_path(args.checkpoint).read_text()
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
    config = OmegaConf.load(io.StringIO(raw))
    if config.get("eval_overrides") is not None:
        config = OmegaConf.merge(config, config.eval_overrides)

    with open_dict(config):
        config.checkpoint = str(args.checkpoint)
        config.seed = args.seed
        config.num_envs = len(keys)
        config.headless = True
        config.multi_gpu = False
        config.output_dir = str(args.output_dir)
        config.experiment_dir = str(args.checkpoint.parent)

        env_cfg = config.manager_env.config
        env_cfg.num_envs = len(keys)
        env_cfg.headless = True
        env_cfg.render_results = not args.no_video_rendering
        env_cfg.enable_cameras = False
        env_cfg.gpu_collision_stack_size_exp = 28
        env_cfg.render_camera = "eval_camera"
        env_cfg.eval_camera_use_env_origin = True
        env_cfg.eval_camera_offset = list(CAMERA_OFFSET)
        env_cfg.eval_camera_target_offset = list(CAMERA_TARGET)
        env_cfg.eval_camera_focal_length = 5.0
        env_cfg.eval_camera_focus_distance = 100.0
        env_cfg.eval_camera_horizontal_aperture = 10.0
        env_cfg.eval_camera_clipping_range = [0.1, 500.0]
        env_cfg.render_width = 640
        env_cfg.render_height = 480
        env_cfg.render_frame_skip = 1
        env_cfg.use_motion_hand_actions = False
        env_cfg.use_latent_residual = False
        env_cfg.use_student_direct_latent = True
        env_cfg.object_usd_path = str(dataset_root / "object_usd")

        motion_cfg = config.manager_env.commands.motion
        motion_cfg.randomize_initial_pose_during_evaluation = True
        motion_cfg.start_from_first_frame = True
        motion_cfg.init_z_offset = 0.05
        motion_cfg.pose_range.x = [0.08, 0.08]
        motion_cfg.pose_range.y = [0.08, 0.08]
        motion_cfg.motion_lib_cfg.motion_file = str(robot_dir)
        motion_cfg.motion_lib_cfg.object_motion_file = str(dataset_root / "objects")
        motion_cfg.motion_lib_cfg.bps_dir = str(dataset_root / "bps")
        motion_cfg.motion_lib_cfg.filter_motion_keys = keys
        motion_cfg.motion_lib_cfg.target_fps = CONTROL_FPS

        for event in env_cfg.get("train_only_events", []):
            config.manager_env.events.pop(event, None)
        for term in env_cfg.get("train_only_terminations", []):
            config.manager_env.terminations.pop(term, None)
        if "recorders" in config.manager_env:
            config.manager_env.recorders.render_envs = None
            config.manager_env.recorders.trajectory = None
    return config


def _build_teacher(config, env, device, checkpoint, torch):
    from gear_sonic.trl.utils import common as trl_common
    from gear_sonic.utils import obs_utils

    module_dims = getattr(config.algo.config, "module_dim", {})
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key in {"policy", "critic"}:
            continue
        dims, names, total = obs_utils.get_group_term_obs_shape(example_obs, key)
        env.config["obs"]["group_obs_dims"][key] = dims
        env.config["obs"]["group_obs_names"][key] = names
        env.config["obs"]["obs_dims"][key] = total
        env.config["robot"]["algo_obs_dim_dict"][key] = total
    env.config["robot"]["actions_dim"] = int(env.config.get("meta_action_dim", 66))

    policy = trl_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dims,
        backbone_kwargs={},
        _resolve=False,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("actor_model_state_dict", payload.get("policy_state_dict"))
    if state is None:
        raise KeyError(f"{checkpoint}: no actor_model_state_dict or policy_state_dict")
    state = dict(state)
    model_has_std = "std" in policy.state_dict()
    if model_has_std and "log_std" in state and "std" not in state:
        state["std"] = torch.exp(state.pop("log_std"))
    elif not model_has_std and "std" in state and "log_std" not in state:
        state["log_std"] = torch.log(state.pop("std"))
    policy.load_state_dict(state)
    return policy.eval()


def _load_generator(checkpoint: Path, device, env, torch):
    import dill
    import hydra

    from sugar_il.wrapper.sugar_il_wrapper import (
        GeneratorWrapper,
        load_generator_policy_state,
    )
    from sugar_il.workspace.run_generator_isaaclab import _validate_generator

    payload = torch.load(
        checkpoint,
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
    )
    policy = hydra.utils.instantiate(payload["cfg"].policy)
    load_generator_policy_state(policy, payload)
    generator = GeneratorWrapper(policy, device)
    _validate_generator(generator.policy, payload["cfg"], env)
    return generator


def _last(value):
    return value[:, -1] if value.ndim == 3 else value


def _termination_reasons(base_env, env_id: int) -> list[str]:
    manager = base_env.termination_manager
    term_dones = getattr(manager, "_last_episode_dones", None)
    if term_dones is None:
        term_dones = getattr(manager, "_term_dones", None)
    if term_dones is None:
        return []
    values = term_dones[env_id].detach().cpu().tolist()
    return [
        name
        for name, triggered in zip(manager.active_terms, values, strict=True)
        if bool(triggered)
    ]


def _episode_data() -> dict[str, list]:
    return {
        "frame_idx": [],
        "motion_step": [],
        **{key: [] for key in TRAIN_FIELDS + AUDIT_FIELDS},
    }


def _append_frame(
    data,
    env,
    generator_obs,
    combined,
    base,
    residual,
    reference_hand,
    flow_latent,
    flow_hand,
    env_id,
):
    robot = env.env.scene["robot"]
    obj = env.env.scene["object"]
    command = env.motion_command
    data["frame_idx"].append(len(data["frame_idx"]))
    data["motion_step"].append(
        int(
            (
                command.motion_start_time_steps[env_id]
                + command.time_steps[env_id]
            ).item()
        )
    )
    values = {
        "hand_primitive_executed_binary": reference_hand,
        "combined_latent_pre_fsq": combined,
        "robot_root_pos_w": robot.data.root_pos_w,
        "robot_root_quat_w": robot.data.root_quat_w,
        "object_root_pos_w": obj.data.root_pos_w,
        "object_root_quat_w": obj.data.root_quat_w,
        "grail_base_latent_pre_fsq": base,
        "sonic_residual_latent": residual,
        "flow_latent_executed": flow_latent,
        "flow_hand_primitive_executed_binary": flow_hand,
        **{key: _last(generator_obs[key]) for key in (
            "object_bps",
            "table_geometry",
            "hand_object_transform_6d",
            "hand_object_contact_force_magnitude",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
        )},
    }
    for key, value in values.items():
        data[key].append(value[env_id].detach().cpu().numpy().copy())


def _validate_arrays(arrays: dict) -> None:
    import numpy as np

    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent per-frame lengths: {sorted(lengths)}")
    for key in ("frame_idx", "motion_step"):
        if arrays[key].ndim != 1:
            raise ValueError(f"{key} has shape {arrays[key].shape}, expected [T]")
    for key, trailing_shape in TRAILING_SHAPES.items():
        value = arrays[key]
        if value.ndim != 2 or value.shape[1:] != trailing_shape:
            raise ValueError(
                f"{key} has shape {value.shape}, expected [T,{trailing_shape[0]}]"
            )
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or Inf")
    for key in ("hand_primitive_executed_binary", "flow_hand_primitive_executed_binary"):
        if not np.isin(arrays[key], (0, 1)).all():
            raise ValueError(f"{key} is not binary")
    expected = (
        arrays["grail_base_latent_pre_fsq"]
        + 0.1 * arrays["sonic_residual_latent"]
    )
    if not np.allclose(arrays["combined_latent_pre_fsq"], expected, atol=1e-5):
        raise ValueError("combined latent is not base + 0.1 * residual")
    for key in ("robot_root_quat_w", "object_root_quat_w"):
        norms = np.linalg.norm(arrays[key], axis=-1)
        if len(norms) and np.max(np.abs(norms - 1.0)) > 1e-3:
            raise ValueError(f"{key} is not normalized")
    for key in ("object_bps", "table_geometry"):
        if len(arrays[key]) and np.max(np.abs(arrays[key] - arrays[key][0])) > 1e-5:
            raise ValueError(f"{key} changes within the episode")


def _save_episode(
    data,
    motion_key,
    reasons,
    video_writer,
    temp_video,
    args,
    seed,
):
    import numpy as np

    if video_writer is not None:
        video_writer.close()
    failed = not reasons or any(reason != "time_out" for reason in reasons)
    destination = args.output_dir / ("failed" if failed else "success")
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{motion_key}_x+0.00_y+0.00"

    arrays = {key: np.asarray(value) for key, value in data.items()}
    _validate_arrays(arrays)
    if temp_video is not None:
        os.replace(temp_video, destination / f"{stem}.mp4")
    payload = {
        "schema_version": 3,
        "motion_key": motion_key,
        "pose_timing": "pre_step_aligned_with_policy_input",
        "pose_quaternion_format": "wxyz",
        "sonic_checkpoint": str(args.checkpoint),
        "generator_checkpoint": str(args.generator_checkpoint),
        "seed": seed,
        "rollout_policy": "flow_matching",
        "hand_label_source": "motion_reference",
        "termination_reasons": reasons,
        "episode_complete": not failed,
        **arrays,
    }
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{stem}.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        temp_pickle = Path(file.name)
    os.replace(temp_pickle, destination / f"{stem}.object_aware.pkl")
    print(
        f"Saved {motion_key}: {len(arrays['frame_idx'])} frames, "
        f"termination={reasons or ['app_closed']} -> {destination}"
    )


def _render_frame(base_env, torch):
    camera = base_env.scene["eval_camera"]
    origins = base_env.scene.env_origins
    eye = origins + torch.tensor(CAMERA_OFFSET, device=base_env.device)
    target = origins + torch.tensor(CAMERA_TARGET, device=base_env.device)
    camera._view._sync_usd_on_fabric_write = True  # noqa: SLF001
    camera.set_world_poses_from_view(eye, target)
    base_env.sim.render()
    base_env.sim.render()
    camera._is_outdated[:] = True  # noqa: SLF001
    camera.update(dt=0.0, force_recompute=True)
    return camera.data.output["rgb"].detach().cpu().numpy()


def _worker(args, app_launcher_cls) -> None:
    robot_dir, dataset_root, _ = _resolve_inputs(args)
    keys = json.loads(args.worker_keys)
    config = _load_config(args, robot_dir, dataset_root, keys)

    args.num_envs = len(keys)
    args.enable_cameras = not args.no_video_rendering
    args.headless = True
    args.multi_gpu = False
    args.distributed = False
    output_dir = args.output_dir
    args.output_dir = str(output_dir)
    args.env_spacing = config.manager_env.config.env_spacing
    launcher = app_launcher_cls(args)
    simulation_app = launcher.app
    args.output_dir = output_dir

    # Isaac Lab/SONIC imports must happen after AppLauncher starts.
    import torch
    from isaaclab.managers import SceneEntityCfg

    from gear_sonic import train_agent_trl
    from gear_sonic.envs.manager_env.mdp.observations import hand_object_transform_6d
    from gear_sonic.utils.common import seeding
    from sugar_il.workspace.run_generator_isaaclab import (
        _generator_observation,
        binary_hand_to_sonic,
    )

    temp_dir = args.output_dir / f".dagger_batch_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    writers = []
    temp_videos = []
    try:
        seeding(args.seed)
        env = train_agent_trl.create_manager_env(config, args.device, args)
        teacher = _build_teacher(config, env, args.device, args.checkpoint, torch)
        generator = _load_generator(args.generator_checkpoint, args.device, env, torch)
        env.reinit_dr()
        env.set_is_evaluating(True)
        obs_dict = env.reset_all()
        for key, value in obs_dict.items():
            if hasattr(value, "to"):
                obs_dict[key] = value.to(args.device)

        motion_keys = list(env.motion_command.motion_lib.curr_motion_keys)
        if len(motion_keys) != len(keys):
            raise RuntimeError(
                f"expected {len(keys)} assigned motions, got {len(motion_keys)}"
            )
        data = [_episode_data() for _ in keys]
        active = torch.ones(len(keys), dtype=torch.bool, device=args.device)
        reasons = [[] for _ in keys]
        if args.no_video_rendering:
            writers = [None] * len(motion_keys)
            temp_videos = [None] * len(motion_keys)
        else:
            import imageio.v2 as imageio

            for motion_key in motion_keys:
                path = temp_dir / f"{motion_key}.mp4"
                temp_videos.append(path)
                writers.append(
                    imageio.get_writer(
                        path,
                        fps=CONTROL_FPS,
                        codec="libx264",
                        quality=5,
                        pixelformat="yuv420p",
                    )
                )

        hand_frame_cfg = SceneEntityCfg("object_to_hand_frame_transformer")
        flow_latents = flow_hands = None
        plan_step = EXECUTION_HORIZON
        with torch.inference_mode():
            while simulation_app.is_running() and active.any():
                generator_obs = _generator_observation(
                    env,
                    generator,
                    obs_dict,
                    hand_frame_cfg,
                    hand_object_transform_6d,
                    torch,
                )
                if plan_step >= EXECUTION_HORIZON:
                    prediction = generator.policy.predict_action(generator_obs)
                    flow_latents = prediction["latent"]
                    flow_hands = prediction["hand_primitive"]
                    expected = ((len(keys), HORIZON, 64), (len(keys), HORIZON, 2))
                    actual = (tuple(flow_latents.shape), tuple(flow_hands.shape))
                    if actual != expected:
                        raise ValueError(f"unexpected Flow output shapes: {actual}")
                    plan_step = 0

                teacher.init_rollout()
                teacher.rollout(obs_dict=obs_dict)
                teacher_mean = teacher.action_mean.detach()
                if teacher_mean.shape != (len(keys), 66):
                    raise ValueError(
                        f"unexpected SONIC action_mean shape: {tuple(teacher_mean.shape)}"
                    )
                residual = teacher_mean[:, :64]
                scale = float(env._latent_residual_scale)  # noqa: SLF001
                residual_mode = env._latent_residual_mode  # noqa: SLF001
                if residual_mode != "pre_quantization" or abs(scale - 0.1) > 1e-8:
                    raise ValueError(
                        "teacher requires pre_quantization latent residual with scale=0.1"
                    )
                scaled_residual = residual * scale
                atm_obs = env._prepare_obs_for_action_transform_module(obs_dict)  # noqa: SLF001
                env.action_transform_module(
                    atm_obs,
                    latent_residual=scaled_residual,
                    latent_residual_mode="pre_quantization",
                )
                atm = env.action_transform_module.actor_module
                combined = _last(
                    atm._last_pre_quantization_latent_flat  # noqa: SLF001
                ).detach()
                base = combined - scaled_residual

                left = env.motion_command.get_hand_action("left_hand")
                right = env.motion_command.get_hand_action("right_hand")
                if left is None or right is None:
                    raise ValueError("motion reference is missing left/right hand actions")
                reference_hand = (torch.stack((left, right), dim=-1) >= 0).to(torch.int8)
                current_flow_latent = flow_latents[:, plan_step]
                current_flow_hand = flow_hands[:, plan_step]

                frames = (
                    None
                    if args.no_video_rendering
                    else _render_frame(env.env, torch)
                )
                for env_id in active.nonzero(as_tuple=True)[0].tolist():
                    if frames is not None:
                        writers[env_id].append_data(frames[env_id])
                    _append_frame(
                        data[env_id],
                        env,
                        generator_obs,
                        combined,
                        base,
                        residual,
                        reference_hand,
                        current_flow_latent,
                        current_flow_hand,
                        env_id,
                    )

                meta_action = torch.cat(
                    (
                        current_flow_latent,
                        binary_hand_to_sonic(current_flow_hand),
                    ),
                    dim=-1,
                )
                obs_dict, _, dones, _ = env.step(
                    {
                        "actions": meta_action,
                        "obs_dict": obs_dict,
                        "action_mode": "direct_latent",
                    }
                )
                plan_step += 1
                done_mask = dones.reshape(-1).bool() & active
                for env_id in done_mask.nonzero(as_tuple=True)[0].tolist():
                    reasons[env_id] = _termination_reasons(env.env, env_id)
                    _save_episode(
                        data[env_id],
                        motion_keys[env_id],
                        reasons[env_id],
                        writers[env_id],
                        temp_videos[env_id],
                        args,
                        args.seed,
                    )
                    active[env_id] = False

        for env_id in active.nonzero(as_tuple=True)[0].tolist():
            _save_episode(
                data[env_id],
                motion_keys[env_id],
                ["app_closed"],
                writers[env_id],
                temp_videos[env_id],
                args,
                args.seed,
            )
        env.env.close()
    finally:
        for writer in writers:
            if writer is None:
                continue
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        simulation_app.close()


def main() -> None:
    is_worker = "--worker" in sys.argv
    app_launcher_cls = None
    if is_worker:
        try:
            from isaaclab.app import AppLauncher
        except ImportError:
            from omni.isaac.lab.app import AppLauncher
        app_launcher_cls = AppLauncher

    parser = _parser(app_launcher_cls)
    args, extra = parser.parse_known_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.generator_checkpoint = args.generator_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "little")
    if args.worker:
        if not args.worker_keys:
            parser.error("--worker-keys is required in worker mode")
        sonic_root = Path(__file__).resolve().parents[3]
        os.chdir(sonic_root)
        for path in (sonic_root, sonic_root / "sugar_il"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        _worker(args, app_launcher_cls)
    else:
        _parent(args, extra)


if __name__ == "__main__":
    main()
