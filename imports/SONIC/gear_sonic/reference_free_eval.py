"""Independent play loop for the goal-conditioned joint-micro-step student."""

import json
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf, open_dict

from gear_sonic.utils.reference_free_data import load_scene_inputs


MODULE = "gear_sonic.envs.manager_env.reference_free"


def prepare_config(config, objects, initial_pose):
    """Replace training managers with the minimal student inference environment."""
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    with open_dict(config):
        manager = config.manager_env
        init_z_offset = manager.commands.motion.get("init_z_offset", 0.05)
        manager.commands = {
            "_target_": "gear_sonic.envs.manager_env.mdp.commands.CommandsCfg",
            "motion": {
                "_target_": f"{MODULE}.ReferenceFreeCommandCfg",
                "objects": objects,
                "initial_pose": initial_pose,
                "init_z_offset": init_z_offset,
                "success_threshold": config.reference_free.success_threshold,
                "debug_vis": config.reference_free.debug_vis,
            },
        }
        for name in list(manager.observations):
            if name not in ("_target_", "proprio_obs", "privileged_obs", "policy_atm"):
                del manager.observations[name]
        for name in ("proprio_obs", "privileged_obs", "policy_atm"):
            manager.observations[name].enable_corruption = False
        manager.observations.privileged_obs.target_object_poses_6d.func = f"{MODULE}:target_object_poses_6d"
        manager.observations.privileged_obs.table_corners_b.params.table_size = objects[0]["table_size"]
        manager.rewards = {"_target_": "builtins.dict"}
        manager.events = {"_target_": "builtins.dict"}
        manager.curriculum = {"_target_": "builtins.dict"}
        manager.terminations = {
            "_target_": "builtins.dict",
            "time_out": {
                "_target_": "isaaclab.managers.TerminationTermCfg",
                "func": "isaaclab.envs.mdp:time_out",
                "time_out": True,
            },
        }
        if config.reference_free.get("terminate_on_success", True):
            manager.terminations.goal_reached = {
                "_target_": "isaaclab.managers.TerminationTermCfg",
                "func": f"{MODULE}:goal_reached",
            }
        # The inherited success recorder queries final reference frames. Replace
        # it while retaining only the explicitly requested video recorder.
        render = manager.recorders.get("render_envs")
        manager.recorders = {
            "_target_": "gear_sonic.envs.manager_env.mdp.recorders.RecordersCfg",
            "dataset_export_mode": 0,
            "export_in_record_pre_reset": False,
            "object_position_success": {"_target_": f"{MODULE}.ReferenceFreeResultRecorderCfg"},
        }
        if render is not None:
            manager.recorders.render_envs = render
        scene = manager.config
        scene.num_envs = int(config.num_envs)
        scene.headless = bool(config.headless)
        scene.object_usd_path = [item["usd"] for item in objects]
        scene.motion_meta_info_path = None
        scene.table_size = objects[0]["table_size"]
        scene.table_position = objects[0]["table_position"]
        scene.table_quat = [1.0, 0.0, 0.0, 0.0]
        scene.table_offset = None
        scene.replicate_physics = False
        scene.multi_object_per_env = False
        scene.use_student_direct_latent = True
        scene.use_motion_hand_actions = False
        scene.student_use_motion_hand_actions = False
    return config


def run(config):
    """Launch Isaac Lab only after routing away from the reference play path."""
    import sys

    from isaaclab.app import AppLauncher

    objects, pose = load_scene_inputs(
        config.reference_free.dataset_dir, config.reference_free.initial_pose_file, int(config.num_envs)
    )
    config = prepare_config(config, objects, pose)
    # Kit must not parse Hydra overrides as its own command-line options.
    hydra_argv = sys.argv
    try:
        sys.argv = sys.argv[:1]
        launcher = AppLauncher(
            headless=bool(config.headless),
            enable_cameras=bool(config.manager_env.config.get("render_results", False)),
            device=str(config.get("device", "cuda:0")),
        )
    finally:
        sys.argv = hydra_argv
    try:
        _play(config, launcher.app)
    except Exception:
        logger.exception("Reference-free evaluation failed")
        raise
    finally:
        launcher.app.close()


def _play(config, simulation_app):
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from gear_sonic.envs.manager_env.reference_free import ReferenceFreeEnvWrapper
    from gear_sonic.trl.utils.common import custom_instantiate
    from gear_sonic.utils import obs_utils
    from gear_sonic.utils.common import seeding

    seeding(config.seed)
    env_cfg = custom_instantiate(config.manager_env)
    env_cfg.seed = config.seed
    env_cfg.sim.device = str(config.get("device", "cuda:0"))
    raw_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = None
    results = []
    completed = set()
    output_dir = Path(config.manager_env.config.save_rendering_dir)
    report_path = Path(config.get("run_once_report_path") or output_dir / "reference_free_results.json")
    try:
        env = ReferenceFreeEnvWrapper(raw_env, env_cfg.config)
        env.set_is_evaluating(True)
        obs = env.reset_all()
        for key, value in obs.items():
            dims, names, total = obs_utils.get_group_term_obs_shape_from_manager(
                raw_env.observation_manager, obs, key
            )
            env.config.obs.group_obs_dims[key] = dims
            env.config.obs.group_obs_names[key] = names
            env.config.obs.obs_dims[key] = total
            env.config.robot.algo_obs_dim_dict[key] = int(np.prod(value.shape[1:]))
        env.config.robot.actions_dim = 66
        policy = custom_instantiate(
            config.algo.config.actor, env_config=env.config, algo_config=config.algo.config,
            module_dim_dict=config.algo.config.get("module_dim", {}), _resolve=False,
        ).to(env.device)
        checkpoint = torch.load(config.checkpoint, map_location=env.device, weights_only=False)
        state_dict = checkpoint.get("actor_model_state_dict", checkpoint.get("policy_state_dict"))
        policy.load_state_dict(state_dict)
        del checkpoint, state_dict
        policy.eval()
        command = raw_env.command_manager.get_term("motion")
        steps = 0
        with torch.inference_mode():
            while simulation_app.is_running():
                policy.init_rollout()
                actions = policy.rollout(obs_dict=obs)
                obs, _, dones, _ = env.step({
                    "actions": policy.action_mean.detach(),
                    "obs_dict": actions["obs_dict"],
                    "action_mode": "direct_latent",
                })
                for result in command.results:
                    env_id = result["env_id"]
                    if config.get("run_once", False) and env_id in completed:
                        continue
                    results.append(result)
                    completed.add(env_id)
                    logger.info(
                        "env={} object={} result={} distance={:.3f}m time={:.2f}s",
                        env_id, result["object_key"], result["termination_reason"],
                        result["final_distance"], result["elapsed_seconds"],
                    )
                command.results.clear()
                steps += 1
                if config.get("run_once", False) and len(completed) == env.num_envs:
                    break
                if config.get("max_render_steps", 0) and steps >= config.max_render_steps:
                    break
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        successes = sum(result["success"] for result in results)
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump({
                "checkpoint": str(config.checkpoint),
                "success_threshold": float(config.reference_free.success_threshold),
                "num_envs": int(config.num_envs),
                "complete": len(completed) == int(config.num_envs),
                "successes": successes,
                "episodes": results,
            }, stream, indent=2, ensure_ascii=False)
        logger.info("Reference-free success: {}/{}; report: {}", successes, len(results), report_path)
        if env is not None:
            raw_env.command_manager.get_term("motion").set_debug_vis(False)
            env.end_render_results()
        raw_env.close()
