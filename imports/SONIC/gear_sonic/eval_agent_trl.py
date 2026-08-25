#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for evaluation but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    import sys
    sys.exit(1)

import filelock  # noqa: I001
import json
import os
import shutil
import subprocess
import sys

sys.path.append(os.getcwd())
import logging
from pathlib import Path

import easydict
import hydra
from hydra import utils
from hydra.core import hydra_config
from loguru import logger
import omegaconf
import yaml

from gear_sonic import train_agent_trl
from gear_sonic.trl.modules.action_chunk import (
    ActionChunkExecutor,
    action_chunk_enabled,
    get_action_chunk_config,
    get_motion_metadata,
    motion_discontinuity,
)
from gear_sonic.trl.utils import common as trl_utils_common
from gear_sonic.trl.utils import scheduler
from gear_sonic.utils import common as rl_utils_common
from gear_sonic.utils import config_utils, obs_utils

config_utils.register_rl_resolvers()


VISUAL_DR_PROFILES = {
    "warm_wood": {
        "floor_color": [0.46, 0.30, 0.18],
        "table_color": [0.38, 0.23, 0.12],
        "object_color": [0.90, 0.08, 0.04],
        "light_color": [1.00, 0.88, 0.72],
        "light_intensity": 2600.0,
        "sky_intensity": 650.0,
    },
    "cool_tile": {
        "floor_color": [0.62, 0.66, 0.70],
        "table_color": [0.12, 0.13, 0.15],
        "object_color": [0.06, 0.72, 0.28],
        "light_color": [0.76, 0.86, 1.00],
        "light_intensity": 3300.0,
        "sky_intensity": 900.0,
    },
    "marble_bright": {
        "floor_color": [0.84, 0.80, 0.70],
        "table_color": [0.88, 0.86, 0.80],
        "object_color": [0.96, 0.66, 0.05],
        "light_color": [1.00, 0.96, 0.86],
        "light_intensity": 3800.0,
        "sky_intensity": 1100.0,
    },
    "dark_lab": {
        "floor_color": [0.16, 0.17, 0.18],
        "table_color": [0.22, 0.24, 0.26],
        "object_color": [0.18, 0.34, 0.95],
        "light_color": [0.78, 0.88, 1.00],
        "light_intensity": 2200.0,
        "sky_intensity": 420.0,
    },
    "green_studio": {
        "floor_color": [0.26, 0.43, 0.33],
        "table_color": [0.62, 0.48, 0.32],
        "object_color": [0.68, 0.05, 0.72],
        "light_color": [0.84, 1.00, 0.90],
        "light_intensity": 2900.0,
        "sky_intensity": 760.0,
    },
}


def _manager_env(env):
    """Return the underlying IsaacLab manager environment."""
    return getattr(env, "env", env)


def _evaluation_motion_keys(env, num_envs):
    """Snapshot the motion key assigned to every evaluation environment."""
    command = getattr(env, "motion_command", None)
    if command is None:
        manager_env = _manager_env(env)
        command_manager = getattr(manager_env, "command_manager", None)
        if command_manager is not None:
            try:
                command = command_manager.get_term("motion")
            except Exception:  # noqa: BLE001
                command = None
    if command is None or not hasattr(command, "motion_ids"):
        return [f"env_{env_id}" for env_id in range(num_envs)]

    motion_ids = command.motion_ids.detach().cpu().tolist()
    motion_lib = getattr(command, "motion_lib", getattr(env, "_motion_lib", None))
    current_keys = getattr(motion_lib, "curr_motion_keys", None)
    all_keys = getattr(motion_lib, "motion_keys", None)
    keys = []
    for env_id in range(num_envs):
        motion_id = int(motion_ids[env_id])
        if current_keys is not None and motion_id < len(current_keys):
            keys.append(str(current_keys[motion_id]))
        elif all_keys is not None and motion_id < len(all_keys):
            keys.append(str(all_keys[motion_id]))
        else:
            keys.append(f"motion_{motion_id}")
    return keys


def _evaluation_reset_reasons(env, env_id):
    """Return all termination terms that fired for one environment."""
    termination_manager = getattr(_manager_env(env), "termination_manager", None)
    if termination_manager is None:
        return ["unknown"]
    reasons = []
    for term_name in termination_manager.active_terms:
        term_value = termination_manager.get_term(term_name)
        if bool(term_value[env_id].item()):
            reasons.append(str(term_name))
    return reasons or ["unknown"]


def _evaluation_step_dt(env):
    """Return one policy/environment step duration in seconds."""
    step_dt = getattr(_manager_env(env), "step_dt", None)
    return float(step_dt) if step_dt is not None else 1.0 / 50.0


def _apply_run_once_motion_cap(config):
    """Cap paired data before scene construction for one-episode play."""
    if not config.get("run_once", False) or not config.get(
        "load_only_num_envs_motions", True
    ):
        return None
    motion_lib_cfg = config.manager_env.commands.motion.motion_lib_cfg
    requested_cap = int(config.num_envs)
    existing_cap = motion_lib_cfg.get("max_unique_motions", None)
    effective_cap = (
        requested_cap
        if existing_cap is None
        else min(int(existing_cap), requested_cap)
    )
    with omegaconf.open_dict(motion_lib_cfg):
        motion_lib_cfg.max_unique_motions = effective_cap
    return effective_cap


def _visual_dr_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _apply_visual_domain_randomization(visual_dr_config) -> None:
    """Apply lightweight USD material and lighting overrides for eval videos."""
    enabled = bool(_visual_dr_get(visual_dr_config, "enabled", True))
    if not enabled:
        return

    profile_name = str(_visual_dr_get(visual_dr_config, "profile", "warm_wood"))
    profile = dict(VISUAL_DR_PROFILES.get(profile_name, VISUAL_DR_PROFILES["warm_wood"]))
    custom = _visual_dr_get(visual_dr_config, "custom", None)
    if custom is not None:
        profile.update(omegaconf.OmegaConf.to_container(custom, resolve=True))

    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdLux, UsdShade
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Visual DR skipped because USD modules are unavailable: {exc}")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        logger.warning("Visual DR skipped because no USD stage is active.")
        return

    def make_material(name, color, roughness=0.72, metallic=0.0):
        mat_path = f"/World/Looks/visual_dr_{name}_{profile_name}"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    materials = {
        "floor": make_material("floor", profile["floor_color"], roughness=0.88),
        "table": make_material("table", profile["table_color"], roughness=0.62),
        "object": make_material("object", profile["object_color"], roughness=0.55),
    }
    counts = {"floor": 0, "table": 0, "object": 0, "light": 0}

    def category_for_path(path):
        path_lower = path.lower()
        if "/world/ground" in path_lower or "/terrain/" in path_lower:
            return "floor"
        if "/table" in path_lower:
            return "table"
        if "/object" in path_lower or "/object_" in path_lower:
            return "object"
        return None

    for prim in stage.Traverse():
        if not prim.IsValid() or prim.GetTypeName() not in {"Mesh", "Cube", "Xform"}:
            continue
        category = category_for_path(str(prim.GetPath()))
        if category is None:
            continue
        try:
            UsdShade.MaterialBindingAPI.Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(materials[category])
            counts[category] += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Failed to bind visual DR material to {prim.GetPath()}: {exc}")

    light_color = profile["light_color"]
    light_vec = Gf.Vec3f(float(light_color[0]), float(light_color[1]), float(light_color[2]))
    light_intensity = float(profile["light_intensity"])
    sky_intensity = float(profile["sky_intensity"])
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name not in {"DistantLight", "DomeLight", "SphereLight", "RectLight"}:
            continue
        try:
            prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f, custom=False).Set(light_vec)
            intensity = sky_intensity if type_name == "DomeLight" else light_intensity
            prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float, custom=False).Set(intensity)
            counts["light"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Failed to update visual DR light {prim.GetPath()}: {exc}")

    if counts["light"] == 0:
        dome = UsdLux.DomeLight.Define(stage, "/World/visual_dr_skyLight")
        dome.CreateColorAttr(light_vec)
        dome.CreateIntensityAttr(sky_intensity)
        counts["light"] = 1

    logger.info(
        "Applied visual DR profile={} floor_prims={} table_prims={} object_prims={} lights={}".format(
            profile_name,
            counts["floor"],
            counts["table"],
            counts["object"],
            counts["light"],
        )
    )


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: omegaconf.OmegaConf):

    hydra_log_path = os.path.join(hydra_config.HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    from gear_sonic.utils import logging as utils_logging

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(utils_logging.HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                raw = file.read()
            # Backward compatibility: rewrite internal repo module paths to release repo paths
            raw = raw.replace("groot.rl.trl.", "gear_sonic.trl.")
            raw = raw.replace("groot.rl.envs.", "gear_sonic.envs.")
            raw = raw.replace("groot.rl.utils.", "gear_sonic.utils.")
            raw = raw.replace("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module.")
            raw = raw.replace("groot.rl.agents.", "gear_sonic.trl.")
            raw = raw.replace("groot/rl/data/", "gear_sonic/data/")
            raw = raw.replace("assets/bm/unitree_description/", "assets/robot_description/")
            raw = raw.replace("1215_bones_seed_filtered", "bones_seed_smpl")
            import io
            train_config = omegaconf.OmegaConf.load(io.StringIO(raw))

            if train_config.eval_overrides is not None:
                train_config = omegaconf.OmegaConf.merge(train_config, train_config.eval_overrides)

            config = omegaconf.OmegaConf.merge(train_config, override_config)
        else:
            config = override_config

        config.experiment_dir = checkpoint.parent
    elif override_config.eval_overrides is not None:
        config = override_config.copy()
        eval_overrides = omegaconf.OmegaConf.to_container(config.eval_overrides, resolve=True)
        for arg in sys.argv[1:]:
            if not arg.startswith("+"):
                key = arg.split("=")[0]
                if key in eval_overrides:
                    del eval_overrides[key]
        config.eval_overrides = omegaconf.OmegaConf.create(eval_overrides)
        config = omegaconf.OmegaConf.merge(config, eval_overrides)
    else:
        config = override_config

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))  # noqa: SIM115
        if config.get("wandb", None) is not None and meta.get("wandb_run"):
            config.wandb.wandb_id = meta["wandb_run"]
            print(f"resume wandb from run: {config.wandb.wandb_id}")  # noqa: T201

    with omegaconf.open_dict(config):
        for event in config.manager_env.config.get("train_only_events", []):
            if event in config.manager_env.events:
                config.manager_env.events.pop(event)
            remove_schedule_keys = []
            for key in config.trainer.get("schedule_dict", {}):
                if event in key:
                    remove_schedule_keys.append(key)
            for key in remove_schedule_keys:
                config.trainer.schedule_dict.pop(key)

        for termination in config.manager_env.config.get("train_only_terminations", []):
            if termination in config.manager_env.terminations:
                config.manager_env.terminations.pop(termination)

        # Checkpoints embed their original environment config, so changing the
        # source YAML alone would leave old checkpoints at 0.10 m during play.
        # Apply the new evaluation default after merging the checkpoint config;
        # users can still override it with
        # ++object_pos_deviation_threshold=<meters>.
        object_pos_term = config.manager_env.terminations.get(
            "object_pos_deviation", None
        )
        if object_pos_term is not None:
            object_pos_threshold = float(
                config.get("object_pos_deviation_threshold", 25) # set to 25 彻底关闭此终止条件
            )
            object_pos_term.params.threshold = object_pos_threshold
            logger.info(
                "Evaluation object_pos_deviation threshold: "
                f"{object_pos_threshold:.3f} m"
            )

        # Paired-motion scene construction consumes the motion library config
        # before the command starts loading clips.  Cap it here so run-once
        # play creates only num_envs paired USDs and deserializes only those
        # robot/object motions.  Applying this after env creation is too late:
        # MultiUsdFileCfg would already contain the entire dataset.
        effective_cap = _apply_run_once_motion_cap(config)
        if effective_cap is not None:
            logger.info(
                "Run-once fast loading enabled: limiting paired motion/object/USD "
                f"data to {effective_cap} entries before environment creation"
            )

    use_encoder = config.get("use_encoder", None)
    if use_encoder is not None:
        encoder_sample_probs = config.manager_env.commands.motion.encoder_sample_probs
        if encoder_sample_probs is not None:
            for encoder in encoder_sample_probs:
                if encoder != use_encoder:
                    encoder_sample_probs[encoder] = 0.0
            print(f"Using encoder: {use_encoder}")  # noqa: T201
            print(f"Encoder sample probs: {encoder_sample_probs}")  # noqa: T201

    simulator_type = "IsaacSim"
    env_config = config.manager_env

    import datetime as dt

    import accelerate
    import torch  # noqa: E402, RUF100

    kwargs = accelerate.InitProcessGroupKwargs(timeout=dt.timedelta(seconds=6000))
    accelerator = accelerate.Accelerator(kwargs_handlers=[kwargs])

    device = str(accelerator.device)
    if accelerator.device.type == "cuda":
        try:
            torch.cuda.set_device(accelerator.local_process_index)
        except Exception:  # noqa: S110, BLE001
            pass

    device = str(accelerator.device)
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    rl_utils_common.seeding(config.seed)

    def _pick_display_gpu_index(default_idx: int = 0) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,display_active,name", "--format=csv,noheader"],
                text=True,
            )
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    idx, active = int(parts[0]), parts[1].lower()
                    if active.startswith("enabled") or active.startswith("on"):
                        return idx
        except Exception:  # noqa: S110, BLE001
            pass
        return default_idx

    render_gpu_idx = _pick_display_gpu_index(default_idx=0)

    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher  # 4.2
        import argparse

        parser = argparse.ArgumentParser(description="Evaluate an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args  # noqa: RUF005
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.enable_cameras = env_config.config.get(
            "render_results", False
        ) or env_config.config.get("enable_cameras", False)

        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        base_kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )
        if args_cli.headless:
            args_cli.kit_args = base_kit_args + " --no-window"
        else:
            args_cli.kit_args = base_kit_args + f" --/renderer/activeGpu={render_gpu_idx}"

        _lock_path = "/tmp/isaaclab_app_launcher.lock"  # noqa: S108
        with filelock.FileLock(_lock_path):
            app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  # noqa: F841

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    unresolved_conf = omegaconf.OmegaConf.to_container(config, resolve=False)  # noqa: F841
    os.chdir(hydra.utils.get_original_cwd())

    ckpt_num = config.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]

    if env_config.config.get("save_rendering_dir", None) is None:
        env_config.config.save_rendering_dir = str(
            checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
        )

    metrics_file = config.get("metrics_file", None)
    if metrics_file is not None:
        metrics_file = Path(metrics_file)
        assert metrics_file.exists(), f"Metrics file {metrics_file} does not exist"
        if metrics_file.exists():
            metrics = json.load(open(metrics_file))  # noqa: SIM115
            all_dict = metrics["eval/all_metrics_dict"]

            # Check if this is grab evaluation (has success_lift)
            has_obj_metrics = "obj_pos_error" in all_dict
            if "success_lift" in all_dict:
                # Grab evaluation: prioritize failed grasps (not lifted) and terminated trajectories
                motion_keys = all_dict["motion_keys"]
                terminated = all_dict["terminated"]
                success_lift = all_dict["success_lift"]
                progress = all_dict.get("progress", [1.0] * len(motion_keys))
                obj_pos_errors = all_dict.get("obj_pos_error", [0.0] * len(motion_keys))

                pairs = []
                for i in range(len(motion_keys)):
                    term = bool(terminated[i]) if i < len(terminated) else False
                    lifted = bool(success_lift[i]) if i < len(success_lift) else False
                    prog = progress[i] if i < len(progress) else 1.0
                    obj_err = obj_pos_errors[i] if i < len(obj_pos_errors) else 0.0
                    priority = 0 if not lifted else (1 if term else 2)
                    pairs.append((motion_keys[i], term, lifted, prog, obj_err, priority))

                pairs_sorted = sorted(pairs, key=lambda x: (x[5], x[3]))
                if len(pairs_sorted) > config.num_envs:
                    pairs_sorted = pairs_sorted[: config.num_envs]

                render_info = []
                for pair in pairs_sorted:
                    motion_key, term, lifted, prog, obj_err, _ = pair
                    status = "FAILED" if not lifted else ("TERMINATED" if term else "SUCCESS")
                    info = [
                        f"{motion_key}",
                        f"lifted: {lifted}",
                        f"progress: {prog:.3f}",
                        f"status: {status}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {obj_err:.4f}m")
                    render_info.append(tuple(info))

                filter_keys = [pair[0] for pair in pairs_sorted]

                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(render_info)
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys
            else:
                # Imitation evaluation: use MPJPE-based sorting
                obj_pos_errors = all_dict.get("obj_pos_error", None)
                success_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        True,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if not all_dict["terminated"][i]
                ]
                render_sort_by = config.get("render_sort_by", "mpjpe_l")
                sort_idx = 4 if render_sort_by == "obj_pos_error" else 1
                success_pair_sorted = sorted(success_pair, key=lambda x: x[sort_idx], reverse=True)
                failed_pair = [
                    (
                        all_dict["motion_keys"][i],
                        all_dict["mpjpe_l"][i],
                        all_dict["mpjpe_g"][i],
                        False,
                        obj_pos_errors[i] if obj_pos_errors is not None else 0.0,
                    )
                    for i in range(len(all_dict["motion_keys"]))
                    if all_dict["terminated"][i]
                ]
                failed_pair_sorted = sorted(failed_pair, key=lambda x: x[sort_idx], reverse=True)
                all_pair = failed_pair_sorted + success_pair_sorted
                if len(all_pair) > config.num_envs:
                    all_pair = all_pair[: config.num_envs]
                render_info = []
                for pair in all_pair:
                    info = [
                        f"{pair[0]}",
                        f"mpjpe_l: {pair[1]:.2f}",
                        f"mpjpe_g: {pair[2]:.2f}",
                        f"success: {pair[3]}",
                    ]
                    if has_obj_metrics:
                        info.append(f"obj_pos_err: {pair[4]:.4f}m")
                    render_info.append(tuple(info))
                with omegaconf.open_dict(env_config.config):
                    env_config.config.render_info = render_info
                    env_config.config.max_render_envs = len(all_pair)
                filter_keys = [pair[0] for pair in all_pair]
                with omegaconf.open_dict(env_config.commands.motion):
                    env_config.commands.motion.filter_motion_keys = filter_keys
                    if "motion_lib_cfg" in env_config.commands.motion:
                        env_config.commands.motion.motion_lib_cfg.filter_motion_keys = filter_keys

    env = train_agent_trl.create_manager_env(config, device, args_cli)

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    policy_backbone_kwargs = {}
    critic_backbone_kwargs = {}
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
        group_obs_dims, group_obs_names, group_obs_total_dim = (
            obs_utils.get_group_term_obs_shape_from_manager(
                env.env.observation_manager, example_obs, key
            )
        )
        env.config["obs"]["group_obs_dims"][key] = group_obs_dims
        env.config["obs"]["group_obs_names"][key] = group_obs_names
        env.config["obs"]["obs_dims"][key] = group_obs_total_dim
        if key not in ["policy", "critic"]:
            env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    meta_action_dim = env.config.get("meta_action_dim", None)
    if meta_action_dim is not None and meta_action_dim > 0:
        env.config["robot"]["actions_dim"] = meta_action_dim
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

    policy = trl_utils_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs=policy_backbone_kwargs,
        _resolve=False,
    ).to(device)

    # Distillation-only policies (for example action-chunk CFM models) do not
    # have or need a critic during evaluation.  Keep the wrapper interface
    # consistent while avoiding an unbound local when critic construction is
    # intentionally skipped.
    value_model = None
    if not getattr(config.algo.config, "distill_only", False):
        value_model = trl_utils_common.custom_instantiate(
            config.algo.config.critic,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=critic_backbone_kwargs,
            _resolve=False,
        ).to(device)

    accelerator.wait_for_everyone()

    args = easydict.EasyDict()
    args.is_main_process = accelerator.is_main_process
    args.global_rank = accelerator.process_index
    args.world_size = accelerator.num_processes
    state = easydict.EasyDict()

    from gear_sonic.trl.trainer import ppo_trainer

    model = ppo_trainer.PolicyAndValueWrapper(policy, value_model)

    checkpoint_path = str(config.checkpoint)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)

    # Load policy state dict with backward compatibility for std/log_std
    if "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = None

    if state_dict is not None:
        model_uses_std = "std" in model.policy.state_dict()
        checkpoint_has_std = "std" in state_dict
        checkpoint_has_log_std = "log_std" in state_dict

        logger.info(f"Model parameterization: {'std' if model_uses_std else 'log_std'}")
        logger.info(
            f"Checkpoint parameterization: {'std' if checkpoint_has_std else 'log_std' if checkpoint_has_log_std else 'unknown'}"  # noqa: E501
        )

        if model_uses_std and checkpoint_has_log_std and not checkpoint_has_std:
            logger.info("Transforming 'log_std' -> 'std' (applying exp) for backward compatibility")
            state_dict["std"] = torch.exp(state_dict.pop("log_std"))
        elif not model_uses_std and checkpoint_has_std and not checkpoint_has_log_std:
            logger.info("Transforming 'std' -> 'log_std' (applying log) for backward compatibility")
            state_dict["log_std"] = torch.log(state_dict.pop("std"))

        model.policy.load_state_dict(state_dict)
        logger.info("Successfully loaded policy state dict")

    state.global_step = checkpoint["state"].global_step

    schedule_wrapper = easydict.EasyDict(env=env, model=model)
    if "schedule_dict" in config.trainer:
        scheduled_params_dict = scheduler.update_scheduled_params(  # noqa: F841
            schedule_wrapper, config.trainer.schedule_dict, state.global_step
        )
    env.reinit_dr()

    global_step = checkpoint["state"].global_step
    exported_policy_path = os.path.join(config.experiment_dir, "exported")
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_onnx_name = f"model_step_{global_step:06d}.onnx"
    new_cp_path = f"{os.path.dirname(config.checkpoint)}/model_step_{global_step:06d}.pt"
    if not os.path.exists(new_cp_path):
        shutil.copy(checkpoint_path, new_cp_path)

    if config.get("export_onnx_only", False):

        def get_example_obs():
            obs_dict = env.reset_all()
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].cpu()
            return obs_dict

        assert config.num_envs == 1, "num_envs must be 1 for exporting onnx"
        from gear_sonic.utils import inference_helpers

        example_obs_dict = get_example_obs()

        # Check if actor has universal-token encoder structure
        has_actor_module = hasattr(model.policy, "actor_module")
        has_encoders = has_actor_module and hasattr(
            model.policy.actor_module, "encoders_to_iterate"
        )

        if "tokenizer" in example_obs_dict and has_encoders:

            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="smpl",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_smpl.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="g1",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_g1.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_module_as_onnx(
                model.policy.actor_module,
                encoder_name="teleop",
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_teleop.onnx"),
                batch_size=1,
            )

            inference_helpers.export_universal_token_encoders_as_onnx(
                model.policy.actor_module,
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_encoder.onnx"),
                batch_size=1,
            )
            inference_helpers.export_universal_token_decoder_as_onnx(
                model.policy.actor_module,
                decoder_name="g1_dyn",
                path=exported_policy_path,
                exported_model_name=exported_onnx_name.replace(".onnx", "_decoder.onnx"),
                batch_size=1,
            )
            print(  # noqa: T201
                f'Exported encoders ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_encoder.onnx"))}'  # noqa: E501
            )
            print(  # noqa: T201
                f'Exported decoder ONNX to {os.path.join(exported_policy_path, exported_onnx_name.replace(".onnx", "_decoder.onnx"))}'  # noqa: E501
            )

        else:
            inference_helpers.export_policy_as_onnx(
                {"actor": model.policy}, exported_policy_path, exported_onnx_name, example_obs_dict
            )

        logger.info(f"Exported policy as onnx to: {os.path.join(exported_policy_path)}")

        # Export configs to YAML
        export_config = {
            "env_config": omegaconf.OmegaConf.to_container(env.config, resolve=True),
            "algo_config": omegaconf.OmegaConf.to_container(config.algo.config, resolve=True),
        }
        config_yaml_path = os.path.join(os.path.dirname(config.checkpoint), "model_config.yaml")
        with open(config_yaml_path, "w") as f:
            yaml.dump(export_config, f, default_flow_style=False)
        logger.info(f"Exported config to: {config_yaml_path}")
        exit()  # noqa: PLR1722

    eval_callbacks = config.get("eval_callbacks", [])
    if isinstance(eval_callbacks, str):
        eval_callbacks = [eval_callbacks]

    callbacks = {}
    for callback_name in eval_callbacks:
        if callback_name == "im_eval":
            with omegaconf.open_dict(config.callbacks.im_eval):
                config.callbacks.im_eval.eval_only = True
                config.callbacks.im_eval.eval_frequency = 1
                config.callbacks.im_eval.output_dir = config.get("eval_output_dir", None)
                config.callbacks.im_eval.log_keys = config.get("log_keys", None)
        if callback_name not in config.callbacks:
            raise ValueError(f"Callback {callback_name} not found")
        callbacks[callback_name] = utils.instantiate(config.callbacks[callback_name])

    for callback_name, callback in callbacks.items():  # noqa: B007
        if hasattr(callback, "model") and callback.model is None:
            callback.model = model

    for callback_name, callback in callbacks.items():  # noqa: B007
        callback.on_step_end(args, state, None, env=env, model=model, accelerator=accelerator)

    camera_rgb_writer = None
    camera_output_writers = {}
    if config.get("run_eval_loop", True):
        env.set_is_evaluating(True)
        obs_dict = env.reset_all()
        visual_dr_config = config.get("visual_domain_randomization", None)
        if visual_dr_config is not None and accelerator.is_main_process:
            _apply_visual_domain_randomization(visual_dr_config)
        model.eval()
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(device)

        eval_step_callbacks = {
            name: cb
            for name, cb in callbacks.items()
            if hasattr(cb, "eval_step") and callable(getattr(cb, "eval_step"))  # noqa: B009
        }
        if eval_step_callbacks:
            logger.info(f"Eval step callbacks enabled: {list(eval_step_callbacks.keys())}")

        step_count = 0
        max_render_steps = config.get("max_render_steps", 0)
        capture_camera_rgb_path = config.get("capture_camera_rgb_path", None)
        capture_camera_rgb_source = config.get("capture_camera_rgb_source", "obs")
        capture_camera_rgb_max_frames = int(config.get("capture_camera_rgb_max_frames", 0) or 0)
        capture_camera_rgb_unnormalize = bool(config.get("capture_camera_rgb_unnormalize", True))
        capture_camera_output_dir = config.get("capture_camera_output_dir", None)
        capture_camera_output_types = list(config.get("capture_camera_output_types", []))
        capture_camera_output_max_frames = int(
            config.get("capture_camera_output_max_frames", capture_camera_rgb_max_frames) or 0
        )
        capture_camera_output_raw_dir = config.get("capture_camera_output_raw_dir", None)
        capture_camera_output_raw_max_frames = int(
            config.get("capture_camera_output_raw_max_frames", capture_camera_output_max_frames) or 0
        )
        camera_rgb_frames = 0
        camera_output_frames = 0
        camera_output_logged_keys = False
        if (
            capture_camera_rgb_path is not None
            or capture_camera_output_dir is not None
            or capture_camera_output_raw_dir is not None
        ) and accelerator.is_main_process:
            import imageio.v2 as imageio
            import numpy as np

            step_dt = getattr(env, "step_dt", None)
            if step_dt is None and hasattr(env, "env"):
                step_dt = getattr(env.env, "step_dt", None)
            fps = max(1, int(round(1.0 / step_dt))) if step_dt else 30
        if capture_camera_rgb_path is not None and accelerator.is_main_process:
            Path(capture_camera_rgb_path).parent.mkdir(parents=True, exist_ok=True)
            camera_rgb_writer = imageio.get_writer(
                capture_camera_rgb_path,
                fps=fps,
                codec="libx264",
                quality=6,
                pixelformat="yuv420p",
            )
        if capture_camera_output_dir is not None and accelerator.is_main_process:
            capture_camera_output_dir = Path(capture_camera_output_dir)
            capture_camera_output_dir.mkdir(parents=True, exist_ok=True)
            for data_type in capture_camera_output_types:
                camera_output_writers[data_type] = imageio.get_writer(
                    str(capture_camera_output_dir / f"{data_type}.mp4"),
                    fps=fps,
                    codec="libx264",
                    quality=6,
                    pixelformat="yuv420p",
                )
        if capture_camera_output_raw_dir is not None and accelerator.is_main_process:
            capture_camera_output_raw_dir = Path(capture_camera_output_raw_dir)
            capture_camera_output_raw_dir.mkdir(parents=True, exist_ok=True)

        def _get_raw_ego_camera():
            raw_env = getattr(env, "env", env)
            scene = getattr(raw_env, "scene", None)
            if scene is None:
                return None
            try:
                camera = scene["ego_camera"]
            except KeyError:
                return None
            return camera

        def _get_raw_ego_camera_frame():
            camera = _get_raw_ego_camera()
            if camera is None:
                return None
            frame = camera.data.output.get("rgb", None)
            if frame is None:
                return None
            frame = frame[0].detach().cpu()
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            return frame

        def _json_safe_camera_info(value):
            if isinstance(value, dict):
                return {str(key): _json_safe_camera_info(val) for key, val in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe_camera_info(item) for item in value]
            return value

        def _to_uint8_rgb(frame):
            frame_np = frame.numpy()
            if frame_np.dtype == np.uint8:
                return frame_np
            if capture_camera_rgb_source == "obs" and capture_camera_rgb_unnormalize:
                mean = np.array([0.485, 0.456, 0.406], dtype=frame_np.dtype)
                std = np.array([0.229, 0.224, 0.225], dtype=frame_np.dtype)
                frame_np = frame_np * std + mean
            if frame_np.max() <= 1.0:
                frame_np = frame_np * 255.0
            return frame_np.clip(0, 255).astype(np.uint8)

        def _to_uint8_camera_output(data_type, frame):
            frame_np = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
            while frame_np.ndim > 3:
                frame_np = frame_np[0]
            if frame_np.ndim == 3 and frame_np.shape[-1] == 1:
                frame_np = frame_np[..., 0]
            if frame_np.ndim == 3 and frame_np.shape[-1] >= 3:
                if frame_np.shape[-1] == 4:
                    frame_np = frame_np[..., :3]
                if frame_np.dtype != np.uint8:
                    if frame_np.max() <= 1.0:
                        frame_np = frame_np * 255.0
                    frame_np = frame_np.clip(0, 255).astype(np.uint8)
                return frame_np

            if data_type in ("semantic_segmentation", "instance_segmentation_fast", "instance_id_segmentation_fast"):
                labels = frame_np.astype(np.int64)
                labels = np.nan_to_num(labels, nan=0).astype(np.int64)
                rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
                positive = labels > 0
                rgb[..., 0] = ((labels * 37) % 255).astype(np.uint8)
                rgb[..., 1] = ((labels * 67) % 255).astype(np.uint8)
                rgb[..., 2] = ((labels * 97) % 255).astype(np.uint8)
                rgb[~positive] = 0
                return rgb

            values = frame_np.astype(np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            finite = np.isfinite(values)
            valid = values[finite]
            if valid.size == 0:
                norm = np.zeros_like(values, dtype=np.uint8)
            else:
                v_min = float(valid.min())
                v_max = float(valid.max())
                if v_max <= v_min:
                    norm = np.zeros_like(values, dtype=np.uint8)
                else:
                    norm = ((values - v_min) / (v_max - v_min) * 255.0).clip(0, 255).astype(np.uint8)
            return np.repeat(norm[..., None], 3, axis=-1)

        def capture_camera_rgb(obs):
            nonlocal camera_rgb_frames
            if camera_rgb_writer is None:
                return
            if capture_camera_rgb_max_frames > 0 and camera_rgb_frames >= capture_camera_rgb_max_frames:
                return

            frame = _get_raw_ego_camera_frame() if capture_camera_rgb_source == "raw" else None
            if frame is None:
                if "camera_rgb" not in obs:
                    return
                frame = obs["camera_rgb"][0].detach().cpu()
                while frame.ndim > 3:
                    frame = frame[0]
            frame_np = _to_uint8_rgb(frame)
            if camera_rgb_frames == 0:
                logger.info(
                    "Captured ego camera frame source={} shape={} min={} max={} mean={:.3f}".format(
                        capture_camera_rgb_source,
                        tuple(frame_np.shape),
                        int(frame_np.min()),
                        int(frame_np.max()),
                        float(frame_np.mean()),
                    )
                )
            camera_rgb_writer.append_data(frame_np)
            camera_rgb_frames += 1

        def capture_camera_outputs():
            nonlocal camera_output_frames, camera_output_logged_keys
            if not camera_output_writers and capture_camera_output_raw_dir is None:
                return
            if capture_camera_output_max_frames > 0 and camera_output_frames >= capture_camera_output_max_frames:
                return
            camera = _get_raw_ego_camera()
            if camera is None:
                return
            if not camera_output_logged_keys:
                logger.info(f"Raw ego camera output keys: {list(camera.data.output.keys())}")
                if capture_camera_output_raw_dir is not None:
                    info_path = capture_camera_output_raw_dir / "camera_data_info.json"
                    with open(info_path, "w", encoding="utf-8") as info_file:
                        json.dump(_json_safe_camera_info(camera.data.info), info_file, indent=2, default=str)
                camera_output_logged_keys = True
            for data_type in capture_camera_output_types:
                frame = camera.data.output.get(data_type, None)
                if frame is None and data_type == "depth":
                    frame = camera.data.output.get("distance_to_image_plane", None)
                if frame is None:
                    continue
                if (
                    capture_camera_output_raw_dir is not None
                    and (
                        capture_camera_output_raw_max_frames <= 0
                        or camera_output_frames < capture_camera_output_raw_max_frames
                    )
                ):
                    raw_np = frame[0].detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame[0])
                    np.savez_compressed(
                        capture_camera_output_raw_dir / f"{camera_output_frames:06d}_{data_type}.npz",
                        data=raw_np,
                    )
                writer = camera_output_writers.get(data_type)
                if writer is None:
                    continue
                frame_np = _to_uint8_camera_output(data_type, frame[0])
                if camera_output_frames == 0:
                    logger.info(
                        "Captured ego camera output {} shape={} min={} max={} mean={:.3f}".format(
                            data_type,
                            tuple(frame_np.shape),
                            int(frame_np.min()),
                            int(frame_np.max()),
                            float(frame_np.mean()),
                        )
                    )
                writer.append_data(frame_np)
            camera_output_frames += 1

        run_once = config.get("run_once", False)
        envs_completed = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
        episode_steps = torch.zeros(
            config.num_envs, dtype=torch.long, device=device
        )
        episode_motion_keys = _evaluation_motion_keys(env, config.num_envs)
        eval_step_dt = _evaluation_step_dt(env)
        log_reset_reasons = bool(config.get("log_reset_reasons", True))
        chunk_executor = None
        previous_motion_metadata = None
        if action_chunk_enabled(config.algo.config):
            chunk_config = get_action_chunk_config(config.algo.config)
            chunk_executor = ActionChunkExecutor(
                policy=model.policy,
                num_envs=config.num_envs,
                horizon=int(chunk_config["horizon"]),
                execute_steps=int(chunk_config["eval_execute_steps"]),
                action_dim=env.config.robot.actions_dim,
                device=device,
            )
        previous_dones = torch.zeros(config.num_envs, dtype=torch.bool, device=device)

        with torch.no_grad():
            while True:
                policy_model = model.policy
                value_model = model.value_model

                actor_state = {}
                if chunk_executor is None:
                    policy_model.init_rollout()
                    actions = policy_model.rollout(obs_dict=obs_dict)
                    action_mean = policy_model.action_mean.detach()
                    decoder_obs_dict = actions["obs_dict"]
                else:
                    motion_metadata = get_motion_metadata(env)
                    discontinuity = motion_discontinuity(
                        motion_metadata, previous_motion_metadata
                    )
                    if not chunk_config.get("replan_on_discontinuity", True):
                        discontinuity = None
                    action_mean = chunk_executor.act(
                        obs_dict,
                        reset_mask=previous_dones,
                        discontinuity_mask=discontinuity,
                    )
                    previous_motion_metadata = motion_metadata
                    # A chunk stores actions only.  The decoder must consume the
                    # observation from the environment step being executed.
                    decoder_obs_dict = obs_dict
                capture_camera_rgb(obs_dict)
                capture_camera_outputs()
                actor_state["actions"] = action_mean
                actor_state["obs_dict"] = decoder_obs_dict

                step_count += 1

                if max_render_steps > 0 and step_count >= max_render_steps:
                    logger.info(f"Reached max_render_steps={max_render_steps}. Exiting.")
                    if hasattr(env, "end_render_results"):
                        env.end_render_results()
                    break

                results = env.step(actor_state)
                obs_dict, rewards, dones, infos = (
                    results[0],
                    results[1],
                    results[2],
                    results[3],
                )  # noqa: F841
                previous_dones = dones.to(device=device, dtype=torch.bool).reshape(-1)
                if run_once:
                    episode_steps[~envs_completed] += 1
                if eval_step_callbacks:
                    all_want_exit = all(
                        cb.eval_step(env, results) for cb in eval_step_callbacks.values()
                    )
                    if all_want_exit:
                        logger.info("All eval step callbacks signaled exit. Exiting evaluation loop.")
                        break

                if run_once:
                    done_mask = dones.to(device=device, dtype=torch.bool).reshape(-1)
                    newly_completed = done_mask & ~envs_completed
                    if log_reset_reasons:
                        for env_id in newly_completed.nonzero(
                            as_tuple=False
                        ).squeeze(-1).tolist():
                            reasons = _evaluation_reset_reasons(env, env_id)
                            displayed_reasons = [
                                (
                                    f"{reason}(参考/真实物体位置偏差 > "
                                    f"{config.object_pos_deviation_threshold:.3f} m)"
                                    if reason == "object_pos_deviation"
                                    else reason
                                )
                                for reason in reasons
                            ]
                            reset_time = float(episode_steps[env_id].item()) * eval_step_dt
                            logger.info(
                                f"第 {env_id + 1} 条数据 "
                                f"({episode_motion_keys[env_id]}) 因为 "
                                f"{', '.join(displayed_reasons)} "
                                f"在第 {reset_time:.2f} 秒 reset"
                            )
                    envs_completed |= done_mask
                    if envs_completed.all():
                        logger.info(
                            f"全部 {config.num_envs} 条数据均已完成一次 episode；"
                            "退出评估 (run_once=True)。"
                        )
                        if hasattr(env, "end_render_results"):
                            env.end_render_results()
                        break

                for obs_key in obs_dict.keys():  # noqa: SIM118
                    obs_dict[obs_key] = obs_dict[obs_key].to(device)

    if camera_rgb_writer is not None:
        camera_rgb_writer.close()
    for writer in camera_output_writers.values():
        writer.close()

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":
    main()
