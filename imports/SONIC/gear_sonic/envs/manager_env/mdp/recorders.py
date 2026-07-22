"""Custom recorder terms for the manager environment MDP."""

from __future__ import annotations

import json
import os
import pickle
from typing import TYPE_CHECKING

import cv2
import imageio
from isaaclab.managers import manager_term_cfg, recorder_manager
from isaaclab.utils import configclass
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from isaaclab import envs


def _xy_offset_suffix(offset: torch.Tensor) -> str:
    """Format a sampled root XY offset for an output filename."""
    x, y = (float(value) for value in offset[:2].detach().cpu())
    x = 0.0 if abs(x) < 0.005 else x
    y = 0.0 if abs(y) < 0.005 else y
    return f"_x{x:+.2f}_y{y:+.2f}"


def _termination_reasons(env, env_id: int) -> tuple[str, ...]:
    """Return every termination term that fired for one environment this step."""
    manager = env.termination_manager
    return tuple(
        name for name in manager.active_terms if bool(manager.get_term(name)[env_id].item())
    )


@configclass
class RecordersCfg(recorder_manager.RecorderManagerBaseCfg):
    """Recorders terms for the MDP."""

    render_envs = None
    running_ref_root_height = None
    trajectory = None


class RenderEnvsRecorderTerm(recorder_manager.RecorderTerm):
    """Recorder term for rendering environments with advanced features like text overlay and frame skipping."""

    cfg: RenderEnvsRecorderCfg

    def __init__(self, cfg: RenderEnvsRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        # Determine save directory (backward compatibility)
        self.save_dir = self.cfg.video_save_path
        logger.info(f"=== Start recording video to {self.save_dir} ===")

        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        self.video_writers = []
        self._writers_closed = False
        self.frame_id = 0
        self.first_render = True
        self._fixed_eye = None
        self._fixed_target = None
        self._completed = torch.zeros(self.env.num_envs, dtype=torch.bool)
        self._successful = torch.zeros(self.env.num_envs, dtype=torch.bool)
        self._termination_reasons: list[tuple[str, ...]] = [
            () for _ in range(self.env.num_envs)
        ]
        self._video_paths: list[str] = []

    def _initialize_writers(self):
        """Initialize video writers for each environment."""
        logger.info(f"Saving rendering to {self.save_dir}")
        # Get configuration parameters with defaults
        self.group_camera = self.env.wrapper.config.get("group_camera", False)
        self.max_render_envs = self.env.wrapper.config.get("max_render_envs", self.env.num_envs)
        if self.group_camera:
            self.max_render_envs = 1  # single video from overview_camera
        self.render_frame_skip = self.env.wrapper.config.get("render_frame_skip", 2)
        self.start_idx = self.env.wrapper.start_idx

        motion = self.env.command_manager.get_term("motion")
        motion_keys = getattr(motion.motion_lib, "curr_motion_keys", [])
        offsets = getattr(motion, "initial_root_pose_offset", None)

        for i in range(self.max_render_envs):
            motion_idx = self.start_idx + i
            stem = motion_keys[i] if i < len(motion_keys) else f"{motion_idx:06d}"
            if self.cfg.append_initial_xy_offset and offsets is not None:
                stem += _xy_offset_suffix(offsets[i])
            file_name = os.path.join(self.save_dir, f"{stem}.mp4")
            fps = 1 / (self.env.step_dt * self.render_frame_skip)
            writer = imageio.get_writer(
                file_name,
                fps=fps,
                codec="libx264",
                quality=self.cfg.video_quality,
                pixelformat="yuv420p",
            )
            self.video_writers.append(writer)
            self._video_paths.append(file_name)

    def record_pre_reset(self, env_ids) -> tuple[str | None, torch.Tensor | dict | None]:
        """Remember whether the first recorded episode ended by timeout."""
        if getattr(self.env, "_suppress_recording", False):
            return None, None
        if not self.video_writers:
            return None, None
        for env_id in env_ids:
            if not self._completed[env_id]:
                self._completed[env_id] = True
                self._successful[env_id] = bool(self.env.reset_time_outs[env_id].item())
                self._termination_reasons[env_id] = _termination_reasons(self.env, env_id)
        return None, None

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record video frames after each step with frame skipping and text overlay support."""
        if getattr(self.env, "_suppress_recording", False):
            return None, None
        if len(self.video_writers) == 0:
            self._initialize_writers()

        # Check if we should render this frame
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)

        camera_name = self.env.wrapper.config.get("render_camera", "eval_camera")
        cam = self.env.scene[camera_name]

        if camera_name == "eval_camera":
            # The legacy third-person camera follows the robot root.
            root_pos = self.env.command_manager.get_term("motion").robot_body_pos_w[:, 0]
            camera_offset = self.env.wrapper.config.get("eval_camera_offset", [2, 2, 1])
            fix_camera = self.env.wrapper.config.get("fix_camera_after_first_frame", False)
            if fix_camera and self._fixed_eye is not None:
                eye, target = self._fixed_eye, self._fixed_target
            elif self.group_camera:
                center = root_pos.mean(dim=0, keepdim=True).expand_as(root_pos)
                eye = center + torch.tensor(camera_offset, device=self.env.device)
                target = center
                if fix_camera:
                    self._fixed_eye = eye.clone()
                    self._fixed_target = center.clone()
            else:
                eye = root_pos + torch.tensor(camera_offset, device=self.env.device)
                target = root_pos
                if fix_camera:
                    self._fixed_eye = eye.clone()
                    self._fixed_target = root_pos.clone()

            # Write world poses to Fabric AND sync to USD so both renderer paths see it.
            cam._view._sync_usd_on_fabric_write = True  # noqa: SLF001
            cam.set_world_poses_from_view(eye, target)

        # Two render calls: 1st flushes pose to render pipeline, 2nd captures at new pose
        if hasattr(self.env, "sim"):
            self.env.sim.render()
            self.env.sim.render()

        # Mark sensor as outdated so update actually re-reads the annotator buffers
        cam._is_outdated[:] = True  # noqa: SLF001
        cam.update(dt=0.0, force_recompute=True)

        # Get RGB data
        rgb_viewer = cam.data.output["rgb"].clone()

        # Get render info if available
        cur_render_info = None
        if self.env.wrapper.config.get("render_info", None) is not None:
            end_idx = self.start_idx + self.max_render_envs
            cur_render_info = self.env.wrapper.config.render_info[self.start_idx : end_idx]

        # Process each environment, loop over the video writers
        if self.frame_id >= 1:
            loop = (
                tqdm(range(self.max_render_envs))
                if self.first_render
                else range(self.max_render_envs)
            )
            for i in loop:
                if self.cfg.save_only_timeouts and self._completed[i]:
                    continue
                frame = rgb_viewer[i].cpu().numpy()

                # Add text overlay if render info is provided
                if cur_render_info is not None and i < len(cur_render_info):
                    for j, text in enumerate(cur_render_info[i]):
                        frame = cv2.putText(
                            frame,
                            str(text),
                            (10, 30 + j * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            1,
                        )

                self.video_writers[i].append_data(frame)
            self.first_render = False

        self.frame_id += 1
        return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self):
        """Explicitly close all video writers."""
        if not self._writers_closed:
            for i, writer in enumerate(self.video_writers):
                try:
                    writer.close()
                    logger.info(f"Closed video writer {i}")
                except Exception as e:  # noqa: BLE001
                    logger.info(f"Error closing video writer {i}: {e}")
            if self.cfg.save_only_timeouts:
                for i, path in enumerate(self._video_paths):
                    if not self._successful[i] and os.path.exists(path):
                        os.remove(path)
                        reasons = ", ".join(self._termination_reasons[i]) or "unknown"
                        logger.info(
                            f"Discarded early-terminated video (env={i}, reasons={reasons}): "
                            f"{path}"
                        )
            self.video_writers.clear()
            self._writers_closed = True
            self.frame_id = 0
            self.first_render = True
            self._fixed_eye = None
            self._fixed_target = None
            logger.info("=== All video writers closed ===")

    def __del__(self):
        """Ensure writers are closed when object is destroyed."""
        self.close_writers()


@configclass
class RenderEnvsRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for environment rendering recorder with advanced features."""

    class_type = RenderEnvsRecorderTerm
    video_save_path: str = None
    video_quality: int = 5
    save_only_timeouts: bool = False
    append_initial_xy_offset: bool = False


class TrajectoryRecorderTerm(recorder_manager.RecorderTerm):
    """Recorder term that saves per-environment trajectory data (joint positions, root pose, object/table state).

    Saves .trajectory.pkl files alongside the video output, enabling kinematic replay
    in multi-scene composite renders.
    """

    cfg: TrajectoryRecorderCfg

    def __init__(self, cfg: TrajectoryRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        self.save_dir = self.cfg.save_path
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"=== TrajectoryRecorder: saving to {self.save_dir} ===")

        self._initialized = False
        self._closed = False
        self._frame_data: dict[int, dict] = {}  # env_idx -> {field: [frames]}
        self.frame_id = 0

    def _initialize(self):
        """Initialize per-env data buffers after environment is ready."""
        self.num_record_envs = self.env.num_envs
        self.start_idx = (
            getattr(self.env.wrapper, "start_idx", 0) if hasattr(self.env, "wrapper") else 0
        )

        # Match video recorder's frame skip to keep trajectory in sync with video
        if hasattr(self.env, "wrapper"):
            self.render_frame_skip = self.env.wrapper.config.get("render_frame_skip", 2)
        else:
            self.render_frame_skip = 2

        # Detect available scene entities
        self._has_object = "object" in self.env.scene.rigid_objects
        self._has_table = "table" in self.env.scene.rigid_objects

        # Get motion command for root pose
        try:
            self._motion_cmd = self.env.command_manager.get_term("motion")
        except Exception:  # noqa: BLE001
            self._motion_cmd = None

        for i in range(self.num_record_envs):
            self._frame_data[i] = self._create_empty_data()

        self._initialized = True

    def _create_empty_data(self) -> dict:
        data = {
            "dof_pos": [],
            "root_pos_w": [],
            "root_quat_w": [],
        }
        if self._has_object:
            data["object_pos_w"] = []
            data["object_quat_w"] = []
        if self._has_table:
            data["table_pos_w"] = []
            data["table_quat_w"] = []
        return data

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record trajectory state after each step, synced with video frame skip."""
        if not self._initialized:
            self._initialize()

        # Skip frames to match video recorder cadence
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "trajectory_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

        robot = self.env.scene["robot"]
        env_origins = self.env.scene.env_origins

        for i in range(self.num_record_envs):
            # Joint positions
            joint_pos = robot.data.joint_pos[i].cpu().numpy().copy()
            self._frame_data[i]["dof_pos"].append(joint_pos)

            # Root position (relative to env origin)
            if self._motion_cmd is not None:
                root_pos = self._motion_cmd.robot_body_pos_w[i, 0].cpu().numpy().copy()
            else:
                root_pos = robot.data.root_pos_w[i].cpu().numpy().copy()
            root_pos_rel = root_pos - env_origins[i].cpu().numpy()
            self._frame_data[i]["root_pos_w"].append(root_pos_rel)

            # Root quaternion (wxyz)
            root_quat = robot.data.root_quat_w[i].cpu().numpy().copy()
            self._frame_data[i]["root_quat_w"].append(root_quat)

            # Object state
            if self._has_object:
                obj = self.env.scene["object"]
                obj_pos = obj.data.root_pos_w[i].cpu().numpy().copy()
                obj_pos_rel = obj_pos - env_origins[i].cpu().numpy()
                obj_quat = obj.data.root_quat_w[i].cpu().numpy().copy()
                self._frame_data[i]["object_pos_w"].append(obj_pos_rel)
                self._frame_data[i]["object_quat_w"].append(obj_quat)

            # Table state
            if self._has_table:
                table = self.env.scene["table"]
                table_pos = table.data.root_pos_w[i].cpu().numpy().copy()
                table_pos_rel = table_pos - env_origins[i].cpu().numpy()
                table_quat = table.data.root_quat_w[i].cpu().numpy().copy()
                self._frame_data[i]["table_pos_w"].append(table_pos_rel)
                self._frame_data[i]["table_quat_w"].append(table_quat)

        self.frame_id += 1
        return "trajectory_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self):
        """Save all trajectory data to pkl files."""
        if self._closed or not self._initialized:
            return
        self._closed = True

        # FPS matches the video (after frame skip)
        effective_fps = 1.0 / (self.env.step_dt * self.render_frame_skip)

        scene_metadata = {}

        for i in range(self.num_record_envs):
            env_idx = self.start_idx + i
            data = self._frame_data[i]

            if not data["dof_pos"]:
                continue

            # Stack frame arrays
            trajectory = {
                "dof_pos": np.array(data["dof_pos"]),
                "root_pos_w": np.array(data["root_pos_w"]),
                "root_quat_w": np.array(data["root_quat_w"]),
                "quat_format": "wxyz",
                "fps": effective_fps,
                "num_joints": data["dof_pos"][0].shape[0],
                "total_frames": len(data["dof_pos"]),
            }

            if data.get("object_pos_w"):
                trajectory["object_pos_w"] = np.array(data["object_pos_w"])
                trajectory["object_quat_w"] = np.array(data["object_quat_w"])
            else:
                trajectory["object_pos_w"] = None
                trajectory["object_quat_w"] = None

            if data.get("table_pos_w"):
                trajectory["table_pos_w"] = np.array(data["table_pos_w"])
                trajectory["table_quat_w"] = np.array(data["table_quat_w"])
            else:
                trajectory["table_pos_w"] = None
                trajectory["table_quat_w"] = None

            # Save pkl
            pkl_path = os.path.join(self.save_dir, f"{env_idx:06d}.trajectory.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved trajectory: {pkl_path} ({trajectory['total_frames']} frames)")

            # Build metadata entry
            meta = {
                "trajectory_file": f"{env_idx:06d}.trajectory.pkl",
                "video_file": f"{env_idx:06d}.mp4",
                "num_frames": trajectory["total_frames"],
                "num_joints": trajectory["num_joints"],
                "fps": effective_fps,
                "has_object": trajectory["object_pos_w"] is not None,
                "has_table": trajectory["table_pos_w"] is not None,
            }

            # Add object USD path if available from config
            if hasattr(self.env, "wrapper"):
                obj_usd = self.env.wrapper.config.get("object_usd_path", None)
                if obj_usd:
                    meta["object_usd_path"] = obj_usd

            scene_metadata[str(env_idx)] = meta

        # Save scene metadata JSON
        meta_path = os.path.join(self.save_dir, "scene_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(scene_metadata, f, indent=2)
        logger.info(f"Saved scene metadata: {meta_path}")
        logger.info("=== TrajectoryRecorder: all data saved ===")

    def __del__(self):
        self.close_writers()


@configclass
class TrajectoryRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for trajectory recording alongside video."""

    class_type = TrajectoryRecorderTerm
    save_path: str = None


class ObjectAwareStateRecorderTerm(recorder_manager.RecorderTerm):
    """Save per-frame object-aware observations and latent adaptor outputs.

    The output is one ``<motion_key>.object_aware.pkl`` per environment.  It
    records the policy observation terms requested by GRAIL Object-Aware
    Tracking, the raw 64+2 meta-action, the actually executed hand primitive,
    and the continuous pre-FSQ ``z + lambda * delta_z`` latent.
    """

    cfg: ObjectAwareStateRecorderCfg

    _OBJECT_REFERENCE_TERMS = (
        "object_pos_b",
        "object_ori_b_6d",
        "target_object_pos",
        "hand_object_transform_6d",
        "finger_tips_force",
        "object_bps",
        "object_pos_delta_multi_future",
        "object_ori_delta_multi_future_6d",
    )
    _PROPRIOCEPTION_TERMS = (
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
        "last_meta_action",
    )

    def __init__(self, cfg: ObjectAwareStateRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env
        self.save_dir = self.cfg.save_path
        os.makedirs(self.save_dir, exist_ok=True)
        self._initialized = False
        self._closed = False
        self.frame_id = 0
        self._frame_data: dict[int, dict[str, list]] = {}
        self._completed = torch.zeros(self.env.num_envs, dtype=torch.bool)
        self._successful = torch.zeros(self.env.num_envs, dtype=torch.bool)
        self._termination_reasons: list[tuple[str, ...]] = [
            () for _ in range(self.env.num_envs)
        ]
        logger.info(f"=== ObjectAwareStateRecorder: saving to {self.save_dir} ===")

    def _initialize(self) -> None:
        self.num_record_envs = self.env.num_envs
        self.start_idx = getattr(self.env.wrapper, "start_idx", 0)
        self.render_frame_skip = self.env.wrapper.config.get("render_frame_skip", 1)
        manager = self.env.observation_manager
        names = manager._group_obs_term_names["policy"]  # noqa: SLF001
        dims = manager._group_obs_term_dim["policy"]  # noqa: SLF001
        self._policy_slices = {}
        offset = 0
        for name, shape in zip(names, dims):
            width = int(np.prod(shape))
            self._policy_slices[name] = (offset, offset + width, tuple(shape))
            offset += width
        self._policy_obs_dim = offset
        self._motion_cmd = self.env.command_manager.get_term("motion")
        offsets = getattr(self._motion_cmd, "initial_root_pose_offset", None)
        self._initial_root_pose_offsets = (
            offsets.detach().cpu().clone() if offsets is not None else None
        )
        if "object" not in self.env.scene.rigid_objects:
            raise RuntimeError("ObjectAwareStateRecorder requires an object rigid body.")
        for i in range(self.num_record_envs):
            self._frame_data[i] = {
                "frame_idx": [],
                "motion_step": [],
                "policy_obs_flat": [],
                "latent_residual_raw": [],
                "latent_residual_scaled": [],
                "hand_primitive_policy_raw": [],
                "hand_primitive_policy_binary": [],
                "hand_primitive_executed": [],
                "hand_primitive_executed_binary": [],
                "combined_latent_pre_fsq": [],
                "robot_root_pos_w": [],
                "robot_root_quat_w": [],
                "object_root_pos_w": [],
                "object_root_quat_w": [],
            }
            for name in self._PROPRIOCEPTION_TERMS + self._OBJECT_REFERENCE_TERMS:
                if name in self._policy_slices:
                    self._frame_data[i][name] = []
        self._initialized = True

    def record_pre_reset(self, env_ids) -> tuple[str | None, torch.Tensor | dict | None]:
        """Remember whether the first recorded episode ended by timeout."""
        if getattr(self.env, "_suppress_recording", False):
            return None, None
        if not self._initialized:
            return None, None
        for env_id in env_ids:
            if not self._completed[env_id] and self._frame_data[env_id]["frame_idx"]:
                self._completed[env_id] = True
                self._successful[env_id] = bool(self.env.reset_time_outs[env_id].item())
                self._termination_reasons[env_id] = _termination_reasons(self.env, env_id)
        return None, None

    @staticmethod
    def _last_step(tensor: torch.Tensor) -> torch.Tensor:
        """Drop a model sequence axis while preserving ordinary 2-D batches."""
        return tensor[:, -1] if tensor.ndim == 3 else tensor

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        if getattr(self.env, "_suppress_recording", False):
            return None, None
        if not self._initialized:
            self._initialize()
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "object_aware_record", torch.ones(
                self.env.num_envs, 1, device=self.env.device
            )

        actor_obs = getattr(self.env, "_object_aware_actor_obs", None)
        residual = getattr(self.env, "_object_aware_latent_residual_raw", None)
        residual_scaled = getattr(self.env, "_object_aware_latent_residual_scaled", None)
        hand_policy = getattr(self.env, "_object_aware_hand_primitive_policy_raw", None)
        hand_executed = getattr(self.env, "_object_aware_hand_primitive_executed", None)
        combined = getattr(self.env, "_object_aware_combined_latent", None)
        required = (actor_obs, residual, residual_scaled, hand_policy, hand_executed, combined)
        if any(value is None for value in required):
            raise RuntimeError(
                "Object-aware state recording requires residual-mode inference and all "
                "diagnostic tensors, but at least one tensor was unavailable."
            )

        actor_obs = self._last_step(actor_obs)
        residual = self._last_step(residual)
        residual_scaled = self._last_step(residual_scaled)
        hand_policy = self._last_step(hand_policy)
        hand_executed = self._last_step(hand_executed)
        combined = self._last_step(combined)
        # The recorder runs post-step, while observations/actions are pre-step.
        # Use the timestamp cached beside the policy tensors to keep alignment.
        motion_steps = getattr(self.env, "_object_aware_motion_step", None)
        if motion_steps is None:
            raise RuntimeError("Object-aware pre-step motion timestamp was not cached.")

        if actor_obs.shape[-1] != self._policy_obs_dim:
            raise RuntimeError(
                f"Policy observation width mismatch: tensor={actor_obs.shape[-1]}, "
                f"terms={self._policy_obs_dim}."
            )

        robot_root_pos_w = getattr(self.env, "_object_aware_robot_root_pos_w", None)
        robot_root_quat_w = getattr(self.env, "_object_aware_robot_root_quat_w", None)
        if robot_root_pos_w is None or robot_root_quat_w is None:
            raise RuntimeError("Object-aware pre-step robot root pose was not cached.")
        object_root_pos_w = getattr(self.env, "_object_aware_object_root_pos_w", None)
        object_root_quat_w = getattr(self.env, "_object_aware_object_root_quat_w", None)
        if object_root_pos_w is None or object_root_quat_w is None:
            raise RuntimeError("Object-aware pre-step object root pose was not cached.")

        for i in range(self.num_record_envs):
            if self.cfg.save_only_timeouts and self._completed[i]:
                continue
            data = self._frame_data[i]
            data["frame_idx"].append(self.frame_id)
            data["motion_step"].append(int(motion_steps[i].item()))
            data["policy_obs_flat"].append(actor_obs[i].detach().cpu().numpy().copy())
            data["latent_residual_raw"].append(residual[i].detach().cpu().numpy().copy())
            data["latent_residual_scaled"].append(
                residual_scaled[i].detach().cpu().numpy().copy()
            )
            data["hand_primitive_policy_raw"].append(
                hand_policy[i].detach().cpu().numpy().copy()
            )
            data["hand_primitive_policy_binary"].append(
                (hand_policy[i] >= 0).to(torch.int8).detach().cpu().numpy().copy()
            )
            data["hand_primitive_executed"].append(
                hand_executed[i].detach().cpu().numpy().copy()
            )
            data["hand_primitive_executed_binary"].append(
                (hand_executed[i] >= 0).to(torch.int8).detach().cpu().numpy().copy()
            )
            data["combined_latent_pre_fsq"].append(
                combined[i].detach().cpu().numpy().copy()
            )
            for name, value in (
                ("robot_root_pos_w", robot_root_pos_w),
                ("robot_root_quat_w", robot_root_quat_w),
            ):
                data[name].append(value[i].detach().cpu().numpy().copy())
            for name, value in (
                ("object_root_pos_w", object_root_pos_w),
                ("object_root_quat_w", object_root_quat_w),
            ):
                data[name].append(value[i].detach().cpu().numpy().copy())
            for name, (start, end, shape) in self._policy_slices.items():
                if name in data:
                    value = actor_obs[i, start:end].reshape(shape)
                    data[name].append(value.detach().cpu().numpy().copy())

        self.frame_id += 1
        return "object_aware_record", torch.ones(
            self.env.num_envs, 1, device=self.env.device
        )

    def close_writers(self) -> None:
        if self._closed or not self._initialized:
            return
        self._closed = True
        motion_keys = getattr(self._motion_cmd.motion_lib, "curr_motion_keys", [])
        scale = float(self.env.wrapper.config.get("latent_residual_scale", 1.0))
        override = bool(self.env.wrapper.config.get("use_motion_hand_actions", False))
        for i, data in self._frame_data.items():
            if not data["frame_idx"]:
                continue
            motion_key = (
                motion_keys[i] if i < len(motion_keys) else f"{self.start_idx + i:06d}"
            )
            output_stem = motion_key
            initial_offset = None
            if self._initial_root_pose_offsets is not None:
                initial_offset = self._initial_root_pose_offsets[i]
                if self.cfg.append_initial_xy_offset:
                    output_stem += _xy_offset_suffix(initial_offset)
            path = os.path.join(self.save_dir, f"{output_stem}.object_aware.pkl")
            if self.cfg.save_only_timeouts and not self._successful[i]:
                if os.path.exists(path):
                    os.remove(path)
                reasons = ", ".join(self._termination_reasons[i]) or "unknown"
                logger.info(
                    f"Discarded early-terminated object-aware data "
                    f"(env={i}, motion={motion_key}, reasons={reasons})"
                )
                continue
            payload = {
                "schema_version": 3,
                "motion_key": motion_key,
                "fps": 1.0 / (self.env.step_dt * self.render_frame_skip),
                "latent_residual_scale": scale,
                "latent_residual_mode": self.env.wrapper.config.get(
                    "latent_residual_mode", "post_quantization"
                ),
                "hand_action_overridden_by_motion": override,
                "hand_binary_convention": {"open": 0, "closed": 1, "threshold": 0.0},
                "pose_timing": "pre_step_aligned_with_policy_input",
                "pose_quaternion_format": "wxyz",
                "policy_observation_term_order": list(self._policy_slices),
                "policy_observation_term_slices": {
                    name: {"start": start, "end": end, "shape": shape}
                    for name, (start, end, shape) in self._policy_slices.items()
                },
            }
            if initial_offset is not None:
                payload["initial_root_pose_offset_xyz"] = initial_offset.numpy().copy()
            array_data = {key: np.asarray(value) for key, value in data.items()}
            payload.update(array_data)
            payload["proprioception"] = {
                name: array_data[name]
                for name in self._PROPRIOCEPTION_TERMS
                if name in array_data
            }
            payload["object_reference"] = {
                name: array_data[name]
                for name in self._OBJECT_REFERENCE_TERMS
                if name in array_data
            }
            with open(path, "wb") as file:
                pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                f"Saved object-aware state: {path} ({len(data['frame_idx'])} frames)"
            )

    def __del__(self):
        self.close_writers()


@configclass
class ObjectAwareStateRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for Object-Aware Tracking tensor recording."""

    class_type = ObjectAwareStateRecorderTerm
    save_path: str = None
    save_only_timeouts: bool = False
    append_initial_xy_offset: bool = False
