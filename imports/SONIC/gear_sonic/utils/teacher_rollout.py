"""Export post-action teacher states in the source motion-library format."""

import json
from pathlib import Path
import shutil

import joblib
from loguru import logger
import numpy as np
from scipy.spatial.transform import Rotation


def _yaw(quat_xyzw):
    x, y, z, w = quat_xyzw.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _turn_direction(angle):
    return "left" if angle > 0.02 else "right" if angle < -0.02 else "straight"


class RolloutReport:
    """Accumulate actual motion and contact events for each first episode."""

    def __init__(self, base_pos, base_quat, object_pos, release_goal):
        self.release_goal = release_goal.copy()
        self.previous_base = base_pos.copy()
        self.previous_yaw = _yaw(base_quat)
        count = len(base_pos)
        self.yaw_change = np.zeros(count)
        self.base_path = np.zeros(count)
        self.base_xy_path = np.zeros(count)
        self.min_goal_distance = np.linalg.norm(object_pos - release_goal, axis=-1)
        self.foot_contacts = np.zeros((count, 2), dtype=bool)
        self.foot_contact_counts = np.zeros((count, 2), dtype=int)
        self.table_collision = np.zeros(count, dtype=bool)

    def update_motion(self, state, active):
        position = state["root_trans_offset"]
        delta = position - self.previous_base
        self.base_path += np.linalg.norm(delta, axis=-1) * active
        self.base_xy_path += np.linalg.norm(delta[:, :2], axis=-1) * active
        yaw = _yaw(state["root_rot"])
        self.yaw_change += ((yaw - self.previous_yaw + np.pi) % (2 * np.pi) - np.pi) * active
        distance = np.linalg.norm(state["object_pos"] - self.release_goal, axis=-1)
        self.min_goal_distance = np.where(active, np.minimum(self.min_goal_distance, distance),
                                          self.min_goal_distance)
        self.previous_base = np.where(active[:, None], position, self.previous_base)
        self.previous_yaw = np.where(active, yaw, self.previous_yaw)

    def update_contacts(self, feet, table, active):
        # Count every transition into contact, including the first observed contact.
        landing = feet & ~self.foot_contacts
        self.foot_contact_counts += landing & active[:, None]
        self.foot_contacts = feet.copy()
        self.table_collision |= table & active

    def result(self, env_id, reference_yaw_change):
        left, right = self.foot_contact_counts[env_id]
        return {
            "turn_direction": _turn_direction(self.yaw_change[env_id]),
            "reference_turn_direction": _turn_direction(reference_yaw_change),
            "yaw_change_rad": float(self.yaw_change[env_id]),
            "release_goal_pos": self.release_goal[env_id].tolist(),
            "min_release_goal_distance_m": float(self.min_goal_distance[env_id]),
            "foot_ground_contacts": {"left": int(left), "right": int(right), "total": int(left + right)},
            "base_path_length_m": float(self.base_path[env_id]),
            "base_xy_path_length_m": float(self.base_xy_path[env_id]),
            "robot_table_collision": bool(self.table_collision[env_id]),
            "contact_force_threshold_n": 1.0,
        }


class TeacherRolloutExporter:
    """Export first episodes that reach timeout, including their pre-reset final state."""

    def __init__(self, env, output_dir):
        self.env = env
        self.output_dir = Path(output_dir).resolve()
        self.command = env.command_manager.get_term("motion")
        self.motion_cfg = self.command.cfg.motion_lib_cfg
        self.source_dir = Path(self.motion_cfg["motion_file"]).parent
        self.frames = [[] for _ in range(env.num_envs)]
        self.saved = set()
        self.finished = set()
        self.motion_keys = [
            str(self.command.motion_lib.curr_motion_keys[motion_id])
            for motion_id in self.command.motion_ids.tolist()
        ]
        self.dof_axis = self.command.motion_lib.mesh_parsers.dof_axis.cpu().numpy()
        robot = env.scene["robot"].data
        origins = env.scene.env_origins
        release_goal, _ = self.command.get_release_target_pose()
        initial = [robot.root_pos_w - origins, robot.root_quat_w[:, [1, 2, 3, 0]],
                   env.scene["object"].data.root_pos_w - origins, release_goal - origins]
        self.metrics = RolloutReport(*(value.detach().cpu().numpy().copy() for value in initial))
        report_path = self.output_dir / "rollout_report.json"
        self.rollout_reports = json.loads(report_path.read_text()) if report_path.is_file() else {}
        for folder in ("robot", "objects", "meta", "bps", "object_usd", "reports"):
            (self.output_dir / folder).mkdir(parents=True, exist_ok=True)

    def record_contacts(self):
        sensors = [self.env.scene[name] for name in (
            "rollout_left_foot_ground_contact", "rollout_right_foot_ground_contact", "rollout_robot_table_contact",
        )]
        contacts = []
        for sensor in sensors:
            # This hook runs immediately after PhysX stepping, before scene.update().
            sensor.update(dt=0.0, force_recompute=True)
            force = sensor.data.force_matrix_w.norm(dim=-1).sum(dim=(1, 2))
            contacts.append((force > 1.0).detach().cpu().numpy())
        active = np.array([i not in self.finished for i in range(self.env.num_envs)])
        self.metrics.update_contacts(np.stack(contacts[:2], axis=-1), contacts[2], active)

    def record(self):
        robot = self.env.scene["robot"].data
        obj = self.env.scene["object"].data
        origins = self.env.scene.env_origins
        object_pose = obj.root_state_w[:, :7]  # rigid-body actor pose, quaternion wxyz
        body_indices = self.command.body_joint_indices[self.command.isaaclab_to_mujoco_dof]
        state = {
            "dof": robot.joint_pos[:, body_indices],
            "hand_dof_pos": robot.joint_pos[:, self.command.extra_joint_indices],
            "root_trans_offset": robot.root_pos_w - origins,
            "root_rot": robot.root_quat_w[:, [1, 2, 3, 0]],  # robot PKL: xyzw
            "object_pos": object_pose[:, :3] - origins,
            "object_quat": object_pose[:, 3:7],
            "hand_actions": self.env._last_meta_action[:, -2:],
        }
        state = {key: value.detach().cpu().numpy().copy() for key, value in state.items()}
        active = np.array([i not in self.finished for i in range(self.env.num_envs)])
        self.metrics.update_motion(state, active)
        for env_id in range(self.env.num_envs):
            if env_id in self.finished:
                continue
            self.frames[env_id].append({key: value[env_id] for key, value in state.items()})
            if self.env.reset_buf[env_id]:
                if self.env.termination_manager.time_outs[env_id]:
                    self.save(env_id)
                self.finished.add(env_id)
                self.frames[env_id] = []

    def save(self, env_id):
        if env_id in self.saved or not self.frames[env_id]:
            return
        key = self.motion_keys[env_id]
        robot_payload = joblib.load(Path(self.motion_cfg["motion_file"]) / f"{key}.pkl")
        object_payload = joblib.load(Path(self.motion_cfg["object_motion_file"]) / f"{key}.pkl")
        robot_key, source_robot = next(iter(robot_payload.items()))
        object_key, source_object = next(iter(object_payload.items()))
        # Sparse yaw repeats each stage's signed total angle at every stage frame.
        # Count each stage once, using the full source even if the rollout ends early.
        yaw_commands = source_robot["policy_command_x_yaw"][:, 1]
        stage_starts = np.r_[True, yaw_commands[1:] != yaw_commands[:-1]]
        reference_yaw_change = float(np.sum(yaw_commands[stage_starts], dtype=np.float64))
        fps = float(source_robot["fps"])
        frames = self.frames[env_id]
        # A source frame spans two 50 Hz policy steps for this 25 Hz dataset.
        # Keep the state at each interval's end; include a partial terminal interval.
        count = int(np.ceil(len(frames) * self.env.step_dt * fps - 1e-9))
        indices = np.minimum(
            np.rint((np.arange(count) + 1) / (fps * self.env.step_dt)).astype(int) - 1,
            len(frames) - 1,
        )
        source_indices = np.minimum(np.arange(count), len(source_robot["dof"]) - 1)
        states = {
            field: np.stack([frame[field] for frame in frames])[indices]
            for field in frames[0]
        }
        # Preserve the source schema, sparse commands and zero SMPL placeholders.
        robot = {
            field: value[source_indices].copy() if isinstance(value, np.ndarray) else value
            for field, value in source_robot.items()
        }
        for field in ("dof", "hand_dof_pos", "root_trans_offset", "root_rot"):
            robot[field] = states[field].astype(source_robot[field].dtype)
        robot["pose_aa"] = np.concatenate(
            [
                Rotation.from_quat(robot["root_rot"]).as_rotvec()[:, None],
                robot["dof"][..., None] * self.dof_axis[None],
            ],
            axis=1,
        ).astype(source_robot["pose_aa"].dtype)
        for hand_id, hand in enumerate(("left", "right")):
            field = f"hand_action_{hand}"
            robot[field] = np.where(states["hand_actions"][:, hand_id] >= 0, 1, -1).astype(
                source_robot[field].dtype
            )
        # Construct object trajectories only from the same sampled simulation steps
        # as the robot. Source object poses must never enter the exported trajectory.
        objects = {
            "root_pos": states["object_pos"][:, None].astype(source_object["root_pos"].dtype),
            "root_quat": states["object_quat"][:, None].astype(source_object["root_quat"].dtype),
            "fps": fps,
        }
        if "scale" in source_object:
            objects["scale"] = source_object["scale"]
        # Contact points are object-local surface labels; retain their source timing.
        for hand in ("left", "right"):
            field = f"contact_points_{hand}_hand"
            objects[field] = {
                frame: source_object[field][int(source_frame)].copy()
                for frame, source_frame in enumerate(source_indices)
                if int(source_frame) in source_object[field]
            }
        joblib.dump({robot_key: robot}, self.output_dir / "robot" / f"{key}.pkl")
        joblib.dump({object_key: objects}, self.output_dir / "objects" / f"{key}.pkl")

        meta = joblib.load(self.source_dir / "meta" / f"{key}.pkl")
        meta["total_frames"] = count
        for field, value in meta.items():
            if field.endswith("_frame") and "source" not in field:
                meta[field] = min(value, count - 1)
            elif field.endswith("_frame_exclusive"):
                meta[field] = min(value, count)
        # Match the dataset semantics: release goal is the object pose immediately
        # before release; carry goal is its pose at the retained carry source frame.
        for prefix, frame in (
            ("place_object_goal", max(meta["place_release_start_frame"] - 1, 0)),
            ("place_object_carry_goal", min(meta["place_carry_source_frame"], count - 1)),
        ):
            meta[f"{prefix}_pos_w"] = objects["root_pos"][frame, 0].copy()
            meta[f"{prefix}_quat_w"] = objects["root_quat"][frame, 0].copy()
        joblib.dump(meta, self.output_dir / "meta" / f"{key}.pkl")
        report = json.loads((self.source_dir / "reports" / f"{key}.json").read_text())
        report.update(
            total_frames=count, fps=fps,
            object_translation_policy="teacher rollout actor position relative to environment origin",
            object_orientation_policy="teacher rollout actor rotation, wxyz",
        )
        report["validation"].update(
            total_frames=count, fps=fps,
            usd_path=str(self.output_dir / "object_usd" / f"{key}.usd"),
        )
        report["rollout"] = self.metrics.result(env_id, reference_yaw_change)
        (self.output_dir / "reports" / f"{key}.json").write_text(json.dumps(report, indent=2) + "\n")
        self.rollout_reports[key] = report["rollout"]
        (self.output_dir / "rollout_report.json").write_text(json.dumps(self.rollout_reports, indent=2) + "\n")

        bps_dir = Path(self.motion_cfg["bps_dir"])
        shutil.copy2(bps_dir / f"{key}.npy", self.output_dir / "bps")
        if (bps_dir / "_basis.npy").exists():
            shutil.copy2(bps_dir / "_basis.npy", self.output_dir / "bps")
        usd_dir = Path(self.env.cfg.config["object_usd_path"])
        shutil.copy2(usd_dir / f"{key}.usd", self.output_dir / "object_usd")
        texture_dir = usd_dir / "textures" / key
        if texture_dir.is_dir():
            shutil.copytree(
                texture_dir, self.output_dir / "object_usd" / "textures" / key,
                dirs_exist_ok=True,
            )
        self.saved.add(env_id)
        self.frames[env_id] = []
        logger.info(f"Saved teacher rollout: {key} ({count} frames at {fps:g} FPS)")

    def close(self):
        # Timeout episodes are saved in record(); never export unfinished episodes here.
        self.frames = [[] for _ in range(self.env.num_envs)]
