"""Reference-free scene state, observations and reset handling for student play."""

from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg, RecorderTerm, RecorderTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat, quat_from_euler_xyz, quat_mul,
)

from gear_sonic.envs.env_utils.joint_utils import get_body_joint_indices, get_hand_joint_indices
from gear_sonic.envs.manager_env.mdp.observations import target_object_poses_6d  # compatibility re-export
from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
from gear_sonic.utils.reference_free_plan import (
    CarryPlannerConfig, PLAN_FPS, sample_carry_plan, table_outward_normal,
)
from gear_sonic.utils.reference_free_task import OfflineCarryTask


class ReferenceFreeCommand(CommandTerm):
    """Own static object geometry and sampled goals, without a motion library."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.robot_anchor_body_index = self.robot.find_bodies("pelvis")[0][0]
        self.motion_lib = None
        self.objects = cfg.objects
        self.results = []

        def tensor(values):
            return torch.tensor(values, device=self.device, dtype=torch.float32)

        self.object_bps = tensor([item["bps"] for item in cfg.objects])
        self.table_pos_w = tensor([item["table_position"] for item in cfg.objects]) + env.scene.env_origins
        # Match TrackingCommand: object and table poses are consumed as wxyz.
        self.table_quat_w = tensor([item["table_quaternion"] for item in cfg.objects])
        self.object_start_pos_w = tensor([item["position"] for item in cfg.objects]) + env.scene.env_origins
        self.object_rest_pos_w = self.object_start_pos_w.clone()
        self.object_start_quat_w = tensor([item["quaternion"] for item in cfg.objects])
        self.goal_quat_w = tensor([item["quaternion"] for item in cfg.objects])
        self.goal_pos_w = self.object_start_pos_w.clone()
        self.carry_pos_w = self.goal_pos_w.clone()
        self.carry_quat_w = self.goal_quat_w.clone()
        self.robot_start_pos_w = torch.zeros_like(self.goal_pos_w)
        self.down_dir = tensor([0.0, 0.0, -1.0]).expand(self.num_envs, -1)
        self.body_indices = get_body_joint_indices(self.robot)
        self.hand_indices = get_hand_joint_indices(self.robot)
        body_order = env.cfg.isaaclab_to_mujoco_mapping["mujoco_to_isaaclab_dof"]
        self.initial_joints = self.robot.data.default_joint_pos.clone()
        self.initial_joints[:, self.body_indices] = tensor(cfg.initial_pose["dof"])[body_order]
        self.initial_joints[:, self.hand_indices] = tensor(cfg.initial_pose["hand_dof_pos"])
        # Only the exported robot root_rot uses xyzw.
        self.initial_quat = tensor(cfg.initial_pose["root_rot"])[[3, 0, 1, 2]]
        self.initial_yaw = euler_xyz_from_quat(self.initial_quat.unsqueeze(0))[2][0]
        self.initial_height = cfg.initial_pose["root_trans_offset"][2] + cfg.init_z_offset
        self.task = None
        if cfg.carry_task:
            self.planner = CarryPlannerConfig(**cfg.planner)
            self.rng = np.random.default_rng(torch.initial_seed())
            self.sampling_normals = tensor([
                table_outward_normal(item["table_quaternion"], item["table_size"], [0.0, 1.0]).tolist()
                for item in cfg.objects
            ])
            self.sampling_tangents = torch.stack((self.sampling_normals[:, 1], -self.sampling_normals[:, 0]), -1)
            self.plans = [None] * self.num_envs
            self.task = OfflineCarryTask(
                self.num_envs, self.device, env.step_dt, self.planner.pickup_settle_s,
                cfg.success_threshold, env.cfg.config.get("lift_contact_force_threshold", 0.1),
            )
            self.sparse_command_state = self.task.contacts
        self.visualizer = None
        if cfg.debug_vis:
            self.visualizer = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/ReferenceFree",
                markers={
                    "robot_range": sim_utils.CuboidCfg(
                        size=(1.0, 1.0, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 1.0)),
                    ),
                    "object_range": sim_utils.CuboidCfg(
                        size=(1.0, 1.0, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.7, 0.1)),
                    ),
                    "goal": sim_utils.SphereCfg(
                        radius=0.035,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                    ),
                    "carry_goal": sim_utils.SphereCfg(
                        radius=0.035,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
                    ),
                },
            ))

    @property
    def command(self):
        return self.goal_pos_w

    @property
    def robot_body_pos_w(self):
        return self.robot.data.body_pos_w

    @property
    def robot_anchor_pos_w(self):
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self):
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    def get_release_target_pose(self):
        return self.goal_pos_w, self.goal_quat_w

    def get_carry_target_pose(self):
        return self.carry_pos_w, self.carry_quat_w

    def get_sparse_policy_commands(self):
        return self.task.read()

    def is_place_generated_phase(self):
        """Use sparse playback's first step as the online generated-phase boundary."""
        if self.task is None:
            raise ValueError("Place phase observations require carry_task=True")
        return self.task.has_playback_started()

    def advance_task(self):
        """Called by the termination term after physics and before auto-reset."""
        hand_force = torch.linalg.vector_norm(
            self._env.scene["object_to_hand_contact_sensor"].data.force_matrix_w, dim=-1
        ).sum(dim=(1, 2))
        table_force = torch.linalg.vector_norm(
            self._env.scene["object_to_table_contact_sensor"].data.force_matrix_w, dim=-1
        ).sum(dim=(1, 2))
        self.task.update(
            self._env.episode_length_buf, hand_force, table_force,
            self._env.scene["object"].data.root_pos_w, self.goal_pos_w,
        )

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = len(env_ids)
        if not count:
            return
        samples = torch.rand(count, 4, device=self.device)
        robot_pos = self.table_pos_w[env_ids].clone()
        robot_pos[:, 0] += 2.0 * samples[:, 0] - 1.0
        robot_pos[:, 1] += 1.5
        if self.task is not None:
            # Preserve the old +Y-side sampling for ordinary tables, rotating
            # both sampling ranges together for tables with other headings.
            normals = self.sampling_normals[env_ids]
            tangents = self.sampling_tangents[env_ids]
            robot_pos[:, :2] = (
                self.table_pos_w[env_ids, :2] + tangents * (2*samples[:, :1]-1) + normals*1.5
            )
        robot_pos[:, 2] = self._env.scene.env_origins[env_ids, 2] + self.initial_height
        direction = self.table_pos_w[env_ids, :2] - robot_pos[:, :2]
        yaw_delta = torch.atan2(direction[:, 1], direction[:, 0]) - self.initial_yaw
        zeros = torch.zeros_like(yaw_delta)
        robot_quat = quat_mul(
            quat_from_euler_xyz(zeros, zeros, yaw_delta), self.initial_quat.expand(count, -1)
        )
        object_pos = self.object_rest_pos_w[env_ids].clone()
        object_pos[:, 0] = self.table_pos_w[env_ids, 0] + samples[:, 1] - 0.5
        object_pos[:, 1] = self.table_pos_w[env_ids, 1] + 0.25 * samples[:, 2]
        if self.task is not None:
            object_pos[:, :2] = (
                self.table_pos_w[env_ids, :2] + tangents*(samples[:, 1:2]-0.5) + normals*(0.25*samples[:, 2:3])
            )
        self.robot_start_pos_w[env_ids] = robot_pos
        self.object_start_pos_w[env_ids] = object_pos
        self.goal_pos_w[env_ids] = object_pos
        self.goal_pos_w[env_ids, 2] += 0.4 + 0.3 * samples[:, 3]
        if self.task is not None:
            plans = []
            object_positions = object_pos.cpu().numpy()
            yaws = torch.atan2(direction[:, 1], direction[:, 0]).cpu().numpy()
            normals_cpu = normals.cpu().numpy()
            for local_id, env_id in enumerate(env_ids.tolist()):
                plan = sample_carry_plan(
                    object_positions[local_id], self.objects[env_id]["quaternion"], yaws[local_id],
                    normals_cpu[local_id], self.rng, self.planner,
                )
                self.plans[env_id] = plan
                plans.append(plan)
            self.task.reset(env_ids, plans)
            for attribute, field in (
                ("goal_pos_w", "release_position"), ("goal_quat_w", "release_quaternion"),
                ("carry_pos_w", "carry_position"), ("carry_quat_w", "carry_quaternion"),
            ):
                getattr(self, attribute)[env_ids] = torch.as_tensor(
                    np.stack([getattr(plan, field) for plan in plans]), device=self.device,
                )
            # Height compensation only affects simulated reset poses.
            object_pos[:, 2] += self.cfg.object_init_z_offset
            self.object_start_pos_w[env_ids] = object_pos
        else:
            self.carry_pos_w[env_ids] = self.goal_pos_w[env_ids]
            self.carry_quat_w[env_ids] = self.goal_quat_w[env_ids]
        zero_vel = torch.zeros(count, 6, device=self.device)
        self.robot.write_joint_state_to_sim(
            self.initial_joints[env_ids], torch.zeros_like(self.initial_joints[env_ids]), env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(torch.cat((robot_pos, robot_quat, zero_vel), -1), env_ids=env_ids)
        self._env.scene["object"].write_root_state_to_sim(
            torch.cat((object_pos, self.object_start_quat_w[env_ids], zero_vel), -1), env_ids=env_ids
        )
        self._env.scene["table"].write_root_state_to_sim(
            torch.cat((self.table_pos_w[env_ids], self.table_quat_w[env_ids], zero_vel), -1), env_ids=env_ids
        )

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis):
        if getattr(self, "visualizer", None) is not None:
            self.visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if self.visualizer is None:
            return
        centers = self.table_pos_w[:, None, :].repeat(1, 7, 1)
        centers[:, 0, 1] += 1.5
        centers[:, 0, 2] = self._env.scene.env_origins[:, 2] + 0.015
        centers[:, 1:5, 2] = self.object_start_pos_w[:, None, 2]
        centers[:, 2, 1] += 0.25
        centers[:, 3, 0] -= 0.5
        centers[:, 4, 0] += 0.5
        centers[:, 3:5, 1] += 0.125
        centers[:, 5] = self.goal_pos_w
        centers[:, 6] = self.carry_pos_w if self.task is not None else self.goal_pos_w
        scales = torch.tensor([
            [2.0, 0.015, 0.015], [1.0, 0.01, 0.01], [1.0, 0.01, 0.01],
            [0.01, 0.25, 0.01], [0.01, 0.25, 0.01], [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0] if self.task is not None else [0.0, 0.0, 0.0],
        ], device=self.device).repeat(self.num_envs, 1)
        orientations = torch.zeros(self.num_envs, 7, 4, device=self.device)
        orientations[..., 0] = 1
        if self.task is not None:
            relative_xy = centers[:, :5, :2] - self.table_pos_w[:, None, :2]
            centers[:, :5, :2] = (
                self.table_pos_w[:, None, :2]
                + relative_xy[..., :1] * self.sampling_tangents[:, None, :]
                + relative_xy[..., 1:2] * self.sampling_normals[:, None, :]
            )
            marker_yaw = torch.atan2(self.sampling_tangents[:, 1], self.sampling_tangents[:, 0])
            zeros = torch.zeros_like(marker_yaw)
            orientations[:, :5] = quat_from_euler_xyz(zeros, zeros, marker_yaw)[:, None, :]
        self.visualizer.visualize(
            translations=centers.reshape(-1, 3), scales=scales,
            orientations=orientations.reshape(-1, 4),
            marker_indices=torch.tensor([0, 1, 1, 1, 1, 2, 3], device=self.device).repeat(self.num_envs),
        )


@configclass
class ReferenceFreeCommandCfg(CommandTermCfg):
    class_type: type = ReferenceFreeCommand
    objects: list = MISSING
    initial_pose: dict = MISSING
    init_z_offset: float = 0.05
    object_init_z_offset: float = 0.0
    carry_task: bool = False
    planner: dict = {}
    success_threshold: float = 0.1
    resampling_time_range: tuple = (1.0e9, 1.0e9)


def goal_reached(env, terminate_on_success=True):
    """Advance carry state before reset, even when early success resets are off."""
    command = env.command_manager.get_term("motion")
    if command.task is not None:
        command.advance_task()
        return command.task.success.clone() if terminate_on_success else torch.zeros_like(command.task.success)
    distance = torch.linalg.vector_norm(env.scene["object"].data.root_pos_w - command.goal_pos_w, dim=-1)
    return distance < command.cfg.success_threshold


class ReferenceFreeResultRecorder(RecorderTerm):
    """Capture terminal positions before Isaac Lab automatically resets the scene."""

    def record_post_step(self):
        env = self._env
        command = env.command_manager.get_term("motion")
        for env_id in env.reset_buf.nonzero(as_tuple=False).flatten().tolist():
            final_pos = env.scene["object"].data.root_pos_w[env_id]
            distance = torch.linalg.vector_norm(final_pos - command.goal_pos_w[env_id]).item()
            result = {
                "env_id": env_id,
                "object_key": command.objects[env_id]["key"],
                "robot_initial_position": command.robot_start_pos_w[env_id].tolist(),
                "object_initial_position": command.object_start_pos_w[env_id].tolist(),
                "target_position": command.goal_pos_w[env_id].tolist(),
                "final_position": final_pos.tolist(),
                "final_distance": distance,
                "elapsed_seconds": env.episode_length_buf[env_id].item() * env.step_dt,
                "success": distance < command.cfg.success_threshold,
                "termination_reason": "time_out" if env.termination_manager.time_outs[env_id].item() else "goal_reached",
            }
            if command.task is not None:
                result.update(command.task.result(env_id))
                plan = command.plans[env_id]
                result.update({
                    "carry_target_position": command.carry_pos_w[env_id].tolist(),
                    "carry_target_quaternion": command.carry_quat_w[env_id].tolist(),
                    "target_quaternion": command.goal_quat_w[env_id].tolist(),
                    "plan": {
                        "fps": PLAN_FPS,
                        "turn_angle": plan.turn_angle,
                        "straight_length": plan.straight_length,
                        "turn_end_seconds": plan.turn_end_frame / PLAN_FPS,
                        "walk_end_seconds": plan.walk_end_frame / PLAN_FPS,
                        "drop_start_seconds": plan.release_start_frame / PLAN_FPS,
                    },
                })
            command.results.append(result)
        return None, None


@configclass
class ReferenceFreeResultRecorderCfg(RecorderTermCfg):
    class_type: type = ReferenceFreeResultRecorder


class ReferenceFreeEnvWrapper(ManagerEnvWrapper):
    """Reuse the direct-latent decoder without reference observation or replay hooks."""

    def process_raw_obs(self, obs, flatten_dict_obs=True):
        return dict(obs)

    def set_is_evaluating(self, is_evaluating=True, **kwargs):
        self.is_evaluating = is_evaluating
