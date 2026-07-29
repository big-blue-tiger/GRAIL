from __future__ import annotations

from pathlib import Path

import dill
import hydra
import torch

from sugar_il.common.geometry import world_pose_to_body
class GeneratorWrapper:
    """Inference adapter with the same world-to-body transform as the dataset."""

    def __init__(
        self,
        policy,
        device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device)
        self.policy = policy.eval().to(self.device)

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str | torch.device = "cuda"):
        with Path(checkpoint_path).open("rb") as file:
            payload = torch.load(file, pickle_module=dill, map_location="cpu", weights_only=False)
        policy = hydra.utils.instantiate(payload["cfg"].policy)
        policy.load_state_dict(payload["state_dicts"]["model"])
        return cls(policy, device)

    @staticmethod
    def _time_axis(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(1) if value.ndim == 2 else value

    def observation_from_world(
        self,
        *,
        object_bps: torch.Tensor,
        table_geometry: torch.Tensor,
        robot_position_w: torch.Tensor,
        robot_quaternion_w: torch.Tensor,
        object_position_w: torch.Tensor,
        object_quaternion_w: torch.Tensor,
        hand_object_transform_6d: torch.Tensor,
        hand_object_contact_force_magnitude: torch.Tensor,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tensors = [object_bps, table_geometry, robot_position_w, robot_quaternion_w, object_position_w, object_quaternion_w, hand_object_transform_6d, hand_object_contact_force_magnitude, base_lin_vel, base_ang_vel, joint_pos, joint_vel]
        tensors = [value.to(self.device, dtype=torch.float32) for value in tensors]
        (object_bps, table_geometry, robot_position_w, robot_quaternion_w, object_position_w, object_quaternion_w, hand_object_transform_6d, hand_object_contact_force_magnitude, base_lin_vel, base_ang_vel, joint_pos, joint_vel) = tensors
        object_pos_b, object_ori_b = world_pose_to_body(robot_position_w, robot_quaternion_w, object_position_w, object_quaternion_w)
        return {key: self._time_axis(value) for key, value in {
            "object_bps": object_bps,
            "table_geometry": table_geometry,
            "object_pos_b": object_pos_b,
            "object_ori_b_6d": object_ori_b,
            "hand_object_transform_6d": hand_object_transform_6d,
            "hand_object_contact_force_magnitude": hand_object_contact_force_magnitude,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }.items()}

    @torch.no_grad()
    def predict_from_world(self, **world_observation):
        prediction = self.policy.predict_action(
            self.observation_from_world(**world_observation)
        )
        return prediction
