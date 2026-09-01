"""Small rigid-body kinematics helpers shared by observation terms."""

from __future__ import annotations

import torch


def relative_twist_at_target_origin_w(
    frame_pos_w: torch.Tensor,
    frame_lin_vel_w: torch.Tensor,
    frame_ang_vel_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_lin_vel_w: torch.Tensor,
    target_ang_vel_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a target twist relative to a moving frame, in world axes."""
    offset_w = target_pos_w - frame_pos_w
    frame_velocity_at_target_w = frame_lin_vel_w + torch.cross(
        frame_ang_vel_w, offset_w, dim=-1
    )
    return (
        target_lin_vel_w - frame_velocity_at_target_w,
        target_ang_vel_w - frame_ang_vel_w,
    )
