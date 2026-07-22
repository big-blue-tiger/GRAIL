"""Coordinate utilities shared by training data and online inference.

SONIC records quaternions in scalar-first (wxyz) order.  Rotation 6D follows
the existing observation convention: the first two *columns* of the matrix,
flattened in row-major order via ``matrix[..., :2].reshape(..., 6)``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def normalize_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    return F.normalize(quaternion, dim=-1, eps=1e-8)


def quaternion_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion_wxyz(quaternion)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quaternion_multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quaternion_inverse_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion_wxyz(quaternion)
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def rotation_6d_columns(quaternion: torch.Tensor) -> torch.Tensor:
    matrix = quaternion_to_matrix_wxyz(quaternion)
    return matrix[..., :2].reshape(quaternion.shape[:-1] + (6,))


def world_pose_to_body(
    robot_position_w: torch.Tensor,
    robot_quaternion_w: torch.Tensor,
    object_position_w: torch.Tensor,
    object_quaternion_w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Express a world-frame pose in the current robot root/body frame."""
    robot_rotation_w = quaternion_to_matrix_wxyz(robot_quaternion_w)
    position_b = torch.matmul(
        robot_rotation_w.transpose(-1, -2),
        (object_position_w - robot_position_w).unsqueeze(-1),
    ).squeeze(-1)
    quaternion_b = quaternion_multiply_wxyz(
        quaternion_inverse_wxyz(robot_quaternion_w),
        normalize_quaternion_wxyz(object_quaternion_w),
    )
    return position_b, rotation_6d_columns(quaternion_b)


def body_pose_to_world(
    robot_position_w: torch.Tensor,
    robot_quaternion_w: torch.Tensor,
    object_position_b: torch.Tensor,
) -> torch.Tensor:
    """Position-only inverse used by validation and downstream diagnostics."""
    rotation_w = quaternion_to_matrix_wxyz(robot_quaternion_w)
    return robot_position_w + torch.matmul(rotation_w, object_position_b.unsqueeze(-1)).squeeze(-1)
