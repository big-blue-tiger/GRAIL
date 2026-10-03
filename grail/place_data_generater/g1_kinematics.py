"""G1 XML joint angles and Kimodo skeleton poses, with explicit axis conventions.

GRAIL/MuJoCo qpos angles are rotations from the XML joint zero. Kimodo's
skeletal local matrices include the XML body's static rotation. Applying a
world transform to qpos must never project or re-extract its joint angles.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from grail.walk_data_tool.concat_walk_motion import ASSET, CSV_JOINTS, joint_mapping

MUJOCO_TO_KIMODO = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
KINEMATICS_VERSION = "g1_xml_joint_zero_v2"


class G1QposAdapter:
    def __init__(self, skeleton):
        """建立 29 个 CSV 关节到骨架的映射,读取 XML 关节轴并校验静态旋转。"""
        self.skeleton = skeleton
        self.indices = [skeleton.bone_index[name.replace("_joint", "_skel")]
                        for name in CSV_JOINTS]
        self.rest = skeleton.rest_pose_local_rot.detach().cpu().double().clone()
        xml = ET.parse(ASSET).getroot()
        parents = {child: parent for parent in xml.iter() for child in parent}
        joints = {j.get("name"): j for j in xml.find("worldbody").iter("joint")}
        axes = []
        for name, index in zip(CSV_JOINTS, self.indices):
            joint = joints[name]
            axis = np.fromstring(joint.get("axis"), sep=" ")
            axis /= np.linalg.norm(axis)
            axes.append(MUJOCO_TO_KIMODO @ axis)
            body = parents[joint]
            quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
            offset = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            expected = MUJOCO_TO_KIMODO @ offset @ MUJOCO_TO_KIMODO.T
            if not np.allclose(self.rest[index].numpy(), expected, atol=2e-5):
                raise ValueError(f"Kimodo/GRAIL XML rest rotation differs at {name}")
        self.axes = np.asarray(axes)
        self.hand_indices = [skeleton.bone_index[name] for name in
                             ("right_wrist_yaw_skel", "right_hand_roll_skel")]

    def local_pose(self, qpos, *, quat_order="wxyz"):
        """将 Z-up qpos (T,36) 转为 Kimodo Y-up 局部旋转 (T,J,3,3) 和 root (T,3)。

        qpos 为 xyz + 指定顺序的 root 四元数 + 29 个 CSV 关节角。
        返回 CPU float64 Tensor,局部旋转包含 XML 静态旋转,位置单位米。
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or not np.isfinite(qpos).all():
            raise ValueError("Expected finite (T,36) G1 qpos")
        if quat_order not in {"wxyz", "xyzw"}:
            raise ValueError("quat_order must be wxyz or xyzw")
        quat = qpos[:, [4, 5, 6, 3]] if quat_order == "wxyz" else qpos[:, 3:7]
        if not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-3):
            raise ValueError("G1 qpos quaternion is not unit length")
        local = self.rest[None].repeat(len(qpos), 1, 1, 1)
        for i, index in enumerate(self.indices):
            rotation = Rotation.from_rotvec(qpos[:, 7+i, None] * self.axes[i]).as_matrix()
            local[:, index] = self.rest[index] @ torch.from_numpy(rotation)
        root_rot = MUJOCO_TO_KIMODO @ Rotation.from_quat(quat).as_matrix() @ MUJOCO_TO_KIMODO.T
        local[:, self.skeleton.root_idx] = torch.from_numpy(root_rot)
        roots = torch.from_numpy(qpos[:, :3] @ MUJOCO_TO_KIMODO.T)
        return local, roots

    def fk(self, qpos, *, quat_order="wxyz"):
        """返回 Kimodo Y-up 世界旋转 (T,J,3,3) 和位置 (T,J,3),均为 float64。"""
        local, roots = self.local_pose(qpos, quat_order=quat_order)
        rots, positions, _ = self.skeleton.fk(local, roots)
        return rots.numpy(), positions.numpy()

    def joint_angles(self, local):
        """去除单帧局部姿态 (J,3,3) 的 XML 静态旋转,提取 29 个 CSV 关节角。"""
        matrices = (self.rest[self.indices].transpose(-1, -2) @ local[self.indices]).numpy()
        skew = np.column_stack([matrices[:, 2, 1] - matrices[:, 1, 2],
                                matrices[:, 0, 2] - matrices[:, 2, 0],
                                matrices[:, 1, 0] - matrices[:, 0, 1]])
        sin = np.sum(skew * self.axes, axis=1)
        cos = np.trace(matrices, axis1=1, axis2=2) - np.einsum("ij,ijk,ik->i", self.axes, matrices, self.axes)
        return np.arctan2(sin, cos)

    def lower_hand(self, qpos, goal_local, goal_root):
        """将源 xyzw qpos 的姿态放到终点,保留关节角、root 高度及倾斜。

        返回下放局部旋转 (J,3,3) 和 Y-up root (3,),水平位置与朝向对齐终点。
        """
        from kimodo.motion_rep.feature_utils import compute_heading_angle

        local, roots = self.local_pose(np.asarray(qpos)[None], quat_order="xyzw")
        _, source_positions, _ = self.skeleton.fk(local, roots)
        _, goal_positions, _ = self.skeleton.fk(goal_local[None], goal_root[None])
        source_heading = float(compute_heading_angle(source_positions[None], self.skeleton)[0, 0])
        goal_heading = float(compute_heading_angle(goal_positions[None], self.skeleton)[0, 0])
        rotation = Rotation.from_euler("y", goal_heading - source_heading).as_matrix()
        lowered = local[0].clone()
        lowered[self.skeleton.root_idx] = lowered.new_tensor(rotation) @ lowered[self.skeleton.root_idx]
        root = goal_root.double().clone()
        root[1] = roots[0, 1]
        return lowered, root


def source_qpos(motion):
    """将原动作转为 Z-up qpos (N,36):xyz + xyzw + 29 个 CSV 顺序关节角。"""
    from grail.walk_data_tool.export_kimodo_end_frame import trajectory_qpos

    values = trajectory_qpos(motion)
    # GRAIL's XML traversal order need not equal Kimodo's CSV joint order.
    indices = joint_mapping()[1]
    values[:, 7:] = values[:, 7:][:, np.argsort(indices)]
    return values


def qpos_to_world(qpos, record):
    """用 record 中的 Y-up 刚体变换恢复 Z-up 世界 qpos (T,36,wxyz)。

    仅改变 root 位置和旋转,29 个关节角逐值保留。
    """
    c2w = np.asarray(record["canonical_to_world_rotation_kimodo"], dtype=float)
    origin = np.asarray(record["canonical_to_world_origin_kimodo"], dtype=float)
    if (c2w.shape != (3, 3) or origin.shape != (3,) or not np.isfinite(c2w).all()
            or not np.isfinite(origin).all() or not np.allclose(c2w.T @ c2w, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(c2w), 1., atol=1e-6)):
        raise ValueError("Invalid canonical-to-world transform")
    rotation = MUJOCO_TO_KIMODO.T @ c2w @ MUJOCO_TO_KIMODO
    translation = origin @ MUJOCO_TO_KIMODO
    result = np.asarray(qpos, dtype=float).copy()
    result[:, :3] = result[:, :3] @ rotation.T + translation
    quats = (Rotation.from_matrix(rotation) * Rotation.from_quat(result[:, [4, 5, 6, 3]])).as_quat()
    result[:, 3:7] = quats[:, [3, 0, 1, 2]]
    return result
