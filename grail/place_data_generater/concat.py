#!/usr/bin/env python3
"""Keep the source through its carry pose, then append the Kimodo suffix."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from grail.walk_data_tool.concat_walk_motion import (
    CONTACT_FIELDS, concat_motion, file_hash, joint_mapping, load_single, validate_fields,
)
from grail.walk_data_tool.export_kimodo_end_frame import load_kinematics
from grail.place_data_generater.constraints import DATA, KIMODO, MODEL_FPS, contact_timing
from grail.place_data_generater.g1_kinematics import (
    G1QposAdapter, KINEMATICS_VERSION, MUJOCO_TO_KIMODO, qpos_to_world, source_qpos,
)


COMMAND_FIELDS = {"policy_command_x_yaw", "policy_command_xy_yaw", "policy_command_drop_button"}
MOTION_META_FIELDS = {
    "table_pos", "table_quat", "table_size", "total_frames",
    "transition_start_frame", "transition_end_frame", "grail_start_frame",
    "residual_fade_out_start_frame", "residual_fade_out_end_frame",
    "place_generated_start_frame", "place_carry_source_frame",
    "place_release_start_frame", "place_output_contact_start_frame", "place_turn_direction",
    "place_object_goal_pos_w", "place_object_goal_quat_w",
    "place_object_carry_goal_pos_w", "place_object_carry_goal_quat_w",
}
YAW_TURN_THRESHOLD_RAD = 0.2
YAW_JITTER_RAD = 0.02
YAW_PAUSE_S = 0.2
YAW_LIMIT_RAD = 1.5 * np.pi


def turn_stage_commands(yaw, fps):
    """Fill each same-direction turn with its signed total angle.

    Unwrap headings, merge same-direction runs across gaps of at most 0.2 s
    with at most 0.02 rad deviation, and retain turns strictly above 0.2 rad.
    A stage spans [start,end): end is the heading endpoint, not another turn
    frame. Small pauses/reversals inside a merged stage share its command.
    """
    yaw = np.unwrap(yaw)
    delta = np.diff(yaw)
    signs = np.sign(np.where(np.abs(delta) > 1e-6, delta, 0))
    edges = np.r_[0, np.flatnonzero(np.diff(signs)) + 1, len(delta)]
    runs = [(int(s), int(e)) for s, e in zip(edges[:-1], edges[1:]) if s < e]
    commands = np.zeros(len(yaw), dtype=np.float32)
    stages = []
    i = 0
    while i < len(runs):
        start, end = runs[i]
        direction = signs[start]
        if direction == 0:
            i += 1
            continue
        j = i + 1
        while j < len(runs):
            next_start, next_end = runs[j]
            if next_start - end > round(YAW_PAUSE_S * fps):
                break
            if np.max(np.abs(yaw[end:next_start + 1] - yaw[end])) > YAW_JITTER_RAD:
                break
            if signs[next_start] == direction:
                end = next_end
                i = j
            j += 1
        total = float(yaw[end] - yaw[start])
        if abs(total) > YAW_TURN_THRESHOLD_RAD:
            commands[start:end] = np.clip(total, -YAW_LIMIT_RAD, YAW_LIMIT_RAD)
            stages.append((start, end))
        i += 1
    return commands, stages


def generate_sparse_commands(robot, objects, meta, name, *, release_start_frame, lookahead_s=0.5):
    """Generate carry displacements and release buttons on the final timeline.

    Read robot positions [T,3] and xyzw root quaternions [T,4]. Require annotated
    boundaries 0 <= a < b < T. Return float32 [dx,dyaw] commands [T,2] in meters/
    radians and buttons [T,1]; commands are active only in [a,b), and buttons
    become one from b onward. Only dx looks ahead by lookahead_s; yaw is the
    signed total of each turn stage above 0.2 rad, limited to +/-1.5*pi.
    Preserve the previous XY deadzone/length limit before selecting dx.

    The release boundary is passed explicitly and also saved for teacher export.
    Also return the object's position and wxyz attitude at b - 1.
    Inputs are not mutated, commands are
    not divided by time, and invalid data raise ValueError with the motion name.
    """
    try:
        pos = np.asarray(robot["root_trans_offset"])
        quat = np.asarray(robot["root_rot"])
        fps = float(robot["fps"])
        a = meta["place_generated_start_frame"]
        b = release_start_frame
        total = len(pos)
        if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                for v in (a, b)) or not 0 <= a < b < total):
            raise ValueError(f"require 0 <= a < b < T; got {a}, {b}, {total}")
        if (pos.shape != (total, 3) or quat.shape != (total, 4)
                or not np.isfinite(pos).all() or not np.isfinite(quat).all()
                or not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4)):
            raise ValueError("invalid root positions or xyzw unit quaternions")
        if not np.isfinite(fps) or fps <= 0 or not np.isfinite(lookahead_s) or lookahead_s <= 0:
            raise ValueError("fps and lookahead_s must be finite and positive")
        obj_pos = np.asarray(objects["root_pos"])
        obj_quat = np.asarray(objects["root_quat"])
        if (obj_pos.shape != (total, 1, 3) or obj_quat.shape != (total, 1, 4)
                or float(objects["fps"]) != fps or not np.isfinite(obj_pos).all()
                or not np.isfinite(obj_quat).all()
                or not np.allclose(np.linalg.norm(obj_quat, axis=-1), 1, atol=1e-4)):
            raise ValueError("expected matching finite single-object trajectory with unit wxyz quaternions")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name}: {error}") from error

    horizon = max(1, round(lookahead_s * fps))
    rot = Rotation.from_quat(quat).as_matrix()
    yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    t = np.arange(a, b)
    j = np.minimum(t + horizon, b - 1)
    delta = pos[j, :2] - pos[t, :2]
    c, s = np.cos(yaw[t]), np.sin(yaw[t])
    xy = np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                          -s * delta[:, 0] + c * delta[:, 1]))
    length = np.linalg.norm(xy, axis=1)
    xy[length < 0.02] = 0
    xy *= np.minimum(1, 0.15 / np.maximum(length, 1e-12))[:, None]
    dyaw, _ = turn_stage_commands(yaw[a:b], fps)
    command = np.zeros((total, 2), dtype=np.float32)
    command[t] = np.column_stack((xy[:, 0], dyaw))
    drop = np.zeros((total, 1), dtype=np.float32)
    drop[b:] = 1
    additions = {
        "place_object_goal_pos_w": obj_pos[b - 1, 0].copy(),
        "place_object_goal_quat_w": obj_quat[b - 1, 0].copy(),
    }
    return {"policy_command_x_yaw": command, "policy_command_drop_button": drop}, additions


def resample_qpos(qpos, source_fps, output_fps):
    """重采样 Z-up/wxyz qpos (T,36):位置和关节角线性插值,root 旋转用 SLERP。"""
    times = np.arange(len(qpos)) / source_fps
    target = np.arange(int(np.ceil(times[-1] * output_fps)) + 1) / output_fps
    target = np.minimum(target, times[-1])
    values = np.column_stack([np.interp(target, times, qpos[:, i])
                              for i in list(range(3)) + list(range(7, 36))])
    rotations = Slerp(times, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(target).as_quat()
    return np.column_stack([values[:, :3], rotations[:, [3, 0, 1, 2]], values[:, 3:]])


def slow_first_turn(qpos, fps, factor):
    """仅扩展首个转身段 [start,end),保持 FPS。

    N 帧扩展为 round(N * factor) 帧,用下一帧 end 作为插值端点但不重复加入。
    位置/关节角线性插值,root 四元数沿最短路径线性插值后归一化。
    """
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("turn_slowdown_factor must be finite and >= 1")
    if factor == 1:
        return qpos
    rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix()
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    _, stages = turn_stage_commands(yaw, fps)
    if not stages:
        return qpos
    start, end = stages[0]
    frames = round((end - start) * factor)
    samples = np.linspace(start, end, frames, endpoint=False)
    left = np.floor(samples).astype(int)
    alpha = (samples - left)[:, None]
    expanded = (1 - alpha) * qpos[left] + alpha * qpos[left + 1]
    # q and -q represent the same orientation; align signs before interpolating.
    qa, qb = qpos[left, 3:7], qpos[left + 1, 3:7].copy()
    qb[np.sum(qa * qb, axis=1) < 0] *= -1
    quat = (1 - alpha) * qa + alpha * qb
    expanded[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return np.concatenate([qpos[:start], expanded, qpos[end:]])


def right_hand_pose(qpos, kinematics, *, quat_order="wxyz"):
    """返回右掌 Z-up 世界位置 (T,3) 和旋转 (T,3,3),包含 base 及所有上游关节。

    FK 使用最后一个腕关节的固定子节点 right_hand_roll_skel。
    """
    adapter = G1QposAdapter(kinematics[0])
    rotations, positions = adapter.fk(qpos, quat_order=quat_order)
    palm = adapter.hand_indices[1]
    return (positions[:, palm] @ MUJOCO_TO_KIMODO,
            MUJOCO_TO_KIMODO.T @ rotations[:, palm] @ MUJOCO_TO_KIMODO)


def robot_segment(qpos, template):
    """将世界 wxyz qpos 转为机器人 PKL,沿用 FPS/dtype/末帧手指,右手闭合。"""
    _, indices, axes = joint_mapping()
    n = len(qpos)
    dof = qpos[:, 7:][:, indices]
    root_rot = qpos[:, [4, 5, 6, 3]]
    result = {"fps": template["fps"],
              "root_trans_offset": qpos[:, :3], "root_rot": root_rot, "dof": dof,
              "pose_aa": np.concatenate([Rotation.from_quat(root_rot).as_rotvec()[:, None],
                                           dof[..., None] * axes[None]], axis=1),
              "smpl_joints": np.zeros((n, 24, 3)),
              "hand_dof_pos": np.repeat(template["hand_dof_pos"][-1:], n, axis=0),
              "hand_action_left": np.repeat(template["hand_action_left"][-1:], n),
              "hand_action_right": np.ones(n)}
    if set(result) != set(template):
        raise ValueError("Robot schema changed; add an explicit suffix adapter")
    for key in result:
        if key != "fps":
            result[key] = result[key].astype(template[key].dtype)
    return result


def seam_segment(original, generated, frames):
    """在原末帧与生成首帧间插入过渡,不重复端点;root 用 SLERP,右手保持闭合。"""
    if frames < 1:
        raise ValueError("Seam needs at least one frame")
    out = {"fps": original["fps"]}
    alpha = np.arange(1, frames + 1, dtype=float) / (frames + 1)
    for key in set(original) - {"fps"}:
        a, b = original[key][-1], generated[key][0]
        if key == "root_rot":
            out[key] = Slerp([0, 1], Rotation.from_quat(np.stack([a, b])))(alpha).as_quat().astype(a.dtype)
        elif key == "pose_aa":
            continue
        elif key == "hand_action_right":
            out[key] = np.ones(frames, dtype=a.dtype)
        else:
            out[key] = ((1 - alpha).reshape((-1,) + (1,) * a.ndim) * a +
                        alpha.reshape((-1,) + (1,) * a.ndim) * b).astype(a.dtype)
    axes = joint_mapping()[2]
    out["pose_aa"] = np.concatenate([
        Rotation.from_quat(out["root_rot"]).as_rotvec()[:, None],
        out["dof"][..., None] * axes[None]], axis=1).astype(original["pose_aa"].dtype)
    return out


def carry_contact_points(objects, carry_frame):
    """按源帧索引读取右手接触点（物体局部坐标）,不从其他帧补猜标注。"""
    if type(carry_frame) is not int or not 0 <= carry_frame < len(objects["root_pos"]):
        raise ValueError(f"Invalid carry_source_frame: {carry_frame}")
    points = np.asarray(objects.get("contact_points_right_hand", {}).get(carry_frame, []))
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or not np.issubdtype(points.dtype, np.number) or not np.isfinite(points).all()):
        raise ValueError(f"Carry frame {carry_frame} must have nonempty finite right-hand contact points")
    return points


def object_segment(original, hand_positions, source_hand_position,
                   hand_rotations, source_hand_rotation, *, carry_frame):
    """按右掌的 Z-up 世界刚体变换更新物体,保留 carry 帧抓握偏移及 wxyz 顺序。

    original 为完整源物体数据,carry_frame 为 manifest 中的源帧索引。
    手掌参考位置/旋转必须来自同一 carry 帧,右手接触点沿用该帧的物体局部坐标。
    """
    contacts = carry_contact_points(original, carry_frame)
    result = {"fps": original["fps"]}
    if "scale" in original:
        result["scale"] = copy.deepcopy(original["scale"])
    source_rotation = Rotation.from_quat(original["root_quat"][carry_frame][:, [1, 2, 3, 0]]).as_matrix()
    # Rotate the object-to-palm offset as well as its orientation to retain the grip.
    rotation_delta = hand_rotations @ source_hand_rotation.T
    offsets = original["root_pos"][carry_frame] - source_hand_position
    result["root_pos"] = (hand_positions[:, None] + np.einsum(
        "tij,oj->toi", rotation_delta, offsets)).astype(original["root_pos"].dtype)
    rotations = rotation_delta[:, None] @ source_rotation[None]
    quats = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
    result["root_quat"] = quats.reshape(len(hand_positions), -1, 4).astype(original["root_quat"].dtype)
    for key in CONTACT_FIELDS & set(original):
        result[key] = ({i: contacts.copy() for i in range(len(hand_positions))}
                       if key == "contact_points_right_hand" else {})
    return result


def repeat_last_frame(motion, frames, last_frame):
    """重复末帧数组及静态字段,last_frame 用于查接触点并将索引重置到新增段。"""
    result = {}
    for key, value in motion.items():
        if key in {"fps", "scale"}:
            result[key] = copy.deepcopy(value)
        elif key in CONTACT_FIELDS:
            result[key] = ({i: copy.deepcopy(value[last_frame]) for i in range(frames)}
                           if last_frame in value else {})
        else:
            result[key] = np.repeat(value[-1:], frames, axis=0)
    return result


def crop_source_prefix(motion, last_frame):
    """保留第 0 帧至 last_frame（含端点）,同步裁剪逐帧数组和接触字典。"""
    result = {}
    for key, value in motion.items():
        if key in {"fps", "scale"}:
            result[key] = copy.deepcopy(value)
        elif key in CONTACT_FIELDS:
            result[key] = {int(frame): copy.deepcopy(points)
                           for frame, points in value.items() if frame <= last_frame}
        else:
            result[key] = value[:last_frame+1].copy()
    return result


def append(dataset, output, name, manifest_path, csv, kimodo_repo=KIMODO, overwrite=False,
           *, kinematics=None, copy_shared_assets=True, turn_slowdown_factor=2.0):
    """保留原段至 carry 帧,追加过渡、Kimodo、0.7 s 保持和 0.7 s 开手下落。

    写出 robot/objects/meta 和 teacher 所需报告并复制资产,返回 robot 文件路径。
    输入与输出目录必须独立。
    批处理可传入复用的 kinematics,并在首条成功后关闭共享资产重复复制。
    首个生成转身段默认扩展到两倍帧数,turn_slowdown_factor=1 时保持原速。
    """
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("Input and output datasets must be separate")
    records = [r for r in json.loads(Path(manifest_path).read_text())["records"] if r["name"] == name]
    if len(records) != 1:
        raise ValueError(f"Expected exactly one manifest record for {name}")
    record = records[0]
    if record.get("kinematics_version") != KINEMATICS_VERSION:
        raise ValueError("Constraints use the old G1 joint-frame conversion. "
                         "Re-export constraints and regenerate Kimodo CSV before appending.")
    robot_path = dataset / "robot" / f"{name}.pkl"
    object_path = dataset / "objects" / f"{name}.pkl"
    motion_key, robot = load_single(robot_path)
    # Derived labels belong to the final timeline, never to an input segment.
    robot = {key: value for key, value in robot.items() if key not in COMMAND_FIELDS}
    object_key, objects = load_single(object_path)
    source_frames = validate_fields(robot, "robot")
    if motion_key != object_key or str(motion_key) != record["motion_key"] or validate_fields(objects, "object") != source_frames:
        raise ValueError("Source motion or object data mismatch")
    if file_hash(robot_path) != record["source_robot_sha256"] or file_hash(object_path) != record["source_objects_sha256"]:
        raise ValueError("Source changed since constraints were generated")
    if robot["fps"] != objects["fps"] or source_frames != record["source_frames"]:
        raise ValueError("Source frame count or FPS mismatch")
    carry_frame = record["carry_source_frame"]
    if type(carry_frame) is not int or not 0 <= carry_frame < source_frames:
        raise ValueError(f"Invalid carry_source_frame: {carry_frame}")
    if robot["hand_action_right"][carry_frame] <= 0:
        raise ValueError(f"Right hand must be closed at carry frame {carry_frame}")
    # Keep the full source for explicit carry-frame references; crop only the retained prefix.
    source_objects = objects
    carry_contact_points(source_objects, carry_frame)
    robot = crop_source_prefix(robot, carry_frame)
    objects = crop_source_prefix(objects, carry_frame)
    n = carry_frame + 1
    walk_report_path = dataset / "reports" / f"{name}.json"
    walk_report = json.loads(walk_report_path.read_text())
    grail_start = walk_report.get("b_start_index")
    if (type(grail_start) is not int or not 0 <= grail_start < n
            or walk_report.get("total_frames") != source_frames
            or walk_report.get("fps") != float(robot["fps"])):
        raise ValueError(f"{walk_report_path}: invalid source GRAIL boundary/frame count/FPS")
    qpos = np.loadtxt(csv, delimiter=",", ndmin=2)
    if qpos.shape != (record["num_frames"], 36) or not np.isfinite(qpos).all():
        raise ValueError(f"{name}: expected finite ({record['num_frames']},36) Kimodo CSV "
                         f"matching the manifest; got shape {qpos.shape}")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-3):
        raise ValueError("Kimodo CSV contains invalid root quaternions")
    if kinematics is None:
        kinematics = load_kinematics(Path(kimodo_repo))
    world_qpos = qpos_to_world(qpos, record)
    fps = float(robot["fps"])
    # Preserve the source FPS and generated poses after restoring world coordinates.
    world_qpos = resample_qpos(world_qpos, MODEL_FPS, fps)
    world_qpos = slow_first_turn(world_qpos, fps, turn_slowdown_factor)
    carry_values = source_qpos(robot)[carry_frame:carry_frame+1]
    hand, hand_rotations = right_hand_pose(world_qpos, kinematics)
    carry_positions, carry_rotations = right_hand_pose(carry_values, kinematics, quat_order="xyzw")
    carry_hand, carry_rotation = carry_positions[0], carry_rotations[0]
    # Join from the retained carry frame, which also defines the start target.
    generated = robot_segment(world_qpos, robot)
    join_frames = max(1, round(0.1 * fps))
    seam = seam_segment(robot, generated, join_frames)
    # Round the 0.7-second hold to source-FPS frames for both robot and object.
    hold_frames = max(1, round(0.7 * fps))
    hold = {k: (v if k == "fps" else np.repeat(v[-1:], hold_frames, axis=0))
            for k, v in generated.items()}
    # Linear joint interpolation does not imply a linear palm path.
    seam_hand, seam_rotations = right_hand_pose(source_qpos(seam), kinematics, quat_order="xyzw")
    obj_seam = object_segment(source_objects, seam_hand, carry_hand, seam_rotations,
                              carry_rotation, carry_frame=carry_frame)
    obj_generated = object_segment(source_objects, hand, carry_hand, hand_rotations,
                                   carry_rotation, carry_frame=carry_frame)
    obj_hold = repeat_last_frame(obj_generated, hold_frames, len(hand)-1)
    # Release during the final 0.7 seconds; preserve the body DOFs and object rotation.
    final_hold_frames = max(1, round(0.7 * fps))
    robot_final_hold = repeat_last_frame(hold, final_hold_frames, len(hold["dof"]) - 1)
    # Release has no contact labels; avoid copying the held object's contact points.
    object_final_hold = repeat_last_frame(
        {k: ({} if k in CONTACT_FIELDS else v) for k, v in obj_hold.items()},
        final_hold_frames, len(obj_hold["root_pos"]) - 1)
    robot_final_hold["hand_action_right"][:] = -1
    robot_final_hold["hand_dof_pos"][:, 7:] = 0  # Right fingers' open pose in the renderer.
    # Appended frames follow the retained endpoint by 1/fps, ..., frames/fps seconds.
    fall_time = np.arange(1, final_hold_frames + 1, dtype=float) / fps
    object_final_hold["root_pos"][..., 2] -= (0.5 * 9.81 * fall_time**2)[:, None]
    # concat_motion validates every segment's schema, FPS, dtype and frame arrays.
    robot_out = concat_motion([robot, seam, generated, hold, robot_final_hold])
    objects_out = concat_motion([objects, obj_seam, obj_generated, obj_hold, object_final_hold], kind="object")
    # Derive output boundaries from the final arrays, after turn interpolation.
    # The added turn frames are already in total_frames: do not add them twice.
    total_frames = len(robot_out["dof"])
    final_hold_start = total_frames - len(robot_final_hold["dof"])
    hold_start = final_hold_start - len(hold["dof"])
    generated_start = hold_start - len(generated["dof"])
    if generated_start != n + len(seam["dof"]):
        raise ValueError(f"{name}: inconsistent concatenated timeline")
    # Assets are copied after validation, without modifying the input snapshot.
    usd = [dataset / "object_usd" / f"{name}{ext}" for ext in (".usd", ".usda")
           if (dataset / "object_usd" / f"{name}{ext}").is_file()]
    if len(usd) != 1:
        raise ValueError("Expected one matching USD/USDA asset")
    assets = [dataset / "meta" / f"{name}.pkl", usd[0]]
    assets += list((dataset / "bps").glob(f"{name}.npy"))
    if copy_shared_assets:
        assets += list((dataset / "bps").glob("_*.npy"))
    for asset in assets:
        if not asset.is_file():
            raise FileNotFoundError(asset)
    outputs = [output / d / f"{name}.pkl" for d in ("robot", "objects", "meta")]
    if not overwrite and any(p.exists() for p in outputs):
        raise FileExistsError("Output exists; pass --overwrite")
    meta = {key: copy.deepcopy(value) for key, value in joblib.load(assets[0]).items()
            if key in MOTION_META_FIELDS}
    # Fade out from the retained endpoint to the first generated frame, including the seam.
    meta.update(total_frames=total_frames,
                grail_start_frame=grail_start,
                residual_fade_out_start_frame=n-1,
                residual_fade_out_end_frame=generated_start,
                place_generated_start_frame=generated_start,
                place_release_start_frame=final_hold_start,
                place_output_contact_start_frame=contact_timing(objects_out)["source_contact_start_frame"],
                # Carry goals stay in the original Z-up world frame; object quats are wxyz.
                place_object_carry_goal_pos_w=source_objects["root_pos"][carry_frame, 0].copy(),
                place_object_carry_goal_quat_w=source_objects["root_quat"][carry_frame, 0].copy(),
                place_carry_source_frame=record["carry_source_frame"])
    labels, sparse_meta = generate_sparse_commands(
        robot_out, objects_out, meta, name, release_start_frame=final_hold_start
    )
    robot_out.update(labels)
    meta.update(sparse_meta)
    validate_fields(robot_out, "robot")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({motion_key: robot_out}, outputs[0])
    joblib.dump({motion_key: objects_out}, outputs[1])
    joblib.dump(meta, outputs[2])
    for asset in assets[1:]:
        target = output / asset.relative_to(dataset)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset, target)
    textures = dataset / "object_usd" / "textures"
    if copy_shared_assets and textures.is_dir():
        shutil.copytree(textures, output / "object_usd" / "textures", dirs_exist_ok=True)
    from grail.datatool.batch_render_replay_clip import validate_motion_input
    report = {
        "name": name, "total_frames": total_frames, "fps": fps,
        "grail_start_frame": grail_start,
        "output_contact_start_frame": meta["place_output_contact_start_frame"],
        "validation": validate_motion_input(str(output), name),
    }
    report_path = output / "reports" / f"{name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return outputs[0]


def main():
    """执行本地拼接并打印输出 robot 文件路径。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=DATA / "pickup_table_walk_concat")
    p.add_argument("--output-dir", type=Path, default=DATA / "pickup_table_place_concat")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--kimodo-repo", type=Path, default=KIMODO)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--turn-slowdown-factor", type=float, default=2.0,
                   help="Expand only the first generated turn by this factor (>=1); 1 disables slowing")
    args = p.parse_args()
    print(append(args.dataset_dir, args.output_dir, args.name,
        args.manifest or args.dataset_dir / "kimodo_place_constraints/manifest.json",
        args.csv, args.kimodo_repo, args.overwrite,
        turn_slowdown_factor=args.turn_slowdown_factor))


if __name__ == "__main__":
    main()
