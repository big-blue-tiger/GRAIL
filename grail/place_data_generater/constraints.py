#!/usr/bin/env python3
"""Export post-grasp G1 constraints, one JSON and manifest record per motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from grail.walk_data_tool.concat_walk_motion import file_hash, load_single, validate_fields
from grail.walk_data_tool.export_kimodo_end_frame import load_kinematics, sample_start
from grail.place_data_generater.g1_kinematics import G1QposAdapter, KINEMATICS_VERSION, source_qpos

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/hf_dataset/data_update/data"
KIMODO = Path("/home/tide/robot/kimodo")
MODEL_FPS = 30
TURNS_DURATION_S = 1


def contact_timing(objects):
    """从非空右手接触标注提取源帧范围及接触时长 D（接触帧数 / FPS）。"""
    frames = sorted(int(frame) for frame, points in
                    objects.get("contact_points_right_hand", {}).items()
                    if np.asarray(points).size > 0)
    if not frames:
        raise ValueError("Source motion has no right-hand contact annotations")
    fps = float(objects["fps"])
    return {"source_contact_start_frame": frames[0],
            "source_contact_end_frame": frames[-1],
            "source_contact_start_s": frames[0] / fps,
            "source_contact_frames": len(frames),
            "source_contact_duration_s": len(frames) / fps}


def sample_destination(qpos, meta, rng):
    """按面积均匀采样桌外 2–3.5 m 扇形,返回 Z-up 世界 XY、距离和偏角。"""
    # sample_start validates the table and selects its outward normal. Its
    # 1..2 m draw is discarded; the angular sector and table side are reused.
    _, _, angle, normal = sample_start(qpos, meta, rng)
    radius = float(np.sqrt(rng.uniform(1.7**2, 3.3**2)))
    tangent = np.array([-normal[1], normal[0]])
    xy = qpos[0, :2] + radius * (np.cos(angle) * normal + np.sin(angle) * tangent)
    return xy, radius, angle


def make_record(dataset, name, seed, kinematics, remote_dir):
    """返回 Kimodo Y-up 行走/下放约束及 manifest,不写文件或启动生成。

    按接触时长 D 取末帧前 D/2 的起始持物姿态,终点下放姿态回看比例均匀采样于 0.6–0.85。
    终点下放姿态位于原到达帧,不再设置独立到达约束或额外下放时长。
    """
    from kimodo.constraints import load_constraints_lst

    skeleton, _, to_axis_angle, _ = kinematics
    adapter = G1QposAdapter(skeleton)
    robot_path = dataset / "robot" / f"{name}.pkl"
    object_path = dataset / "objects" / f"{name}.pkl"
    key, motion = load_single(robot_path)
    object_key, objects = load_single(object_path)
    n = validate_fields(motion, "robot")
    if key != object_key or validate_fields(objects, "object") != n or motion["fps"] != objects["fps"]:
        raise ValueError("Robot/object key, frame count or FPS differs")
    if motion["hand_action_right"][-1] <= 0:
        raise ValueError("Right hand is not closed at the grasp endpoint")
    source = source_qpos(motion)
    contact = contact_timing(objects)
    contact_frames = contact["source_contact_frames"]
    duration = contact["source_contact_duration_s"]
    carry_source_frame = max(0, n - 1 - round(contact_frames / 2))
    qpos = source[carry_source_frame:carry_source_frame+1]
    meta = joblib.load(dataset / "meta" / f"{name}.pkl")
    digest = hashlib.sha256(f"{seed}:{name}:{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    target_xy, distance, angle = sample_destination(qpos, meta, rng)
    lower_pose_lookback_ratio = float(rng.uniform(0.6, 0.8))
    lower_source_frame = max(0, n - 1 - round(contact_frames * lower_pose_lookback_ratio))
    source_local, source_roots = adapter.local_pose(qpos, quat_order="xyzw")
    start_local = source_local[0].clone()
    start_root_world = source_roots[0].clone()
    # Canonical initial heading is +Z, while the source/world transform is saved
    # for applying the generated suffix back to the untouched grasp sequence.
    # Match the model's hip-vector heading, including source root roll/pitch.
    from kimodo.motion_rep.feature_utils import compute_heading_angle
    _, source_positions, _ = skeleton.fk(source_local, source_roots)
    heading = float(compute_heading_angle(source_positions[None], skeleton)[0, 0])
    c2w = Rotation.from_euler("y", heading).as_matrix()
    start_local[0] = (start_local[0].new_tensor(c2w.T) @ start_local[0])
    origin = start_root_world.numpy().copy()
    origin[1] = 0.0
    start_root = start_root_world - start_root_world.new_tensor(origin)
    displacement_world = np.array([target_xy[1] - qpos[0, 1], 0.,
                                   target_xy[0] - qpos[0, 0]])
    displacement = displacement_world @ c2w
    goal_root = start_root + start_root.new_tensor(displacement)
    goal_local = start_local.clone()
    goal_heading = float(np.arctan2(displacement[0], displacement[2]))
    goal_local[0] = (goal_local[0].new_tensor(Rotation.from_euler("y", goal_heading).as_matrix())
                     @ goal_local[0])
    lowered_local, lowered_root = adapter.lower_hand(source[lower_source_frame], goal_local, goal_root)
    lower = int(round(TURNS_DURATION_S * MODEL_FPS)) + int(round(distance / 0.4 * MODEL_FPS))
    total = lower + 1
    full = {"type": "fullbody", "frame_indices": [0, lower],
            "local_joints_rot": to_axis_angle(
                torch.stack([start_local, lowered_local])).tolist(),
            "root_positions": torch.stack([start_root, lowered_root]).tolist()}
    root2d = {"type": "root2d", "frame_indices": [0, lower],
              "smooth_root_2d": [start_root[[0, 2]].tolist(), lowered_root[[0, 2]].tolist()]}
    constraints = [root2d, full]
    parsed = load_constraints_lst(constraints, skeleton)
    restored = parsed[1].global_joints_positions[0].numpy() @ c2w.T + origin
    source_pose_error = float(np.abs(restored-source_positions[0].numpy()).max())
    durations = repr((total + 0.001) / MODEL_FPS)
    prompt = "A person turns very slowly and gradually without any sudden rotation, then walks slowly while holding an object in the right hand, gradually lowers the right hand while approaching the destination, and stops."
    remote = f"{remote_dir}/{name}"
    command = shlex.join(["python", "-m", "kimodo.scripts.generate", prompt,
        "--model", "Kimodo-G1-RP-v1", "--duration", durations,
        "--constraints", remote + ".json", "--num_samples", "1", "--seed", str(seed),
        "--output", f"{remote_dir}/motions/{name}"])
    record = {"name": name, "motion_key": str(key), "kinematics_version": KINEMATICS_VERSION,
              "source_pose_fk_error_m": source_pose_error,
              "source_robot_sha256": file_hash(robot_path),
              "source_objects_sha256": file_hash(object_path),
              "source_frames": n, "source_fps": float(motion["fps"]),
              "target_xy": target_xy.tolist(), "distance_m": distance,
              "sector_angle_deg": float(np.rad2deg(angle)),
              **contact,
              "carry_source_frame": carry_source_frame,
              "carry_pose_lookback_s": duration / 2,
              "lower_source_frame": lower_source_frame,
              "lower_pose_lookback_ratio": lower_pose_lookback_ratio,
              "lower_pose_lookback_s": duration * lower_pose_lookback_ratio,
              # Retain arrival_frame as an alias for existing manifest consumers.
              "arrival_frame": lower, "lower_frame": lower, "num_frames": total,
              "model_fps": MODEL_FPS, "seed": seed,
              "canonical_to_world_rotation_kimodo": c2w.tolist(),
              "canonical_to_world_origin_kimodo": origin.tolist(),
              "command": command}
    return constraints, record


def main():
    """批量导出约束 JSON、manifest 和生成命令,有失败或无成功记录时返回非零。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=DATA / "pickup_table_walk_concat")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--pattern", default="*.pkl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kimodo-repo", type=Path, default=KIMODO)
    p.add_argument("--remote-dir", default="/workspace/kimodo/data/wostep/place_walk")
    args = p.parse_args()
    dataset = args.dataset_dir.resolve()
    output = (args.output_dir or dataset / "kimodo_place_constraints").resolve()
    output.mkdir(parents=True, exist_ok=True)
    kinematics = load_kinematics(args.kimodo_repo)
    records, failures = [], {}
    for path in sorted((dataset / "robot").glob(args.pattern)):
        try:
            constraints, record = make_record(dataset, path.stem, args.seed,
                                               kinematics, args.remote_dir)
            (output / f"{path.stem}.json").write_text(json.dumps(constraints, indent=2) + "\n")
            records.append(record)
            print(f"OK {path.stem}: {record['distance_m']:.2f} m")
        except Exception as exc:
            failures[path.stem] = str(exc)
            (output / f"{path.stem}.json").unlink(missing_ok=True)
            print(f"FAIL {path.stem}: {exc}")
    (output / "manifest.json").write_text(json.dumps({"records": records, "failures": failures}, indent=2) + "\n")
    (output / "generate_commands.sh").write_text("#!/bin/bash\nset -euo pipefail\n" +
        "\n".join(r["command"] for r in records) + "\n")
    if failures or not records:
        raise SystemExit(f"Exported {len(records)} constraints; {len(failures)} failures")


if __name__ == "__main__":
    main()
