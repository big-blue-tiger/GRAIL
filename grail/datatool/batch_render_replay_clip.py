#!/usr/bin/env python3
"""Clean a motion library by kinematically replaying reference motions.

The cleaner uses the existing visualization conversion/replay path.  It writes
the reference robot and object state directly into one Isaac Sim session and
never constructs or loads a neural-network policy.

Example::

    python -m grail.datatool.batch_render_replay_clip \
        --data_dir data/hf_dataset/data_update/data/pickup_table \
        --output_dir /tmp/pickup_table_cleaned \
        --quat_convention xyzw \
        --no_record_video
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import glob
import json
import os
import pickle
import shutil
import sys
import threading
import time
import traceback

import joblib
import numpy as np

from grail.datatool.cleaning_metrics import (
    CleaningThresholds,
    analyze_gait,
    analyze_object_lift,
    analyze_penetration,
    classify_motion,
    effective_penetration_mask,
)

# Per-motion progress heartbeat. Updated before each motion and each frame.
# A watchdog thread force-exits the process if no heartbeat for WATCHDOG_TIMEOUT
# seconds, so a hung USD load / sim step can't stall the job indefinitely.
_last_progress_time = time.time()
_last_progress_label = "init"
WATCHDOG_TIMEOUT = 180.0  # seconds

# A contact is treated as penetration only after it exceeds the normal contact
# tolerance.  PhysX reports touching/resting contacts with separation values
# close to zero; negative values represent overlap depth.
PENETRATION_DEPTH_THRESHOLD_M = 0.01

_MONITORED_HAND_LINKS = frozenset(
    {
        "left_hand_palm_link",
        "left_hand_thumb_0_link",
        "left_hand_thumb_1_link",
        "left_hand_thumb_2_link",
        "left_hand_index_0_link",
        "left_hand_index_1_link",
        "left_hand_middle_0_link",
        "left_hand_middle_1_link",
        "right_hand_palm_link",
        "right_hand_thumb_0_link",
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
        "right_hand_index_0_link",
        "right_hand_index_1_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
    }
)

_MONITORED_BODY_LINKS = frozenset(
    {
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
    }
)

# Kept in memory for downstream processing inserted before the existing
# IsaacSim force-exit. Keys are motion_key values and arrays have one entry per
# original trajectory frame.
penetration_frames_by_motion = {}


def _path_matches_link(path, robot_root, link_name):
    """Return whether a USD path belongs to a monitored robot link."""
    path = str(path or "")
    link_root = f"{robot_root}/{link_name}"
    return path == link_root or path.startswith(f"{link_root}/")


def _monitored_contact_pair(path0, path1, robot_root):
    """Return True only for a hand-vs-body pair in either contact order."""
    hand0 = any(_path_matches_link(path0, robot_root, link) for link in _MONITORED_HAND_LINKS)
    hand1 = any(_path_matches_link(path1, robot_root, link) for link in _MONITORED_HAND_LINKS)
    body0 = any(_path_matches_link(path0, robot_root, link) for link in _MONITORED_BODY_LINKS)
    body1 = any(_path_matches_link(path1, robot_root, link) for link in _MONITORED_BODY_LINKS)
    return (hand0 and body1) or (hand1 and body0)


def _separation_is_penetration(separation, threshold=PENETRATION_DEPTH_THRESHOLD_M):
    """Classify a PhysX separation value using the configured depth threshold."""
    try:
        return float(separation) <= -float(threshold)
    except (TypeError, ValueError):
        return False


def _heartbeat(label: str) -> None:
    global _last_progress_time, _last_progress_label
    _last_progress_time = time.time()
    _last_progress_label = label


def _watchdog_loop():
    while True:
        time.sleep(15.0)
        idle = time.time() - _last_progress_time
        if idle > WATCHDOG_TIMEOUT:
            sys.stderr.write(
                f"\n[WATCHDOG] No progress for {idle:.0f}s at '{_last_progress_label}'. "
                f"Forcing non-zero exit so the scheduler flags this job.\n"
            )
            sys.stderr.flush()
            os._exit(2)


def start_watchdog():
    t = threading.Thread(target=_watchdog_loop, daemon=True, name="render-watchdog")
    t.start()
    return t


def reconstruct_filter_keys(metrics, render_sort_by="obj_pos_error"):
    """Reconstruct filter_keys order from Phase 1 metrics.

    Replicates eval_agent_trl.py logic for env_idx -> motion_key mapping.
    """
    all_dict = metrics.get("eval/all_metrics_dict", {})
    motion_keys = all_dict.get("motion_keys", [])
    if not motion_keys:
        raise ValueError("No motion_keys in metrics")

    terminated = all_dict.get("terminated", [])
    mpjpe_l = all_dict.get("mpjpe_l", [0.0] * len(motion_keys))
    mpjpe_g = all_dict.get("mpjpe_g", [0.0] * len(motion_keys))
    obj_pos_errors = all_dict.get("obj_pos_error", None)

    sort_idx = 4 if render_sort_by == "obj_pos_error" else 1

    success_pair = [
        (motion_keys[i], mpjpe_l[i], mpjpe_g[i], True, obj_pos_errors[i] if obj_pos_errors else 0.0)
        for i in range(len(motion_keys))
        if not terminated[i]
    ]
    failed_pair = [
        (
            motion_keys[i],
            mpjpe_l[i],
            mpjpe_g[i],
            False,
            obj_pos_errors[i] if obj_pos_errors else 0.0,
        )
        for i in range(len(motion_keys))
        if terminated[i]
    ]

    success_sorted = sorted(success_pair, key=lambda x: x[sort_idx], reverse=True)
    failed_sorted = sorted(failed_pair, key=lambda x: x[sort_idx], reverse=True)

    all_pair = failed_sorted + success_sorted
    filter_keys = [p[0] for p in all_pair]
    success_set = {p[0] for p in success_pair}

    return filter_keys, success_set


def build_render_plan(
    shard_dir,
    object_usd_dir,
    output_dir,
    skip_existing=False,
    traj_dir=None,
    record_video=True,
):
    """Build list of (env_idx, motion_key, traj_path, usd_path, output_path)."""
    metrics_path = os.path.join(shard_dir, "metrics_eval.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    filter_keys, success_set = reconstruct_filter_keys(metrics)
    if traj_dir is None:
        traj_dir = os.path.join(shard_dir, "trajectories")

    # Original env→motion_key map (load order during phase1 eval). The recorder
    # writes `{env_idx:06d}.trajectory.pkl` keyed on this load order, so we
    # must look up trajectories by THIS mapping — NOT the sort-permuted
    # filter_keys above. (Bug fix: previously used enumerate(filter_keys),
    # which paired success motions with the wrong env's trajectory whenever
    # any motion failed and got pushed to position 0 by reconstruct_filter_keys.)
    motion_keys_load_order = metrics.get("eval/all_metrics_dict", {}).get("motion_keys", [])
    motion_to_env = {mk: i for i, mk in enumerate(motion_keys_load_order)}

    plan = []
    stats = {"skipped": 0, "missing_traj": 0, "missing_usd": 0}

    for motion_key in filter_keys:
        if motion_key not in success_set:
            continue
        env_idx = motion_to_env.get(motion_key)
        if env_idx is None:
            stats["missing_traj"] += 1
            continue

        traj_path = os.path.join(traj_dir, f"{motion_key}.trajectory.pkl")
        if not os.path.exists(traj_path):
            traj_path = os.path.join(traj_dir, f"{env_idx:06d}.trajectory.pkl")
        if not os.path.exists(traj_path):
            stats["missing_traj"] += 1
            continue

        output_path = os.path.join(output_dir, f"{motion_key}.mp4")
        if record_video and skip_existing and os.path.exists(output_path):
            stats["skipped"] += 1
            continue

        usd_path = os.path.join(object_usd_dir, f"{motion_key}.usd")
        if not os.path.exists(usd_path):
            usd_path = os.path.join(object_usd_dir, f"{motion_key}.usda")
        if not os.path.exists(usd_path):
            stats["missing_usd"] += 1
            usd_path = None

        plan.append((env_idx, motion_key, traj_path, usd_path, output_path))

    return plan, stats, len(filter_keys), len(success_set)


def compute_start_frame_skip(total_frames, requested_skip):
    """Clamp the optional initial frame skip to a valid frame range."""
    requested_skip = max(0, int(requested_skip))
    if total_frames <= 0:
        return 0
    return min(requested_skip, total_frames - 1)


def compute_penetration_segment_start(
    penetration_frames, post_penetration_frames=20
):
    """Return the first frame to export after the last penetration.

    The returned index is intentionally ``last_penetration + 20`` (rather
    than ``+ 21``), matching the frame-counting convention used by the
    replay logs.  ``None`` means that no penetration was detected.
    """
    post_penetration_frames = int(post_penetration_frames)
    if post_penetration_frames < 0:
        raise ValueError("post_penetration_frames must be non-negative")
    flags = np.asarray(penetration_frames).reshape(-1)
    penetration_indices = np.flatnonzero(flags != 0)
    if penetration_indices.size == 0:
        return None
    return int(penetration_indices[-1]) + int(post_penetration_frames)


def _slice_frame_aligned_mapping(data, start_frame, end_frame, total_frames):
    """Copy a motion mapping and slice every frame-aligned field.

    Raw robot motion dictionaries contain frame arrays such as ``dof`` and
    ``pose_aa``, scalar metadata such as ``fps``, and occasionally auxiliary
    values.  Only values whose leading dimension equals the source frame count
    are sliced; all other values are copied unchanged.
    """
    segmented = {}
    for key, value in data.items():
        if key == "total_frames":
            segmented[key] = int(end_frame - start_frame)
        elif isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_frames:
            segmented[key] = value[start_frame:end_frame].copy()
        elif isinstance(value, list) and len(value) == total_frames:
            segmented[key] = copy.deepcopy(value[start_frame:end_frame])
        else:
            segmented[key] = copy.deepcopy(value)
    return segmented


def _load_raw_robot_motion(source_robot_dir, motion_key):
    """Load one original robot motion and retain its original outer key."""
    path = os.path.join(source_robot_dir, f"{motion_key}.pkl")
    if not os.path.isfile(path):
        return None, None
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"cannot load raw robot motion {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"raw robot motion {path} is not a dictionary")
    if motion_key in payload:
        source_key = motion_key
        motion = payload[motion_key]
    elif len(payload) == 1:
        source_key, motion = next(iter(payload.items()))
    else:
        raise ValueError(
            f"raw robot motion {path} has no unique motion entry for {motion_key}"
        )
    if not isinstance(motion, dict) or "dof" not in motion or "root_trans_offset" not in motion:
        raise ValueError(
            f"{path} is not an original robot motion (expected dof/root_trans_offset); "
            "it may be a trajectory pkl"
        )
    return source_key, motion


def _load_raw_object_motion(source_object_dir, motion_key):
    """Load one original object motion and retain its original outer key."""
    path = os.path.join(source_object_dir, f"{motion_key}.pkl")
    if not os.path.isfile(path):
        return None, None
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"cannot load raw object motion {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"raw object motion {path} is not a dictionary")
    if motion_key in payload:
        source_key = motion_key
        motion = payload[motion_key]
    elif len(payload) == 1:
        source_key, motion = next(iter(payload.items()))
    else:
        raise ValueError(
            f"raw object motion {path} has no unique motion entry for {motion_key}"
        )
    if (
        not isinstance(motion, dict)
        or "root_pos" not in motion
        or "root_quat" not in motion
    ):
        raise ValueError(
            f"{path} is not an original object motion (expected root_pos/root_quat)"
        )
    return source_key, motion


def _slice_object_motion_mapping(data, start_frame, end_frame, total_frames):
    """Slice object motion fields and reindex per-frame contact dictionaries."""
    segmented = _slice_frame_aligned_mapping(data, start_frame, end_frame, total_frames)
    for key in ("contact_points_left_hand", "contact_points_right_hand"):
        contact_points = data.get(key)
        if not isinstance(contact_points, dict):
            continue
        if all(isinstance(frame, (int, np.integer)) for frame in contact_points):
            segmented[key] = {
                int(frame) - start_frame: copy.deepcopy(points)
                for frame, points in contact_points.items()
                if start_frame <= int(frame) < end_frame
            }
        else:
            segmented[key] = copy.deepcopy(contact_points)
    return segmented


def _has_raw_robot_motion(source_robot_dir):
    """Return whether a directory contains at least one original robot pkl."""
    if not source_robot_dir or not os.path.isdir(source_robot_dir):
        return False
    for filename in sorted(os.listdir(source_robot_dir)):
        if not filename.endswith(".pkl") or filename.endswith(".trajectory.pkl"):
            continue
        try:
            payload = joblib.load(os.path.join(source_robot_dir, filename))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        entries = payload.values()
        if any(
            isinstance(entry, dict)
            and "dof" in entry
            and "root_trans_offset" in entry
            for entry in entries
        ):
            return True
    return False


def resolve_source_robot_dir(source_robot_dir, object_usd_dir):
    """Resolve the raw robot directory, including the dataset backup layout."""
    if source_robot_dir is not None:
        return source_robot_dir
    data_dir = os.path.dirname(os.path.abspath(object_usd_dir))
    default_dir = os.path.join(data_dir, "robot")
    if _has_raw_robot_motion(default_dir):
        return default_dir

    backup_dir = os.path.join(data_dir, "robot_old")
    if _has_raw_robot_motion(backup_dir):
        print(
            f"[SEGMENT] {default_dir} has no raw robot motions; using {backup_dir}",
            flush=True,
        )
        return backup_dir
    return default_dir


def resolve_source_object_dir(source_object_dir, object_usd_dir):
    """Resolve the raw object directory next to the object USD directory."""
    if source_object_dir is not None:
        return source_object_dir
    data_dir = os.path.dirname(os.path.abspath(object_usd_dir))
    return os.path.join(data_dir, "objects")


def export_penetration_segment(
    traj,
    motion_key,
    penetration_frames,
    output_dir,
    source_robot_dir,
    post_penetration_frames=20,
):
    """Export the post-penetration suffix in original robot-motion format.

    Returns the output path when a segment is written, otherwise ``None``.
    A motion with no penetration is deliberately not exported because it has
    no penetration-relative start frame.  Likewise, a penetration occurring
    fewer than ``post_penetration_frames`` frames before the end leaves no
    valid suffix to write.
    """
    total_frames = int(traj["total_frames"])
    flags = np.asarray(penetration_frames).reshape(-1)
    if flags.size != total_frames:
        raise ValueError(
            f"penetration_frames has {flags.size} entries, expected {total_frames}"
        )
    start_frame = compute_penetration_segment_start(
        flags, post_penetration_frames=post_penetration_frames
    )
    if start_frame is None:
        print(f"  [SEGMENT] {motion_key}: no penetration, skipped", flush=True)
        return None
    if start_frame >= total_frames:
        print(
            f"  [SEGMENT] {motion_key}: start frame {start_frame} >= "
            f"total_frames {total_frames}, skipped",
            flush=True,
        )
        return None

    if not source_robot_dir:
        raise ValueError(
            "source_robot_dir is required when segmented_output_dir is enabled"
        )
    source_key, raw_motion = _load_raw_robot_motion(source_robot_dir, motion_key)
    if raw_motion is None:
        print(
            f"  [SEGMENT] {motion_key}: raw robot file not found in "
            f"{source_robot_dir}, skipped",
            flush=True,
        )
        return None

    raw_total_frames = int(np.asarray(raw_motion["dof"]).shape[0])
    if raw_total_frames != total_frames:
        raise ValueError(
            f"{motion_key}: trajectory has {total_frames} frames but raw robot motion "
            f"has {raw_total_frames}"
        )

    os.makedirs(output_dir, exist_ok=True)
    # Keep the original filename exactly: <motion_key>.pkl.
    output_path = os.path.join(output_dir, f"{motion_key}.pkl")
    segmented = _slice_frame_aligned_mapping(
        raw_motion, start_frame, total_frames, raw_total_frames
    )
    joblib.dump({source_key: segmented}, output_path)
    print(
        f"  [SEGMENT] {motion_key}: frames {start_frame}:{total_frames} "
        f"({total_frames - start_frame} frames, raw robot format) -> {output_path}",
        flush=True,
    )
    return output_path


def export_object_penetration_segment(
    traj,
    motion_key,
    penetration_frames,
    output_dir,
    source_object_dir,
    post_penetration_frames=20,
):
    """Export the post-penetration suffix in original object-motion format."""
    total_frames = int(traj["total_frames"])
    flags = np.asarray(penetration_frames).reshape(-1)
    if flags.size != total_frames:
        raise ValueError(
            f"penetration_frames has {flags.size} entries, expected {total_frames}"
        )
    start_frame = compute_penetration_segment_start(
        flags, post_penetration_frames=post_penetration_frames
    )
    if start_frame is None:
        print(f"  [SEGMENT_OBJECT] {motion_key}: no penetration, skipped", flush=True)
        return None
    if start_frame >= total_frames:
        print(
            f"  [SEGMENT_OBJECT] {motion_key}: start frame {start_frame} >= "
            f"total_frames {total_frames}, skipped",
            flush=True,
        )
        return None

    if not source_object_dir:
        raise ValueError(
            "source_object_dir is required when segmented_object_output_dir is enabled"
        )
    source_key, raw_motion = _load_raw_object_motion(source_object_dir, motion_key)
    if raw_motion is None:
        print(
            f"  [SEGMENT_OBJECT] {motion_key}: raw object file not found in "
            f"{source_object_dir}, skipped",
            flush=True,
        )
        return None

    raw_pos = np.asarray(raw_motion["root_pos"])
    raw_quat = np.asarray(raw_motion["root_quat"])
    if raw_pos.ndim == 0 or raw_quat.ndim == 0:
        raise ValueError(
            f"{motion_key}: object root_pos/root_quat must be frame arrays"
        )
    raw_total_frames = int(raw_pos.shape[0])
    if raw_quat.shape[0] != raw_total_frames:
        raise ValueError(
            f"{motion_key}: object root_pos has {raw_total_frames} frames but "
            f"root_quat has {raw_quat.shape[0]}"
        )
    if raw_total_frames != total_frames:
        raise ValueError(
            f"{motion_key}: trajectory has {total_frames} frames but raw object motion "
            f"has {raw_total_frames}"
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{motion_key}.pkl")
    segmented = _slice_object_motion_mapping(
        raw_motion, start_frame, total_frames, raw_total_frames
    )
    joblib.dump({source_key: segmented}, output_path)
    print(
        f"  [SEGMENT_OBJECT] {motion_key}: frames {start_frame}:{total_frames} "
        f"({total_frames - start_frame} frames, raw object format) -> {output_path}",
        flush=True,
    )
    return output_path


def _single_motion_entry(payload, motion_key, label):
    """Return ``(outer_key, entry)`` from a motion-library pickle payload."""
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload is not a dictionary")
    if motion_key in payload:
        outer_key, entry = motion_key, payload[motion_key]
    elif len(payload) == 1:
        outer_key, entry = next(iter(payload.items()))
    else:
        raise ValueError(f"{label} payload has no unique entry for {motion_key}")
    if not isinstance(entry, dict):
        raise ValueError(f"{label} motion entry is not a dictionary")
    return outer_key, entry


def _object_frame_array(value, width, label):
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} must have shape (T,{width}) or (T,1,{width}), got {array.shape}")
    return array


def validate_motion_input(data_dir, motion_key):
    """Validate one raw robot/object pair before launching Isaac Sim."""
    data_dir = os.path.abspath(data_dir)
    robot_path = os.path.join(data_dir, "robot", f"{motion_key}.pkl")
    object_path = os.path.join(data_dir, "objects", f"{motion_key}.pkl")
    meta_path = os.path.join(data_dir, "meta", f"{motion_key}.pkl")
    usd_path = next(
        (
            os.path.join(data_dir, "object_usd", f"{motion_key}{ext}")
            for ext in (".usd", ".usda")
            if os.path.isfile(os.path.join(data_dir, "object_usd", f"{motion_key}{ext}"))
        ),
        None,
    )
    missing = [
        label
        for label, path in (
            ("robot", robot_path),
            ("objects", object_path),
            ("meta", meta_path),
            ("object_usd", usd_path),
        )
        if not path or not os.path.isfile(path)
    ]
    if missing:
        raise ValueError("missing paired input: " + ", ".join(missing))

    robot_key, robot = _single_motion_entry(joblib.load(robot_path), motion_key, "robot")
    object_key, obj = _single_motion_entry(joblib.load(object_path), motion_key, "object")
    for field in ("dof", "root_trans_offset", "root_rot"):
        if field not in robot:
            raise ValueError(f"robot motion is missing {field}")
    for field in ("root_pos", "root_quat"):
        if field not in obj:
            raise ValueError(f"object motion is missing {field}")

    dof = np.asarray(robot["dof"])
    root_pos = np.asarray(robot["root_trans_offset"])
    root_rot = np.asarray(robot["root_rot"])
    if dof.ndim != 2 or dof.shape[1] not in (29, 43):
        raise ValueError(f"robot dof must have shape (T,29) or (T,43), got {dof.shape}")
    total_frames = int(dof.shape[0])
    if total_frames < 2:
        raise ValueError("robot motion must contain at least two frames")
    if root_pos.shape != (total_frames, 3):
        raise ValueError(f"root_trans_offset must have shape ({total_frames},3), got {root_pos.shape}")
    if root_rot.shape != (total_frames, 4):
        raise ValueError(f"root_rot must have shape ({total_frames},4), got {root_rot.shape}")
    if "hand_dof_pos" in robot:
        hand_dof = np.asarray(robot["hand_dof_pos"])
        if hand_dof.shape != (total_frames, 14):
            raise ValueError(
                f"hand_dof_pos must have shape ({total_frames},14), got {hand_dof.shape}"
            )

    object_pos = _object_frame_array(obj["root_pos"], 3, "object root_pos")
    object_quat = _object_frame_array(obj["root_quat"], 4, "object root_quat")
    if object_pos.shape[0] != total_frames or object_quat.shape[0] != total_frames:
        raise ValueError(
            f"robot/object frame mismatch: robot={total_frames}, "
            f"object_pos={object_pos.shape[0]}, object_quat={object_quat.shape[0]}"
        )

    required_arrays = {
        "robot dof": dof,
        "robot root_trans_offset": root_pos,
        "robot root_rot": root_rot,
        "object root_pos": object_pos,
        "object root_quat": object_quat,
    }
    if "hand_dof_pos" in robot:
        required_arrays["robot hand_dof_pos"] = np.asarray(robot["hand_dof_pos"])
    for label, array in required_arrays.items():
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{label} is non-numeric or contains NaN/Inf")

    # Validate every numeric frame-aligned auxiliary field as well.  These
    # fields are preserved during clean export and sliced during repair, so a
    # corrupt pose_aa/smpl_joints/hand-action array must not slip through just
    # because the minimum replay fields are valid.
    for entry_label, entry in (("robot", robot), ("object", obj)):
        for field, value in entry.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_frames:
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise ValueError(f"{entry_label} {field} contains NaN/Inf")

    try:
        meta = joblib.load(meta_path)
    except Exception as exc:
        raise ValueError(f"cannot load paired meta file: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError("meta payload is not a dictionary")

    fps = float(robot.get("fps", obj.get("fps", 30.0)))
    object_fps = float(obj.get("fps", fps))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid robot fps: {fps}")
    if not np.isfinite(object_fps) or not np.isclose(fps, object_fps):
        raise ValueError(f"robot/object fps mismatch: {fps} vs {object_fps}")
    return {
        "motion_key": motion_key,
        "robot_key": str(robot_key),
        "object_key": str(object_key),
        "total_frames": total_frames,
        "fps": fps,
        "usd_path": usd_path,
    }


def _atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def _refresh_report_summary(report):
    summary = {"clean": 0, "repaired": 0, "rejected": 0, "input_error": 0, "pending": 0}
    reason_counts = {}
    for entry in report.get("motions", {}).values():
        status = entry.get("status", "pending")
        summary[status] = summary.get(status, 0) + 1
        reason_code = str(entry.get("reason_code", "unknown"))
        reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
    report["summary"] = summary
    report["statistics"] = {
        "total": len(report.get("motions", {})),
        "by_status": dict(summary),
        "by_reason_code": dict(sorted(reason_counts.items())),
    }
    report["updated_at_unix"] = time.time()


def print_clean_statistics(report):
    """Print final counts by processing status and detailed outcome."""
    _refresh_report_summary(report)
    statistics = report["statistics"]
    print("\nCleaning statistics:", flush=True)
    print(f"  total: {statistics['total']}", flush=True)
    print("  by status:", flush=True)
    for status in ("clean", "repaired", "rejected", "input_error", "pending"):
        print(f"    - {status}: {statistics['by_status'].get(status, 0)}", flush=True)
    for status in sorted(set(statistics["by_status"]) - {
        "clean", "repaired", "rejected", "input_error", "pending"
    }):
        print(f"    - {status}: {statistics['by_status'][status]}", flush=True)
    print("  by case (reason_code):", flush=True)
    if statistics["by_reason_code"]:
        for reason_code, count in statistics["by_reason_code"].items():
            print(f"    - {reason_code}: {count}", flush=True)
    else:
        print("    - none: 0", flush=True)


def update_clean_report(report, report_path, motion_key=None, entry=None):
    if motion_key is not None and entry is not None:
        report.setdefault("motions", {})[motion_key] = entry
    _refresh_report_summary(report)
    _atomic_write_json(report_path, report)


def _atomic_copy(source, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_path = f"{destination}.tmp"
    shutil.copy2(source, tmp_path)
    os.replace(tmp_path, destination)


def _copy_motion_assets(data_dir, output_dir, motion_key):
    """Copy static metadata, USD, textures, and BPS data for an accepted motion."""
    meta_src = os.path.join(data_dir, "meta", f"{motion_key}.pkl")
    _atomic_copy(meta_src, os.path.join(output_dir, "meta", f"{motion_key}.pkl"))

    for ext in (".usd", ".usda"):
        usd_src = os.path.join(data_dir, "object_usd", f"{motion_key}{ext}")
        if os.path.isfile(usd_src):
            _atomic_copy(usd_src, os.path.join(output_dir, "object_usd", f"{motion_key}{ext}"))

    bps_src = os.path.join(data_dir, "bps", f"{motion_key}.npy")
    if os.path.isfile(bps_src):
        _atomic_copy(bps_src, os.path.join(output_dir, "bps", f"{motion_key}.npy"))

    texture_root = os.path.join(data_dir, "object_usd", "textures")
    nested_src = os.path.join(texture_root, motion_key)
    nested_dst = os.path.join(output_dir, "object_usd", "textures", motion_key)
    if os.path.isdir(nested_src):
        os.makedirs(nested_dst, exist_ok=True)
        for root, _, filenames in os.walk(nested_src):
            rel = os.path.relpath(root, nested_src)
            for filename in filenames:
                src = os.path.join(root, filename)
                dst_root = nested_dst if rel == "." else os.path.join(nested_dst, rel)
                _atomic_copy(src, os.path.join(dst_root, filename))
    for src in glob.glob(os.path.join(texture_root, f"{motion_key}_*")):
        if os.path.isfile(src):
            _atomic_copy(
                src,
                os.path.join(output_dir, "object_usd", "textures", os.path.basename(src)),
            )


def _atomic_joblib_dump(payload, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_path = f"{destination}.tmp"
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, destination)


def verify_exported_pair(output_dir, motion_key):
    robot_path = os.path.join(output_dir, "robot", f"{motion_key}.pkl")
    object_path = os.path.join(output_dir, "objects", f"{motion_key}.pkl")
    _, robot = _single_motion_entry(joblib.load(robot_path), motion_key, "exported robot")
    _, obj = _single_motion_entry(joblib.load(object_path), motion_key, "exported object")
    dof = np.asarray(robot["dof"])
    object_pos = _object_frame_array(obj["root_pos"], 3, "exported object root_pos")
    object_quat = _object_frame_array(obj["root_quat"], 4, "exported object root_quat")
    if dof.ndim != 2 or dof.shape[0] != object_pos.shape[0] or dof.shape[0] != object_quat.shape[0]:
        raise ValueError(
            f"exported robot/object frame mismatch: {dof.shape[0]}, "
            f"{object_pos.shape[0]}, {object_quat.shape[0]}"
        )
    for label, array in (("dof", dof), ("object_pos", object_pos), ("object_quat", object_quat)):
        if not np.isfinite(array).all():
            raise ValueError(f"exported {label} contains NaN/Inf")
    robot_fps = float(robot.get("fps", 30.0))
    object_fps = float(obj.get("fps", robot_fps))
    if not np.isclose(robot_fps, object_fps):
        raise ValueError(f"exported fps mismatch: {robot_fps} vs {object_fps}")
    return int(dof.shape[0])


def export_accepted_motion(data_dir, output_dir, motion_key, crop_start_frame=0):
    """Export a clean motion verbatim or a synchronously cropped robot/object pair."""
    robot_src = os.path.join(data_dir, "robot", f"{motion_key}.pkl")
    object_src = os.path.join(data_dir, "objects", f"{motion_key}.pkl")
    robot_dst = os.path.join(output_dir, "robot", f"{motion_key}.pkl")
    object_dst = os.path.join(output_dir, "objects", f"{motion_key}.pkl")

    if crop_start_frame <= 0:
        _atomic_copy(robot_src, robot_dst)
        _atomic_copy(object_src, object_dst)
    else:
        robot_key, robot = _single_motion_entry(joblib.load(robot_src), motion_key, "robot")
        object_key, obj = _single_motion_entry(joblib.load(object_src), motion_key, "object")
        total_frames = int(np.asarray(robot["dof"]).shape[0])
        if not 0 < crop_start_frame < total_frames:
            raise ValueError(
                f"invalid crop_start_frame {crop_start_frame} for {total_frames} frames"
            )
        cropped_robot = _slice_frame_aligned_mapping(
            robot, crop_start_frame, total_frames, total_frames
        )
        cropped_object = _slice_object_motion_mapping(
            obj, crop_start_frame, total_frames, total_frames
        )
        _atomic_joblib_dump({robot_key: cropped_robot}, robot_dst)
        _atomic_joblib_dump({object_key: cropped_object}, object_dst)

    _copy_motion_assets(data_dir, output_dir, motion_key)
    return verify_exported_pair(output_dir, motion_key)


def render_all(
    plan,
    resolution=(1920, 1080),
    camera_offset=(-3.54, 0.0, 1.2),
    camera_target=(0.0, 0.0, 0.8),
    headless=True,
    start_frame_skip=0,
    segmented_output_dir=None,
    source_robot_dir=None,
    post_penetration_frames=20,
    record_video=True,
    segmented_object_output_dir=None,
    source_object_dir=None,
    clean_data_dir=None,
    clean_output_dir=None,
    clean_report=None,
    clean_report_path=None,
    thresholds=None,
):
    """Replay all trajectories with one IsaacSim session, optionally recording video."""
    thresholds = thresholds or CleaningThresholds()
    if record_video:
        import imageio
    import torch

    # ---- Launch IsaacSim ----
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=headless, enable_cameras=record_video)
    simulation_app = launcher.app

    import carb
    import isaaclab.sim as sim_utils
    from omni.physx import get_physx_simulation_interface
    import omni.usd
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.sim import SimulationContext
    from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdLux, UsdShade

    if record_video:
        from isaaclab.sensors import Camera, CameraCfg

    penetration_frames_by_motion.clear()

    # Inline G1_43DOF_CFG to avoid importing groot.rl (which drags in unsynced code)
    G1_43DOF_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="imports/SONIC/gear_sonic/data/robots/g1/g1_43dof_s3.usda",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76)),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    # ---- Physics simulation (CPU, kinematic only) ----
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 50.0,
        render_interval=1,
        device="cpu",
        use_fabric=True,
        gravity=(0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)

    # ---- Ground + lighting ----
    ground_cfg = sim_utils.GroundPlaneCfg(size=(500.0, 500.0))
    ground_cfg.func("/World/ground", ground_cfg)

    dome_cfg = sim_utils.DomeLightCfg(
        color=(0.45, 0.55, 0.75),
        intensity=1500.0,
    )
    dome_cfg.func("/World/DomeLight", dome_cfg)

    stage = omni.usd.get_context().get_stage()
    dome_prim = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
    if dome_prim.GetPrim().IsValid():
        dome_prim.GetTextureFileAttr().Set("")

    # ---- Robot (43-DOF, HOI) ----
    env_path = "/World/envs/env_0"
    env_prim = stage.DefinePrim(env_path, "Xform")

    robot_cfg = G1_43DOF_CFG.copy()
    robot_path = f"{env_path}/Robot"
    robot_cfg.spawn.func(robot_path, robot_cfg.spawn, translation=(0, 0, 0))
    # IsaacLab disables contact processing globally unless a ContactSensor is
    # present.  This replay script uses the lower-level PhysX contact report
    # API directly, so enable processing after the USD spawner has activated
    # the robot contact reporters and before physics initialization.
    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)

    # ---- Camera ----
    camera = None
    if record_video:
        w, h = resolution
        camera_cfg = CameraCfg(
            prim_path="/World/OverviewCamera",
            offset=CameraCfg.OffsetCfg(
                pos=camera_offset, rot=(1, 0, 0, 0), convention="world"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=5.0,
                focus_distance=100.0,
                horizontal_aperture=10.0,
                clipping_range=(0.1, 500.0),
            ),
            width=w,
            height=h,
        )
        camera = Camera(camera_cfg)

    # ---- Articulation wrapper ----
    robot_cfg.prim_path = "/World/envs/env_.*/Robot"
    art = Articulation(robot_cfg)

    # ---- Initialize ----
    sim.reset()
    if camera is not None:
        camera.reset()
    art.reset()
    print(
        f"[DOF] Articulation num_joints = {int(art.num_joints)} "
        f"(body-only trajectories will be zero-padded to this width)",
        flush=True,
    )
    required_body_names = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")
    missing_body_names = [name for name in required_body_names if name not in art.body_names]
    if missing_body_names:
        raise RuntimeError("Missing gait-analysis bodies: " + ", ".join(missing_body_names))
    body_indices = {name: art.body_names.index(name) for name in required_body_names}

    # Contact reports are used only as an observation channel.  The robot USD
    # is spawned with activate_contact_sensors=True, so PhysX emits contact
    # data for its rigid bodies without changing the replay dynamics.
    robot_root = f"{env_path}/Robot"
    contact_frame_state = {"max_penetration": 0.0}

    def _contact_path(value):
        """Convert a PhysX actor/collider id or USD path to a string path."""
        if value is None:
            return ""
        path_string = getattr(value, "pathString", None)
        if path_string:
            return str(path_string)
        if isinstance(value, str):
            return value
        try:
            return str(PhysicsSchemaTools.intToSdfPath(int(value)))
        except (TypeError, ValueError, RuntimeError):
            return str(value)

    def _best_contact_path(header, actor_name, collider_name):
        """Prefer the actor/collider path that identifies a monitored link."""
        candidates = [
            _contact_path(getattr(header, actor_name, None)),
            _contact_path(getattr(header, collider_name, None)),
        ]
        for candidate in candidates:
            if any(
                _path_matches_link(candidate, robot_root, link)
                for link in _MONITORED_HAND_LINKS | _MONITORED_BODY_LINKS
            ):
                return candidate
        return next((candidate for candidate in candidates if candidate), "")

    def _on_contact_report(contact_headers, contact_data):
        """Accumulate the deepest monitored hand-body contact for this update."""
        for header in contact_headers:
            path0 = _best_contact_path(header, "actor0", "collider0")
            path1 = _best_contact_path(header, "actor1", "collider1")
            if not _monitored_contact_pair(path0, path1, robot_root):
                continue

            offset = int(getattr(header, "contact_data_offset", 0))
            count = int(getattr(header, "num_contact_data", 0))
            for contact in contact_data[offset : offset + count]:
                separation = getattr(contact, "separation", None)
                try:
                    depth = max(0.0, -float(separation))
                except (TypeError, ValueError):
                    continue
                contact_frame_state["max_penetration"] = max(
                    contact_frame_state["max_penetration"], depth
                )

    try:
        physx_interface = get_physx_simulation_interface()
        contact_report_subscription = physx_interface.subscribe_contact_report_events(
            _on_contact_report
        )
    except Exception as exc:
        raise RuntimeError("Failed to subscribe to Isaac Sim PhysX contact reports") from exc

    missing_links = [
        link
        for link in sorted(_MONITORED_HAND_LINKS | _MONITORED_BODY_LINKS)
        if not stage.GetPrimAtPath(f"{robot_root}/{link}").IsValid()
    ]
    if missing_links:
        raise RuntimeError(
            "Missing monitored G1 link prims: " + ", ".join(missing_links)
        )
    print(
        f"[PENETRATION] Monitoring {len(_MONITORED_HAND_LINKS)} hand links against "
        f"{len(_MONITORED_BODY_LINKS)} body links; threshold="
        f"{thresholds.penetration_depth_m * 1000:.1f} mm",
        flush=True,
    )

    if camera is not None:
        cam_device = getattr(camera, "_device", "cpu")
        eye = torch.tensor([list(camera_offset)], dtype=torch.float32, device=cam_device)
        tgt = torch.tensor([list(camera_target)], dtype=torch.float32, device=cam_device)
        camera.set_world_poses_from_view(eye, tgt)

        # ---- Warmup shaders ----
        for _ in range(5):
            sim.step(render=True)
            camera.update(dt=0.0)
        print("IsaacSim initialized, shaders warmed up")
    else:
        print("IsaacSim initialized without camera/video recording")
    start_watchdog()
    _heartbeat("render_loop_start")

    # ---- Render loop ----
    current_obj_usd = None
    prev_obj_path = None
    total = len(plan)
    succeeded = 0
    failed = 0
    start_time = time.time()

    # --- Scene-yaw helpers (match multi_scene_render.py cosmetic rotation) ---
    def _quat_mul_wxyz(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    def _rotate_xy(x, y, cos_a, sin_a):
        return x * cos_a - y * sin_a, x * sin_a + y * cos_a

    for idx, (env_idx, motion_key, traj_path, usd_path, output_path) in enumerate(plan):
        elapsed = time.time() - start_time
        eta = (elapsed / max(idx, 1)) * (total - idx)
        print(
            f"\n[{idx+1}/{total}] {motion_key} (env {env_idx:06d}) ETA: {eta/60:.0f}m", flush=True
        )
        _heartbeat(f"motion_{idx}_{motion_key}_load_traj")

        try:
            # Load trajectory
            with open(traj_path, "rb") as f:
                traj = pickle.load(f)

            # ---- Swap object USD if needed ----
            # Unique prim path per iteration: RemovePrim doesn't reliably clear USD
            # references at the same path, so prior object geometry leaks through.
            obj_path = f"{env_path}/Object_{idx:06d}"
            if usd_path != current_obj_usd:
                # Delete previous iteration's object prim
                if prev_obj_path and stage.GetPrimAtPath(prev_obj_path).IsValid():
                    stage.RemovePrim(prev_obj_path)
                # Delete old table
                for p in [f"{env_path}/Table"] + [f"{env_path}/TableLeg_{j}" for j in range(4)]:
                    if stage.GetPrimAtPath(p).IsValid():
                        stage.RemovePrim(p)

                # Spawn new object
                if usd_path and os.path.exists(usd_path):
                    # No rigid_props: we set xform ops directly each frame for
                    # kinematic replay. RigidBody with kinematic_enabled=True
                    # caches its own pose via PhysX/Fabric and silently overrides
                    # our xformOp updates after the first sim step, which
                    # produced stale/inconsistent stair orientations across
                    # frames. Raw USD reference lets orient_op.Set() take effect.
                    obj_spawn_cfg = sim_utils.UsdFileCfg(
                        usd_path=os.path.abspath(usd_path),
                    )
                    obj_spawn_cfg.func(obj_path, obj_spawn_cfg, translation=(0, 0, 0))

                    # Assign fallback color to meshes whose materials are missing or
                    # bound to orphaned/broken shaders (some USDs reference MDL textures
                    # that don't ship with the asset).
                    obj_prim = stage.GetPrimAtPath(obj_path)
                    if obj_prim.IsValid():
                        for desc in Usd.PrimRange(obj_prim):
                            if desc.GetTypeName() == "Mesh":
                                mesh = UsdGeom.Mesh(desc)
                                binding_api = UsdShade.MaterialBindingAPI(desc)
                                bound_mat = binding_api.GetDirectBinding().GetMaterial()
                                needs_fallback = False
                                if not bound_mat.GetPrim().IsValid():
                                    needs_fallback = True
                                else:
                                    # Material exists — check that its surface shader
                                    # actually resolves to an asset. MDL shaders with a
                                    # missing info:mdl:sourceAsset render invisible.
                                    try:
                                        surface = bound_mat.ComputeSurfaceSource()
                                        shader = (
                                            surface[0] if isinstance(surface, tuple) else surface
                                        )
                                        if shader and shader.GetPrim().IsValid():
                                            src_input = shader.GetInput("info:mdl:sourceAsset")
                                            if src_input is not None:
                                                src_asset = src_input.Get()
                                                if src_asset is None or not str(src_asset):
                                                    needs_fallback = True
                                        else:
                                            needs_fallback = True
                                    except Exception:
                                        needs_fallback = True
                                if needs_fallback:
                                    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.6, 0.65)])

                # Spawn table if trajectory has table data above ground
                if traj.get("table_pos_w") is not None and float(traj["table_pos_w"][0][2]) >= 0.1:
                    table_pos = traj["table_pos_w"][0].copy()
                    table_z = float(table_pos[2])
                    table_w, table_d, table_t = 2.0, 0.6, 0.04

                    table_prim = stage.DefinePrim(f"{env_path}/Table", "Cube")
                    cube = UsdGeom.Cube(table_prim)
                    cube.GetSizeAttr().Set(1.0)
                    xf = UsdGeom.Xformable(table_prim)
                    # Translate MUST come before Scale (USD applies outermost-first;
                    # [T,S] -> scale local point, then translate in parent frame)
                    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, table_z))
                    xf.AddScaleOp().Set(Gf.Vec3f(table_w, table_d, table_t))
                    cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.35, 0.2)])

                    # Table legs
                    leg_height = table_z - table_t / 2.0
                    if leg_height > 0:
                        leg_w = 0.04
                        hw, hd = table_w / 2.0, table_d / 2.0
                        inset, dinset = 0.03, 0.10
                        corners = [
                            (hw - inset - leg_w / 2, hd - dinset - leg_w / 2),
                            (hw - inset - leg_w / 2, -(hd - dinset - leg_w / 2)),
                            (-(hw - inset - leg_w / 2), hd - dinset - leg_w / 2),
                            (-(hw - inset - leg_w / 2), -(hd - dinset - leg_w / 2)),
                        ]
                        for j, (lx, ly) in enumerate(corners):
                            leg_path = f"{env_path}/TableLeg_{j}"
                            leg_prim = stage.DefinePrim(leg_path, "Cube")
                            leg_cube = UsdGeom.Cube(leg_prim)
                            leg_cube.GetSizeAttr().Set(1.0)
                            leg_xf = UsdGeom.Xformable(leg_prim)
                            # Translate before Scale (same convention as multi_scene_render.py)
                            leg_xf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, leg_height / 2.0))
                            leg_xf.AddScaleOp().Set(Gf.Vec3f(leg_w, leg_w, leg_height))
                            leg_cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.28, 0.15)])

                current_obj_usd = usd_path
                prev_obj_path = obj_path
                # Let IsaacSim process the new prims
                sim.step(render=record_video)
            else:
                # Same USD as previous iteration — reuse the existing prim path
                obj_path = prev_obj_path or obj_path

            # ---- Determine scene center ----
            has_table = (
                traj.get("table_pos_w") is not None and float(traj["table_pos_w"][0][2]) >= 0.1
            )
            has_obj = traj.get("object_pos_w") is not None

            if has_table:
                cx, cy = float(traj["table_pos_w"][0][0]), float(traj["table_pos_w"][0][1])
            elif has_obj:
                cx, cy = float(traj["object_pos_w"][0][0]), float(traj["object_pos_w"][0][1])
            else:
                mid = traj["total_frames"] // 2
                cx, cy = float(traj["root_pos_w"][mid][0]), float(traj["root_pos_w"][mid][1])

            # ---- Scene yaw (cosmetic rotation mirroring multi_scene_render.py) ----
            # Stairs/terrain/sitting motions are stored in a canonical source frame;
            # the preview renderer rotates the whole scene (robot + object XY & quat)
            # so the natural rise/seat direction aligns with the camera view. The
            # relative pose between robot and object is preserved (rigid-body yaw).
            path_blob = f"{motion_key}|{traj_path}|{usd_path}|{output_path}".lower()
            obj_on_ground = has_obj and abs(float(traj["object_pos_w"][0][2])) < 0.05
            # Stair/sitting motions: source motion lib stores a FIXED object quat
            # that, applied alone, tips the USD in world frame. Matching the
            # reference preview renderer (fancy-eval-render/multi_scene_render.py),
            # we compose an extra Z-axis yaw before applying (135° for stairs, 180°
            # for sitting) to restore upright orientation and align camera framing.
            if "sitting" in path_blob:
                scene_yaw = float(np.pi)
            elif "stair" in path_blob:
                scene_yaw = float(135.0 * np.pi / 180.0)
            elif obj_on_ground:
                scene_yaw = float(135.0 * np.pi / 180.0)
            else:
                scene_yaw = 0.0
            override_obj_quat_identity = False
            scene_cos = float(np.cos(scene_yaw))
            scene_sin = float(np.sin(scene_yaw))
            scene_yaw_quat = np.array(
                [np.cos(scene_yaw / 2.0), 0.0, 0.0, np.sin(scene_yaw / 2.0)],
                dtype=np.float32,
            )
            if idx == 0 or scene_yaw != 0:
                print(f"  scene_yaw={np.degrees(scene_yaw):.0f}deg", flush=True)

            # ---- Replay trajectory frames ----
            fps = traj.get("fps", 25.0)
            total_frames = traj["total_frames"]
            skip = compute_start_frame_skip(total_frames, start_frame_skip) if record_video else 0
            if record_video and skip and idx == 0:
                print(f"[FRAME] Skipping first {skip} frame(s) per --start_frame_skip", flush=True)

            writer = None
            if record_video:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                try:
                    writer = imageio.get_writer(
                        output_path, fps=fps, codec="libx264", quality=5, pixelformat="yuv420p"
                    )
                except Exception as e:
                    print(f"  FAILED to open writer: {e}")
                    failed += 1
                    continue

            # Detection covers every source frame, including frames omitted
            # from the video by --start_frame_skip.
            penetration_depths = np.zeros(total_frames, dtype=np.float32)
            pelvis_positions = np.zeros((total_frames, 3), dtype=np.float32)
            left_foot_positions = np.zeros((total_frames, 3), dtype=np.float32)
            right_foot_positions = np.zeros((total_frames, 3), dtype=np.float32)

            # Robot articulation expects `num_art_dofs` joints. Trajectories may be
            # body-only (e.g. 29 DOF for G1 terrain/sitting) while the articulation is
            # 43-DOF (body + hands) — pad the trailing hand slots with zeros so
            # write_joint_state_to_sim gets the right shape. Works for any robot whose
            # traj DOFs are a leading subset of articulation DOFs.
            num_art_dofs = int(art.num_joints)
            traj_dofs = int(traj["dof_pos"].shape[1])
            pad_dofs = num_art_dofs - traj_dofs
            if pad_dofs < 0:
                raise RuntimeError(
                    f"traj has {traj_dofs} DOFs but articulation only has {num_art_dofs} — "
                    f"check robot config matches trajectory source"
                )
            if idx == 0:
                if pad_dofs > 0:
                    print(
                        f"[DOF] Padding {traj_dofs}-DOF trajectories to "
                        f"{num_art_dofs}-DOF articulation ({pad_dofs} zero hand DOFs)",
                        flush=True,
                    )
                else:
                    print(
                        f"[DOF] Trajectory DOFs match articulation "
                        f"({num_art_dofs}) — no padding needed",
                        flush=True,
                    )

            for f in range(total_frames):
                if f % 25 == 0:
                    _heartbeat(f"motion_{idx}_{motion_key}_frame_{f}")
                contact_frame_state["max_penetration"] = 0.0
                # Robot state (pad to 43 DOFs when traj is body-only 29)
                dof_row = traj["dof_pos"][f]
                if pad_dofs > 0:
                    dof_row = np.concatenate([dof_row, np.zeros(pad_dofs, dtype=dof_row.dtype)])
                joint_pos = torch.from_numpy(dof_row).float().unsqueeze(0)
                joint_vel = torch.zeros_like(joint_pos)
                root_state = torch.zeros(1, 13, device="cpu")

                root_pos = traj["root_pos_w"][f].copy()
                root_pos[0] -= cx
                root_pos[1] -= cy
                if scene_yaw != 0.0:
                    rx, ry = _rotate_xy(root_pos[0], root_pos[1], scene_cos, scene_sin)
                    root_pos[0], root_pos[1] = rx, ry
                root_state[0, :3] = torch.from_numpy(root_pos).float()
                root_quat_f = np.asarray(traj["root_quat_w"][f], dtype=np.float32)
                if scene_yaw != 0.0:
                    root_quat_f = _quat_mul_wxyz(scene_yaw_quat, root_quat_f)
                root_state[0, 3:7] = torch.from_numpy(root_quat_f).float()

                art.write_joint_state_to_sim(joint_pos, joint_vel)
                art.write_root_state_to_sim(root_state)

                # Object state
                obj_prim = stage.GetPrimAtPath(obj_path)
                if has_obj and obj_prim.IsValid():
                    obj_pos = traj["object_pos_w"][f].copy()
                    if override_obj_quat_identity:
                        obj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        obj_quat = np.asarray(traj["object_quat_w"][f], dtype=np.float32)  # wxyz
                    ox = float(obj_pos[0]) - cx
                    oy = float(obj_pos[1]) - cy
                    if scene_yaw != 0.0:
                        ox, oy = _rotate_xy(ox, oy, scene_cos, scene_sin)
                        obj_quat = _quat_mul_wxyz(scene_yaw_quat, obj_quat)

                    xformable = UsdGeom.Xformable(obj_prim)
                    ops = xformable.GetOrderedXformOps()
                    translate_op = orient_op = None
                    for op in ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            translate_op = op
                        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                            orient_op = op
                    if translate_op is None:
                        translate_op = xformable.AddTranslateOp()
                    if orient_op is None:
                        orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
                    translate_op.Set(Gf.Vec3d(ox, oy, float(obj_pos[2])))
                    # obj_quat is wxyz end-to-end (convert_hoi_to_motion_lib.py stores
                    # source wxyz; motion_lib passes bytes verbatim to IsaacLab's
                    # write_root_state_to_sim which expects wxyz). USD Gf.Quatf takes
                    # (real, i, j, k) == (w, x, y, z), so [0..3] maps through directly.
                    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                        orient_op.Set(
                            Gf.Quatf(
                                float(obj_quat[0]),
                                float(obj_quat[1]),
                                float(obj_quat[2]),
                                float(obj_quat[3]),
                            )
                        )
                    else:
                        orient_op.Set(
                            Gf.Quatd(
                                float(obj_quat[0]),
                                float(obj_quat[1]),
                                float(obj_quat[2]),
                                float(obj_quat[3]),
                            )
                        )
                    if f == skip and idx == 0:
                        # Confirm the orient op actually took the stored value
                        post_ops = xformable.GetOrderedXformOps()
                        for o in post_ops:
                            try:
                                print(f"[DBG-POST] {o.GetOpName()} = {o.Get()}", flush=True)
                            except Exception:
                                print(f"[DBG-POST] {o.GetOpName()} (err)", flush=True)

                # Table state
                table_prim = stage.GetPrimAtPath(f"{env_path}/Table")
                if has_table and table_prim.IsValid():
                    tp = traj["table_pos_w"][f].copy()
                    tx = float(tp[0]) - cx
                    ty = float(tp[1]) - cy
                    tz = float(tp[2])

                    table_xf = UsdGeom.Xformable(table_prim)
                    t_ops = table_xf.GetOrderedXformOps()
                    t_translate = None
                    for op in t_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            t_translate = op
                    if t_translate is None:
                        t_translate = table_xf.AddTranslateOp()
                    t_translate.Set(Gf.Vec3d(tx, ty, tz))

                    # Update legs
                    leg_height = tz - 0.02
                    if leg_height > 0:
                        leg_w = 0.04
                        hw, hd = 1.0, 0.4
                        inset, dinset = 0.03, 0.10
                        corners = [
                            (tx + hw - inset - leg_w / 2, ty + hd - dinset - leg_w / 2),
                            (tx + hw - inset - leg_w / 2, ty - (hd - dinset - leg_w / 2)),
                            (tx - (hw - inset - leg_w / 2), ty + hd - dinset - leg_w / 2),
                            (tx - (hw - inset - leg_w / 2), ty - (hd - dinset - leg_w / 2)),
                        ]
                        for j, (lx, ly) in enumerate(corners):
                            leg_prim = stage.GetPrimAtPath(f"{env_path}/TableLeg_{j}")
                            if leg_prim.IsValid():
                                leg_xf = UsdGeom.Xformable(leg_prim)
                                l_ops = leg_xf.GetOrderedXformOps()
                                l_tr = None
                                for op in l_ops:
                                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                        l_tr = op
                                if l_tr is None:
                                    l_tr = leg_xf.AddTranslateOp()
                                l_tr.Set(Gf.Vec3d(lx, ly, leg_height / 2.0))

                # PhysX contact reports are emitted during a physics update;
                # SimulationContext.forward() only flushes kinematics/Fabric.
                # Probe contacts without rendering, then restore the exact
                # replay state before the frame is captured.
                sim.step(render=False)

                penetration_depths[f] = contact_frame_state["max_penetration"]

                # The probe may depenetrate dynamic articulation bodies.  Put
                # the recorded state back before rendering so collision
                # response never changes the replay video.
                art.write_joint_state_to_sim(joint_pos, joint_vel)
                art.write_root_state_to_sim(root_state)
                sim.forward()
                art.update(sim_cfg.dt)
                body_pos_w = art.data.body_pos_w[0]
                pelvis_positions[f] = (
                    body_pos_w[body_indices["pelvis"]].detach().cpu().numpy()
                )
                left_foot_positions[f] = (
                    body_pos_w[body_indices["left_ankle_roll_link"]].detach().cpu().numpy()
                )
                right_foot_positions[f] = (
                    body_pos_w[body_indices["right_ankle_roll_link"]].detach().cpu().numpy()
                )

                # --start_frame_skip affects only video output, never detection.
                if record_video and f >= skip:
                    sim.render()
                    camera.update(dt=0.0)
                    rgb = camera.data.output["rgb"][0].cpu().numpy()
                    writer.append_data(rgb)

            if writer is not None:
                writer.close()
            penetration_frames = effective_penetration_mask(penetration_depths, thresholds)
            penetration_frames_by_motion[motion_key] = penetration_frames.astype(np.uint8)
            render_time = (time.time() - start_time) / max(idx + 1, 1)
            if record_video:
                print(
                    f"  OK: {output_path} ({total_frames - skip} frames, {render_time:.1f}s avg)",
                    flush=True,
                )
            else:
                print(
                    f"  OK: {motion_key} (video disabled, {total_frames} frames processed, "
                    f"{render_time:.1f}s avg)",
                    flush=True,
                )
            lift_metrics = analyze_object_lift(traj["object_pos_w"], fps, thresholds)
            gait_metrics = analyze_gait(
                pelvis_positions,
                left_foot_positions,
                right_foot_positions,
                fps,
                thresholds,
            )
            penetration_metrics = analyze_penetration(penetration_depths, fps, thresholds)

            decision = classify_motion(lift_metrics, gait_metrics, penetration_metrics)
            status = decision["status"]
            reason_code = decision["reason_code"]
            crop_start = decision["crop_start_frame"]

            output_frames = 0
            if status in ("clean", "repaired"):
                if not clean_data_dir or not clean_output_dir:
                    raise RuntimeError("clean data/output directories are required for export")
                output_frames = export_accepted_motion(
                    clean_data_dir,
                    clean_output_dir,
                    motion_key,
                    crop_start_frame=crop_start,
                )

            report_entry = {
                "status": status,
                "reason_code": reason_code,
                "gait_warning": bool(gait_metrics["warning"]),
                "original_frames": int(total_frames),
                "output_frames": int(output_frames),
                "fps": float(fps),
                "crop_start_frame": int(crop_start),
                "crop_seconds": float(crop_start / fps),
                "lift": lift_metrics,
                "gait": gait_metrics,
                "penetration": penetration_metrics,
            }
            if clean_report is not None and clean_report_path:
                update_clean_report(
                    clean_report,
                    clean_report_path,
                    motion_key=motion_key,
                    entry=report_entry,
                )
            print(
                f"  [CLEAN] status={status} reason={reason_code} "
                f"frames={total_frames}->{output_frames}",
                flush=True,
            )
            succeeded += 1
            _heartbeat(f"motion_{idx}_{motion_key}_done")
        except Exception as e:
            failed += 1
            print(f"  FAILED {motion_key}: {e}", flush=True)
            traceback.print_exc()
            # Remove any partial output so a retry re-renders it cleanly
            try:
                if record_video and os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            if clean_report is not None and clean_report_path:
                update_clean_report(
                    clean_report,
                    clean_report_path,
                    motion_key=motion_key,
                    entry={
                        "status": "input_error",
                        "reason_code": "runtime_error",
                        "error": str(e),
                    },
                )
            _heartbeat(f"motion_{idx}_{motion_key}_failed")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch render complete: {succeeded}/{total} succeeded, {failed} failed")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/max(total,1):.1f}s per trajectory)")
    if clean_report is not None:
        print_clean_statistics(clean_report)

    # IsaacSim teardown (simulation_app.close()) can hang in USD/Hydra cleanup after
    # rendering many frames. Match eval_agent_trl.py and force-exit the process —
    # all per-motion outputs and report updates were already flushed.  Preserve
    # a non-zero process result when any replay/export failed so schedulers and
    # --resume workflows can distinguish a complete run from a partial one.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failed else 0)


def build_arg_parser():
    defaults = CleaningThresholds()
    parser = argparse.ArgumentParser(
        description="Clean motion-library reference motions with direct kinematic replay"
    )
    parser.add_argument("--data_dir", required=True, help="Input motion-library directory")
    parser.add_argument("--output_dir", required=True, help="Output cleaned motion-library directory")
    parser.add_argument(
        "--quat_convention",
        choices=("xyzw", "wxyz", "auto"),
        default="xyzw",
        help="Input robot root_rot convention (pickup_table default: xyzw)",
    )
    parser.add_argument("--motion_keys", default=None, help="Comma-separated motion keys")
    parser.add_argument("--max_motions", type=int, default=0, help="Maximum motions (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from clean_report.json")
    parser.add_argument("--dry_run", action="store_true", help="Preflight inputs without Isaac Sim")
    parser.add_argument("--diagnostic_video_dir", default=None)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--camera_offset", type=float, nargs=3, default=[-3.54, 0.0, 1.2])
    parser.add_argument("--camera_target", type=float, nargs=3, default=[0.0, 0.0, 0.8])
    parser.add_argument("--headless", action="store_true", default=True)
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument("--record_video", dest="record_video", action="store_true")
    video_group.add_argument("--no_record_video", dest="record_video", action="store_false")
    parser.set_defaults(record_video=False)

    parser.add_argument("--penetration_depth", type=float, default=defaults.penetration_depth_m)
    parser.add_argument(
        "--single_frame_penetration_depth",
        type=float,
        default=defaults.single_frame_penetration_depth_m,
    )
    parser.add_argument(
        "--penetration_min_frames", type=int, default=defaults.penetration_min_frames
    )
    parser.add_argument(
        "--post_penetration_frames", type=int, default=defaults.post_penetration_frames
    )
    parser.add_argument("--max_crop_seconds", type=float, default=defaults.max_crop_seconds)
    parser.add_argument("--max_crop_ratio", type=float, default=defaults.max_crop_ratio)
    parser.add_argument(
        "--min_remaining_seconds", type=float, default=defaults.min_remaining_seconds
    )
    parser.add_argument(
        "--initial_lift_window_seconds",
        type=float,
        default=defaults.initial_lift_window_seconds,
    )
    parser.add_argument(
        "--final_lift_window_seconds",
        type=float,
        default=defaults.final_lift_window_seconds,
    )
    parser.add_argument(
        "--median_lift_threshold", type=float, default=defaults.median_lift_threshold_m
    )
    parser.add_argument(
        "--lifted_frame_threshold", type=float, default=defaults.lifted_frame_threshold_m
    )
    parser.add_argument("--lifted_frame_ratio", type=float, default=defaults.lifted_frame_ratio)
    parser.add_argument(
        "--foot_contact_height", type=float, default=defaults.foot_contact_height_m
    )
    parser.add_argument(
        "--foot_contact_speed", type=float, default=defaults.foot_contact_speed_mps
    )
    parser.add_argument(
        "--contact_debounce_frames", type=int, default=defaults.contact_debounce_frames
    )
    parser.add_argument("--locomotion_speed", type=float, default=defaults.locomotion_speed_mps)
    parser.add_argument(
        "--locomotion_min_seconds", type=float, default=defaults.locomotion_min_seconds
    )
    parser.add_argument(
        "--locomotion_min_displacement",
        type=float,
        default=defaults.locomotion_min_displacement_m,
    )
    parser.add_argument("--micro_step_length", type=float, default=defaults.micro_step_length_m)
    parser.add_argument("--shuffle_min_steps", type=int, default=defaults.shuffle_min_steps)
    parser.add_argument(
        "--shuffle_cadence",
        type=float,
        default=defaults.shuffle_cadence_steps_per_second,
    )
    parser.add_argument(
        "--shuffle_micro_step_ratio",
        type=float,
        default=defaults.shuffle_micro_step_ratio,
    )
    return parser


def thresholds_from_args(args):
    thresholds = replace(
        CleaningThresholds(),
        penetration_depth_m=args.penetration_depth,
        single_frame_penetration_depth_m=args.single_frame_penetration_depth,
        penetration_min_frames=args.penetration_min_frames,
        post_penetration_frames=args.post_penetration_frames,
        max_crop_seconds=args.max_crop_seconds,
        max_crop_ratio=args.max_crop_ratio,
        min_remaining_seconds=args.min_remaining_seconds,
        initial_lift_window_seconds=args.initial_lift_window_seconds,
        final_lift_window_seconds=args.final_lift_window_seconds,
        median_lift_threshold_m=args.median_lift_threshold,
        lifted_frame_threshold_m=args.lifted_frame_threshold,
        lifted_frame_ratio=args.lifted_frame_ratio,
        foot_contact_height_m=args.foot_contact_height,
        foot_contact_speed_mps=args.foot_contact_speed,
        contact_debounce_frames=args.contact_debounce_frames,
        locomotion_speed_mps=args.locomotion_speed,
        locomotion_min_seconds=args.locomotion_min_seconds,
        locomotion_min_displacement_m=args.locomotion_min_displacement,
        micro_step_length_m=args.micro_step_length,
        shuffle_min_steps=args.shuffle_min_steps,
        shuffle_cadence_steps_per_second=args.shuffle_cadence,
        shuffle_micro_step_ratio=args.shuffle_micro_step_ratio,
    )
    numeric = thresholds.to_dict()
    if any(not np.isfinite(value) for value in numeric.values()):
        raise ValueError("all thresholds must be finite")
    if thresholds.penetration_depth_m <= 0 or thresholds.single_frame_penetration_depth_m <= 0:
        raise ValueError("penetration thresholds must be positive")
    if thresholds.single_frame_penetration_depth_m < thresholds.penetration_depth_m:
        raise ValueError("single-frame penetration depth must be >= penetration depth")
    if not 0 <= thresholds.max_crop_ratio <= 1:
        raise ValueError("max_crop_ratio must be in [0,1]")
    if not 0 <= thresholds.lifted_frame_ratio <= 1:
        raise ValueError("lifted_frame_ratio must be in [0,1]")
    if not 0 <= thresholds.shuffle_micro_step_ratio <= 1:
        raise ValueError("shuffle_micro_step_ratio must be in [0,1]")
    return thresholds


def _load_or_create_report(data_dir, output_dir, motion_keys, thresholds, resume):
    report_path = os.path.join(output_dir, "clean_report.json")
    if resume:
        if not os.path.isfile(report_path):
            raise ValueError("--resume requires an existing clean_report.json")
        with open(report_path, encoding="utf-8") as file:
            report = json.load(file)
        if os.path.abspath(report.get("source_data_dir", "")) != data_dir:
            raise ValueError("resume report source_data_dir does not match --data_dir")
        if report.get("thresholds") != thresholds.to_dict():
            raise ValueError("resume thresholds differ from the existing report")
    else:
        report = {
            "format_version": 1,
            "source_data_dir": data_dir,
            "output_data_dir": output_dir,
            "thresholds": thresholds.to_dict(),
            "motions": {},
            "created_at_unix": time.time(),
        }
    for motion_key in motion_keys:
        report.setdefault("motions", {}).setdefault(
            motion_key, {"status": "pending", "reason_code": "not_processed"}
        )
    update_clean_report(report, report_path)
    return report, report_path


def main():
    args = build_arg_parser().parse_args()
    thresholds = thresholds_from_args(args)
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    if data_dir == output_dir:
        raise ValueError("--output_dir must differ from --data_dir")
    robot_dir = os.path.join(data_dir, "robot")
    if not os.path.isdir(robot_dir):
        raise ValueError(f"input robot directory does not exist: {robot_dir}")

    output_exists_nonempty = os.path.isdir(output_dir) and bool(os.listdir(output_dir))
    if output_exists_nonempty and not args.resume:
        raise ValueError("--output_dir is non-empty; use a new directory or pass --resume")
    os.makedirs(output_dir, exist_ok=True)

    all_motion_keys = sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(robot_dir)
        if filename.endswith(".pkl") and not filename.endswith(".trajectory.pkl")
    )
    if args.motion_keys:
        requested = [key.strip() for key in args.motion_keys.split(",") if key.strip()]
        unknown = sorted(set(requested) - set(all_motion_keys))
        if unknown:
            raise ValueError("unknown motion keys: " + ", ".join(unknown))
        motion_keys = requested
    else:
        motion_keys = all_motion_keys
    if args.max_motions > 0:
        motion_keys = motion_keys[: args.max_motions]
    if not motion_keys:
        raise ValueError("no motions selected")

    report, report_path = _load_or_create_report(
        data_dir, output_dir, motion_keys, thresholds, args.resume
    )
    completed = {"clean", "repaired", "rejected"}
    pending_keys = []
    for motion_key in motion_keys:
        previous = report["motions"].get(motion_key, {})
        if args.resume and previous.get("status") in completed:
            continue
        if (
            args.resume
            and previous.get("status") == "input_error"
            and previous.get("reason_code") != "runtime_error"
        ):
            continue
        try:
            metadata = validate_motion_input(data_dir, motion_key)
            report["motions"][motion_key] = {
                "status": "pending",
                "reason_code": "preflight_passed",
                "original_frames": metadata["total_frames"],
                "fps": metadata["fps"],
            }
            pending_keys.append(motion_key)
        except Exception as exc:
            report["motions"][motion_key] = {
                "status": "input_error",
                "reason_code": "preflight_error",
                "error": str(exc),
            }
        update_clean_report(report, report_path)

    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Selected: {len(motion_keys)}, pending replay: {len(pending_keys)}")
    if args.dry_run or not pending_keys:
        print(f"Preflight complete. Report: {report_path}")
        print_clean_statistics(report)
        return

    work_dir = os.path.join(output_dir, ".clean_work")
    from grail.visualization.prepare_vis_shard import convert_motion_lib_to_trajectories

    converted = convert_motion_lib_to_trajectories(
        data_dir,
        work_dir,
        max_motions=0,
        motion_filter=set(pending_keys),
        quat_convention=args.quat_convention,
    )
    missing_converted = sorted(set(pending_keys) - set(converted))
    for motion_key in missing_converted:
        update_clean_report(
            report,
            report_path,
            motion_key=motion_key,
            entry={
                "status": "input_error",
                "reason_code": "trajectory_conversion_failed",
            },
        )

    diagnostic_dir = args.diagnostic_video_dir or os.path.join(output_dir, "diagnostic_videos")
    plan, stats, _, _ = build_render_plan(
        work_dir,
        os.path.join(data_dir, "object_usd"),
        diagnostic_dir,
        traj_dir=os.path.join(work_dir, "trajectories"),
        record_video=args.record_video,
    )
    plan = [item for item in plan if item[1] in set(converted)]
    plan.sort(key=lambda item: (item[3] or "", item[1]))
    print(f"Replay plan: {len(plan)} motions; stats={stats}")
    if not plan:
        print("Nothing to replay after conversion.")
        print_clean_statistics(report)
        return

    config_src = os.path.join(data_dir, "object_usd", "config.yaml")
    if os.path.isfile(config_src):
        _atomic_copy(config_src, os.path.join(output_dir, "object_usd", "config.yaml"))

    width, height = map(int, args.resolution.lower().split("x"))
    render_all(
        plan,
        resolution=(width, height),
        camera_offset=tuple(args.camera_offset),
        camera_target=tuple(args.camera_target),
        headless=args.headless,
        post_penetration_frames=thresholds.post_penetration_frames,
        record_video=args.record_video,
        clean_data_dir=data_dir,
        clean_output_dir=output_dir,
        clean_report=report,
        clean_report_path=report_path,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
