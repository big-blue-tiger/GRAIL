#!/usr/bin/env python3
"""Batch render trajectory replays with a single IsaacSim session.

Initializes IsaacSim once, then renders each trajectory sequentially —
swapping object USDs between renders. This avoids the ~90s init overhead
per trajectory that would occur when calling multi_scene_render.py as
a subprocess.

Usage (on cluster with GPU + IsaacSim):
    python -u batch_render_replay.py \
        --shard_dir /path/to/eval/step_019500/export_shard_0 \
        --object_usd_dir /path/to/exported/step_019500/merged/object_usd \
        --output_dir /path/to/exported/step_019500/merged/vis \
        --segmented_output_dir /path/to/exported/step_019500/merged/penetration_segments \
        --segmented_object_output_dir /path/to/exported/step_019500/merged/penetration_object_segments \
        --skip_existing

    # Replay and detect penetrations without creating an IsaacLab camera/video:
    python -u batch_render_replay_clip.py \
        --shard_dir /path/to/eval/step_019500/export_shard_0 \
        --object_usd_dir /path/to/exported/step_019500/merged/object_usd \
        --output_dir /path/to/exported/step_019500/merged/vis \
        --no_record_video
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import joblib
import numpy as np

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
):
    """Replay all trajectories with one IsaacSim session, optionally recording video."""
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
                if not _separation_is_penetration(separation):
                    continue
                depth = -float(separation)
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
        f"{PENETRATION_DEPTH_THRESHOLD_M * 1000:.1f} mm",
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
            penetration_frames = np.zeros(total_frames, dtype=np.uint8)

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

                if contact_frame_state["max_penetration"] >= PENETRATION_DEPTH_THRESHOLD_M:
                    penetration_frames[f] = 1

                # The probe may depenetrate dynamic articulation bodies.  Put
                # the recorded state back before rendering so collision
                # response never changes the replay video.
                art.write_joint_state_to_sim(joint_pos, joint_vel)
                art.write_root_state_to_sim(root_state)
                sim.forward()

                # --start_frame_skip affects only video output, never detection.
                if record_video and f >= skip:
                    sim.render()
                    camera.update(dt=0.0)
                    rgb = camera.data.output["rgb"][0].cpu().numpy()
                    writer.append_data(rgb)

            if writer is not None:
                writer.close()
            penetration_frames_by_motion[motion_key] = penetration_frames
            succeeded += 1
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
            penetration_indices = np.where(penetration_frames == 1)[0]

            print(
                f"  [PENETRATION] {int(penetration_frames.sum())}/{total_frames} frame(s)",
                flush=True,
            )

            print(
                f"  [PENETRATION] indices: {penetration_indices.tolist()}",
                flush=True,
            )
            if segmented_output_dir is not None:
                export_penetration_segment(
                    traj,
                    motion_key,
                    penetration_frames,
                    segmented_output_dir,
                    source_robot_dir,
                    post_penetration_frames=post_penetration_frames,
                )
            if segmented_object_output_dir is not None:
                export_object_penetration_segment(
                    traj,
                    motion_key,
                    penetration_frames,
                    segmented_object_output_dir,
                    source_object_dir,
                    post_penetration_frames=post_penetration_frames,
                )
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
            _heartbeat(f"motion_{idx}_{motion_key}_failed")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch render complete: {succeeded}/{total} succeeded, {failed} failed")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/max(total,1):.1f}s per trajectory)")

    # IsaacSim teardown (simulation_app.close()) can hang in USD/Hydra cleanup after
    # rendering many frames. Match eval_agent_trl.py and force-exit the process —
    # all .mp4 outputs were already flushed by the per-motion writer.close() calls,
    # so this is safe and returns exit code 0 for scheduler completion.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Batch render trajectory replays (in-process)")
    parser.add_argument(
        "--shard_dir", required=True, help="Path to shard directory containing metrics_eval.json"
    )
    parser.add_argument(
        "--traj_dir",
        default=None,
        help="Directory with *.trajectory.pkl files (default: {shard_dir}/trajectories)",
    )
    parser.add_argument(
        "--object_usd_dir", required=True, help="Directory with {motion_key}.usd files"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for rendered .mp4 files"
    )
    parser.add_argument(
        "--resolution", default="1920x1080", help="Output video resolution (default: 1920x1080)"
    )
    parser.add_argument(
        "--camera_offset",
        type=float,
        nargs=3,
        default=[-3.54, 0.0, 1.2],
        help="Camera position [x y z]",
    )
    parser.add_argument(
        "--camera_target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.8],
        help="Camera look-at point [x y z]",
    )
    parser.add_argument(
        "--start_frame_skip",
        type=int,
        default=0,
        help="Initial frames to omit from each output video (default: 0)",
    )
    parser.add_argument(
        "--segmented_output_dir",
        default=None,
        help=(
            "Optional directory for post-penetration raw robot-motion exports. "
            "Each output is named {motion_key}.pkl; motions with no valid "
            "post-penetration suffix are skipped."
        ),
    )
    parser.add_argument(
        "--segmented_object_output_dir",
        default=None,
        help=(
            "Optional directory for post-penetration raw object-motion exports. "
            "Each output is named {motion_key}.pkl; motions with no valid "
            "post-penetration suffix are skipped. The source is objects/ next "
            "to --object_usd_dir."
        ),
    )
    parser.add_argument(
        "--source_robot_dir",
        default=None,
        help=(
            "Directory containing original robot/{motion_key}.pkl files. "
            "Defaults to the robot directory next to --object_usd_dir, with "
            "robot_old/ used when robot/ contains no raw motions."
        ),
    )
    parser.add_argument(
        "--post_penetration_frames",
        type=int,
        default=20,
        help="Frames added to the last penetration index for segmentation (default: 20)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="Skip trajectories with existing output videos"
    )
    record_video_group = parser.add_mutually_exclusive_group()
    record_video_group.add_argument(
        "--record_video",
        dest="record_video",
        action="store_true",
        help="Record output videos (default)",
    )
    record_video_group.add_argument(
        "--no_record_video",
        dest="record_video",
        action="store_false",
        help="Disable cameras and video recording",
    )
    parser.set_defaults(record_video=True)
    parser.add_argument(
        "--dry_run", action="store_true", help="Print render plan without executing"
    )
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    plan, stats, n_total, n_success = build_render_plan(
        args.shard_dir,
        args.object_usd_dir,
        args.output_dir,
        skip_existing=args.skip_existing,
        traj_dir=args.traj_dir,
        record_video=args.record_video,
    )
    w, h = map(int, args.resolution.split("x"))

    print(f"Shard: {args.shard_dir}")
    print(f"Filter keys: {n_total} total, {n_success} successful")
    print(f"Render plan: {len(plan)} trajectories")
    for k, v in stats.items():
        if v:
            print(f"  {k}: {v}")

    if args.dry_run:
        for env_idx, mk, tp, up, op in plan[:10]:
            print(f"  env {env_idx:06d} -> {mk}")
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more")
        return

    if not plan:
        print("Nothing to render.")
        return

    # Sort by object USD path to minimize USD swapping
    plan.sort(key=lambda x: (x[3] or "", x[1]))

    source_robot_dir = None
    if args.segmented_output_dir is not None:
        source_robot_dir = resolve_source_robot_dir(args.source_robot_dir, args.object_usd_dir)
    source_object_dir = None
    if args.segmented_object_output_dir is not None:
        source_object_dir = resolve_source_object_dir(None, args.object_usd_dir)

    render_all(
        plan,
        resolution=(w, h),
        camera_offset=tuple(args.camera_offset),
        camera_target=tuple(args.camera_target),
        headless=args.headless,
        start_frame_skip=args.start_frame_skip,
        segmented_output_dir=args.segmented_output_dir,
        source_robot_dir=source_robot_dir,
        segmented_object_output_dir=args.segmented_object_output_dir,
        source_object_dir=source_object_dir,
        post_penetration_frames=args.post_penetration_frames,
        record_video=args.record_video,
    )


if __name__ == "__main__":
    main()
