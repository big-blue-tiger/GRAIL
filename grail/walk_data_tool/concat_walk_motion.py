#!/usr/bin/env python3
"""Prepend Kimodo G1 CSV to a paired GRAIL motion, anchoring the latter in world space.

Four-point PCHIP with configurable added frames (default 10); no resampling, IK, or height correction. See README.md next to
this script. Pure NumPy/SciPy operations are separate from the lazy SONIC FK adapter.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/hf_dataset/data_update/data"
ASSET = ROOT / "imports/SONIC/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
DEFAULT_NAME = "pickup_table__alcohol_12__001"
TRANSITION_FRAMES = 10
# Kimodo exports/mujoco.py traverses worldbody joints in this order.
CSV_JOINTS = tuple(
    [f"{side}_{part}_joint" for side in ("left", "right") for part in
     ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [f"{side}_{part}_joint" for side in ("left", "right") for part in
       ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
CSV_AXES = np.eye(3)[[1, 0, 2, 1, 1, 0] * 2 + [2, 0, 1] + [1, 0, 2, 1, 0, 1, 2] * 2]
POSITION_FIELDS = {"root_trans_offset", "root_pos", "body_pos"}
QUAT_FIELDS = {"root_rot", "body_rot"}  # robot xyzw, never object root_quat
VELOCITY_FIELDS = {"root_lin_vel", "root_ang_vel", "body_lin_vel", "body_ang_vel"}
LOCAL_FIELDS = {"dof", "dof_pos", "joint_pos", "dof_vel", "hand_dof_pos",
                "hand_action_left", "hand_action_right"}
ROBOT_FIELDS = POSITION_FIELDS | QUAT_FIELDS | VELOCITY_FIELDS | LOCAL_FIELDS | {
    "pose_aa", "smpl_joints"}
CONTACT_FIELDS = {"contact_points_left_hand", "contact_points_right_hand"}
OBJECT_FIELDS = {"root_pos", "root_quat"} | CONTACT_FIELDS
WALK_FIELDS = {"root_trans_offset", "root_rot", "dof", "pose_aa", "smpl_joints",
               "hand_dof_pos", "hand_action_left", "hand_action_right"}


@dataclass(frozen=True)
class BoundaryFrame:
    root_quat_xyzw: np.ndarray
    feet_world: np.ndarray  # (2, 3), left then right


@dataclass(frozen=True)
class SE2Alignment:
    yaw: float
    translation_xy: np.ndarray

    @property
    def rotation(self):
        return Rotation.from_euler("z", self.yaw)


def pelvis_yaw(quat_xyzw):
    matrix = Rotation.from_quat(quat_xyzw).as_matrix()
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def compute_se2_alignment(a_end: BoundaryFrame, b_start: BoundaryFrame) -> SE2Alignment:
    delta = pelvis_yaw(b_start.root_quat_xyzw) - pelvis_yaw(a_end.root_quat_xyzw)
    delta = float(np.arctan2(np.sin(delta), np.cos(delta)))
    rotation = Rotation.from_euler("z", delta).as_matrix()[:2, :2]
    center_a = np.asarray(a_end.feet_world)[:, :2].mean(axis=0)
    center_b = np.asarray(b_start.feet_world)[:, :2].mean(axis=0)
    return SE2Alignment(delta, center_b - rotation @ center_a)


def validate_fields(motion, kind):
    """Explicit temporal schemas; new fields require a deliberate adapter policy."""
    allowed = ROBOT_FIELDS if kind == "robot" else OBJECT_FIELDS
    static = {"fps"} if kind == "robot" else {"fps", "scale"}
    unknown = set(motion) - allowed - static
    if unknown:
        raise ValueError(f"Unsupported {kind} fields: {sorted(unknown)}; add a field policy")
    n = len(motion["root_trans_offset" if kind == "robot" else "root_pos"])
    if n < 1 or not np.isfinite(motion["fps"]) or motion["fps"] <= 0:
        raise ValueError("Motion must have frames and a finite positive fps")
    for key in set(motion) & allowed:
        value = motion[key]
        if key in CONTACT_FIELDS:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a frame-indexed dictionary")
            for frame, points in value.items():
                if not isinstance(frame, (int, np.integer)) or not 0 <= frame < n:
                    raise ValueError(f"Invalid contact frame {frame} for {n} frames")
                points = np.asarray(points)
                if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                    raise ValueError(f"Invalid contact points at {key}[{frame}]")
        elif (not isinstance(value, np.ndarray) or value.ndim == 0
              or value.shape[0] != n or not np.isfinite(value).all()):
            raise ValueError(f"Invalid frame array {key}; expected {n} finite frames")
    shapes = ({"root_trans_offset": (n, 3), "root_rot": (n, 4), "dof": (n, 29),
               "pose_aa": (n, 30, 3), "smpl_joints": (n, 24, 3),
               "hand_dof_pos": (n, 14), "hand_action_left": (n,), "hand_action_right": (n,)}
              if kind == "robot" else {})
    for key, shape in shapes.items():
        if key in motion and motion[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}, got {motion[key].shape}")
    for key in ((QUAT_FIELDS & set(motion)) if kind == "robot" else {"root_quat"}):
        if motion[key].shape[-1] != 4 or not np.allclose(
            np.linalg.norm(motion[key], axis=-1), 1, atol=1e-4
        ):
            raise ValueError(f"{key} must contain unit quaternions")
    return n


def transform_motion_se2(motion: dict, alignment: SE2Alignment) -> dict:
    validate_fields(motion, "robot")
    result = copy.deepcopy(motion)
    rotation = alignment.rotation
    matrix = rotation.as_matrix()
    for key in POSITION_FIELDS | VELOCITY_FIELDS:
        if key in motion:
            values = motion[key] @ matrix.T
            if key in POSITION_FIELDS:
                values[..., :2] += alignment.translation_xy
            result[key] = values.astype(motion[key].dtype)
    for key in QUAT_FIELDS:
        if key in motion:
            values = motion[key]
            result[key] = (rotation * Rotation.from_quat(values.reshape(-1, 4))).as_quat().reshape(
                values.shape).astype(values.dtype)
    if "pose_aa" in motion:
        result["pose_aa"][:, 0] = (
            rotation * Rotation.from_rotvec(motion["pose_aa"][:, 0])
        ).as_rotvec().astype(motion["pose_aa"].dtype)
    # smpl_joints is an exporter placeholder, not a world-space skeleton.
    if "smpl_joints" in motion and np.any(motion["smpl_joints"]):
        raise ValueError("Nonzero smpl_joints needs a dedicated skeleton adapter")
    return result


def concat_motion(motions: Sequence[dict], *, kind="robot") -> dict:
    """Direct time concatenation; alignment/transition strategies run before this."""
    if not motions:
        raise ValueError("At least one motion is required")
    lengths = [validate_fields(m, kind) for m in motions]
    first = motions[0]
    if any(set(m) != set(first) or m["fps"] != first["fps"] for m in motions):
        raise ValueError("All segments must have identical fields and fps; no implicit resampling")
    result = {}
    for key, value in first.items():
        if key in CONTACT_FIELDS:
            result[key] = {}
            offset = 0
            for motion, length in zip(motions, lengths):
                result[key].update({int(k) + offset: copy.deepcopy(v) for k, v in motion[key].items()})
                offset += length
        elif key in {"fps", "scale"}:
            if any(not np.array_equal(m[key], value) for m in motions):
                raise ValueError(f"Static field {key} differs across segments")
            result[key] = copy.deepcopy(value)
        else:
            if any(m[key].dtype != value.dtype or m[key].shape[1:] != value.shape[1:]
                   for m in motions):
                raise ValueError(f"Incompatible dtype or shape for {key}")
            result[key] = np.concatenate([m[key] for m in motions], axis=0)
    validate_fields(result, kind)
    return result


def build_transition(a: dict, b: dict, num_frames: int = TRANSITION_FRAMES) -> dict:
    """Return the PCHIP window, including N existing frames per side.

    Knots: A[-N-1], A[-1], B[0], B[N], at times 0,N,2N+1,3N+1.
    Exclude the outer anchors; insert N new frames in a 3N-frame window. Quaternion
    components use sign-continuous PCHIP followed by normalization. Original
    internal anchor values are restored exactly, including quaternion signs.
    """
    validate_fields(a, "robot")
    validate_fields(b, "robot")
    if not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    if min(len(a["dof"]), len(b["dof"])) <= num_frames:
        raise ValueError(f"PCHIP requires at least {num_frames + 1} frames in each motion")
    if set(a) != set(b) or a["fps"] != b["fps"]:
        raise ValueError("Transition endpoints must have identical fields and fps")
    knots = np.array([0, num_frames, 2 * num_frames + 1, 3 * num_frames + 1])
    times = np.arange(1, knots[-1])
    result = {"fps": a["fps"]}
    for key in set(a) - {"fps", "pose_aa"}:
        if a[key].dtype != b[key].dtype or a[key].shape[1:] != b[key].shape[1:]:
            raise ValueError(f"Incompatible dtype or shape for {key}")
        controls = np.stack([a[key][-num_frames - 1], a[key][-1],
                             b[key][0], b[key][num_frames]]).astype(np.float64)
        if key in QUAT_FIELDS:
            controls /= np.linalg.norm(controls, axis=-1, keepdims=True)
            for i in range(1, 4):
                controls[i] = np.where(
                    np.sum(controls[i - 1] * controls[i], axis=-1, keepdims=True) < 0,
                    -controls[i], controls[i])
        values = PchipInterpolator(knots, controls, axis=0, extrapolate=False)(times)
        if key in QUAT_FIELDS:
            norms = np.linalg.norm(values, axis=-1, keepdims=True)
            if np.any(norms < 1e-12):
                raise ValueError("PCHIP produced a degenerate quaternion")
            values /= norms
        result[key] = values.astype(a[key].dtype)
    if "pose_aa" in a:
        axes = joint_mapping()[2]
        result["pose_aa"] = np.concatenate([
            Rotation.from_quat(result["root_rot"]).as_rotvec()[:, None],
            result["dof"][..., None] * axes[None]], axis=1).astype(a["pose_aa"].dtype)
    for key in set(a) - {"fps"}:
        result[key][knots[1] - 1] = a[key][-1]
        result[key][knots[2] - 1] = b[key][0]
    validate_fields(result, "robot")
    return result


def slice_robot_motion(motion, start=None, stop=None):
    """Slice all robot frame fields together, leaving static metadata intact."""
    return {key: value if key == "fps" else value[start:stop]
            for key, value in motion.items()}


def crop_paired_motion(robot, objects, source_frame, transition_frames):
    """Crop robot/object time together; contacts remain zero-based frame indices."""
    total = validate_fields(robot, "robot")
    if validate_fields(objects, "object") != total or robot["fps"] != objects["fps"]:
        raise ValueError("Robot/object frame counts or fps differ")
    if type(source_frame) is not int or not 0 <= source_frame < total:
        raise ValueError(f"Invalid source_frame: {source_frame}")
    if total - source_frame <= transition_frames:
        raise ValueError(f"Cropping at {source_frame} leaves {total - source_frame} frames; "
                         f"PCHIP requires at least {transition_frames + 1}")
    cropped_objects = {}
    for key, value in objects.items():
        if key in CONTACT_FIELDS:
            cropped_objects[key] = {int(frame) - source_frame: points
                                    for frame, points in value.items() if frame >= source_frame}
        elif key in {"fps", "scale"}:
            cropped_objects[key] = value
        else:
            cropped_objects[key] = value[source_frame:]
    return slice_robot_motion(robot, start=source_frame), cropped_objects


def constraint_source_frame(manifest_path, robot_path, motion_key, robot):
    """Resolve an exact source/key pair and reject stale constraint metadata."""
    if manifest_path is None:
        return 0
    manifest = json.loads(Path(manifest_path).read_text())
    matches = [record for record in manifest["records"]
               if Path(record["source"]).resolve() == Path(robot_path).resolve()
               and record["motion_key"] == str(motion_key)]
    if len(matches) != 1:
        raise ValueError(f"Expected one constraint record for {robot_path} / {motion_key}; "
                         f"found {len(matches)}")
    record = matches[0]
    if (record.get("source_sha256") != file_hash(robot_path)
            or record.get("source_num_frames") != len(robot["dof"])
            or record.get("source_fps") != robot["fps"]):
        raise ValueError("Constraint manifest does not match source hash/frame count/fps; re-export constraints")
    frame = record.get("source_frame")
    if type(frame) is not int or not 0 <= frame < len(robot["dof"]):
        raise ValueError(f"Invalid source_frame in constraint manifest: {frame}")
    return frame


def joint_mapping(asset=ASSET):
    joints = [j for j in ET.parse(asset).getroot().find("worldbody").iter("joint")
              if j.get("type") != "free"]
    names = [j.get("name") for j in joints]
    if len(names) != len(set(names)) or set(names) != set(CSV_JOINTS):
        raise ValueError("Target model does not have the expected 29 G1 joints")
    indices = [CSV_JOINTS.index(name) for name in names]
    axes = np.array([np.fromstring(j.attrib["axis"], sep=" ") for j in joints])
    if not np.array_equal(axes, CSV_AXES[indices]):
        raise ValueError("Target joint axes differ from Kimodo G1 CSV convention")
    return names, np.asarray(indices), axes


def load_kimodo_csv(path, template, *, asset=ASSET):
    validate_fields(template, "robot")
    if set(template) != WALK_FIELDS | {"fps"}:
        raise ValueError("CSV adapter requires the standard GRAIL robot fields; add an adapter for extras")
    if np.any(template["smpl_joints"]):
        raise ValueError("Expected zero smpl_joints placeholders")
    qpos = np.loadtxt(path, delimiter=",", ndmin=2)
    if qpos.shape[1] != 36 or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError("Expected finite, headerless (T,36) Kimodo G1 CSV")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-4):
        raise ValueError("CSV root quaternion is not unit length")
    names, indices, axes = joint_mapping(asset)
    n = len(qpos)
    result = {"fps": template["fps"], "root_trans_offset": qpos[:, :3],
              "root_rot": qpos[:, [4, 5, 6, 3]], "dof": qpos[:, 7:][:, indices].copy()}
    for i, name in enumerate(names):
        if "_wrist_" in name:
            result["dof"][:, i] = 0
    result["pose_aa"] = np.concatenate([
        Rotation.from_quat(result["root_rot"]).as_rotvec()[:, None],
        result["dof"][..., None] * axes[None]], axis=1)
    result["smpl_joints"] = np.zeros((n, 24, 3))
    for key in ("hand_dof_pos", "hand_action_left", "hand_action_right"):
        result[key] = np.repeat(template[key][:1], n, axis=0) if key == "hand_dof_pos" else np.ones(n)
    for key in WALK_FIELDS:
        result[key] = result[key].astype(template[key].dtype)
    validate_fields(result, "robot")
    return result


def prepend_static_objects(objects, length):
    validate_fields(objects, "object")
    prefix = {}
    for key, value in objects.items():
        if key in CONTACT_FIELDS:
            prefix[key] = {}
        elif key in {"root_pos", "root_quat"}:
            prefix[key] = np.repeat(value[:1], length, axis=0)
        else:
            prefix[key] = copy.deepcopy(value)
    return concat_motion([prefix, objects], kind="object")


def sonic_config(dataset_dir=None):
    from omegaconf import OmegaConf

    config = {"asset": {"assetRoot": str(ASSET.parent), "assetFileName": ASSET.name},
              "extend_config": [], "multi_thread": False, "target_fps": 50,
              "randomize_heading": False, "randomize_wrist_poses": False,
              "cat_upper_body_poses": False, "freeze_frame_aug": False,
              "zero_root_xy": False, "body_indexes_data": list(range(30))}
    if dataset_dir is not None:
        config.update({"motion_file": str(dataset_dir / "robot"),
                       "object_motion_file": str(dataset_dir / "objects"),
                       "bps_dir": str(dataset_dir / "bps"), "smpl_motion_file": "dummy"})
    return OmegaConf.create(config)


class GrailFK:
    def __init__(self):
        sys.path.insert(0, str(ROOT / "imports/SONIC"))
        from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

        self.model = Humanoid_Batch(sonic_config())
        self.foot_indices = [self.model.body_names.index(f"{side}_ankle_roll_link")
                             for side in ("left", "right")]

    def boundary(self, motion, frame):
        import torch

        if frame < 0:
            frame += len(motion["dof"])
        if not 0 <= frame < len(motion["dof"]):
            raise IndexError(f"Boundary frame out of range: {frame}")
        pose = torch.from_numpy(motion["pose_aa"][frame:frame + 1].copy()).float()[None]
        trans = torch.from_numpy(motion["root_trans_offset"][frame:frame + 1].copy()).float()[None]
        with torch.no_grad():
            out = self.model.fk_batch(pose, trans, return_full=False, interpolate_data=False)
        return BoundaryFrame(motion["root_rot"][frame],
                             out.global_translation[0, 0, self.foot_indices].numpy())


def alignment_metrics(a_end, b_start):
    feet_delta = a_end.feet_world - b_start.feet_world
    yaw = pelvis_yaw(a_end.root_quat_xyzw) - pelvis_yaw(b_start.root_quat_xyzw)
    return {"foot_center_xy_error_m": float(np.linalg.norm(feet_delta[:, :2].mean(axis=0))),
            "pelvis_yaw_error_rad": abs(float(np.arctan2(np.sin(yaw), np.cos(yaw)))),
            "left_right_foot_xy_error_m": np.linalg.norm(feet_delta[:, :2], axis=1).tolist(),
            "left_right_foot_z_delta_m": feet_delta[:, 2].tolist()}


def load_single(path):
    payload = joblib.load(path)
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(f"Expected exactly one motion in {path}")
    return next(iter(payload.items()))


def assert_equal(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError("Dictionary keys changed")
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, np.ndarray):
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise AssertionError("Array values or dtype changed")
    elif actual != expected:
        raise AssertionError("Static value changed")


def validate_preserved_suffix(result, original, offset, start_frame=0):
    for key, value in original.items():
        if key in CONTACT_FIELDS:
            assert_equal(result[key], {int(k) + offset: v for k, v in value.items()})
        elif key in {"fps", "scale"}:
            assert_equal(result[key], value)
        else:
            assert_equal(result[key][offset + start_frame:], value[start_frame:])


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_dataset(walk_csv, dataset_dir, motion_name, output_dir, overwrite=False,
                   transition_frames=TRANSITION_FRAMES, constraint_manifest=None):
    if not isinstance(transition_frames, int) or transition_frames < 1:
        raise ValueError("transition_frames must be a positive integer")
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    walk_csv = Path(walk_csv).resolve()
    if output_dir == dataset_dir or dataset_dir in output_dir.parents or output_dir in dataset_dir.parents:
        raise ValueError("Output must be separate from the source dataset")
    if Path(motion_name).name != motion_name or motion_name in {"", ".", ".."}:
        raise ValueError("motion-name must be a filename stem")
    robot_path = dataset_dir / "robot" / f"{motion_name}.pkl"
    object_path = dataset_dir / "objects" / f"{motion_name}.pkl"
    robot_key, robot = load_single(robot_path)
    object_key, objects = load_single(object_path)
    if robot_key != object_key:
        raise ValueError("Robot and object internal motion keys differ")
    n_b = validate_fields(robot, "robot")
    if validate_fields(objects, "object") != n_b or robot["fps"] != objects["fps"]:
        raise ValueError("Robot/object frame counts or fps differ")
    original_frames = n_b
    source_frame = constraint_source_frame(constraint_manifest, robot_path, robot_key, robot)
    robot, objects = crop_paired_motion(robot, objects, source_frame, transition_frames)
    n_b = len(robot["dof"])
    assets = [Path("meta") / f"{motion_name}.pkl", Path("bps") / f"{motion_name}.npy"]
    usd = [p.relative_to(dataset_dir) for ext in (".usd", ".usda")
           if (p := dataset_dir / "object_usd" / f"{motion_name}{ext}").is_file()]
    if len(usd) != 1:
        raise ValueError("Expected one matching USD asset")
    assets += usd
    assets += [p.relative_to(dataset_dir) for p in (dataset_dir / "bps").glob("_*.npy")]
    for path in assets:
        if not (dataset_dir / path).is_file():
            raise FileNotFoundError(dataset_dir / path)
    meta = joblib.load(dataset_dir / assets[0])
    if not {"table_pos", "table_quat", "table_size"} <= set(meta):
        raise ValueError("Table metadata is incomplete")
    generated = [Path("robot") / robot_path.name, Path("objects") / object_path.name,
                 Path("reports") / f"{motion_name}.json"]
    for path in assets + generated:
        destination = output_dir / path
        if destination.resolve() != destination or (destination.exists() and not overwrite):
            raise FileExistsError(f"Refusing symlink or existing output: {destination}")
    walk = load_kimodo_csv(walk_csv, robot)
    n_a = len(walk["dof"])
    fk = GrailFK()
    b_start = fk.boundary(robot, 0)
    alignment = compute_se2_alignment(fk.boundary(walk, n_a - 1), b_start)
    aligned = transform_motion_se2(walk, alignment)
    transition = build_transition(aligned, robot, num_frames=transition_frames)
    b_start_index = n_a + transition_frames
    combined = concat_motion([slice_robot_motion(aligned, stop=-transition_frames),
                              transition, slice_robot_motion(robot, start=transition_frames)])
    combined_objects = prepend_static_objects(objects, b_start_index)
    output_meta = copy.deepcopy(meta)
    # Zero-based, inclusive bounds for the entire build_transition window,
    # including rewritten A/B frames, not just the newly inserted frames.
    output_meta.update({
        "total_frames": len(combined["dof"]),
        "source_start_frame": source_frame,
        "source_total_frames": original_frames,
        "transition_start_frame": n_a - transition_frames,
        "transition_end_frame": n_a - transition_frames + len(transition["dof"]) - 1,
    })
    metrics = alignment_metrics(fk.boundary(aligned, n_a - 1), b_start)
    if metrics["foot_center_xy_error_m"] >= 1e-5 or metrics["pelvis_yaw_error_rad"] >= 1e-5:
        raise ValueError(f"SE(2) alignment failed: {metrics}")
    report = {"sources": {"walk_csv": str(walk_csv), "robot": str(robot_path),
                           "objects": str(object_path)}, "motion_name": motion_name,
              "internal_motion_key": robot_key, "frames_a": n_a, "frames_b": n_b,
              "source_frame": source_frame, "source_num_frames": original_frames,
              "constraint_manifest": str(Path(constraint_manifest).resolve()) if constraint_manifest else None,
              "source_to_output_frame_offset": b_start_index - source_frame,
              "original_unchanged_from_frame": source_frame + transition_frames,
              "transition_frames": transition_frames, "transition_start_index": n_a,
              "transition_method": "four-point PCHIP; sign-continuous normalized quaternion components",
              "replacement_start_index": n_a - transition_frames,
              "replacement_stop_index_exclusive": b_start_index + transition_frames,
              "control_frame_indices": [n_a - transition_frames - 1, n_a - 1,
                                        b_start_index, b_start_index + transition_frames],
              "total_frames": b_start_index + n_b, "b_start_index": b_start_index, "fps": robot["fps"],
              "timing_policy": "keep all CSV frames; interpret at B fps; no resampling",
              "csv_source_fps": None, "wrist_policy": "A wrists zero before PCHIP; seam window follows four control frames",
              "quaternions": {"csv": "wxyz", "robot": "xyzw", "objects": "wxyz"},
              "alignment": {"delta_yaw_rad": alignment.yaw,
                            "translation_xy_m": alignment.translation_xy.tolist()},
              "metrics": metrics, "b_suffix_exact": False, "b_unchanged_from_frame": transition_frames,
              "static_assets_sha256": {str(p): file_hash(dataset_dir / p) for p in assets[1:]}}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".concat-motion-", dir=output_dir.parent) as temp:
        staging = Path(temp)
        for path in assets + generated:
            (staging / path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({robot_key: combined}, staging / generated[0])
        joblib.dump({object_key: combined_objects}, staging / generated[1])
        for path in assets:
            if path == assets[0]:
                joblib.dump(output_meta, staging / path)
                assert_equal(joblib.load(staging / path), output_meta)
                report["output_meta_sha256"] = file_hash(staging / path)
                continue
            shutil.copy2(dataset_dir / path, staging / path)
            if file_hash(staging / path) != report["static_assets_sha256"][str(path)]:
                raise AssertionError(f"Asset copy mismatch: {path}")
        _, reloaded = load_single(staging / generated[0])
        _, reloaded_objects = load_single(staging / generated[1])
        validate_preserved_suffix(reloaded, robot, b_start_index, start_frame=transition_frames)
        for key in set(robot) - {"fps"}:
            assert_equal(reloaded[key][:n_a - transition_frames], aligned[key][:-transition_frames])
            assert_equal(reloaded[key][n_a - 1], aligned[key][-1])
            assert_equal(reloaded[key][b_start_index], robot[key][0])
        validate_preserved_suffix(reloaded_objects, objects, b_start_index)
        sys.path.insert(0, str(ROOT))
        from grail.datatool.batch_render_replay_clip import validate_motion_input

        report["dataset_validation"] = validate_motion_input(str(staging), motion_name)
        report["dataset_validation"]["usd_path"] = str(output_dir / usd[0])
        (staging / generated[2]).write_text(json.dumps(report, indent=2) + "\n")
        for path in assets + generated:
            (output_dir / path).parent.mkdir(parents=True, exist_ok=True)
            (staging / path).replace(output_dir / path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk-csv", type=Path, default=ROOT / "data/kimodo/demo/qpos.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DATA_ROOT / "pickup_table_cleaned_succeeded")
    parser.add_argument("--motion-name", default=DEFAULT_NAME)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "pickup_table_walk_concat")
    parser.add_argument("--constraint-manifest", type=Path,
                        help="Read and validate source_frame from the new constraint manifest; omitted = frame 0")
    parser.add_argument("--transition-frames", type=int, default=TRANSITION_FRAMES,
                        help="Inserted frames N and context per side; knots A[-N-1], A[-1], B[0], B[N] (default: 10)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = export_dataset(**vars(args))
    print(json.dumps({"output_dir": str(args.output_dir), "total_frames": report["total_frames"],
                      "fps": report["fps"], **report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
