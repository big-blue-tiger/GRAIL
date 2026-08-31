"""Pure metrics used by the reference-motion cleaning replay.

This module intentionally has no Isaac Sim/Isaac Lab imports.  Keeping the
classification logic independent from the replay backend makes the numerical
thresholds testable with small synthetic trajectories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CleaningThresholds:
    """Numerical defaults for the balanced reference-motion cleaner."""

    penetration_depth_m: float = 0.01
    single_frame_penetration_depth_m: float = 0.02
    penetration_min_frames: int = 2
    post_penetration_frames: int = 20
    max_crop_seconds: float = 2.0
    max_crop_ratio: float = 0.20
    min_remaining_seconds: float = 4.0

    initial_lift_window_seconds: float = 0.5
    final_lift_window_seconds: float = 1.0
    median_lift_threshold_m: float = 0.10
    lifted_frame_threshold_m: float = 0.08
    lifted_frame_ratio: float = 0.80

    foot_contact_height_m: float = 0.05
    foot_contact_speed_mps: float = 0.15
    contact_debounce_frames: int = 3
    locomotion_speed_mps: float = 0.12
    locomotion_min_seconds: float = 1.5
    locomotion_min_displacement_m: float = 0.30
    micro_step_length_m: float = 0.15
    shuffle_min_steps: int = 4
    shuffle_cadence_steps_per_second: float = 2.5
    shuffle_micro_step_ratio: float = 0.60

    def to_dict(self) -> dict:
        return asdict(self)


def _json_float(value: float) -> float:
    """Return a regular finite Python float for JSON-facing metrics."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"metric is not finite: {value}")
    return value


def true_intervals(mask: np.ndarray) -> list[list[int]]:
    """Return inclusive ``[start, end]`` intervals for true runs in a mask."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [[int(start), int(end)] for start, end in zip(starts, ends, strict=True)]


def effective_penetration_mask(
    penetration_depth_m: np.ndarray,
    thresholds: CleaningThresholds,
) -> np.ndarray:
    """Debounce per-frame penetration depths using the balanced rules."""
    depths = np.asarray(penetration_depth_m, dtype=np.float64).reshape(-1)
    if not np.isfinite(depths).all():
        raise ValueError("penetration depths contain NaN or Inf")

    shallow = depths >= thresholds.penetration_depth_m
    effective = depths >= thresholds.single_frame_penetration_depth_m
    min_frames = max(1, int(thresholds.penetration_min_frames))
    for start, end in true_intervals(shallow):
        if end - start + 1 >= min_frames:
            effective[start : end + 1] = True
    return effective


def analyze_penetration(
    penetration_depth_m: np.ndarray,
    fps: float,
    thresholds: CleaningThresholds,
) -> dict:
    """Classify a penetration trace and calculate a repairable prefix crop."""
    depths = np.asarray(penetration_depth_m, dtype=np.float64).reshape(-1)
    if depths.size == 0:
        raise ValueError("penetration trace is empty")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    effective = effective_penetration_mask(depths, thresholds)
    bad_indices = np.flatnonzero(effective)
    max_depth = float(np.max(depths))
    metrics = {
        "max_depth_m": _json_float(max_depth),
        "effective_frame_count": int(effective.sum()),
        "effective_frames": bad_indices.astype(int).tolist(),
        "effective_intervals": true_intervals(effective),
        "repairable": True,
        "crop_start_frame": 0,
        "crop_seconds": 0.0,
        "crop_budget_frames": int(
            min(round(thresholds.max_crop_seconds * fps), np.floor(depths.size * thresholds.max_crop_ratio))
        ),
    }
    if bad_indices.size == 0:
        metrics["classification"] = "clean"
        return metrics

    crop_start = int(bad_indices[-1]) + int(thresholds.post_penetration_frames)
    crop_budget = metrics["crop_budget_frames"]
    min_remaining = int(np.ceil(thresholds.min_remaining_seconds * fps))
    repairable = (
        crop_start <= crop_budget
        and depths.size - crop_start >= min_remaining
        and not bool(effective[crop_start:].any())
    )
    metrics.update(
        {
            "classification": "repaired" if repairable else "rejected",
            "repairable": bool(repairable),
            "crop_start_frame": crop_start,
            "crop_seconds": _json_float(crop_start / fps),
            "remaining_frames": max(0, int(depths.size - crop_start)),
        }
    )
    return metrics


def analyze_object_lift(
    object_pos: np.ndarray,
    fps: float,
    thresholds: CleaningThresholds,
) -> dict:
    """Check that the object remains lifted in the final reference window."""
    positions = np.asarray(object_pos, dtype=np.float64)
    if positions.ndim == 3 and positions.shape[1] == 1:
        positions = positions[:, 0, :]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"object_pos must have shape (T,3) or (T,1,3), got {positions.shape}")
    if positions.shape[0] < 2 or not np.isfinite(positions).all():
        raise ValueError("object_pos is too short or contains NaN/Inf")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    initial_frames = max(1, int(np.ceil(thresholds.initial_lift_window_seconds * fps)))
    final_frames = max(1, int(np.ceil(thresholds.final_lift_window_seconds * fps)))
    if positions.shape[0] < max(initial_frames, final_frames):
        raise ValueError(
            f"object trajectory has {positions.shape[0]} frames, but lift windows require "
            f"at least {max(initial_frames, final_frames)}"
        )

    initial_z = float(np.median(positions[:initial_frames, 2]))
    final_lift = positions[-final_frames:, 2] - initial_z
    median_lift = float(np.median(final_lift))
    lifted_ratio = float(np.mean(final_lift >= thresholds.lifted_frame_threshold_m))
    success = (
        median_lift >= thresholds.median_lift_threshold_m
        and lifted_ratio >= thresholds.lifted_frame_ratio
    )
    return {
        "success": bool(success),
        "initial_z_m": _json_float(initial_z),
        "initial_window_frames": initial_frames,
        "final_window_frames": final_frames,
        "final_median_lift_m": _json_float(median_lift),
        "final_lifted_frame_ratio": _json_float(lifted_ratio),
    }


def _median_filter_three(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 3:
        return values.copy()
    padded = np.pad(values, ((1, 1), (0, 0)), mode="edge")
    return np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=0), axis=0)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    window = max(1, int(window))
    if window == 1 or values.size < 2:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _speed(positions: np.ndarray, fps: float, dims: slice | list[int]) -> np.ndarray:
    selected = positions[:, dims]
    delta = np.diff(selected, axis=0, prepend=selected[[0]])
    return np.linalg.norm(delta, axis=1) * fps


def _debounce_contact(raw_contact: np.ndarray, min_frames: int) -> np.ndarray:
    result = np.zeros_like(np.asarray(raw_contact, dtype=bool))
    for start, end in true_intervals(raw_contact):
        if end - start + 1 >= max(1, int(min_frames)):
            result[start : end + 1] = True
    return result


def _bridge_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if max_gap_frames <= 0:
        return result
    false_runs = true_intervals(~result)
    for start, end in false_runs:
        if start == 0 or end == result.size - 1:
            continue
        if end - start + 1 <= max_gap_frames:
            result[start : end + 1] = True
    return result


def _extract_steps(contact: np.ndarray, positions: np.ndarray, foot: str) -> list[dict]:
    """Extract complete contact->swing->contact events for one foot."""
    contact = np.asarray(contact, dtype=bool)
    steps: list[dict] = []
    takeoff = None
    for frame in range(1, contact.size):
        if contact[frame - 1] and not contact[frame]:
            takeoff = frame - 1
        elif not contact[frame - 1] and contact[frame] and takeoff is not None:
            touchdown = frame
            length = float(np.linalg.norm(positions[touchdown, :2] - positions[takeoff, :2]))
            steps.append(
                {
                    "foot": foot,
                    "takeoff_frame": int(takeoff),
                    "touchdown_frame": int(touchdown),
                    "length_m": _json_float(length),
                }
            )
            takeoff = None
    return steps


def analyze_gait(
    pelvis_pos: np.ndarray,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    fps: float,
    thresholds: CleaningThresholds,
) -> dict:
    """Detect sustained high-cadence, short-step locomotion."""
    arrays = [
        np.asarray(pelvis_pos, dtype=np.float64),
        np.asarray(left_foot_pos, dtype=np.float64),
        np.asarray(right_foot_pos, dtype=np.float64),
    ]
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if any(a.ndim != 2 or a.shape[1] != 3 for a in arrays):
        raise ValueError("pelvis and foot trajectories must all have shape (T,3)")
    if len({a.shape[0] for a in arrays}) != 1 or arrays[0].shape[0] < 2:
        raise ValueError("pelvis and foot trajectories must have the same non-trivial length")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("gait trajectories contain NaN or Inf")

    pelvis, left, right = (_median_filter_three(a) for a in arrays)
    feet_z = np.concatenate([left[:, 2], right[:, 2]])
    ground_z = float(np.percentile(feet_z, 5.0))
    left_speed = _speed(left, fps, slice(None))
    right_speed = _speed(right, fps, slice(None))
    left_contact = _debounce_contact(
        (left[:, 2] <= ground_z + thresholds.foot_contact_height_m)
        & (left_speed < thresholds.foot_contact_speed_mps),
        thresholds.contact_debounce_frames,
    )
    right_contact = _debounce_contact(
        (right[:, 2] <= ground_z + thresholds.foot_contact_height_m)
        & (right_speed < thresholds.foot_contact_speed_mps),
        thresholds.contact_debounce_frames,
    )

    steps = _extract_steps(left_contact, left, "left") + _extract_steps(
        right_contact, right, "right"
    )
    steps.sort(key=lambda item: item["touchdown_frame"])
    for step in steps:
        step["is_micro_step"] = bool(step["length_m"] < thresholds.micro_step_length_m)

    pelvis_speed = _speed(pelvis, fps, [0, 1])
    pelvis_speed = _moving_average(pelvis_speed, max(1, int(round(0.2 * fps))))
    locomotion_mask = pelvis_speed >= thresholds.locomotion_speed_mps
    locomotion_mask = _bridge_short_gaps(locomotion_mask, int(round(0.2 * fps)))

    min_segment_frames = max(2, int(np.ceil(thresholds.locomotion_min_seconds * fps)))
    segments: list[dict] = []
    severe = False
    warning = False
    for start, end in true_intervals(locomotion_mask):
        frame_count = end - start + 1
        displacement = float(np.linalg.norm(pelvis[end, :2] - pelvis[start, :2]))
        if frame_count < min_segment_frames or displacement < thresholds.locomotion_min_displacement_m:
            continue
        duration = frame_count / fps
        segment_steps = [
            step for step in steps if start <= step["touchdown_frame"] <= end
        ]
        lengths = [step["length_m"] for step in segment_steps]
        step_count = len(segment_steps)
        cadence = step_count / duration
        median_length = float(np.median(lengths)) if lengths else 0.0
        micro_ratio = (
            float(np.mean([step["is_micro_step"] for step in segment_steps]))
            if segment_steps
            else 0.0
        )
        checks = {
            "enough_steps": step_count >= thresholds.shuffle_min_steps,
            "high_cadence": cadence >= thresholds.shuffle_cadence_steps_per_second,
            "short_median_step": bool(lengths)
            and median_length < thresholds.micro_step_length_m,
            "high_micro_step_ratio": micro_ratio >= thresholds.shuffle_micro_step_ratio,
        }
        segment_severe = all(checks.values())
        # A warning is useful for borderline clips, but a lack of enough steps
        # alone must not warn on ordinary short locomotion segments.
        segment_warning = checks["enough_steps"] and sum(checks.values()) >= 3 and not segment_severe
        severe = severe or segment_severe
        warning = warning or segment_warning
        segments.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_seconds": _json_float(duration),
                "pelvis_displacement_m": _json_float(displacement),
                "step_count": step_count,
                "cadence_steps_per_second": _json_float(cadence),
                "median_step_length_m": _json_float(median_length),
                "micro_step_ratio": _json_float(micro_ratio),
                "severe": bool(segment_severe),
                "warning": bool(segment_warning),
            }
        )

    return {
        "severe": bool(severe),
        "warning": bool(warning),
        "ground_z_m": _json_float(ground_z),
        "step_count": len(steps),
        "steps": steps,
        "locomotion_segments": segments,
    }


def classify_motion(lift_metrics: dict, gait_metrics: dict, penetration_metrics: dict) -> dict:
    """Apply the documented rejection precedence to the three metric groups."""
    if not lift_metrics.get("success", False):
        return {"status": "rejected", "reason_code": "reject_not_lifted", "crop_start_frame": 0}
    if gait_metrics.get("severe", False):
        return {"status": "rejected", "reason_code": "reject_shuffle", "crop_start_frame": 0}
    penetration_class = penetration_metrics.get("classification")
    if penetration_class == "rejected":
        return {"status": "rejected", "reason_code": "reject_penetration", "crop_start_frame": 0}
    if penetration_class == "repaired":
        return {
            "status": "repaired",
            "reason_code": "repaired_prefix_penetration",
            "crop_start_frame": int(penetration_metrics["crop_start_frame"]),
        }
    if penetration_class != "clean":
        raise ValueError(f"unknown penetration classification: {penetration_class}")
    return {"status": "clean", "reason_code": "passed_all_checks", "crop_start_frame": 0}
