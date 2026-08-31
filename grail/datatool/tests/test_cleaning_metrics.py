from __future__ import annotations

import joblib
import numpy as np
import pytest

from grail.datatool.batch_render_replay_clip import (
    export_accepted_motion,
    validate_motion_input,
)
from grail.datatool.cleaning_metrics import (
    CleaningThresholds,
    analyze_gait,
    analyze_object_lift,
    analyze_penetration,
    classify_motion,
    effective_penetration_mask,
)
from grail.visualization.prepare_vis_shard import (
    G1_MUJOCO_TO_ISAACLAB_DOF,
    convert_motion_lib_to_trajectories,
)


def test_penetration_debounce_and_single_frame_override():
    thresholds = CleaningThresholds()
    depths = np.zeros(12)
    depths[2] = 0.011
    assert not effective_penetration_mask(depths, thresholds).any()

    depths[5:7] = 0.011
    mask = effective_penetration_mask(depths, thresholds)
    assert np.flatnonzero(mask).tolist() == [5, 6]

    depths[:] = 0
    depths[9] = 0.021
    assert np.flatnonzero(effective_penetration_mask(depths, thresholds)).tolist() == [9]


def test_penetration_crop_budget_boundary():
    thresholds = CleaningThresholds()
    depths = np.zeros(250)
    depths[0:2] = 0.011
    repaired = analyze_penetration(depths, 25.0, thresholds)
    assert repaired["classification"] == "repaired"
    assert repaired["crop_start_frame"] == 21

    depths[:] = 0
    depths[30:32] = 0.011
    rejected = analyze_penetration(depths, 25.0, thresholds)
    assert rejected["crop_start_frame"] == 51
    assert rejected["crop_budget_frames"] == 50
    assert rejected["classification"] == "rejected"


def test_object_lift_requires_median_and_eighty_percent():
    thresholds = CleaningThresholds()
    positions = np.zeros((250, 3), dtype=np.float32)
    positions[-25:, 2] = 0.12
    assert analyze_object_lift(positions, 25.0, thresholds)["success"]

    brief = positions.copy()
    brief[-25:, 2] = 0.0
    brief[-5:, 2] = 0.20
    metrics = analyze_object_lift(brief, 25.0, thresholds)
    assert not metrics["success"]
    assert metrics["final_lifted_frame_ratio"] == pytest.approx(0.2)

    dropped = positions.copy()
    dropped[-25:, 2] = 0.12
    dropped[-10:, 2] = 0.0
    assert not analyze_object_lift(dropped, 25.0, thresholds)["success"]


def _synthetic_gait(step_interval, step_length, total_frames, pelvis_distance, fps=25):
    pelvis = np.zeros((total_frames, 3), dtype=np.float64)
    pelvis[:, 0] = np.linspace(0.0, pelvis_distance, total_frames)
    pelvis[:, 2] = 0.8
    feet = [np.zeros((total_frames, 3)), np.zeros((total_frames, 3))]
    feet[0][:, 1] = 0.1
    feet[1][:, 1] = -0.1
    current = [0.0, 0.0]
    events = []
    foot = 0
    for start in range(5, total_frames - 5, step_interval):
        end = min(start + 4, total_frames - 1)
        current[foot] += step_length
        events.append((foot, start, end, current[foot]))
        foot = 1 - foot
    for foot_index, positions in enumerate(feet):
        x = 0.0
        cursor = 0
        for event_foot, start, end, target in events:
            if event_foot != foot_index:
                continue
            positions[cursor:start, 0] = x
            for offset, frame in enumerate(range(start, end + 1)):
                alpha = (offset + 1) / (end - start + 1)
                positions[frame, 0] = x + (target - x) * alpha
                positions[frame, 2] = 0.10 * np.sin(np.pi * alpha)
            x = target
            cursor = end + 1
        positions[cursor:, 0] = x
    return pelvis, feet[0], feet[1]


def test_gait_distinguishes_normal_micro_steps_and_stationary_adjustment():
    thresholds = CleaningThresholds()
    normal = analyze_gait(
        *_synthetic_gait(15, 0.30, 125, 1.2), fps=25.0, thresholds=thresholds
    )
    assert not normal["severe"]

    shuffle = analyze_gait(
        *_synthetic_gait(8, 0.08, 100, 0.7), fps=25.0, thresholds=thresholds
    )
    assert shuffle["severe"]
    assert shuffle["locomotion_segments"][0]["cadence_steps_per_second"] >= 2.5
    assert shuffle["locomotion_segments"][0]["micro_step_ratio"] == 1.0

    adjustment = analyze_gait(
        *_synthetic_gait(8, 0.08, 100, 0.1), fps=25.0, thresholds=thresholds
    )
    assert not adjustment["severe"]
    assert adjustment["locomotion_segments"] == []


def test_rejection_precedence():
    penetration = {"classification": "repaired", "crop_start_frame": 21}
    assert classify_motion({"success": False}, {"severe": True}, penetration)[
        "reason_code"
    ] == "reject_not_lifted"
    assert classify_motion({"success": True}, {"severe": True}, penetration)[
        "reason_code"
    ] == "reject_shuffle"
    assert classify_motion({"success": True}, {"severe": False}, penetration) == {
        "status": "repaired",
        "reason_code": "repaired_prefix_penetration",
        "crop_start_frame": 21,
    }


def _write_motion_library(base, motion_key="sample", total_frames=50):
    for directory in ("robot", "objects", "meta", "object_usd", "bps"):
        (base / directory).mkdir(parents=True, exist_ok=True)
    dof = np.tile(np.arange(29, dtype=np.float32), (total_frames, 1))
    root_rot = np.zeros((total_frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0  # xyzw identity
    robot = {
        "dof": dof,
        "root_trans_offset": np.zeros((total_frames, 3), dtype=np.float32),
        "root_rot": root_rot,
        "pose_aa": np.zeros((total_frames, 30, 3), dtype=np.float32),
        "hand_dof_pos": np.ones((total_frames, 14), dtype=np.float32),
        "fps": 25.0,
    }
    contacts = {frame: [float(frame)] for frame in range(total_frames)}
    obj = {
        "root_pos": np.zeros((total_frames, 1, 3), dtype=np.float32),
        "root_quat": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (total_frames, 1, 1)
        ),
        "fps": 25.0,
        "contact_points_left_hand": contacts,
        "contact_points_right_hand": contacts,
    }
    joblib.dump({motion_key: robot}, base / "robot" / f"{motion_key}.pkl")
    joblib.dump({motion_key: obj}, base / "objects" / f"{motion_key}.pkl")
    joblib.dump({"table_pos": np.zeros(3)}, base / "meta" / f"{motion_key}.pkl")
    (base / "object_usd" / f"{motion_key}.usd").write_bytes(b"usd-placeholder")
    np.save(base / "bps" / f"{motion_key}.npy", np.zeros((4, 3), dtype=np.float32))
    return dof


def test_preflight_conversion_and_synchronized_export(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    shard = tmp_path / "shard"
    source_dof = _write_motion_library(source)
    metadata = validate_motion_input(str(source), "sample")
    assert metadata["total_frames"] == 50

    keys = convert_motion_lib_to_trajectories(
        str(source), str(shard), motion_filter={"sample"}, quat_convention="xyzw"
    )
    assert keys == ["sample"]
    import pickle

    with open(shard / "trajectories" / "sample.trajectory.pkl", "rb") as file:
        trajectory = pickle.load(file)
    np.testing.assert_array_equal(
        trajectory["dof_pos"][:, :29], source_dof[:, G1_MUJOCO_TO_ISAACLAB_DOF]
    )
    np.testing.assert_array_equal(trajectory["root_quat_w"][0], [1.0, 0.0, 0.0, 0.0])
    assert trajectory["dof_pos"].shape == (50, 43)

    output_frames = export_accepted_motion(str(source), str(output), "sample", 10)
    assert output_frames == 40
    robot = joblib.load(output / "robot" / "sample.pkl")["sample"]
    obj = joblib.load(output / "objects" / "sample.pkl")["sample"]
    assert robot["dof"].shape[0] == obj["root_pos"].shape[0] == 40
    assert sorted(obj["contact_points_left_hand"])[0] == 0
    assert sorted(obj["contact_points_left_hand"])[-1] == 39
    assert (output / "meta" / "sample.pkl").is_file()
    assert (output / "object_usd" / "sample.usd").is_file()
    assert (output / "bps" / "sample.npy").is_file()
