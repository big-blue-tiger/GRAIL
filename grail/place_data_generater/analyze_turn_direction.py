"""Classify post-grasp turns from accumulated pelvis heading in Z-up robot PKLs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


DATASET = Path(__file__).resolve().parents[2] / "data/hf_dataset/data_update/data/pickup_table_place_concat"
ANNOTATIONS = {
    "pickup_table__alcohol_1__004": "right",
    "pickup_table__alcohol_2__001": "right",
    "pickup_table__alcohol_2__005": "right",
    "pickup_table__alcohol_11__000": "left",
}


def classify_turn(root_rot_xyzw, carry_frame):
    """Positive accumulated world-Z heading is left; negative is right.

    Begin at the retained source carry frame and include the appended motion.
    Unwrap the heading before subtraction so turns exceeding 180 degrees retain
    their actual direction. The appended hold/release freezes the root pose.
    """
    quaternions = np.asarray(root_rot_xyzw)
    if (quaternions.ndim != 2 or quaternions.shape[1] != 4
            or not np.isfinite(quaternions).all()):
        raise ValueError("root_rot must be finite (T,4) xyzw quaternions")
    if (not isinstance(carry_frame, (int, np.integer))
            or not 0 <= carry_frame < len(quaternions)):
        raise ValueError("carry_frame must index the source carry pose")
    rotation = Rotation.from_quat(quaternions[carry_frame:]).as_matrix()
    yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    change = float(yaw[-1] - yaw[0])
    direction = "left" if change > 0 else "right" if change < 0 else "straight"
    return direction, float(np.rad2deg(change))


def analyze(dataset):
    dataset = Path(dataset)
    paths = sorted((dataset / "robot").glob("*.pkl"))
    if not paths:
        raise ValueError(f"No robot PKLs in {dataset / 'robot'}")
    records = []
    for path in paths:
        payload = joblib.load(path)
        if len(payload) != 1:
            raise ValueError(f"{path}: expected one robot motion")
        robot = next(iter(payload.values()))
        meta = joblib.load(dataset / "meta" / path.name)
        start = meta["place_carry_source_frame"]
        direction, angle = classify_turn(robot["root_rot"], start)
        records.append({"name": path.stem, "carry_frame": int(start),
                        "yaw_change_deg": angle, "direction": direction})
    counts = Counter(record["direction"] for record in records)
    validations = [{"name": record["name"], "expected": ANNOTATIONS[record["name"]],
                    "actual": record["direction"],
                    "matches": record["direction"] == ANNOTATIONS[record["name"]]}
                   for record in records if record["name"] in ANNOTATIONS]
    return {
        "dataset": str(dataset.resolve()), "total": len(records),
        "counts": {direction: counts[direction] for direction in ("left", "right", "straight")},
        "percentages": {direction: 100 * counts[direction] / len(records)
                        for direction in ("left", "right", "straight")},
        "annotation_checks": validations, "motions": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
