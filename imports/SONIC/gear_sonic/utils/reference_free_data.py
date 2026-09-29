"""Static scene inputs for the joint-micro-step student evaluation."""

from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
import pickle

import joblib
import numpy as np


def load_scene_inputs(dataset_dir, initial_pose_file, num_envs):
    """Select paired assets and retain only the object poses needed at reset."""
    root = Path(dataset_dir).resolve()
    groups = defaultdict(list)
    for path in sorted((root / "object_usd").glob("*.usd")):
        groups[path.stem.rsplit("__", 1)[0]].append(path.stem)
    keys = [key for row in zip_longest(*groups.values()) for key in row if key is not None]
    if len(keys) < num_envs:
        raise ValueError(f"Need {num_envs} paired objects, found {len(keys)} in {root}")

    objects = []
    for key in keys[:num_envs]:
        meta = joblib.load(root / "meta" / f"{key}.pkl")
        trajectories = joblib.load(root / "objects" / f"{key}.pkl")
        trajectory = next(iter(trajectories.values()))
        objects.append({
            "key": key,
            "usd": str(root / "object_usd" / f"{key}.usd"),
            "bps": np.load(root / "bps" / f"{key}.npy").tolist(),
            "position": np.asarray(trajectory["root_pos"])[0, 0].tolist(),
            "quaternion": np.asarray(trajectory["root_quat"])[0, 0].tolist(),
            "table_position": np.asarray(meta["table_pos"]).tolist(),
            "table_quaternion": np.asarray(meta["table_quat"]).tolist(),
            "table_size": np.asarray(meta["table_size"]).tolist(),
        })
        del trajectory, trajectories

    with Path(initial_pose_file).open("rb") as stream:
        frame = pickle.load(stream)
    pose = {key: np.asarray(frame[key]).tolist() for key in (
        "dof", "hand_dof_pos", "root_rot", "root_trans_offset"
    )}
    return objects, pose
