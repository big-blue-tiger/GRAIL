#!/usr/bin/env python3
"""Recompute changed-object hand contacts by replaying the reference in MuJoCo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mujoco
import numpy as np


CONTACT_KEYS = {
    "left": "contact_points_left_hand",
    "right": "contact_points_right_hand",
}


def unwrap_single_entry(data: object, path: Path) -> dict:
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"Expected one wrapped motion in {path}")
    motion = next(iter(data.values()))
    if not isinstance(motion, dict):
        raise ValueError(f"Expected a motion dictionary in {path}")
    return motion


def load_replay_model(model_path: Path, unitree_model_dir: Path) -> mujoco.MjModel:
    """Load an SBTO-expanded MJCF while restoring its omitted relative asset roots."""
    xml = model_path.read_text(encoding="utf-8")
    xml = xml.replace(
        'meshdir="assets/"',
        f'meshdir="{(unitree_model_dir / "assets").resolve()}/"',
    )
    xml = xml.replace('file="meshes/', 'file="../meshes/')
    return mujoco.MjModel.from_xml_string(xml)


class ContactReplay:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.object_geom_id = model.geom("obj").id
        self.actuated_qpos_addresses = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joints) != 1:
            raise ValueError(f"Expected exactly one object free joint, got {len(free_joints)}")
        self.object_qpos_address = int(model.jnt_qposadr[free_joints[0]])
        self.object_rotation = np.empty(9, dtype=np.float64)

    def _hand_side(self, geom_id: int) -> str | None:
        geom_name = self.model.geom(geom_id).name or ""
        body_name = self.model.body(int(self.model.geom_bodyid[geom_id])).name or ""
        names = f"{geom_name} {body_name}"
        if "left_hand" in names:
            return "left"
        if "right_hand" in names:
            return "right"
        return None

    def _set_frame(self, robot: dict, object_motion: dict, frame: int) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[:3] = robot["root_trans_offset"][frame]
        # GRAIL robot PKLs store root_rot in XYZW; MuJoCo expects WXYZ.
        self.data.qpos[3:7] = robot["root_rot"][frame, [3, 0, 1, 2]]

        dof = np.asarray(robot["dof"][frame])
        if "hand_dof_pos" in robot:
            dof = np.concatenate([dof, np.asarray(robot["hand_dof_pos"][frame])])
        if dof.shape != (len(self.actuated_qpos_addresses),):
            raise ValueError(
                f"Expected {len(self.actuated_qpos_addresses)} robot DOFs, got {dof.shape}"
            )
        self.data.qpos[self.actuated_qpos_addresses] = dof

        qpos_address = self.object_qpos_address
        self.data.qpos[qpos_address : qpos_address + 3] = object_motion["root_pos"][
            frame, 0
        ]
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = object_motion[
            "root_quat"
        ][frame, 0]
        mujoco.mj_forward(self.model, self.data)

    def frame_contacts(self) -> dict[str, np.ndarray]:
        qpos_address = self.object_qpos_address
        mujoco.mju_quat2Mat(
            self.object_rotation,
            self.data.qpos[qpos_address + 3 : qpos_address + 7],
        )
        world_to_object = self.object_rotation.reshape(3, 3).T
        object_position = self.data.geom_xpos[self.object_geom_id]
        points = {"left": [], "right": []}

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = map(int, contact.geom)
            if geom1 == self.object_geom_id:
                other_geom = geom2
            elif geom2 == self.object_geom_id:
                other_geom = geom1
            else:
                continue
            side = self._hand_side(other_geom)
            if side is not None:
                points[side].append(
                    world_to_object @ (contact.pos - object_position)
                )

        result = {}
        for side in ("left", "right"):
            if points[side]:
                center = np.mean(points[side], axis=0, dtype=np.float64)
                result[side] = center[None].astype(np.float32)
            else:
                result[side] = np.empty((0, 3), dtype=np.float32)
        return result

    def replay(self, robot: dict, object_motion: dict) -> dict[str, dict[int, np.ndarray]]:
        num_frames = len(robot["dof"])
        if object_motion["root_pos"].shape[0] != num_frames:
            raise ValueError(
                "Robot and object frame counts differ: "
                f"{num_frames} and {object_motion['root_pos'].shape[0]}"
            )
        contacts = {"left": {}, "right": {}}
        for frame in range(num_frames):
            self._set_frame(robot, object_motion, frame)
            frame_contacts = self.frame_contacts()
            for side in ("left", "right"):
                contacts[side][frame] = frame_contacts[side]
        return contacts


def contact_frame_set(contacts: object) -> set[int]:
    if not isinstance(contacts, dict):
        return set()
    return {int(frame) for frame, points in contacts.items() if len(points) > 0}


def repair_file(
    replay: ContactReplay,
    robot_path: Path,
    object_path: Path,
    dry_run: bool,
) -> dict[str, tuple[int, int, int]]:
    robot_wrapped = joblib.load(robot_path)
    object_wrapped = joblib.load(object_path)
    robot = unwrap_single_entry(robot_wrapped, robot_path)
    object_motion = unwrap_single_entry(object_wrapped, object_path)
    recomputed = replay.replay(robot, object_motion)

    comparison = {}
    for side, key in CONTACT_KEYS.items():
        old_frames = contact_frame_set(object_motion.get(key))
        new_frames = contact_frame_set(recomputed[side])
        comparison[side] = (
            len(old_frames),
            len(new_frames),
            len(old_frames & new_frames),
        )
        object_motion[key] = recomputed[side]

    if not dry_run:
        temporary_path = object_path.with_suffix(object_path.suffix + ".tmp")
        try:
            joblib.dump(object_wrapped, temporary_path, compress=True)
            os.replace(temporary_path, object_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return comparison


def parse_args() -> argparse.Namespace:
    sbto_root = Path("../sbto")
    template_run = (
        sbto_root
        / "datas/grail2sbto_dataset_table_fixed_z/pickup_table__apple_0__003"
        / "refined/2026_07_14__15_56_36__pickup_table__apple_0__003"
    )
    parser = argparse.ArgumentParser(
        description="Recompute per-frame hand/object contacts using MuJoCo."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw_grail_chaged_obj/pickup_table"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=template_run / "mj_model.xml",
    )
    parser.add_argument(
        "--unitree-model-dir",
        type=Path,
        default=sbto_root / "sbto/models/unitree_g1",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_replay_model(args.model, args.unitree_model_dir)
    replay = ContactReplay(model)
    object_paths = sorted((args.dataset_dir / "objects").glob("*.pkl"))
    if not object_paths:
        raise ValueError(f"No object PKLs found under {args.dataset_dir}")

    totals = {
        "left": np.zeros(3, dtype=np.int64),
        "right": np.zeros(3, dtype=np.int64),
    }
    for object_path in object_paths:
        robot_path = args.dataset_dir / "robot" / object_path.name
        if not robot_path.is_file():
            raise FileNotFoundError(f"Missing matching robot motion: {robot_path}")
        comparison = repair_file(replay, robot_path, object_path, args.dry_run)
        for side in ("left", "right"):
            totals[side] += comparison[side]

    mode = "Checked" if args.dry_run else "Updated"
    print(f"{mode} {len(object_paths)} object PKLs")
    for side in ("left", "right"):
        old, new, overlap = totals[side]
        print(f"{side}: old={old}, recomputed={new}, overlap={overlap}")


if __name__ == "__main__":
    main()
