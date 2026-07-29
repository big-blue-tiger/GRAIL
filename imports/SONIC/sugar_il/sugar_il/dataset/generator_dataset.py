from __future__ import annotations

import copy
import glob
import pickle
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from sugar_il.common.geometry import world_pose_to_body
from sugar_il.dataset.base_dataset import BaseLowdimDataset
from sugar_il.model.common.normalizer import LinearNormalizer


REQUIRED_ARRAYS = {
    "robot_root_pos_w": (3,),
    "robot_root_quat_w": (4,),
    "object_root_pos_w": (3,),
    "object_root_quat_w": (4,),
    "object_bps": (10,),
    "table_geometry": (4,),
    "hand_object_transform_6d": (9,),
    "hand_object_contact_force_magnitude": (8,),
    "base_lin_vel": (3,),
    "base_ang_vel": (3,),
    "joint_pos": (43,),
    "joint_vel": (43,),
    "combined_latent_pre_fsq": (64,),
    "hand_primitive_executed_binary": (2,),
}


def _resolve_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    resolved: set[Path] = set()
    for value in values:
        value = Path(value).expanduser()
        if value.is_dir():
            resolved.update(value.glob("*.object_aware.pkl"))
        elif any(char in str(value) for char in "*?[]"):
            resolved.update(Path(path) for path in glob.glob(str(value)))
        elif value.is_file():
            resolved.add(value)
        else:
            raise FileNotFoundError(f"Object-aware dataset path does not exist: {value}")
    if not resolved:
        raise ValueError("No *.object_aware.pkl files were found")
    return sorted(path.resolve() for path in resolved)


def _load_episode(path: Path, required_frames: int) -> dict:
    # Pickle can execute code. Inputs are expected to be trusted files produced by SONIC.
    with path.open("rb") as file:
        episode = pickle.load(file)
    if episode.get("schema_version") != 3:
        raise ValueError(f"{path}: expected schema_version=3")
    if episode.get("pose_timing") != "pre_step_aligned_with_policy_input":
        raise ValueError(f"{path}: poses are not pre-step aligned")
    if episode.get("pose_quaternion_format") != "wxyz":
        raise ValueError(f"{path}: expected wxyz quaternions")
    if (
        "hand_object_contact_force_magnitude" not in episode
        and "finger_tips_force" in episode
    ):
        legacy_force = np.asarray(episode["finger_tips_force"])
        if legacy_force.ndim != 2 or legacy_force.shape[1] != 24:
            raise ValueError(
                f"{path}: legacy finger_tips_force has shape {legacy_force.shape}, "
                "expected [T,24]"
            )
        episode["hand_object_contact_force_magnitude"] = np.linalg.norm(
            legacy_force.reshape(len(legacy_force), 8, 3), axis=-1
        )

    lengths = set()
    for key, trailing_shape in REQUIRED_ARRAYS.items():
        if key not in episode:
            raise KeyError(f"{path}: missing {key}")
        value = np.asarray(episode[key])
        if value.ndim != 2 or value.shape[1:] != trailing_shape:
            raise ValueError(f"{path}: {key} has shape {value.shape}, expected [T,{trailing_shape[0]}]")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} contains non-finite values")
        lengths.add(len(value))
        episode[key] = value
    if len(lengths) != 1:
        raise ValueError(f"{path}: per-frame arrays have inconsistent lengths")
    length = lengths.pop()
    if length < required_frames + 1:
        raise ValueError(
            f"{path}: needs at least {required_frames + 1} frames, got {length}"
        )
    hands = episode["hand_primitive_executed_binary"]
    if not np.isin(hands, (0, 1)).all():
        raise ValueError(f"{path}: executed hand primitives are not binary")
    for key in ("robot_root_quat_w", "object_root_quat_w"):
        norms = np.linalg.norm(episode[key], axis=-1)
        if np.max(np.abs(norms - 1.0)) > 1e-3:
            raise ValueError(f"{path}: {key} is not normalized")
    for key in ("object_bps", "table_geometry"):
        if np.max(np.abs(episode[key] - episode[key][0])) > 1e-5:
            raise ValueError(f"{path}: {key} changes within one trajectory")
    episode["_path"] = path
    return episode


class GeneratorDataset(BaseLowdimDataset):
    """Contiguous 40-frame windows from SONIC schema-v3 recordings."""

    def __init__(
        self,
        pickle_paths: str | Path | Sequence[str | Path],
        horizon: int = 40,
        val_ratio: float = 0.05,
        seed: int = 42,
        _episodes: list[dict] | None = None,
        _is_validation: bool = False,
    ):
        if horizon != 40:
            raise ValueError("Object-aware flow matching uses a fixed 40-frame horizon")
        self.horizon = horizon
        self.seed = seed
        self.val_ratio = val_ratio
        action_span = horizon
        all_episodes = (
            _episodes
            if _episodes is not None
            else [
                _load_episode(path, action_span)
                for path in _resolve_paths(pickle_paths)
            ]
        )
        if _episodes is None:
            order = np.arange(len(all_episodes))
            np.random.default_rng(seed).shuffle(order)
            n_val = 0 if len(order) < 2 or val_ratio <= 0 else min(len(order) - 1, max(1, round(len(order) * val_ratio)))
            val_ids = set(order[:n_val].tolist())
            self._train_episodes = [ep for i, ep in enumerate(all_episodes) if i not in val_ids]
            self._val_episodes = [ep for i, ep in enumerate(all_episodes) if i in val_ids]
        else:
            self._train_episodes = [] if _is_validation else all_episodes
            self._val_episodes = all_episodes if _is_validation else []
        self.episodes = self._val_episodes if _is_validation else self._train_episodes
        self.indices = [
            (episode_id, t)
            for episode_id, episode in enumerate(self.episodes)
            for t in range(1, len(episode["combined_latent_pre_fsq"]) - action_span + 1)
        ]

    def get_validation_dataset(self):
        return GeneratorDataset(
            [],
            self.horizon,
            self.val_ratio,
            self.seed,
            self._val_episodes,
            True,
        )

    @staticmethod
    def _pose(episode: dict, t: int):
        robot_pos = torch.from_numpy(episode["robot_root_pos_w"][t]).float()
        robot_quat = torch.from_numpy(episode["robot_root_quat_w"][t]).float()
        object_pos = torch.from_numpy(episode["object_root_pos_w"][t]).float()
        object_quat = torch.from_numpy(episode["object_root_quat_w"][t]).float()
        return world_pose_to_body(robot_pos, robot_quat, object_pos, object_quat)

    def __getitem__(self, index: int) -> Dict[str, Dict[str, torch.Tensor]]:
        episode_id, t = self.indices[index]
        episode = self.episodes[episode_id]
        object_pos, object_ori = self._pose(episode, t)
        obs = {
            "object_bps": torch.from_numpy(episode["object_bps"][t]).float().unsqueeze(0),
            "table_geometry": torch.from_numpy(
                episode["table_geometry"][t]
            ).float().unsqueeze(0),
            "object_pos_b": object_pos.unsqueeze(0),
            "object_ori_b_6d": object_ori.unsqueeze(0),
            "hand_object_transform_6d": torch.from_numpy(episode["hand_object_transform_6d"][t]).float().unsqueeze(0),
            "hand_object_contact_force_magnitude": torch.from_numpy(
                episode["hand_object_contact_force_magnitude"][t]
            ).float().unsqueeze(0),
            **{
                key: torch.from_numpy(episode[key][t]).float().unsqueeze(0)
                for key in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")
            },
        }
        action = {
            "latent": torch.from_numpy(
                episode["combined_latent_pre_fsq"][
                    t : t + self.horizon
                ]
            ).float(),
            "hand_primitive": torch.from_numpy(
                episode["hand_primitive_executed_binary"][
                    t : t + self.horizon
                ]
            ).float(),
        }
        return {"obs": obs, "action": action}

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        if not self.indices:
            raise ValueError("Cannot fit a normalizer on an empty dataset")
        fields: dict[str, list[np.ndarray]] = {}
        latents = []
        for i in range(len(self)):
            sample = self[i]
            for key, value in sample["obs"].items():
                fields.setdefault(key, []).append(value.numpy())
            latents.append(sample["action"]["latent"].numpy())
        data = {key: np.concatenate(values, axis=0) for key, values in fields.items()}
        data["latent"] = np.concatenate(latents, axis=0)
        normalizer = LinearNormalizer()
        normalizer.fit(data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.indices)
