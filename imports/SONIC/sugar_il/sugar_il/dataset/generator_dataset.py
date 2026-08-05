from __future__ import annotations

import copy
import glob
import pickle
import warnings
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from sugar_il.common.geometry import world_pose_to_body
from sugar_il.dataset.base_dataset import BaseLowdimDataset
from sugar_il.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


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


class _WeightedStats:
    """Streaming per-feature statistics without materializing all windows."""

    def __init__(self):
        self.minimum = None
        self.maximum = None
        self.total = None
        self.total_squared = None
        self.count = 0.0

    def update(self, values: np.ndarray | torch.Tensor, weights=None):
        values = torch.as_tensor(values, dtype=torch.float64).reshape(-1, values.shape[-1])
        if weights is None:
            weights = torch.ones(len(values), dtype=torch.float64)
        else:
            weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        selected = weights > 0
        if not selected.any():
            return
        values = values[selected]
        weights = weights[selected].unsqueeze(-1)
        batch_minimum = values.min(dim=0).values
        batch_maximum = values.max(dim=0).values
        self.minimum = (
            batch_minimum
            if self.minimum is None
            else torch.minimum(self.minimum, batch_minimum)
        )
        self.maximum = (
            batch_maximum
            if self.maximum is None
            else torch.maximum(self.maximum, batch_maximum)
        )
        weighted_total = (values * weights).sum(dim=0)
        weighted_total_squared = (values.square() * weights).sum(dim=0)
        self.total = (
            weighted_total if self.total is None else self.total + weighted_total
        )
        self.total_squared = (
            weighted_total_squared
            if self.total_squared is None
            else self.total_squared + weighted_total_squared
        )
        self.count += weights.sum().item()

    def tensors(self):
        if self.count <= 0:
            raise ValueError("Cannot finalize empty statistics")
        mean = self.total / self.count
        if self.count > 1:
            variance = (
                self.total_squared - self.total.square() / self.count
            ) / (self.count - 1)
            std = variance.clamp_min(0).sqrt()
        else:
            std = torch.full_like(mean, float("nan"))
        return tuple(
            value.to(dtype=torch.float32)
            for value in (self.minimum, self.maximum, mean, std)
        )


def _normalizer_from_stats(
    statistics: dict[str, _WeightedStats],
    mode: str,
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-4,
    fit_offset: bool = True,
) -> LinearNormalizer:
    if mode not in ("limits", "gaussian"):
        raise ValueError(f"Unsupported normalizer mode: {mode}")
    normalizer = LinearNormalizer()
    for key, stats in statistics.items():
        input_min, input_max, input_mean, input_std = stats.tensors()
        if mode == "limits":
            if fit_offset:
                input_range = input_max - input_min
                ignore_dim = input_range < range_eps
                safe_range = input_range.clone()
                safe_range[ignore_dim] = output_max - output_min
                scale = (output_max - output_min) / safe_range
                offset = output_min - scale * input_min
                offset[ignore_dim] = (
                    (output_max + output_min) / 2 - input_min[ignore_dim]
                )
            else:
                if output_max <= 0 or output_min >= 0:
                    raise ValueError("limits without offset requires a signed output range")
                output_abs = min(abs(output_min), abs(output_max))
                input_abs = torch.maximum(input_min.abs(), input_max.abs())
                ignore_dim = input_abs < range_eps
                safe_abs = input_abs.clone()
                safe_abs[ignore_dim] = output_abs
                scale = output_abs / safe_abs
                offset = torch.zeros_like(input_mean)
        else:
            ignore_dim = input_std < range_eps
            safe_std = input_std.clone()
            safe_std[ignore_dim] = 1
            scale = 1 / safe_std
            offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

        field = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={
                "min": input_min,
                "max": input_max,
                "mean": input_mean,
                "std": input_std,
            },
        )
        normalizer[key] = field
    return normalizer


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


def _load_episode(path: Path, required_frames: int) -> dict | None:
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
        warnings.warn(
            f"Skipping {path}: needs at least {required_frames + 1} frames, got {length}",
            stacklevel=2,
        )
        return None
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
                episode
                for path in _resolve_paths(pickle_paths)
                if (episode := _load_episode(path, action_span)) is not None
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
        statistics: dict[str, _WeightedStats] = {}

        def update(key, values, weights=None):
            statistics.setdefault(key, _WeightedStats()).update(values, weights)

        direct_observation_keys = (
            "object_bps",
            "table_geometry",
            "hand_object_transform_6d",
            "hand_object_contact_force_magnitude",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
        )
        for episode in self.episodes:
            length = len(episode["combined_latent_pre_fsq"])
            num_starts = length - self.horizon
            observation_slice = slice(1, num_starts + 1)
            for key in direct_observation_keys:
                update(key, np.asarray(episode[key][observation_slice], dtype=np.float32))

            robot_position = torch.from_numpy(
                episode["robot_root_pos_w"][observation_slice]
            ).float()
            robot_quaternion = torch.from_numpy(
                episode["robot_root_quat_w"][observation_slice]
            ).float()
            object_position = torch.from_numpy(
                episode["object_root_pos_w"][observation_slice]
            ).float()
            object_quaternion = torch.from_numpy(
                episode["object_root_quat_w"][observation_slice]
            ).float()
            object_position_b, object_orientation_b = world_pose_to_body(
                robot_position,
                robot_quaternion,
                object_position,
                object_quaternion,
            )
            update("object_pos_b", object_position_b)
            update("object_ori_b_6d", object_orientation_b)

            # A frame appears once for every overlapping action window that
            # contains it. Accumulate those multiplicities as weights instead
            # of allocating [num_windows, horizon, latent_dim].
            difference = np.zeros(length + 1, dtype=np.int64)
            starts = np.arange(1, num_starts + 1)
            np.add.at(difference, starts, 1)
            np.add.at(difference, starts + self.horizon, -1)
            latent_weights = np.cumsum(difference[:-1])
            update(
                "latent",
                np.asarray(episode["combined_latent_pre_fsq"], dtype=np.float32),
                latent_weights,
            )

        return _normalizer_from_stats(statistics, mode=mode, **kwargs)

    def __len__(self) -> int:
        return len(self.indices)
