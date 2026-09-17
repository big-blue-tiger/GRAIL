"""Shared motion-metadata residual blending for teacher play and training."""

from pathlib import Path
import logging
import numbers
import json

import joblib
import torch


def motion_source_step_ratio(command, key):
    """Convert source PKL frame indices to the motion command's simulation steps."""
    source = command.motion_lib._motion_data_load[key]
    if "path" in source:
        source = next(iter(joblib.load(source["path"]).values()))
    source_fps = float(source["fps"])
    if not 0 < source_fps < float("inf"):
        raise ValueError(f"{key}: invalid motion fps {source_fps}")
    return command.motion_lib._sim_fps / source_fps


class KimodoTrackingBoundary:
    """Cache report boundaries; ordinary motions remain tracked throughout.

    b_start_index is the first GRAIL frame, not the beginning or end of the
    PCHIP replacement window. Rebuild the tensor when the library reloads keys.
    """

    def __init__(self):
        self._keys = None
        self._boundaries = {}

    def before_grail(self, command, device):
        keys = command.motion_lib.curr_motion_keys
        if keys is not self._keys:
            motion_file = Path(command.cfg.motion_lib_cfg["motion_file"])
            robot_dir = motion_file if motion_file.is_dir() else motion_file.parent
            bounds = []
            for key in keys:
                source = command.motion_lib._motion_data_load[key]
                # Directory mode keys are PKL stems, not internal motion keys.
                source_path = Path(source["path"]) if "path" in source else motion_file
                name = source_path.stem if source_path.is_file() else str(key)
                path = robot_dir.parent / "reports" / f"{name}.json"
                if path not in self._boundaries:
                    boundary = float("inf")
                    if path.is_file():
                        report = json.loads(path.read_text())
                        if report.get("sources", {}).get("walk_csv") is not None:
                            start = report.get("b_start_index")
                            total = report.get("total_frames")
                            fps = report.get("fps")
                            if (type(start) is not int or type(total) is not int
                                    or not 0 <= start < total
                                    or not isinstance(fps, (int, float))
                                    or not 0 < fps < float("inf")):
                                raise ValueError(f"{path}: invalid Kimodo boundary/frame count/fps")
                            ratio = motion_source_step_ratio(command, key)
                            if abs(ratio * fps - command.motion_lib._sim_fps) > 1e-5:
                                raise ValueError(f"{path}: report fps does not match source motion")
                            boundary = start * ratio
                    self._boundaries[path] = boundary
                bounds.append(self._boundaries[path])
            self._frame_bounds = torch.tensor(bounds, device=device, dtype=torch.float64)
            self._keys = keys
        frame = command.motion_start_time_steps + command.time_steps
        return frame < self._frame_bounds[command.motion_ids]


class EvalResidualTransition:
    """Scale only latent residuals using zero-based, inclusive motion frames."""

    def __init__(self, env):
        self.env = env
        self._keys = None
        self._bounds = {}

    def weights(self, device):
        """Snapshot current per-env weights, including random motion start offsets."""
        env = self.env
        if not (
            getattr(env, "_use_latent_residual", False)
            or getattr(env, "_use_student_direct_latent", False)
        ):
            return torch.ones(env.num_envs, device=device)
        command = env.motion_command
        keys = command.motion_lib.curr_motion_keys
        # The motion library replaces this array when it reloads motions.
        if keys is not self._keys:
            motion_file = Path(command.cfg.motion_lib_cfg["motion_file"])
            robot_dir = motion_file if motion_file.is_dir() else motion_file.parent
            meta_dir = robot_dir.parent / "meta"
            bounds = []
            for key in keys:
                path = meta_dir / f"{key}.pkl"
                if path not in self._bounds:
                    meta = joblib.load(path) if path.is_file() else {}
                    start = meta.get("transition_start_frame")
                    end = meta.get("transition_end_frame")
                    bound = (-1, -1)
                    if start is not None and end is not None:
                        if not (
                            isinstance(start, numbers.Integral)
                            and isinstance(end, numbers.Integral)
                            and 0 <= start < end
                        ):
                            raise ValueError(
                                f"{path}: transition frames must be integers with "
                                f"0 <= start < end; got {start!r}, {end!r}"
                            )
                        # Metadata indexes source frames, whereas command time
                        # is measured in simulation steps (often 50 Hz).
                        step_ratio = motion_source_step_ratio(command, key)
                        bound = (start * step_ratio, end * step_ratio)
                        logging.getLogger(__name__).info(
                            "Residual transition %s: frames %s -> %s", key, start, end
                        )
                    self._bounds[path] = bound
                bounds.append(self._bounds[path])
            self._frame_bounds = torch.tensor(bounds, device=device)
            self._keys = keys

        start, end = self._frame_bounds[command.motion_ids].unbind(-1)
        frame = command.motion_start_time_steps + command.time_steps
        duration = torch.where(start >= 0, end - start, torch.ones_like(start))
        weight = ((frame - start).float() / duration).clamp(0, 1)
        weight = torch.where(start >= 0, weight, torch.ones_like(weight))
        return weight

    def apply(self, actions, *, weight=None, residual=False):
        """Blend residual actions; explicit residual=True also supports DAgger teachers."""
        env = self.env
        if not residual and (
            not getattr(env, "_use_latent_residual", False)
            or getattr(env, "_use_student_direct_latent", False)
        ):
            return actions
        latent_dim = env.config.get("tokenizer_action_dim")
        if latent_dim is None:
            if residual and not getattr(env, "_use_latent_residual", False):
                return actions
            raise ValueError("Residual transition requires tokenizer_action_dim")
        if weight is None:
            weight = self.weights(actions.device)
        result = actions.clone()
        result[..., :latent_dim] *= weight.to(actions.dtype).unsqueeze(-1)
        return result
