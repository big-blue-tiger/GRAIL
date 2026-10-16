"""Shared motion-metadata residual blending for teacher play and training."""

from pathlib import Path
import logging
import math
import numbers

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


class PlaceCarryBoundary:
    """Cache carry steps for BC masking; non-place motions keep the legacy mask."""

    def __init__(self):
        """Create an empty carry-boundary cache for teacher BC masking.

        The first before_carry call loads metadata. Replacing the motion
        library's current key list invalidates the cached step boundaries.
        """
        self._keys = None

    def before_carry(self, command, device):
        """Return which environments precede the annotated source carry frame.

        Convert source frame indices to simulation steps using source FPS and
        ceil, then compare against motion_start_time_steps + time_steps.
        Return a boolean [N] mask on device and rebuild it on library reload.
        Motions without a carry annotation retain an infinite boundary; invalid
        annotated frame/total-frame pairs raise ValueError.
        """
        keys = command.motion_lib.curr_motion_keys
        if keys is not self._keys:
            motion_file = Path(command.cfg.motion_lib_cfg["motion_file"])
            robot_dir = motion_file if motion_file.is_dir() else motion_file.parent
            bounds = []
            for key in keys:
                source = command.motion_lib._motion_data_load[key]
                if "path" in source:
                    name = Path(source["path"]).stem
                else:
                    name = motion_file.stem if motion_file.is_file() else str(key)
                path = robot_dir.parent / "meta" / f"{name}.pkl"
                meta = joblib.load(path) if path.is_file() else {}
                boundary = float("inf")
                if "place_carry_source_frame" in meta:
                    frame = meta["place_carry_source_frame"]
                    total = meta.get("total_frames")
                    if (isinstance(frame, bool) or isinstance(total, bool)
                            or not isinstance(frame, numbers.Integral)
                            or not isinstance(total, numbers.Integral)
                            or not 0 <= frame < total):
                        raise ValueError(f"{path}: invalid carry frame/total_frames")
                    boundary = math.ceil(frame * motion_source_step_ratio(command, key))
                bounds.append(boundary)
            self._carry_steps = torch.tensor(bounds, device=device, dtype=torch.float64)
            self._keys = keys
        absolute_steps = command.motion_start_time_steps + command.time_steps
        return absolute_steps < self._carry_steps[command.motion_ids]


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
                    step_ratio = None
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
                    fade_out_bound = (-1, -1)
                    if ("residual_fade_out_start_frame" in meta
                            or "residual_fade_out_end_frame" in meta):
                        fade_out_start = meta.get("residual_fade_out_start_frame")
                        fade_out_end = meta.get("residual_fade_out_end_frame")
                        total = meta.get("total_frames")
                        if not (
                            all(isinstance(value, numbers.Integral) and not isinstance(value, bool)
                                for value in (fade_out_start, fade_out_end, total))
                            and 0 <= fade_out_start < fade_out_end < total
                        ):
                            raise ValueError(
                                f"{path}: residual fade-out frames must be paired integers with "
                                f"0 <= start < end < total_frames; got "
                                f"{fade_out_start!r}, {fade_out_end!r}, {total!r}"
                            )
                        if step_ratio is None:
                            step_ratio = motion_source_step_ratio(command, key)
                        fade_out_bound = (
                            int(fade_out_start) * step_ratio, int(fade_out_end) * step_ratio
                        )
                        logging.getLogger(__name__).info(
                            "Residual fade-out %s: frames %s -> %s", key, fade_out_start, fade_out_end
                        )
                    self._bounds[path] = (*bound, *fade_out_bound)
                bounds.append(self._bounds[path])
            self._frame_bounds = torch.tensor(bounds, device=device)
            self._keys = keys

        start, end, fade_out_start, fade_out_end = self._frame_bounds[command.motion_ids].unbind(-1)
        frame = command.motion_start_time_steps + command.time_steps
        duration = torch.where(start >= 0, end - start, torch.ones_like(start))
        weight = ((frame - start).float() / duration).clamp(0, 1)
        weight = torch.where(start >= 0, weight, torch.ones_like(weight))
        fade_out_duration = torch.where(
            fade_out_start >= 0, fade_out_end - fade_out_start, torch.ones_like(fade_out_start)
        )
        fade_out_weight = 1 - ((frame - fade_out_start).float() / fade_out_duration).clamp(0, 1)
        fade_out_weight = torch.where(
            fade_out_start >= 0, fade_out_weight, torch.ones_like(fade_out_weight)
        )
        return weight * fade_out_weight

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
