"""Shared motion-metadata residual blending for teacher play and training."""

from pathlib import Path
import logging
import numbers

import joblib
import torch


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
                        source = command.motion_lib._motion_data_load[key]
                        if "path" in source:
                            source = next(iter(joblib.load(source["path"]).values()))
                        source_fps = float(source["fps"])
                        if not 0 < source_fps < float("inf"):
                            raise ValueError(f"{key}: invalid motion fps {source_fps}")
                        step_ratio = command.motion_lib._sim_fps / source_fps
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
