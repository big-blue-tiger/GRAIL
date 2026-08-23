"""Runtime utilities for stateless action-chunk policy execution."""

from __future__ import annotations

import torch


def get_action_chunk_config(policy_or_config):
    """Return a plain action-chunk config mapping, defaulting to disabled."""
    config = getattr(policy_or_config, "algo_config", policy_or_config)
    if config is None:
        return {}
    chunk_config = config.get("action_chunk", {})
    return chunk_config or {}


def action_chunk_enabled(policy_or_config):
    return bool(get_action_chunk_config(policy_or_config).get("enabled", False))


def get_motion_metadata(env):
    """Return current `(motion_id, absolute_frame)` tensors when available."""
    command = getattr(env, "motion_command", None)
    if command is None:
        return None
    return (
        command.motion_ids.detach().clone(),
        (command.motion_start_time_steps + command.time_steps).detach().clone(),
    )


def motion_discontinuity(current, previous):
    if current is None or previous is None:
        return None
    motion_id, frame = current
    previous_motion, previous_frame = previous
    return (motion_id != previous_motion) | (frame != previous_frame + 1)


class ActionChunkExecutor:
    """Per-environment action cache and cursor with subset replanning.

    The executor intentionally stores no observations.  Callers must attach the
    current environment observation to every single-step action sent to the
    environment wrapper.
    """

    def __init__(
        self,
        policy,
        num_envs,
        horizon,
        execute_steps,
        action_dim,
        device,
    ):
        self.policy = policy
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.execute_steps = int(execute_steps)
        self.action_dim = int(action_dim)
        if not 0 < self.execute_steps <= self.horizon:
            raise ValueError(
                f"execute_steps must be in [1,{self.horizon}], got {self.execute_steps}"
            )
        self.device = torch.device(device)
        self.cache = torch.zeros(
            self.num_envs,
            self.horizon,
            self.action_dim,
            device=self.device,
        )
        self.cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.model_calls = 0
        self.replanned_envs = 0

    def reset(self, env_ids=None):
        if env_ids is None:
            self.valid.zero_()
            self.cursor.zero_()
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.valid[env_ids] = False
        self.cursor[env_ids] = 0

    def _subset_observations(self, obs_dict, env_ids):
        subset = {}
        for key, value in obs_dict.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == self.num_envs:
                subset[key] = value.index_select(0, env_ids)
            else:
                subset[key] = value
        return subset

    @torch.no_grad()
    def act(
        self,
        obs_dict,
        reset_mask=None,
        discontinuity_mask=None,
        active_mask=None,
    ):
        """Return one `[num_envs, action_dim]` frame and advance active cursors."""
        if active_mask is None:
            active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            active_mask = active_mask.to(device=self.device, dtype=torch.bool)
        invalidate = torch.zeros_like(active_mask)
        if reset_mask is not None:
            invalidate |= reset_mask.to(device=self.device, dtype=torch.bool)
        if discontinuity_mask is not None:
            invalidate |= discontinuity_mask.to(device=self.device, dtype=torch.bool)
        self.valid[invalidate] = False
        self.cursor[invalidate] = 0

        need_plan = active_mask & (~self.valid | (self.cursor >= self.execute_steps))
        env_ids = need_plan.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            chunk = self.policy.predict_action_chunk(
                self._subset_observations(obs_dict, env_ids)
            )
            expected = (env_ids.numel(), self.horizon, self.action_dim)
            if tuple(chunk.shape) != expected:
                raise RuntimeError(
                    f"predict_action_chunk returned {tuple(chunk.shape)}, expected {expected}"
                )
            self.cache.index_copy_(0, env_ids, chunk.to(self.cache))
            self.cursor[env_ids] = 0
            self.valid[env_ids] = True
            self.model_calls += 1
            self.replanned_envs += int(env_ids.numel())

        output = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
            dtype=self.cache.dtype,
        )
        active_ids = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if active_ids.numel() > 0:
            if not self.valid[active_ids].all():
                raise RuntimeError("Active action-chunk environment has no valid cached plan")
            output[active_ids] = self.cache[
                active_ids, self.cursor[active_ids]
            ]
            self.cursor[active_ids] += 1
        return output
