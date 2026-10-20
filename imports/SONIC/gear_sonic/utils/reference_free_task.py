"""Tensor-only playback and physical outcome tracking for offline carry plans."""

import math

import torch

from gear_sonic.utils.reference_free_plan import PLAN_FPS
from gear_sonic.utils.sparse_command import SparseCommandState


class OfflineCarryTask:
    """A fixed clock started once after pickup; no position-based replanning."""

    def __init__(self, num_envs, device, step_dt, settle_s, success_threshold, force_threshold=0.1):
        self.device = device
        self.step_dt = float(step_dt)
        self.settle_steps = math.ceil(settle_s / self.step_dt - 1e-9)
        self.success_threshold = success_threshold
        self.contacts = SparseCommandState(num_envs, device, force_threshold=force_threshold)
        self.steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.start_step = torch.full_like(self.steps, -1)
        self.release_frame = torch.ones_like(self.steps)
        self.sparse = torch.zeros(num_envs, 1, 2, device=device)
        self.drop = torch.ones(num_envs, 1, 1, device=device)
        self.success = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.event_steps = {
            name: torch.full_like(self.steps, -1)
            for name in ("picked_up", "arrived", "drop_requested", "released", "dropped")
        }

    def reset(self, env_ids, plans):
        """Replace only reset environments; other playback clocks keep running."""
        capacity = max(len(plan.sparse) for plan in plans)
        if capacity > self.sparse.shape[1]:
            sparse = self.sparse.new_zeros(len(self.steps), capacity, 2)
            drop = self.drop.new_ones(len(self.steps), capacity, 1)
            sparse[:, :self.sparse.shape[1]] = self.sparse
            drop[:, :self.drop.shape[1]] = self.drop
            self.sparse, self.drop = sparse, drop
        for env_id, plan in zip(env_ids.tolist(), plans):
            length = len(plan.sparse)
            self.sparse[env_id] = 0
            self.drop[env_id] = 1
            self.sparse[env_id, :length] = torch.as_tensor(plan.sparse, device=self.device)
            self.drop[env_id, :length] = torch.as_tensor(plan.drop, device=self.device)
            self.release_frame[env_id] = plan.release_start_frame
        self.contacts.reset(env_ids)
        self.steps[env_ids] = 0
        self.start_step[env_ids] = -1
        self.success[env_ids] = False
        for values in self.event_steps.values():
            values[env_ids] = -1

    def has_playback_started(self):
        """Enter the carry-plan phase at its first playback step, until reset.

        Contact loss, zero sparse commands and release do not rewind this phase.
        """
        return (self.start_step >= 0) & (self.steps >= self.start_step)

    def read(self):
        """Pure zero-order hold lookup, including a latched final drop button."""
        elapsed = self.steps - self.start_step
        playing = self.has_playback_started()
        frames = torch.floor(elapsed * (self.step_dt * PLAN_FPS) + 1e-6).long()
        frames = torch.minimum(frames.clamp_min(0), self.release_frame)
        ids = torch.arange(len(self.steps), device=self.device)
        sparse = self.sparse[ids, frames]
        drop = self.drop[ids, frames]
        enabled = playing & ~self.contacts.dropped
        return torch.where(enabled[:, None], sparse, 0.0), torch.where(playing[:, None], drop, 0.0)

    def update(self, steps, hand_force, table_force, object_position, release_position):
        """Advance once before termination/reset, using the previously observed drop."""
        if torch.equal(steps, self.steps):
            return
        _, applied_drop = self.read()
        self.contacts.update(hand_force, table_force, applied_drop)
        new_pickup = self.contacts.picked_up & (self.start_step < 0)
        self.start_step[new_pickup] = steps[new_pickup] + self.settle_steps
        in_goal = torch.linalg.vector_norm(object_position - release_position, dim=-1) < self.success_threshold
        self.contacts.release_goal_reached |= (
            in_goal & self.contacts.carrying & ~self.contacts.dropped
            & (applied_drop[:, 0] < 0.5)
        )
        physically_released = (
            self.contacts.released & ~self.contacts.dropped
            & (applied_drop[:, 0] >= 0.5) & (hand_force < self.contacts.force_threshold)
        )
        self.success |= self.contacts.release_goal_reached & physically_released & in_goal
        self.steps.copy_(steps)
        _, next_drop = self.read()
        events = {
            "picked_up": self.contacts.picked_up,
            "arrived": self.contacts.release_goal_reached,
            "drop_requested": next_drop[:, 0] >= 0.5,
            "released": physically_released,
            "dropped": self.contacts.dropped,
        }
        for name, happened in events.items():
            values = self.event_steps[name]
            first = happened & (values < 0)
            values[first] = steps[first]

    def result(self, env_id):
        """Small JSON-ready outcome, independent of whether the clock finished."""
        result = {"success": bool(self.success[env_id].item())}
        for name, values in self.event_steps.items():
            step = int(values[env_id].item())
            result[name] = step >= 0
            result[f"{name}_seconds"] = step * self.step_dt if step >= 0 else None
        start = int(self.start_step[env_id].item())
        result["playback_start_seconds"] = start * self.step_dt if start >= 0 else None
        return result
