"""Physical carry contact state shared by tracking and reference-free play."""

import torch


class SparseCommandState:
    """Track pickup/release once per control step; observation reads are pure."""

    def __init__(self, num_envs, device, *, force_threshold=0.1, stable_steps=3):
        self.force_threshold = force_threshold
        self.stable_steps = stable_steps
        self.pickup_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.release_count = torch.zeros_like(self.pickup_count)
        self.lost_contact_count = torch.zeros_like(self.pickup_count)
        self.picked_up = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.released = torch.zeros_like(self.picked_up)
        self.dropped = torch.zeros_like(self.picked_up)
        self.carrying = torch.zeros_like(self.picked_up)
        self.release_goal_reached = torch.zeros_like(self.picked_up)

    def reset(self, env_ids):
        for value in (self.pickup_count, self.release_count, self.lost_contact_count,
                      self.picked_up, self.released, self.dropped, self.carrying,
                      self.release_goal_reached):
            value[env_ids] = 0

    def update(self, hand_force, table_force, drop_button):
        contact = hand_force >= self.force_threshold
        drop = drop_button.reshape(-1) >= 0.5
        lifted = contact & (table_force < self.force_threshold) & ~drop
        self.pickup_count = torch.where(lifted, self.pickup_count + 1, 0)
        self.picked_up |= self.pickup_count >= self.stable_steps
        releasing = self.picked_up & drop & ~contact
        self.release_count = torch.where(releasing, self.release_count + 1, 0)
        self.released |= self.release_count >= self.stable_steps
        lost = self.picked_up & ~drop & ~contact
        self.lost_contact_count = torch.where(lost, self.lost_contact_count + 1, 0)
        self.dropped |= self.lost_contact_count >= self.stable_steps
        self.carrying = self.picked_up & contact & ~drop & ~self.released

    def command(self, x_yaw, drop_button):
        enabled = self.carrying & (drop_button.reshape(-1) < 0.5)
        return torch.where(enabled[:, None], x_yaw, 0.0)
