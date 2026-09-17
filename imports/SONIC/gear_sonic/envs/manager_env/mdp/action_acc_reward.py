"""Action smoothness costs evaluated once per control step."""

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg


class ActionAccL2(ManagerTermBase):
    """Squared target joint-position second difference, summed over joint_pos dims.

    Compute a[t] - 2*a[t-1] + a[t-2] without control-period normalization.
    Use processed JointPositionAction targets to include per-joint scales,
    offsets and clipping. For revolute joints the raw cost has units of rad^2,
    rather than the rad^2/s^4 of the actual joint_acc_l2 cost.
    The first two calls after reset collect history without charging a cost.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._joint_pos = env.action_manager.get_term("joint_pos")
        self._previous = torch.zeros_like(self._joint_pos.processed_actions)
        self._previous_previous = torch.zeros_like(self._previous)
        self._history_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._previous[ids] = 0
        self._previous_previous[ids] = 0
        self._history_count[ids] = 0

    def __call__(self, env):
        action = self._joint_pos.processed_actions
        acceleration = (
            action - 2 * self._previous + self._previous_previous
        )
        cost = acceleration.square().sum(dim=-1)
        cost = torch.where(self._history_count >= 2, cost, 0.0)
        self._previous_previous.copy_(self._previous)
        self._previous.copy_(action)
        self._history_count.add_(1).clamp_(max=2)
        return cost
