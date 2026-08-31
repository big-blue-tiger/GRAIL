"""Helpers for keeping policy-action observations episode-local."""

from __future__ import annotations

import torch


def zero_actions_at_episode_start(
    actions: torch.Tensor,
    episode_length_buf: torch.Tensor | None,
) -> torch.Tensor:
    """Return actions with rows at the first observation of an episode zeroed.

    Isaac Lab resets observation-history buffers before it computes the first
    observation of a new episode. Wrapper-owned action buffers are cleared
    later, however, so their previous-episode values must be masked at the
    observation source. ``torch.where`` avoids mutating the wrapper buffer and
    avoids a device synchronization from branching on ``reset_mask.any()``.
    """
    if episode_length_buf is None:
        return actions
    if actions.shape[0] != episode_length_buf.shape[0]:
        raise ValueError(
            "actions and episode_length_buf must have the same batch size, "
            f"got {actions.shape[0]} and {episode_length_buf.shape[0]}"
        )

    reset_mask = episode_length_buf == 0
    while reset_mask.ndim < actions.ndim:
        reset_mask = reset_mask.unsqueeze(-1)
    zero = torch.zeros((), dtype=actions.dtype, device=actions.device)
    return torch.where(reset_mask, zero, actions)
