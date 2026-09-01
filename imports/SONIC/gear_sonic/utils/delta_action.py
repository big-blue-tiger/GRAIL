"""Shared delta meta-action accumulation helpers."""

from __future__ import annotations

import torch


def accumulate_delta_meta_action(
    previous_absolute_action: torch.Tensor,
    delta_action: torch.Tensor,
    *,
    latent_dim: int = 64,
    hand_min: float = -1.0,
    hand_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate a delta command and clamp only its hand primitive slice.

    Returns the projected absolute command and its effective change relative
    to ``previous_absolute_action``.  The effective delta differs from the raw
    model output only when a hand primitive reaches a configured bound.
    """
    if previous_absolute_action.shape != delta_action.shape:
        raise ValueError(
            "previous absolute action and delta action must have identical shapes, "
            f"got {tuple(previous_absolute_action.shape)} and {tuple(delta_action.shape)}"
        )
    if previous_absolute_action.ndim < 1:
        raise ValueError("delta meta actions must have at least one dimension")
    action_dim = previous_absolute_action.shape[-1]
    if not 0 < latent_dim < action_dim:
        raise ValueError(
            f"latent_dim must be in [1, action_dim - 1], got {latent_dim} for {action_dim}D"
        )
    if hand_min >= hand_max:
        raise ValueError(
            f"hand_min must be smaller than hand_max, got {hand_min} >= {hand_max}"
        )

    raw_absolute = previous_absolute_action + delta_action
    absolute_action = torch.cat(
        [
            raw_absolute[..., :latent_dim],
            raw_absolute[..., latent_dim:].clamp(min=hand_min, max=hand_max),
        ],
        dim=-1,
    )
    return absolute_action, absolute_action - previous_absolute_action
