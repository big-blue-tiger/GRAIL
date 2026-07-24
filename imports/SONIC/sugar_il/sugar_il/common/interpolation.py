from __future__ import annotations

import torch
import torch.nn.functional as F


INTERPOLATION_RATE = 3


def interpolate_action_frames(
    value: torch.Tensor,
    rate: int = INTERPOLATION_RATE,
) -> torch.Tensor:
    """Linearly resample [B, T, D] actions, retaining every input waypoint."""
    if value.ndim != 3:
        raise ValueError(f"Expected action tensor [B,T,D], got {tuple(value.shape)}")
    if rate < 1:
        raise ValueError(f"Interpolation rate must be positive, got {rate}")
    if rate == 1 or value.shape[1] <= 1:
        return value
    output_frames = (value.shape[1] - 1) * rate + 1
    return F.interpolate(
        value.transpose(1, 2),
        size=output_frames,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
