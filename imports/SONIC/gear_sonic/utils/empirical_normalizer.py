"""Cumulative latent statistics with explicit updates at PPO iteration boundaries."""

import math

import torch
from torch import nn
import torch.distributed as dist


class EmpiricalNormalizer(nn.Module):
    """Per-feature population moments; forward/inverse never update statistics.

    Uses the Instinct-RL empirical merge formula and ``std + eps`` in both
    directions. Floating-point round trips are approximate, not bitwise exact.
    """

    def __init__(self, size, eps=1e-2, until=None):
        super().__init__()
        if size <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Normalizer size and eps must be positive and finite")
        if until is not None and (not isinstance(until, int) or until < 0):
            raise ValueError("Normalizer until must be a nonnegative sample count or None")
        self.eps = float(eps)
        self.until = until
        self.register_buffer("_mean", torch.zeros(size))
        self.register_buffer("_var", torch.ones(size))
        self.register_buffer("_std", torch.ones(size))
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))

    def _transform(self, value, inverse):
        # Do the affine transform in at least FP32 even under mixed precision.
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        value = value.to(dtype=dtype)
        mean = self._mean.to(dtype=dtype)
        scale = self._std.to(dtype=dtype) + self.eps
        return value * scale + mean if inverse else (value - mean) / scale

    def normalize(self, value):
        return self._transform(value, inverse=False)

    def inverse(self, value):
        return self._transform(value, inverse=True)

    def forward(self, value):
        return self.normalize(value)

    @torch.no_grad()
    def update_from_moments(self, count, mean, variance):
        """Merge one new population exactly once; return accepted sample count."""
        count = int(count)
        if count < 0:
            raise ValueError("Sample count must be nonnegative")
        if count == 0 or (self.until is not None and self.count.item() >= self.until):
            return 0
        mean = mean.to(device=self._mean.device, dtype=torch.float64)
        variance = variance.to(device=self._var.device, dtype=torch.float64)
        if mean.shape != self._mean.shape or variance.shape != self._var.shape:
            raise ValueError("Latent moment shape does not match normalizer")
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise FloatingPointError("Non-finite latent normalization moments")
        total = self.count.item() + count
        rate = count / total
        old_mean = self._mean.double()
        old_var = self._var.double()
        delta = mean - old_mean
        new_mean = old_mean + rate * delta
        new_var = old_var + rate * (variance - old_var + delta * (mean - new_mean))
        self._mean.copy_(new_mean)
        self._var.copy_(new_var.clamp_min(0))
        self._std.copy_(self._var.sqrt())
        self.count.fill_(total)
        return count

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Trainer's legacy loader uses strict=False. Missing normalization state
        # must nevertheless fail rather than silently reinterpret learned actions.
        absent = [prefix + key for key in ("_mean", "_var", "_std", "count") if prefix + key not in state_dict]
        if absent:
            error_msgs.append("Missing empirical latent normalization state: " + ", ".join(absent))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class RolloutLatentMoments:
    """Trainer-owned pending moments, outside DDP's model buffer broadcasts."""

    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.snapshot = {key: value.clone() for key, value in normalizer.named_buffers()}
        self.total = torch.zeros_like(normalizer._mean, dtype=torch.float64)
        self.square_total = torch.zeros_like(self.total)
        self.count = torch.zeros_like(normalizer.count)
        self.committed = False

    @torch.no_grad()
    def collect(self, latent):
        if self.committed:
            raise RuntimeError("Cannot collect moments after the rollout was committed")
        if latent.shape[-1] != self.total.numel():
            raise ValueError("Unexpected latent feature dimension")
        flat = latent.detach().reshape(-1, self.total.numel()).double()
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Non-finite decoder latent target; statistics were not updated")
        self.total.add_(flat.sum(dim=0))
        self.square_total.add_(flat.square().sum(dim=0))
        self.count.add_(flat.shape[0])

    @torch.no_grad()
    def commit(self):
        if self.committed:
            raise RuntimeError("Rollout latent moments have already been committed")
        for key, value in self.normalizer.named_buffers():
            if not torch.equal(value, self.snapshot[key]):
                raise RuntimeError("Latent normalizer changed during rollout/PPO replay")
        if dist.is_available() and dist.is_initialized():
            for value in (self.count, self.total, self.square_total):
                dist.all_reduce(value)
        self.committed = True
        count = self.count.item()
        if count == 0:
            return 0
        mean = self.total / count
        variance = (self.square_total / count - mean.square()).clamp_min(0)
        return self.normalizer.update_from_moments(count, mean, variance)
