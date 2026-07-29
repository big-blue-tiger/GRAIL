from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int = 0,
    num_training_steps: int | None = None,
    last_epoch: int = -1,
):
    """Create the small set of learning-rate schedules used by Sugar-IL."""

    name = name.lower()
    warmup_steps = max(0, int(num_warmup_steps))

    if name == "constant":
        lr_lambda = lambda _: 1.0
    elif name == "constant_with_warmup":
        lr_lambda = lambda step: min(1.0, step / max(1, warmup_steps))
    elif name == "cosine":
        if num_training_steps is None:
            raise ValueError("cosine requires num_training_steps")
        training_steps = int(num_training_steps)
        if training_steps <= warmup_steps:
            raise ValueError("num_training_steps must exceed num_warmup_steps")

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / (training_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    else:
        raise ValueError(f"Unsupported lr_scheduler: {name}")

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
