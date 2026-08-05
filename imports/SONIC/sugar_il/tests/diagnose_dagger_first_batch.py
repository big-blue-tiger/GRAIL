#!/usr/bin/env python3
"""Isolate the DAgger generator's first two CUDA training micro-batches."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import time

from accelerate import Accelerator
import dill
import hydra
import torch


OBSERVATION_DIMS = {
    "object_bps": 10,
    "table_geometry": 4,
    "object_pos_b": 3,
    "object_ori_b_6d": 6,
    "hand_object_transform_6d": 9,
    "hand_object_contact_force_magnitude": 8,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "joint_pos": 43,
    "joint_vel": 43,
}


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def report(stage: str, started: float) -> float:
    synchronize()
    now = time.monotonic()
    allocated = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    reserved = torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0
    print(
        f"[isolate] {stage}: {now - started:.3f}s, "
        f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB",
        flush=True,
    )
    return now


def make_batch(batch_size: int, device: torch.device) -> dict:
    return {
        "obs": {
            key: torch.randn(batch_size, 1, dim, device=device)
            for key, dim in OBSERVATION_DIMS.items()
        },
        "action": {
            "latent": torch.randn(batch_size, 40, 64, device=device),
            "hand_primitive": torch.randint(
                0, 2, (batch_size, 40, 2), device=device
            ).float(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mixed-precision", default="fp16")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    started = time.monotonic()
    payload = torch.load(
        checkpoint, map_location="cpu", pickle_module=dill, weights_only=False
    )
    mark = report("checkpoint loaded", started)

    cfg = payload["cfg"]
    states = payload["state_dicts"]
    model = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(states["model"])
    optimizer = model.get_optimizer(**cfg.optimizer)
    optimizer.load_state_dict(states["optimizer"])
    mark = report("model and optimizer restored on CPU", mark)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()
    optimizer.zero_grad()
    mark = report("model and optimizer prepared", mark)

    for batch_idx in range(2):
        batch = make_batch(args.batch_size, accelerator.device)
        mark = report(f"batch {batch_idx} materialized", mark)
        should_step = batch_idx == 1
        context = nullcontext() if should_step else accelerator.no_sync(model)
        with context:
            losses = model(batch, training=True, normalized=True)
            mark = report(f"batch {batch_idx} forward", mark)
            accelerator.backward(losses["loss"] / 2)
            mark = report(f"batch {batch_idx} backward", mark)
        if should_step:
            accelerator.clip_grad_norm_(model.parameters(), 0.5)
            mark = report("gradient clipping", mark)
            optimizer.step()
            mark = report("optimizer step", mark)
            optimizer.zero_grad()
        print(
            f"[isolate] batch {batch_idx} complete: loss={losses['loss'].item():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
