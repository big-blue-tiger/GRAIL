"""GPU-resident storage for the latest object-aware DAgger rollout."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch

from gear_sonic.trl.modules.data_utils import RolloutStorage


class FlowRolloutStorage(RolloutStorage):
    """Store one parallel rollout and expose contiguous flow targets."""

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        observation_shapes: Mapping[str, tuple[int, ...]],
        horizon: int = 40,
        device: str | torch.device = "cuda",
    ) -> None:
        if capacity <= horizon:
            raise ValueError(f"capacity must exceed horizon ({horizon})")
        super().__init__(num_envs, capacity, device=device)
        self.horizon = horizon
        self.observation_keys = tuple(observation_shapes)
        for key, shape in observation_shapes.items():
            self.register_key(key, shape=shape)
        self.register_key("teacher_latent", shape=(64,))
        self.register_key("teacher_hand", shape=(2,))
        self.register_key("valid", shape=(1,), dtype=torch.bool)
        self.register_key("dones", shape=(1,), dtype=torch.bool)
        self.register_key("time_outs", shape=(1,), dtype=torch.bool)
        self.register_key("episode_starts", shape=(1,), dtype=torch.bool)
        self.register_key("teacher_execution", shape=(1,), dtype=torch.bool)
        self.is_normalized = False

    def append(
        self,
        observation: Mapping[str, torch.Tensor],
        teacher_latent: torch.Tensor,
        teacher_hand: torch.Tensor,
        valid: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        episode_starts: torch.Tensor,
        teacher_execution: torch.Tensor,
    ) -> None:
        values = {
            **observation,
            "teacher_latent": teacher_latent,
            "teacher_hand": teacher_hand,
            "valid": valid[:, None],
            "dones": dones[:, None],
            "time_outs": time_outs[:, None],
            "episode_starts": episode_starts[:, None],
            "teacher_execution": teacher_execution[:, None],
        }
        for key, value in values.items():
            self.update_key(key, value.detach())
        self.increment_step()

    def clear(self) -> None:
        super().clear()
        self.is_normalized = False
        for key in ("valid", "dones", "time_outs", "episode_starts"):
            getattr(self, key).zero_()

    @torch.no_grad()
    def normalize_(self, normalizer) -> None:
        """Normalize each stored frame once before constructing windows.

        Action windows overlap by ``horizon`` frames, so normalizing after
        window construction repeats the same latent transformation up to 40
        times. The online storage is no longer used for simulation after a
        rollout, making an in-place pass both safe and substantially cheaper.
        """
        if self.is_normalized:
            return
        for key in self.observation_keys:
            if key == "last_hand_primitive":
                continue
            values = getattr(self, key)[: self.step]
            values.copy_(normalizer[key].normalize(values))
        latents = self.teacher_latent[: self.step]
        latents.copy_(normalizer["latent"].normalize(latents))
        self.is_normalized = True

    def window_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return valid ``(start_step, env_id)`` pairs for 40-step targets."""
        num_starts = self.step - self.horizon + 1
        if num_starts <= 1:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty

        # Match GeneratorDataset: reset frame zero is not a training input.
        starts = torch.arange(1, num_starts, device=self.device)
        offsets = torch.arange(self.horizon, device=self.device)
        steps = starts[:, None] + offsets[None]
        valid = self.valid[: self.step, :, 0]
        window_valid = valid[steps].all(dim=1)
        start_grid, env_grid = torch.meshgrid(
            starts,
            torch.arange(self.num_envs, device=self.device),
            indexing="ij",
        )
        return start_grid[window_valid], env_grid[window_valid]

    def make_batch(
        self, start_steps: torch.Tensor, env_ids: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        target_steps = start_steps[:, None] + torch.arange(
            self.horizon, device=self.device
        )
        target_envs = env_ids[:, None].expand_as(target_steps)
        observation = {
            key: getattr(self, key)[start_steps, env_ids] for key in self.observation_keys
        }
        return {
            "obs": observation,
            "action": {
                "latent": self.teacher_latent[target_steps, target_envs],
                "hand_primitive": self.teacher_hand[target_steps, target_envs],
            },
        }

    def batches(self, batch_size: int, epochs: int) -> Iterator[dict]:
        start_steps, env_ids = self.window_indices()
        for _ in range(epochs):
            order = torch.randperm(len(start_steps), device=self.device)
            for offset in range(0, len(order), batch_size):
                selected = order[offset : offset + batch_size]
                yield self.make_batch(start_steps[selected], env_ids[selected])

    def window_counts(self) -> tuple[int, int, int]:
        starts, env_ids = self.window_indices()
        teacher = self.teacher_execution[starts, env_ids, 0]
        return len(starts), int(teacher.sum().item()), int((~teacher).sum().item())
