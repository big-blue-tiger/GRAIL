# Partially based on NVLabs ProtoMotions (Apache 2.0):
# https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
"""Data-management utilities for on-policy PPO rollouts.

Provides :class:`RolloutStorage` for on-policy PPO and
:class:`ActionChunkDaggerBuffer` for delayed, continuous Teacher chunks.
"""

import torch
from torch import Tensor, nn


class ActionChunkDaggerBuffer:
    """GPU time-major ring for delayed continuous Teacher action chunks."""

    def __init__(
        self,
        num_envs,
        capacity,
        horizon,
        rollout_steps,
        action_dim,
        observation_shapes,
        device="cpu",
    ):
        self.num_envs = int(num_envs)
        self.capacity = int(capacity)
        self.horizon = int(horizon)
        self.rollout_steps = int(rollout_steps)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        if self.capacity <= self.horizon + self.rollout_steps:
            raise ValueError(
                "ActionChunkDaggerBuffer capacity must be strictly greater than "
                f"horizon + rollout_steps ({self.horizon + self.rollout_steps})"
            )
        self.observations = {
            key: torch.zeros(
                self.capacity,
                self.num_envs,
                *tuple(shape),
                device=self.device,
            )
            for key, shape in observation_shapes.items()
        }
        self.teacher_target = torch.zeros(
            self.capacity,
            self.num_envs,
            self.action_dim,
            device=self.device,
        )
        metadata_shape = (self.capacity, self.num_envs)
        self.episode_uid = torch.zeros(metadata_shape, dtype=torch.long, device=self.device)
        self.motion_id = torch.zeros(metadata_shape, dtype=torch.long, device=self.device)
        self.reference_frame = torch.zeros(metadata_shape, dtype=torch.long, device=self.device)
        self.transition_id = torch.full(
            metadata_shape, -1, dtype=torch.long, device=self.device
        )
        self.written_valid = torch.zeros(
            metadata_shape, dtype=torch.bool, device=self.device
        )
        self.frame_finite = torch.zeros(
            metadata_shape, dtype=torch.bool, device=self.device
        )
        self.done_after_action = torch.zeros(
            metadata_shape, dtype=torch.bool, device=self.device
        )
        self.read_pos = 0
        self.write_pos = 0
        self.size = 0

    @property
    def ready(self):
        return self.size >= self.horizon + self.rollout_steps - 1

    def clear(self):
        self.read_pos = 0
        self.write_pos = 0
        self.size = 0
        self.written_valid.zero_()
        self.transition_id.fill_(-1)

    def append(
        self,
        observations,
        teacher_target,
        episode_uid,
        motion_id,
        reference_frame,
        transition_id,
        frame_finite,
        done_after_action,
    ):
        """Append a `[steps, envs, ...]` block without reallocating storage."""
        steps = int(teacher_target.shape[0])
        expected_prefix = (steps, self.num_envs)
        if teacher_target.shape != (*expected_prefix, self.action_dim):
            raise ValueError(
                f"teacher_target shape {tuple(teacher_target.shape)} does not match "
                f"{(*expected_prefix, self.action_dim)}"
            )
        if self.size + steps > self.capacity:
            raise RuntimeError(
                f"Action chunk ring overflow: size={self.size}, append={steps}, "
                f"capacity={self.capacity}"
            )
        indices = (
            torch.arange(steps, device=self.device, dtype=torch.long) + self.write_pos
        ) % self.capacity
        for key, storage in self.observations.items():
            if key not in observations:
                raise KeyError(f"Missing action chunk observation '{key}'")
            source = observations[key]
            expected_shape = (steps, self.num_envs, *storage.shape[2:])
            if tuple(source.shape) != expected_shape:
                raise ValueError(
                    f"Action chunk observation '{key}' has shape {tuple(source.shape)}, "
                    f"expected {expected_shape}"
                )
            storage.index_copy_(0, indices, source.to(storage))
        self.teacher_target.index_copy_(0, indices, teacher_target.to(self.teacher_target))
        for storage, value in (
            (self.episode_uid, episode_uid),
            (self.motion_id, motion_id),
            (self.reference_frame, reference_frame),
            (self.transition_id, transition_id),
            (self.frame_finite, frame_finite),
            (self.done_after_action, done_after_action),
        ):
            if value.shape != expected_prefix:
                raise ValueError(
                    f"Metadata shape {tuple(value.shape)} does not match {expected_prefix}"
                )
            storage.index_copy_(0, indices, value.to(storage))
        self.written_valid[indices] = True
        self.write_pos = (self.write_pos + steps) % self.capacity
        self.size += steps

    def pop(self):
        """Emit the oldest rollout-sized anchor batch, or ``None`` while warming up."""
        if not self.ready:
            return None
        anchor_offsets = torch.arange(
            self.rollout_steps, device=self.device, dtype=torch.long
        )
        horizon_offsets = torch.arange(
            self.horizon, device=self.device, dtype=torch.long
        )
        anchor_indices = (self.read_pos + anchor_offsets) % self.capacity
        window_indices = (
            self.read_pos + anchor_offsets[:, None] + horizon_offsets[None, :]
        ) % self.capacity

        observations = {
            key: value[anchor_indices].transpose(0, 1).contiguous()
            for key, value in self.observations.items()
        }
        target = self.teacher_target[window_indices].permute(2, 0, 1, 3).contiguous()

        def gather_metadata(storage):
            return storage[window_indices].permute(2, 0, 1).contiguous()

        written = gather_metadata(self.written_valid)
        finite = gather_metadata(self.frame_finite)
        transition = gather_metadata(self.transition_id)
        episode = gather_metadata(self.episode_uid)
        motion = gather_metadata(self.motion_id)
        frame = gather_metadata(self.reference_frame)
        expected_transition = transition[..., :1] + horizon_offsets
        expected_frame = frame[..., :1] + horizon_offsets
        validity_checks = {
            "written": written.all(dim=-1),
            "finite": finite.all(dim=-1),
            "transition": (transition == expected_transition).all(dim=-1),
            "episode": (episode == episode[..., :1]).all(dim=-1),
            "motion": (motion == motion[..., :1]).all(dim=-1),
            "reference_frame": (frame == expected_frame).all(dim=-1),
        }
        valid = torch.ones_like(validity_checks["written"])
        for check in validity_checks.values():
            valid &= check

        # These slots are no longer required by any future anchor.
        self.written_valid[anchor_indices] = False
        self.transition_id[anchor_indices] = -1
        self.read_pos = (self.read_pos + self.rollout_steps) % self.capacity
        self.size -= self.rollout_steps
        return {
            "obs_dict": observations,
            "diffusion_target": target,
            "diffusion_target_valid": valid,
            "anchor_transition_id": transition[..., 0],
            "validity_checks": validity_checks,
        }


class RolloutStorage(nn.Module):
    """On-policy rollout buffer for PPO with dynamic key registration.

    Stores up to ``num_transitions_per_env`` steps for each of the
    ``num_envs`` parallel environments.  Keys (e.g. "obs", "actions",
    "rewards") are registered on demand via :meth:`register_key`, which
    allocates a zero-filled ``nn.Module`` buffer of the appropriate shape so
    that the storage follows ``.to(device)`` calls.

    Args:
        num_envs: Number of parallel simulation environments.
        num_transitions_per_env: Maximum rollout length (horizon).
        device: PyTorch device string for all allocated buffers.
    """

    def __init__(self, num_envs, num_transitions_per_env, device="cpu"):

        super().__init__()

        self.device = device

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        # self.saved_hidden_states_a = None
        # self.saved_hidden_states_c = None

        self.step = 0
        self.stored_keys = []

    def register_key(self, key: str, shape=(), dtype=torch.float):
        """Allocate and register a new storage buffer for the given key.

        The buffer has shape ``(num_transitions_per_env, num_envs, *shape)``
        and is registered as a non-persistent ``nn.Module`` buffer so it moves
        with ``.to(device)`` but is not saved in ``state_dict``.

        Args:
            key: Unique name for the data field (e.g. ``"obs"``, ``"rewards"``).
            shape: Per-transition, per-environment shape tuple.
            dtype: Tensor dtype for the buffer.

        Raises:
            AssertionError: If ``key`` is already registered or ``shape`` is
                not a list or tuple.
        """
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not hasattr(self, key), key
        assert isinstance(shape, list | tuple), f"shape must be a list or tuple, got {type(shape)}"
        buffer = torch.zeros(
            (self.num_transitions_per_env, self.num_envs) + shape, dtype=dtype, device=self.device
        )
        self.register_buffer(key, buffer, persistent=False)
        self.stored_keys.append(key)

    def increment_step(self):
        """Advance the write cursor by one transition step."""
        self.step += 1

    def update_key(self, key: str, data: Tensor):
        """Write ``data`` into the buffer at the current step for ``key``.

        Args:
            key: Previously registered buffer key.
            data: Tensor of shape ``(num_envs, *key_shape)``.  Must not
                require gradients.

        Raises:
            AssertionError: If ``data.requires_grad`` is True or the buffer
                is full (step >= num_transitions_per_env).
        """
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        assert self.step < self.num_transitions_per_env, "Rollout buffer overflow"
        getattr(self, key)[self.step].copy_(data)

    def batch_update_data(self, key: str, data: Tensor):
        """Overwrite the entire buffer for ``key`` with ``data``.

        Useful for writing pre-computed values (e.g. advantages, returns)
        after a full rollout has been collected.

        Args:
            key: Previously registered buffer key.
            data: Tensor of shape
                ``(num_transitions_per_env, num_envs, *key_shape)``.  Must
                not require gradients.
        """
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        getattr(self, key)[:] = data
        # self.store_dict[key] += self.total_sum()

    def _save_hidden_states(self, hidden_states):
        assert NotImplementedError
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device)
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device)
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        """Reset the write cursor to the start of the buffer."""
        self.step = 0

    def get_statistics(self):
        """Return buffer statistics (not implemented)."""
        raise NotImplementedError

    def query_key(self, key: str):
        """Return the full buffer tensor for ``key``.

        Args:
            key: Previously registered buffer key.

        Returns:
            Tensor of shape ``(num_transitions_per_env, num_envs, *key_shape)``.

        Raises:
            AssertionError: If ``key`` has not been registered.
        """
        assert hasattr(self, key), key
        return getattr(self, key)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Yield randomly shuffled mini-batches over all stored transitions.

        The full buffer (flattened across transitions and environments) is
        shuffled once per call and then sliced into ``num_mini_batches``
        equal-sized chunks, repeated ``num_epochs`` times.

        Args:
            num_mini_batches: Number of mini-batches per epoch.
            num_epochs: Number of times to iterate over the shuffled data.

        Yields:
            Dict mapping each registered key to a mini-batch tensor of shape
            ``(mini_batch_size, *key_shape)``.
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        _buffer_dict = {key: getattr(self, key)[:].flatten(0, 1) for key in self.stored_keys}

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                _batch_buffer_dict = {key: _buffer_dict[key][batch_idx] for key in self.stored_keys}
                yield _batch_buffer_dict
