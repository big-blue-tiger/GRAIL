from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.trl.modules.action_chunk import ActionChunkExecutor
from gear_sonic.trl.modules.actor_critic_modules import Actor
from gear_sonic.trl.modules.data_utils import ActionChunkDaggerBuffer
from gear_sonic.trl.modules.diffusion_policy_modules import (
    EncoderVectorSingleStepTransformerFlowPolicy,
    EncoderVectorTransformerFlowPolicy,
)


def _make_model(
    num_layers=14,
    num_inference_steps=1,
    attention_dropout=0.1,
    gradient_checkpointing=False,
):
    config = {
        "proprio_input_dim": 805,
        "proprio_feature_dim": 512,
        "proprio_hidden_dims": [1024],
        "privileged_input_dim": 48,
        "privileged_feature_dim": 512,
        "privileged_hidden_dims": [512],
        "cond_dim": 1024,
        "output_dim": ["robot_action_dim"],
        "action_horizon": 40,
        "transformer_hidden_dim": 512,
        "transformer_num_layers": num_layers,
        "transformer_num_heads": 8,
        "transformer_attention_dropout": attention_dropout,
        "transformer_gradient_checkpointing": gradient_checkpointing,
        "diffusion_objective": "flow_matching",
        "num_inference_steps": num_inference_steps,
    }
    env_config = SimpleNamespace(
        robot=SimpleNamespace(
            actions_dim=66,
            algo_obs_dim_dict={"proprio_obs": (5, 161), "privileged_obs": 48},
        )
    )
    return EncoderVectorTransformerFlowPolicy(
        obs_dim_dict={"proprio_obs": (5, 161), "privileged_obs": 48},
        module_config_dict=config,
        module_dim_dict={},
        env_config=env_config,
        process_output_dim=True,
    )


def test_transformer_parameter_count_and_chunk_shapes():
    model = _make_model(num_layers=14)
    assert sum(parameter.numel() for parameter in model.parameters()) == 38_919_746
    assert model.load_state_dict(model.state_dict(), strict=True).missing_keys == []

    small_model = _make_model(num_layers=1)
    observations = {
        "proprio_obs": torch.randn(2, 5, 161),
        "privileged_obs": torch.randn(2, 48),
    }
    target = torch.randn(2, 40, 66)
    captured_timestep_shapes = []
    original_predict_velocity = small_model._predict_velocity

    def capture_timestep(noisy_action, timestep, condition_tokens):
        captured_timestep_shapes.append(tuple(timestep.shape))
        return original_predict_velocity(noisy_action, timestep, condition_tokens)

    small_model._predict_velocity = capture_timestep
    result = small_model(
        observations,
        compute_aux_loss=True,
        diffusion_target=target,
        diffusion_target_valid=torch.tensor([True, False]),
    )
    assert result["action_mean"].shape == (2, 66)
    assert result["aux_losses"]["diffusion_flow"].isfinite()
    assert captured_timestep_shapes[0] == (2,)
    result["aux_losses"]["diffusion_flow"].backward()
    assert all(
        parameter.grad is not None
        for parameter in small_model.parameters()
        if parameter.requires_grad
    )
    with torch.no_grad():
        assert small_model.sample_action_chunk(observations).shape == (2, 40, 66)


def test_four_step_euler_constant_velocity_and_time_buckets():
    model = _make_model(num_layers=1, num_inference_steps=4)
    observations = {
        "proprio_obs": torch.zeros(2, 5, 161),
        "privileged_obs": torch.zeros(2, 48),
    }
    buckets = []

    def constant_velocity(noisy_action, timestep, _condition_tokens):
        buckets.append(timestep.detach().clone())
        return torch.ones_like(noisy_action)

    model._predict_velocity = constant_velocity
    torch.manual_seed(7)
    initial_noise = torch.randn(2, 40, 66)
    torch.manual_seed(7)
    output = model.sample_action_chunk(observations)
    assert torch.allclose(output, initial_noise + 1.0)
    assert [int(value[0]) for value in buckets] == [0, 250, 500, 750]


def test_transformer_flow_time_uses_inverted_beta_distribution():
    model = _make_model(num_layers=1)

    class _FixedBeta:
        @staticmethod
        def sample(shape):
            assert shape == (2,)
            return torch.tensor([0.2, 0.8])

    model.beta_dist = _FixedBeta()
    sampled_time = model._sample_flow_time(
        (2,), device=torch.device("cpu"), dtype=torch.float32
    )
    expected = (1.0 - torch.tensor([0.2, 0.8])) * model.noise_s
    assert torch.allclose(sampled_time, expected)


def test_transformer_gradient_checkpointing_preserves_dropout_rng_and_gradients():
    regular = _make_model(num_layers=1, attention_dropout=0.1)
    checkpointed = _make_model(
        num_layers=1,
        attention_dropout=0.1,
        gradient_checkpointing=True,
    )
    checkpointed.load_state_dict(regular.state_dict())
    regular.train()
    checkpointed.train()
    observations = {
        "proprio_obs": torch.randn(2, 5, 161),
        "privileged_obs": torch.randn(2, 48),
    }
    target = torch.randn(2, 40, 66)
    valid = torch.ones(2, dtype=torch.bool)

    def forward_backward(model):
        torch.manual_seed(1234)
        result = model(
            observations,
            compute_aux_loss=True,
            diffusion_target=target,
            diffusion_target_valid=valid,
            update_running_stats=False,
        )
        loss = result["aux_losses"]["diffusion_flow"]
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), gradients

    regular_loss, regular_gradients = forward_backward(regular)
    checkpointed_loss, checkpointed_gradients = forward_backward(checkpointed)

    assert torch.allclose(regular_loss, checkpointed_loss)
    assert regular_gradients.keys() == checkpointed_gradients.keys()
    for name in regular_gradients:
        assert torch.allclose(
            regular_gradients[name],
            checkpointed_gradients[name],
            atol=1.0e-5,
            rtol=1.0e-5,
        ), name


def test_single_step_transformer_uses_regular_dagger_tensor_shapes():
    config = {
        "proprio_input_dim": 805,
        "proprio_feature_dim": 512,
        "proprio_hidden_dims": [1024],
        "privileged_input_dim": 48,
        "privileged_feature_dim": 512,
        "privileged_hidden_dims": [512],
        "cond_dim": 1024,
        "output_dim": ["robot_action_dim"],
        "action_horizon": 1,
        "transformer_hidden_dim": 512,
        "transformer_num_layers": 1,
        "transformer_num_heads": 8,
        "diffusion_objective": "flow_matching",
        "num_inference_steps": 1,
    }
    env_config = SimpleNamespace(
        robot=SimpleNamespace(
            actions_dim=66,
            algo_obs_dim_dict={"proprio_obs": (5, 161), "privileged_obs": 48},
        )
    )
    model = EncoderVectorSingleStepTransformerFlowPolicy(
        obs_dim_dict={"proprio_obs": (5, 161), "privileged_obs": 48},
        module_config_dict=config,
        module_dim_dict={},
        env_config=env_config,
        process_output_dim=True,
    )
    observations = {
        "proprio_obs": torch.randn(2, 3, 5, 161),
        "privileged_obs": torch.randn(2, 3, 48),
    }

    assert not model.is_action_chunk_policy
    with torch.no_grad():
        assert model(observations).shape == (2, 3, 66)

    result = model(
        observations,
        compute_aux_loss=True,
        diffusion_target=torch.randn(2, 3, 66),
    )
    assert result["action_mean"].shape == (2, 3, 66)
    assert result["aux_losses"]["diffusion_flow"].isfinite()
    result["aux_losses"]["diffusion_flow"].backward()


def test_regular_rollout_storage_accepts_structured_shape_override():
    from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer

    trainer = TRLPPOTrainer.__new__(TRLPPOTrainer)
    trainer.config = {
        "num_steps_per_env": 2,
        "rollout_storage_obs_shapes": {"proprio_obs": [5, 161]},
    }
    trainer.env = SimpleNamespace(
        num_envs=2,
        config=SimpleNamespace(
            num_envs=2,
            robot=SimpleNamespace(
                algo_obs_dim_dict={"proprio_obs": 805, "privileged_obs": 48}
            ),
        ),
    )
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.algo_obs_dim_dict = trainer.env.config.robot.algo_obs_dim_dict
    trainer.num_steps_per_env = 2
    trainer.rollout_storage_obs_shapes = trainer.config[
        "rollout_storage_obs_shapes"
    ]
    trainer.action_chunk_enabled = False
    trainer.num_critics = 1
    trainer.num_act = 66
    trainer.use_symmetry = False
    trainer.learn_normalized_actions = False
    trainer.camera_resolution = None
    trainer._setup_episode_tracking = lambda: None

    trainer._setup_storage()

    assert trainer.storage.proprio_obs.shape == (2, 2, 5, 161)
    trainer.storage.update_key("proprio_obs", torch.zeros(2, 5, 161))


def test_actor_clear_rollout_is_idempotent():
    graph_leaf = torch.tensor(1.0, requires_grad=True)
    actor = SimpleNamespace(
        has_aux_loss=True,
        distribution=object(),
        aux_losses={"flow": graph_leaf.square()},
        aux_loss_coef={"flow": 1.0},
    )
    Actor.clear_rollout(actor)
    Actor.clear_rollout(actor)
    assert actor.distribution is None
    assert actor.aux_losses is None
    assert actor.aux_loss_coef is None


def test_actor_clear_forward_state_releases_training_graph_references():
    graph_leaf = torch.tensor(1.0, requires_grad=True)
    actor = SimpleNamespace(
        has_aux_loss=True,
        distribution=object(),
        aux_losses={"flow": graph_leaf.square()},
        aux_loss_coef={"flow": 1.0},
    )

    Actor.clear_forward_state(actor)

    assert actor.distribution is None
    assert actor.aux_losses is None
    assert actor.aux_loss_coef is None


def test_ring_alignment_zero_values_and_episode_boundary():
    buffer = ActionChunkDaggerBuffer(
        num_envs=2,
        capacity=64,
        horizon=40,
        rollout_steps=16,
        action_dim=3,
        observation_shapes={"proprio_obs": (2,), "privileged_obs": (1,)},
    )
    for block in range(4):
        start = block * 16
        transition = torch.arange(start, start + 16).view(16, 1).expand(16, 2)
        target = transition.unsqueeze(-1).expand(16, 2, 3).float().clone()
        target[:, 1] = 0.0  # Numerical zero is a valid Teacher target.
        episode = torch.zeros(16, 2, dtype=torch.long)
        episode[transition[:, 0] >= 40, 0] = 1
        done_after_action = torch.zeros(16, 2, dtype=torch.bool)
        done_after_action[transition[:, 0] == 39, 0] = True
        buffer.append(
            observations={
                "proprio_obs": torch.zeros(16, 2, 2),
                "privileged_obs": torch.zeros(16, 2, 1),
            },
            teacher_target=target,
            episode_uid=episode,
            motion_id=torch.zeros(16, 2, dtype=torch.long),
            reference_frame=transition,
            transition_id=transition,
            frame_finite=torch.ones(16, 2, dtype=torch.bool),
            done_after_action=done_after_action,
        )

    batch = buffer.pop()
    assert batch["diffusion_target"].shape == (2, 16, 40, 3)
    assert batch["diffusion_target"][0, 15, -1, 0].item() == 54
    assert batch["diffusion_target_valid"][0, 0]
    assert not batch["diffusion_target_valid"][0, 1:].any()
    assert batch["diffusion_target_valid"][1].all()
    assert (batch["diffusion_target"][1] == 0).all()
    assert buffer.size == 48

    transition = torch.arange(64, 80).view(16, 1).expand(16, 2)
    target = transition.unsqueeze(-1).expand(16, 2, 3).float()
    buffer.append(
        observations={
            "proprio_obs": torch.zeros(16, 2, 2),
            "privileged_obs": torch.zeros(16, 2, 1),
        },
        teacher_target=target,
        episode_uid=torch.tensor([1, 0]).view(1, 2).expand(16, 2),
        motion_id=torch.zeros(16, 2, dtype=torch.long),
        reference_frame=transition,
        transition_id=transition,
        frame_finite=torch.ones(16, 2, dtype=torch.bool),
        done_after_action=torch.zeros(16, 2, dtype=torch.bool),
    )
    wrapped = buffer.pop()
    assert wrapped["diffusion_target"][0, 15, -1, 0].item() == 70
    assert buffer.size == 48


def test_ring_preserves_structured_runtime_proprio_shape():
    from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer

    trainer = TRLPPOTrainer.__new__(TRLPPOTrainer)
    trainer.env = SimpleNamespace(num_envs=2)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.action_chunk_capacity = 64
    trainer.action_chunk_horizon = 40
    trainer.num_steps_per_env = 16
    trainer.num_act = 66
    # This flattened metadata caused the original bug; runtime tensors must
    # remain authoritative for ring allocation.
    trainer.algo_obs_dim_dict = {"proprio_obs": 805, "privileged_obs": 48}
    runtime_observations = {
        "proprio_obs": torch.zeros(2, 5, 161),
        "privileged_obs": torch.zeros(2, 48),
    }
    trainer._initialize_action_chunk_buffer(runtime_observations)
    buffer = trainer.action_chunk_buffer
    transition = torch.arange(16).view(16, 1).expand(16, 2)
    buffer.append(
        observations={
            "proprio_obs": torch.zeros(16, 2, 5, 161),
            "privileged_obs": torch.zeros(16, 2, 48),
        },
        teacher_target=torch.zeros(16, 2, 66),
        episode_uid=torch.zeros(16, 2, dtype=torch.long),
        motion_id=torch.zeros(16, 2, dtype=torch.long),
        reference_frame=transition,
        transition_id=transition,
        frame_finite=torch.ones(16, 2, dtype=torch.bool),
        done_after_action=torch.zeros(16, 2, dtype=torch.bool),
    )
    assert buffer.observations["proprio_obs"].shape == (64, 2, 5, 161)

    with pytest.raises(ValueError, match="expected.*5, 161"):
        buffer.append(
            observations={
                "proprio_obs": torch.zeros(16, 2, 805),
                "privileged_obs": torch.zeros(16, 2, 48),
            },
            teacher_target=torch.zeros(16, 2, 66),
            episode_uid=torch.zeros(16, 2, dtype=torch.long),
            motion_id=torch.zeros(16, 2, dtype=torch.long),
            reference_frame=transition + 16,
            transition_id=transition + 16,
            frame_finite=torch.ones(16, 2, dtype=torch.bool),
            done_after_action=torch.zeros(16, 2, dtype=torch.bool),
        )


class _FakeChunkAccelerator:
    def __init__(self, num_processes=1):
        self.device = torch.device("cpu")
        self.num_processes = num_processes
        self.no_sync_calls = 0

    @staticmethod
    def backward(loss):
        loss.backward()

    def no_sync(self, _model):
        self.no_sync_calls += 1
        return nullcontext()


class _ToyChunkTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.forward_calls = 0

    def forward(self, modes, input_kwargs):
        assert modes == ["policy_chunk_distill"]
        self.forward_calls += 1
        kwargs = input_kwargs["policy_chunk_distill"]
        feature = kwargs["obs_dict"]["feature"]
        target = kwargs["diffusion_target"]
        valid = kwargs["diffusion_target_valid"].to(target.dtype)
        prediction = self.weight * feature.reshape(-1, 1, 1).expand_as(target)
        per_chunk_loss = (prediction - target).square().mean(dim=(-2, -1))
        loss = (per_chunk_loss * valid).sum() / valid.sum().clamp_min(1.0)
        return {
            "policy_chunk_distill": {
                "aux_losses": {"diffusion_flow": loss},
                "aux_loss_coef": {"diffusion_flow": 1.0},
            }
        }


def _make_toy_chunk_trainer(model, micro_batch_size, epochs=3, minibatches=4):
    from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer

    trainer = TRLPPOTrainer.__new__(TRLPPOTrainer)
    trainer.action_chunk_horizon = 2
    trainer.num_act = 1
    trainer.action_chunk_num_epochs = epochs
    trainer.action_chunk_num_minibatches = minibatches
    trainer.action_chunk_train_micro_batch_size = micro_batch_size
    trainer.accelerator = _FakeChunkAccelerator()
    trainer.policy_model = SimpleNamespace(
        update_action_chunk_normalizers=lambda *_args, **_kwargs: None
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer.config = {"aux_loss_scale": 1.0}
    trainer._train_mode = model.train
    trainer._gradient_clipping = lambda: torch.tensor(0.0)
    return trainer


def _make_toy_chunk_batch(valid):
    count = valid.numel()
    feature = torch.linspace(-1.0, 1.0, count).unsqueeze(-1)
    target = torch.stack((feature.squeeze(-1), feature.squeeze(-1) + 0.5), dim=-1)
    return {
        "obs_dict": {"feature": feature},
        "diffusion_target": target.unsqueeze(-1),
        "diffusion_target_valid": valid,
    }


def test_action_chunk_microbatch_matches_logical_minibatch_updates():
    valid = torch.ones(10, dtype=torch.bool)
    batch = _make_toy_chunk_batch(valid)
    full_model = _ToyChunkTrainModel()
    micro_model = _ToyChunkTrainModel()
    micro_model.load_state_dict(full_model.state_dict())
    full_trainer = _make_toy_chunk_trainer(full_model, micro_batch_size=None)
    micro_trainer = _make_toy_chunk_trainer(micro_model, micro_batch_size=2)

    torch.manual_seed(99)
    full_metrics = full_trainer._train_action_chunk_batch(full_model, batch)
    torch.manual_seed(99)
    micro_metrics = micro_trainer._train_action_chunk_batch(micro_model, batch)

    assert full_metrics["optimizer_steps"] == 12
    assert micro_metrics["optimizer_steps"] == 12
    assert torch.allclose(full_model.weight, micro_model.weight, atol=1.0e-6)
    assert micro_trainer.accelerator.no_sync_calls > 0


@pytest.mark.parametrize("local_valid_count", [0, 1])
def test_action_chunk_microbatch_handles_short_and_empty_ddp_rank(local_valid_count):
    valid = torch.zeros(2, dtype=torch.bool)
    valid[:local_valid_count] = True
    batch = _make_toy_chunk_batch(valid)
    model = _ToyChunkTrainModel()
    trainer = _make_toy_chunk_trainer(
        model, micro_batch_size=1, epochs=1, minibatches=1
    )
    trainer.accelerator = _FakeChunkAccelerator(num_processes=2)
    trainer._all_reduce_count = lambda count: torch.full_like(count, 2)
    trainer._all_reduce_max_count = lambda count: torch.full_like(count, 2)

    metrics = trainer._train_action_chunk_batch(model, batch)

    assert metrics["optimizer_steps"] == 1
    assert metrics["global_valid"] == 2
    # A shorter rank performs a final dummy backward so DDP can synchronize.
    assert model.forward_calls == local_valid_count + 1


class _FakeChunkPolicy:
    def __init__(self):
        self.calls = 0

    def predict_action_chunk(self, obs_dict):
        self.calls += 1
        batch = obs_dict["state"].shape[0]
        frame = torch.arange(40).reshape(1, 40, 1).expand(batch, 40, 3)
        return frame.float() + self.calls * 100.0


def test_horizon_one_executor_replans_every_step():
    class SingleStepPolicy:
        def __init__(self):
            self.calls = 0

        def predict_action_chunk(self, obs_dict):
            self.calls += 1
            batch = obs_dict["state"].shape[0]
            return torch.full((batch, 1, 3), float(self.calls))

    policy = SingleStepPolicy()
    executor = ActionChunkExecutor(policy, 2, 1, 1, 3, "cpu")
    observations = {"state": torch.zeros(2, 1)}

    outputs = [executor.act(observations)[:, 0] for _ in range(3)]

    assert policy.calls == 3
    assert [output.tolist() for output in outputs] == [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]


def test_executor_uses_twenty_frames_and_replans_only_reset_env():
    cadence_policy = _FakeChunkPolicy()
    cadence_executor = ActionChunkExecutor(cadence_policy, 2, 40, 20, 3, "cpu")
    observations = {"state": torch.zeros(2, 1)}
    for _ in range(45):
        cadence_executor.act(observations)
    assert cadence_policy.calls == 3

    policy = _FakeChunkPolicy()
    executor = ActionChunkExecutor(policy, 2, 40, 20, 3, "cpu")
    outputs = [executor.act(observations)[:, 0] for _ in range(10)]
    assert policy.calls == 1
    assert outputs[0].tolist() == [100.0, 100.0]
    assert outputs[-1].tolist() == [109.0, 109.0]

    output = executor.act(
        observations, reset_mask=torch.tensor([True, False])
    )[:, 0]
    assert policy.calls == 2
    assert output.tolist() == [200.0, 110.0]

    for _ in range(9):
        executor.act(observations)
    output = executor.act(observations)[:, 0]
    assert policy.calls == 3
    assert output.tolist() == [210.0, 300.0]
