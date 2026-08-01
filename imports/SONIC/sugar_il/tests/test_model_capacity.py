import pytest
import torch

from sugar_il.model.encoder.generator_state_encoder import GeneratorStateObsEncoder
from sugar_il.model.flowmatching.transformer_for_action_flow_matching import (
    HandPrimitiveHead,
    TransformerForActionFlowMatching,
)
from sugar_il.wrapper.sugar_il_wrapper import load_generator_policy_state


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@pytest.mark.parametrize(
    ("layers", "expected_total"),
    (
        (12, 60_207_682),
        (16, 77_017_666),
        (20, 93_827_650),
    ),
)
def test_power_of_two_capacity_presets(layers, expected_total):
    with torch.device("meta"):
        encoder = GeneratorStateObsEncoder(
            feature_dim=512,
            token_hidden_dim=512,
        )
        flow = TransformerForActionFlowMatching(
            n_emb=512,
            n_layer=layers,
            n_head=8,
        )
        hand = HandPrimitiveHead(hidden_size=512, num_heads=8, num_layers=2)

    assert parameter_count(encoder) == 863_232
    assert parameter_count(hand) == 8_430_594
    assert sum(map(parameter_count, (encoder, flow, hand))) == expected_total


def test_original_capacity_defaults_remain_available():
    with torch.device("meta"):
        encoder = GeneratorStateObsEncoder()
        flow = TransformerForActionFlowMatching()
        hand = HandPrimitiveHead()

    assert sum(map(parameter_count, (encoder, flow, hand))) == 15_161_666


@pytest.mark.parametrize("time_conditioning", ("token", "adaln"))
def test_optional_time_conditioning_preserves_output_contract(time_conditioning):
    model = TransformerForActionFlowMatching(
        n_emb=64,
        n_layer=1,
        n_head=8,
        ffn_ratio=2,
        time_conditioning=time_conditioning,
        use_action_time_encoder=True,
        velocity_head_layers=2,
    )
    trajectory = torch.randn(2, 40, 64)
    condition = torch.randn(2, 3, 64)
    velocity, maps = model(trajectory, torch.tensor([0, 500]), condition)

    assert velocity.shape == (2, 40, 64)
    assert maps is None


def test_encoder_hidden_width_preserves_three_token_contract():
    encoder = GeneratorStateObsEncoder(feature_dim=64, token_hidden_dim=128)
    observations = {
        "object_bps": torch.randn(2, 1, 10),
        "table_geometry": torch.randn(2, 1, 4),
        "object_pos_b": torch.randn(2, 1, 3),
        "object_ori_b_6d": torch.randn(2, 1, 6),
        "hand_object_transform_6d": torch.randn(2, 1, 16),
        "hand_object_contact_force_magnitude": torch.randn(2, 1, 1),
        "base_lin_vel": torch.randn(2, 1, 3),
        "base_ang_vel": torch.randn(2, 1, 3),
        "joint_pos": torch.randn(2, 1, 43),
        "joint_vel": torch.randn(2, 1, 43),
    }

    assert encoder(observations, training=False).shape == (2, 3, 64)


def test_hidden_size_must_be_divisible_by_attention_heads():
    with pytest.raises(ValueError, match="divisible"):
        TransformerForActionFlowMatching(n_emb=512, n_head=10)


def test_checkpoint_loader_prefers_ema_and_supports_old_checkpoints():
    source = torch.nn.Linear(2, 2)
    ema = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(1)
        ema.weight.fill_(2)

    selected = load_generator_policy_state(
        target,
        {"state_dicts": {"model": source.state_dict(), "ema_model": ema.state_dict()}},
    )
    assert selected == "ema_model"
    assert torch.equal(target.weight, ema.weight)

    selected = load_generator_policy_state(
        target,
        {"state_dicts": {"model": source.state_dict()}},
    )
    assert selected == "model"
    assert torch.equal(target.weight, source.weight)
