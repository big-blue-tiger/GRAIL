from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest
import torch

from gear_sonic.trl.modules.diffusion_policy_modules import EncoderVectorMlpPolicy


def _make_model(target_normalization="none"):
    config = {
        "proprio_input_dim": 805,
        "proprio_feature_dim": 8,
        "proprio_hidden_dims": [16],
        "privileged_input_dim": 48,
        "privileged_feature_dim": 8,
        "privileged_hidden_dims": [16],
        "cond_dim": 16,
        "output_dim": ["robot_action_dim"],
        "target_normalization": target_normalization,
        "mlp_hidden_dims": [32, 24],
        "bc_loss_coef": 0.7,
    }
    env_config = SimpleNamespace(
        robot=SimpleNamespace(
            actions_dim=66,
            algo_obs_dim_dict={"proprio_obs": (5, 161), "privileged_obs": 48},
        )
    )
    return EncoderVectorMlpPolicy(
        obs_dim_dict={"proprio_obs": (5, 161), "privileged_obs": 48},
        module_config_dict=config,
        module_dim_dict={},
        env_config=env_config,
        process_output_dim=True,
    )


def _observations(batch=2, time=None):
    prefix = (batch,) if time is None else (batch, time)
    return {
        "proprio_obs": torch.randn(*prefix, 5, 161),
        "privileged_obs": torch.randn(*prefix, 48),
    }


def test_direct_bc_rollout_is_deterministic_and_has_no_diffusion_branches():
    model = _make_model()
    observations = _observations(time=3)
    with torch.no_grad():
        first = model(observations)
        second = model(observations)

    assert first.shape == (2, 3, 66)
    assert torch.equal(first, second)
    assert not hasattr(model, "time_encoder")
    assert not hasattr(model, "denoiser")
    assert any(name.startswith("action_head") for name, _ in model.named_parameters())


def test_direct_bc_loss_returns_prediction_and_backpropagates():
    model = _make_model()
    observations = _observations()
    target = torch.randn(2, 66)
    result = model(observations, compute_aux_loss=True, diffusion_target=target)

    assert result["action_mean"].shape == (2, 66)
    assert result["aux_losses"]["latent_bc_mse"].isfinite()
    assert result["aux_loss_coef"] == {"latent_bc_mse": 0.7}
    assert torch.allclose(
        result["aux_losses"]["latent_bc_mse"],
        torch.nn.functional.mse_loss(result["action_mean"], target),
    )

    result["aux_losses"]["latent_bc_mse"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_direct_bc_validates_target_presence_shape_and_input_shape():
    model = _make_model()
    observations = _observations()
    with pytest.raises(ValueError, match="requires diffusion_target"):
        model(observations, compute_aux_loss=True)
    with pytest.raises(ValueError, match="target dim mismatch"):
        model(observations, compute_aux_loss=True, diffusion_target=torch.randn(2, 65))
    with pytest.raises(ValueError, match="batch shape"):
        model(observations, compute_aux_loss=True, diffusion_target=torch.randn(3, 66))
    with pytest.raises(ValueError, match="proprio_obs dim mismatch"):
        model(
            {"proprio_obs": torch.randn(2, 5, 160), "privileged_obs": torch.randn(2, 48)}
        )


def test_pickup_table_bc_config_preserves_structured_proprio_storage_shape():
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/distill/"
                "robocasa_pickup_table_mlp_decoder_latent_vector_obs"
            ],
        )

    assert list(config.algo.config.rollout_storage_obs_shapes.proprio_obs) == [5, 161]
