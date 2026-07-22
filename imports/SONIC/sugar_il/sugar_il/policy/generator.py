from __future__ import annotations

import inspect
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from sugar_il.model.common.module_attr_mixin import ModuleAttrMixin
from sugar_il.model.common.normalizer import LinearNormalizer
from sugar_il.model.diffusion.transformer_for_action_diffusion import HandPrimitiveHead, TransformerForActionDiffusion


class Generator(ModuleAttrMixin):
    def __init__(self, shape_meta: dict, noise_scheduler, obs_encoder, num_inference_steps: int = 16, n_layer: int = 12, n_head: int = 8, p_drop_attn: float = 0.1, hand_loss_weight: float = 1.0, **kwargs):
        super().__init__()
        horizon = int(shape_meta["action"]["horizon"])
        latent_dim = int(shape_meta["action"].get("latent_dim", 64))
        obs_shape, _ = obs_encoder.output_shape()
        hidden_size = obs_shape[-1]
        self.obs_encoder = obs_encoder
        self.model = TransformerForActionDiffusion(latent_dim, latent_dim, horizon, n_layer, n_head, hidden_size, 5, p_drop_attn)
        self.hand_head = HandPrimitiveHead(horizon, hidden_size, n_head, 2, p_drop_attn)
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_horizon = horizon
        self.latent_dim = latent_dim
        self.num_inference_steps = num_inference_steps
        self.hand_loss_weight = hand_loss_weight

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in obs.items():
            result[key] = value if key == "last_hand_primitive" else self.normalizer[key].normalize(value)
        return result

    def conditional_sample(self, condition: torch.Tensor, gen_attn_map: bool = False):
        batch_size = condition.shape[0]
        trajectory = torch.randn(batch_size, self.action_horizon, self.latent_dim, device=self.device, dtype=self.dtype)
        try:
            self.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        except TypeError:
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
        attention_maps = {}
        for timestep in self.noise_scheduler.timesteps:
            prediction, maps = self.model(trajectory, timestep, condition, gen_attn_map=gen_attn_map)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory).prev_sample
            if gen_attn_map:
                attention_maps[int(timestep)] = maps
        return trajectory, attention_maps

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], gen_attn_map: bool = False) -> Dict[str, torch.Tensor]:
        normalized_obs = self._normalize_obs(obs_dict)
        condition = self.obs_encoder(normalized_obs, training=False)
        normalized_latent, attention_maps = self.conditional_sample(condition, gen_attn_map)
        latent = self.normalizer["latent"].unnormalize(normalized_latent)
        hand_logits = self.hand_head(condition)
        hand_probability = torch.sigmoid(hand_logits)
        hand_primitive = (hand_probability >= 0.5).to(latent.dtype)
        result = {
            "latent": latent,
            "hand_logits": hand_logits,
            "hand_probability": hand_probability,
            "hand_primitive": hand_primitive,
            "action": torch.cat((latent, hand_primitive), dim=-1),
        }
        if gen_attn_map:
            result["attention_maps"] = attention_maps
        return result

    def compute_loss(self, batch: dict, training: bool = True) -> dict[str, torch.Tensor]:
        normalized_obs = self._normalize_obs(batch["obs"])
        trajectory = self.normalizer["latent"].normalize(batch["action"]["latent"])
        hand_target = batch["action"]["hand_primitive"].float()
        condition = self.obs_encoder(normalized_obs, training=training)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (trajectory.shape[0],), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        prediction, _ = self.model(noisy_trajectory, timesteps, condition)
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            diffusion_target = noise
        elif prediction_type == "sample":
            diffusion_target = trajectory
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        latent_loss = F.mse_loss(prediction, diffusion_target)
        hand_logits = self.hand_head(condition)
        hand_loss = F.binary_cross_entropy_with_logits(hand_logits, hand_target)
        total_loss = latent_loss + self.hand_loss_weight * hand_loss
        return {"loss": total_loss, "latent_loss": latent_loss, "hand_loss": hand_loss, "hand_logits": hand_logits}

    def forward(self, batch: dict, training: bool = True):
        return self.compute_loss(batch, training)["loss"]

    def get_optimizer(self, lr: float, weight_decay: float, betas: Tuple[float, float]):
        decay, no_decay = [], []
        for parameter in self.parameters():
            if parameter.requires_grad:
                (decay if parameter.dim() >= 2 else no_decay).append(parameter)
        kwargs = {"lr": lr, "betas": betas}
        if "fused" in inspect.signature(torch.optim.AdamW).parameters and torch.cuda.is_available():
            kwargs["fused"] = True
        return torch.optim.AdamW(({"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}), **kwargs)
