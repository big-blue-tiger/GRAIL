from __future__ import annotations

import inspect
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Beta

from sugar_il.model.common.module_attr_mixin import ModuleAttrMixin
from sugar_il.model.common.normalizer import LinearNormalizer
from sugar_il.model.flowmatching.transformer_for_action_flow_matching import (
    HandPrimitiveHead,
    TransformerForActionFlowMatching,
)


class Generator(ModuleAttrMixin):
    """Object-aware action generator trained with conditional flow matching."""

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder,
        num_inference_steps: int = 4,
        n_layer: int = 12,
        n_head: int = 8,
        p_drop_attn: float = 0.1,
        frequency_embedding_size: int = 256,
        ffn_ratio: int = 4,
        ffn_activation: str = "gelu",
        time_conditioning: str = "token",
        use_action_time_encoder: bool = False,
        velocity_head_layers: int = 1,
        zero_init_velocity_head: bool = False,
        hand_num_layers: int = 2,
        hand_loss_weight: float = 1.0,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
    ):
        super().__init__()
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if num_timestep_buckets <= 0:
            raise ValueError("num_timestep_buckets must be positive")
        if noise_beta_alpha <= 0 or noise_beta_beta <= 0:
            raise ValueError("Beta distribution parameters must be positive")
        if not 0 < noise_s <= 1:
            raise ValueError("noise_s must be in (0, 1]")

        horizon = int(shape_meta["action"]["horizon"])
        latent_dim = int(shape_meta["action"].get("latent_dim", 64))
        obs_shape, _ = obs_encoder.output_shape()
        hidden_size = obs_shape[-1]

        self.obs_encoder = obs_encoder
        self.model = TransformerForActionFlowMatching(
            input_dim=latent_dim,
            output_dim=latent_dim,
            action_horizon=horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=hidden_size,
            max_cond_tokens=5,
            p_drop_attn=p_drop_attn,
            frequency_embedding_size=frequency_embedding_size,
            ffn_ratio=ffn_ratio,
            ffn_activation=ffn_activation,
            time_conditioning=time_conditioning,
            use_action_time_encoder=use_action_time_encoder,
            velocity_head_layers=velocity_head_layers,
            zero_init_velocity_head=zero_init_velocity_head,
        )
        self.hand_head = HandPrimitiveHead(
            horizon=horizon,
            hidden_size=hidden_size,
            num_heads=n_head,
            num_layers=hand_num_layers,
            dropout=p_drop_attn,
        )
        self.normalizer = LinearNormalizer()
        self.action_horizon = horizon
        self.latent_dim = latent_dim
        self.num_inference_steps = int(num_inference_steps)
        self.num_timestep_buckets = int(num_timestep_buckets)
        self.noise_s = float(noise_s)
        self.hand_loss_weight = float(hand_loss_weight)
        self.time_distribution = Beta(
            torch.tensor(float(noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(noise_beta_beta), dtype=torch.float32, device="cpu"),
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs(
        self,
        obs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: (
                value
                if key == "last_hand_primitive"
                else self.normalizer[key].normalize(value)
            )
            for key, value in obs.items()
        }

    def sample_time(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        time = self.time_distribution.sample((batch_size,))
        return ((1 - time) * self.noise_s).to(device=device, dtype=dtype)

    def discretize_time(self, time: torch.Tensor) -> torch.Tensor:
        buckets = (time * self.num_timestep_buckets).long()
        return buckets.clamp_(0, self.num_timestep_buckets - 1)

    def conditional_sample(
        self,
        condition: torch.Tensor,
        gen_attn_map: bool = False,
    ):
        batch_size = condition.shape[0]
        trajectory = torch.randn(
            batch_size,
            self.action_horizon,
            self.latent_dim,
            device=self.device,
            dtype=self.dtype,
        )
        dt = 1.0 / self.num_inference_steps
        attention_maps = {}

        for step in range(self.num_inference_steps):
            time = torch.full(
                (batch_size,),
                step / self.num_inference_steps,
                device=self.device,
                dtype=torch.float32,
            )
            time_bucket = self.discretize_time(time)
            velocity, maps = self.model(
                trajectory,
                time_bucket,
                condition,
                gen_attn_map=gen_attn_map,
            )
            trajectory = trajectory + dt * velocity
            if gen_attn_map:
                attention_maps[int(time_bucket[0].item())] = maps

        return trajectory, attention_maps

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        gen_attn_map: bool = False,
    ) -> Dict[str, torch.Tensor]:
        normalized_obs = self._normalize_obs(obs_dict)
        condition = self.obs_encoder(normalized_obs, training=False)
        normalized_latent, attention_maps = self.conditional_sample(
            condition,
            gen_attn_map,
        )
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

    def compute_loss(
        self,
        batch: dict,
        training: bool = True,
        normalized: bool = False,
    ) -> dict[str, torch.Tensor]:
        if normalized:
            normalized_obs = batch["obs"]
            trajectory = batch["action"]["latent"]
        else:
            normalized_obs = self._normalize_obs(batch["obs"])
            trajectory = self.normalizer["latent"].normalize(
                batch["action"]["latent"]
            )
        hand_target = batch["action"]["hand_primitive"].float()
        condition = self.obs_encoder(normalized_obs, training=training)

        noise = torch.randn_like(trajectory)
        time = self.sample_time(
            trajectory.shape[0],
            trajectory.device,
            trajectory.dtype,
        )
        path_time = time[:, None, None]
        noisy_trajectory = (1 - path_time) * noise + path_time * trajectory
        target_velocity = trajectory - noise
        time_bucket = self.discretize_time(time)

        predicted_velocity, _ = self.model(
            noisy_trajectory,
            time_bucket,
            condition,
        )
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)
        hand_logits = self.hand_head(condition)
        hand_loss = F.binary_cross_entropy_with_logits(
            hand_logits,
            hand_target,
        )
        total_loss = flow_loss + self.hand_loss_weight * hand_loss
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "hand_loss": hand_loss,
            "hand_logits": hand_logits,
        }

    def forward(
        self,
        batch: dict,
        training: bool = True,
        normalized: bool = False,
    ) -> dict[str, torch.Tensor]:
        """DDP entry point for training and validation."""
        return self.compute_loss(
            batch,
            training,
            normalized=normalized,
        )

    def get_optimizer(
        self,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ):
        decay, no_decay = [], []
        for parameter in self.parameters():
            if parameter.requires_grad:
                (decay if parameter.dim() >= 2 else no_decay).append(parameter)

        kwargs = {"lr": lr, "betas": betas}
        supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        if supports_fused and torch.cuda.is_available():
            kwargs["fused"] = True
        return torch.optim.AdamW(
            (
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ),
            **kwargs,
        )
