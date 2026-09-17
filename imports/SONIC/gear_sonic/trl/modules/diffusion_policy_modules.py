"""Diffusion-style latent policies for SONIC distillation.

This module follows the conditioning pattern used by real-stanford's
diffusion_policy: encode observations, denoise an action vector conditioned on
those observations, and train with flow-matching or DDPM objectives.  Both
RGB and structured-vector condition encoders use the same diffusion head.  The
action vector here is the SONIC decoder-input latent plus hand command.
"""

import math

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from torch.utils import checkpoint as checkpoint_utils
from torchvision import models

from gear_sonic.utils.empirical_normalizer import EmpiricalNormalizer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb_scale)
        emb = x.float().unsqueeze(-1) * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class _RmsNorm(nn.Module):
    """Minimal RMSNorm matching the old SUGAR RDT parameterization."""

    def __init__(self, dim, eps=1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight.to(value)


class _RdtTimestepEmbedder(nn.Module):
    """Old-SUGAR/RDT sinusoidal timestep embedding followed by a two-layer MLP."""

    def __init__(self, hidden_dim, frequency_dim=256):
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps):
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class _RdtSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.q_norm = _RmsNorm(self.head_dim)
        self.k_norm = _RmsNorm(self.head_dim)
        self.attn_dropout = float(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, value):
        batch, length, hidden = value.shape
        qkv = self.qkv(value).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, val = qkv.unbind(0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            val,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        return self.proj_dropout(self.proj(attended))


class _RdtCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.key_value = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)
        self.q_norm = _RmsNorm(self.head_dim)
        self.k_norm = _RmsNorm(self.head_dim)
        self.attn_dropout = float(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, value, condition):
        batch, length, hidden = value.shape
        cond_length = condition.shape[1]
        query = self.query(value).reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_value = self.key_value(condition).reshape(
            batch, cond_length, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        key, cond_value = key_value.unbind(0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            cond_value,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        return self.proj_dropout(self.proj(attended))


class _RdtFeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, value):
        return self.fc2(F.gelu(self.fc1(value), approximate="tanh"))


class _RdtBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, attention_dropout):
        super().__init__()
        self.norm1 = _RmsNorm(hidden_dim)
        self.self_attention = _RdtSelfAttention(
            hidden_dim, num_heads, attention_dropout
        )
        self.norm2 = _RmsNorm(hidden_dim)
        self.cross_attention = _RdtCrossAttention(
            hidden_dim, num_heads, attention_dropout
        )
        self.norm3 = _RmsNorm(hidden_dim)
        self.feed_forward = _RdtFeedForward(hidden_dim)

    def forward(self, value, condition):
        value = value + self.self_attention(self.norm1(value))
        value = value + self.cross_attention(self.norm2(value), condition)
        return value + self.feed_forward(self.norm3(value))


class _ActionChunkRdt(nn.Module):
    """Strict old-SUGAR RDT block layout adapted to CFM action chunks."""

    def __init__(
        self,
        action_dim,
        action_horizon,
        hidden_dim=512,
        num_layers=14,
        num_heads=8,
        attention_dropout=0.1,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.action_embedding = nn.Linear(self.action_dim, self.hidden_dim)
        self.action_position = nn.Parameter(
            torch.empty(1, self.action_horizon, self.hidden_dim)
        )
        self.time_embedding = _RdtTimestepEmbedder(self.hidden_dim, frequency_dim=256)
        self.condition_position = nn.Parameter(torch.empty(1, 3, self.hidden_dim))
        self.blocks = nn.ModuleList(
            [
                _RdtBlock(self.hidden_dim, num_heads, attention_dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = _RmsNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.action_dim)
        self.apply(self._initialize)
        nn.init.normal_(self.action_position, mean=0.0, std=0.02)
        nn.init.normal_(self.condition_position, mean=0.0, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, _RmsNorm):
            nn.init.ones_(module.weight)

    def forward(self, noisy_action, timestep_bucket, observation_tokens):
        if noisy_action.shape[-2:] != (self.action_horizon, self.action_dim):
            raise ValueError(
                "RDT noisy action must end in "
                f"[{self.action_horizon},{self.action_dim}], got {tuple(noisy_action.shape)}"
            )
        prefix = noisy_action.shape[:-2]
        flat_count = math.prod(prefix) if prefix else 1
        action = noisy_action.reshape(flat_count, self.action_horizon, self.action_dim)
        condition = observation_tokens.reshape(flat_count, 2, self.hidden_dim)
        timestep = timestep_bucket.reshape(flat_count)
        time_token = self.time_embedding(timestep).unsqueeze(1).to(condition)
        condition = torch.cat([condition, time_token], dim=1) + self.condition_position
        value = self.action_embedding(action) + self.action_position
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                value = checkpoint_utils.checkpoint(
                    block,
                    value,
                    condition,
                    use_reentrant=False,
                )
            else:
                value = block(value, condition)
        output = self.output(self.final_norm(value))
        return output.reshape(*prefix, self.action_horizon, self.action_dim)


def _build_mlp(input_dim, hidden_dims, output_dim, activation_name="SiLU"):
    activation_cls = getattr(nn, activation_name)
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_cls())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class EncoderRgbDiffusionPolicy(nn.Module):
    """Condition on ego RGB and diffuse latent actions.

    Expected inputs:
    - ``image_key``: first-person RGB image with shape ``[..., H, W, 3]``.

    During training, the trainer passes ``diffusion_target`` with shape
    ``[..., action_dim]``.  The module returns a dict compatible with
    ``Actor(has_aux_loss=True)``.  During rollout/eval it samples a denoised
    action vector and returns it as the action mean used by the outer actor.
    """

    def __init__(
        self,
        obs_dim_dict=None,
        module_config_dict=None,
        module_dim_dict=None,
        env_config=None,
        algo_config=None,  # noqa: ARG002
        process_output_dim=False,
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()
        module_config_dict = module_config_dict or {}

        self.image_key = module_config_dict.get("image_key", "camera_rgb")
        self.state_key = module_config_dict.get("state_key", None)
        self.diffusion_target_key = module_config_dict.get(
            "diffusion_target_key", "diffusion_target"
        )
        self.action_dim = self._resolve_action_dim(
            module_config_dict.get("output_dim", ["robot_action_dim"]),
            module_dim_dict or {},
            env_config,
            process_output_dim,
        )

        activation = module_config_dict.get("activation", "SiLU")
        image_feature_dim = int(module_config_dict.get("image_feature_dim", 128))
        state_token_dim = int(module_config_dict.get("state_token_dim", 128))
        cond_dim = int(module_config_dict.get("cond_dim", 512))
        self.cond_dim = cond_dim
        timestep_dim = int(module_config_dict.get("timestep_dim", 128))
        timestep_hidden_dim = int(
            module_config_dict.get("timestep_hidden_dim", timestep_dim * 4)
        )
        self.timestep_feature_dim = int(
            module_config_dict.get("timestep_feature_dim", timestep_dim)
        )

        self.skip_image_encoder = bool(module_config_dict.get("_skip_image_encoder", False))
        self.image_encoder = (
            nn.Identity()
            if self.skip_image_encoder
            else self._build_image_encoder(module_config_dict, image_feature_dim)
        )
        self.use_state_token = self.state_key is not None
        if self.use_state_token:
            if obs_dim_dict is None and env_config is not None:
                obs_dim_dict = env_config.robot.algo_obs_dim_dict
            self.state_dim = int(
                module_config_dict.get("state_input_dim", obs_dim_dict[self.state_key])
            )
            self.state_encoder = _build_mlp(
                self.state_dim,
                module_config_dict.get("state_hidden_dims", [128, 128]),
                state_token_dim,
                activation,
            )
            self.state_normalization = module_config_dict.get(
                "state_normalization", "standardize"
            )
            self.state_norm_momentum = float(module_config_dict.get("state_norm_momentum", 0.05))
            self.state_norm_clip = float(module_config_dict.get("state_norm_clip", 5.0))
            self.state_std_eps = float(module_config_dict.get("state_std_eps", 1.0e-4))
            self.register_buffer("state_mean", torch.zeros(self.state_dim))
            self.register_buffer("state_var", torch.ones(self.state_dim))
            self.register_buffer("state_updates", torch.zeros((), dtype=torch.long))
        else:
            self.state_dim = 0
            state_token_dim = 0
        self.normalize_image = bool(module_config_dict.get("normalize_image", True))
        if not self.skip_image_encoder:
            self.register_buffer(
                "image_mean",
                torch.tensor(module_config_dict.get("image_mean", [0.485, 0.456, 0.406])).view(
                    1, 3, 1, 1
                ),
            )
            self.register_buffer(
                "image_std",
                torch.tensor(module_config_dict.get("image_std", [0.229, 0.224, 0.225])).view(
                    1, 3, 1, 1
                ),
            )
        self.cond_encoder = _build_mlp(
            image_feature_dim + state_token_dim,
            module_config_dict.get("cond_hidden_dims", [512]),
            cond_dim,
            activation,
        )
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_dim),
            nn.Linear(timestep_dim, timestep_hidden_dim),
            getattr(nn, activation)(),
            nn.Linear(timestep_hidden_dim, self.timestep_feature_dim),
        )
        self.denoiser = _build_mlp(
            self.action_dim + cond_dim + self.timestep_feature_dim,
            module_config_dict.get("denoiser_hidden_dims", [1024, 1024, 512]),
            self.action_dim,
            activation,
        )

        self.num_train_timesteps = int(module_config_dict.get("num_train_timesteps", 100))
        self.num_inference_steps = int(module_config_dict.get("num_inference_steps", 16))
        self.diffusion_loss_coef = float(module_config_dict.get("diffusion_loss_coef", 1.0))
        self.diffusion_objective = module_config_dict.get(
            "diffusion_objective", "flow_matching"
        )

        self.target_normalization = module_config_dict.get(
            "target_normalization", "standardize"
        )
        self.target_norm_momentum = float(module_config_dict.get("target_norm_momentum", 0.05))
        self.target_norm_clip = float(module_config_dict.get("target_norm_clip", 5.0))
        self.target_std_eps = float(module_config_dict.get("target_std_eps", 1.0e-4))

        self.noise_beta_alpha = float(module_config_dict.get("noise_beta_alpha", 1.5))
        self.noise_beta_beta = float(module_config_dict.get("noise_beta_beta", 1.0))
        self.noise_s = float(module_config_dict.get("noise_s", 0.999))
        self.num_timestep_buckets = int(module_config_dict.get("num_timestep_buckets", 1000))
        self.beta_dist = Beta(
            torch.tensor(self.noise_beta_alpha, dtype=torch.float32, device="cpu"),
            torch.tensor(self.noise_beta_beta, dtype=torch.float32, device="cpu"),
        )

        self.register_buffer("target_mean", torch.zeros(self.action_dim))
        self.register_buffer("target_var", torch.ones(self.action_dim))
        self.register_buffer("target_updates", torch.zeros((), dtype=torch.long))

        beta_start = float(module_config_dict.get("beta_start", 1.0e-4))
        beta_end = float(module_config_dict.get("beta_end", 2.0e-2))
        betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    @staticmethod
    def _resolve_action_dim(output_dim_cfg, module_dim_dict, env_config, process_output_dim):
        if isinstance(output_dim_cfg, int):
            return int(output_dim_cfg)
        total_dim = 0
        for item in output_dim_cfg:
            if item == "robot_action_dim" and process_output_dim:
                total_dim += int(env_config.robot.actions_dim)
            elif isinstance(item, int | float):
                total_dim += int(item)
            elif item in module_dim_dict:
                total_dim += int(module_dim_dict[item])
            else:
                raise ValueError(f"Unknown output dim entry: {item}")
        return total_dim

    def _build_image_encoder(self, module_config_dict, image_feature_dim):
        resnet_type = module_config_dict.get("resnet_type", "resnet18")
        pretrained = bool(module_config_dict.get("pretrained", True))
        trainable = bool(module_config_dict.get("trainable", True))
        if resnet_type == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            resnet_feature_dim = 512
        elif resnet_type == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            resnet_feature_dim = 512
        elif resnet_type == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            resnet_feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        features = nn.Sequential(*list(resnet.children())[:-2])
        if not trainable:
            for param in features.parameters():
                param.requires_grad = False
        activation_cls = getattr(nn, module_config_dict.get("activation", "SiLU"))
        return nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(resnet_feature_dim, image_feature_dim),
            activation_cls(),
        )

    def _encode_image(self, image):
        if image.ndim not in (4, 5):
            raise ValueError(f"{self.image_key} must be [B,H,W,C] or [B,T,H,W,C], got {image.shape}")
        prefix_shape = image.shape[:-3]
        image = image.reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2).contiguous().float()
        if self.normalize_image:
            if image.detach().amax().item() > 2.0:
                image = image / 255.0
            image = (image - self.image_mean.to(image)) / self.image_std.to(image)
        image_feat = self.image_encoder(image)
        return image_feat.reshape(*prefix_shape, -1)

    @torch.no_grad()
    def _update_state_stats(self, state):
        if not self.use_state_token or self.state_normalization != "standardize":
            return
        flat = state.detach().reshape(-1, self.state_dim).float()
        if flat.numel() == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False).clamp_min(self.state_std_eps**2)
        if int(self.state_updates.item()) == 0:
            self.state_mean.copy_(batch_mean.to(self.state_mean))
            self.state_var.copy_(batch_var.to(self.state_var))
        else:
            momentum = self.state_norm_momentum
            self.state_mean.lerp_(batch_mean.to(self.state_mean), momentum)
            self.state_var.lerp_(batch_var.to(self.state_var), momentum)
        self.state_updates += 1

    def _normalize_state(self, state):
        if self.state_normalization == "none":
            return state
        if self.state_normalization != "standardize":
            raise ValueError(f"Unsupported state_normalization={self.state_normalization}")
        mean = self.state_mean.to(device=state.device, dtype=state.dtype)
        std = self.state_var.clamp_min(self.state_std_eps**2).sqrt().to(
            device=state.device, dtype=state.dtype
        )
        normalized = (state - mean) / std
        if self.state_norm_clip > 0:
            normalized = normalized.clamp(-self.state_norm_clip, self.state_norm_clip)
        return normalized

    def _encode_state(self, obs_dict, update_stats=False):
        if not self.use_state_token:
            return None
        state = obs_dict[self.state_key].float()
        if state.shape[-1] != self.state_dim:
            state = state.reshape(*state.shape[:-1], -1)
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"{self.state_key} dim mismatch: got {state.shape[-1]}, expected {self.state_dim}"
            )
        if update_stats:
            with torch.no_grad():
                self._update_state_stats(state)
        return self.state_encoder(self._normalize_state(state))

    def _encode_condition(self, obs_dict, update_state_stats=False):
        image_feat = self._encode_image(obs_dict[self.image_key])
        if not self.use_state_token:
            return self.cond_encoder(image_feat)
        state_feat = self._encode_state(obs_dict, update_stats=update_state_stats)
        if state_feat.shape[:-1] != image_feat.shape[:-1]:
            state_feat = state_feat.reshape(*image_feat.shape[:-1], -1)
        return self.cond_encoder(torch.cat([image_feat, state_feat], dim=-1))

    @torch.no_grad()
    def _update_target_stats(self, target):
        if self.target_normalization != "standardize":
            return
        flat = target.detach().reshape(-1, self.action_dim).float()
        if flat.numel() == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False).clamp_min(self.target_std_eps**2)
        if int(self.target_updates.item()) == 0:
            self.target_mean.copy_(batch_mean.to(self.target_mean))
            self.target_var.copy_(batch_var.to(self.target_var))
        else:
            momentum = self.target_norm_momentum
            self.target_mean.lerp_(batch_mean.to(self.target_mean), momentum)
            self.target_var.lerp_(batch_var.to(self.target_var), momentum)
        self.target_updates += 1

    def _normalize_target(self, target):
        if self.target_normalization == "none":
            return target
        if self.target_normalization != "standardize":
            raise ValueError(f"Unsupported target_normalization={self.target_normalization}")
        mean = self.target_mean.to(device=target.device, dtype=target.dtype)
        std = self.target_var.clamp_min(self.target_std_eps**2).sqrt().to(
            device=target.device, dtype=target.dtype
        )
        normalized = (target - mean) / std
        if self.target_norm_clip > 0:
            normalized = normalized.clamp(-self.target_norm_clip, self.target_norm_clip)
        return normalized

    def _denormalize_target(self, normalized):
        if self.target_normalization == "none":
            return normalized
        mean = self.target_mean.to(device=normalized.device, dtype=normalized.dtype)
        std = self.target_var.clamp_min(self.target_std_eps**2).sqrt().to(
            device=normalized.device, dtype=normalized.dtype
        )
        if self.target_norm_clip > 0:
            normalized = normalized.clamp(-self.target_norm_clip, self.target_norm_clip)
        return normalized * std + mean

    def _sample_flow_time(self, prefix_shape, device, dtype):
        prefix_shape = tuple(prefix_shape)
        flat_count = max(int(math.prod(prefix_shape)), 1)
        sample = self.beta_dist.sample((flat_count,)).to(device=device, dtype=dtype)
        sample = sample.reshape(prefix_shape)
        return (1.0 - sample) * self.noise_s

    def _extract_target(self, kwargs):
        if self.diffusion_target_key in kwargs:
            return kwargs[self.diffusion_target_key]
        return kwargs.get("diffusion_target", None)

    def _q_sample(self, clean_action, noise, timesteps):
        alpha_bar = self.alphas_cumprod[timesteps].to(clean_action)
        while alpha_bar.ndim < clean_action.ndim:
            alpha_bar = alpha_bar.unsqueeze(-1)
        return alpha_bar.sqrt() * clean_action + (1.0 - alpha_bar).sqrt() * noise

    def _predict_noise(self, noisy_action, timesteps, cond):
        time_feat = self.time_encoder(timesteps)
        if time_feat.shape[:-1] != noisy_action.shape[:-1]:
            time_feat = time_feat.reshape(*noisy_action.shape[:-1], -1)
        denoise_input = torch.cat([noisy_action, cond, time_feat], dim=-1)
        return self.denoiser(denoise_input)

    def _sample(self, cond):
        if self.diffusion_objective == "flow_matching":
            return self._sample_flow(cond)
        if self.diffusion_objective != "ddpm_noise":
            raise ValueError(f"Unsupported diffusion_objective={self.diffusion_objective}")
        return self._sample_ddpm(cond)

    def _sample_flow(self, cond):
        sample = torch.randn(*cond.shape[:-1], self.action_dim, device=cond.device, dtype=cond.dtype)
        step_count = max(int(self.num_inference_steps), 1)
        dt = 1.0 / step_count
        for step in range(step_count):
            t_cont = step / float(step_count)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            t = torch.full(cond.shape[:-1], t_discretized, device=cond.device, dtype=torch.long)
            pred_velocity = self._predict_noise(sample, t, cond)
            sample = sample + dt * pred_velocity
        return self._denormalize_target(sample)

    def _sample_ddpm(self, cond):
        sample = torch.randn(*cond.shape[:-1], self.action_dim, device=cond.device, dtype=cond.dtype)
        step_count = min(self.num_inference_steps, self.num_train_timesteps)
        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            step_count,
            device=cond.device,
        ).round().long()
        for i, t_value in enumerate(timesteps):
            t = torch.full(cond.shape[:-1], int(t_value.item()), device=cond.device, dtype=torch.long)
            pred_noise = self._predict_noise(sample, t, cond)
            alpha_bar = self.alphas_cumprod[t_value].to(sample)
            pred_x0 = (sample - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            if i == len(timesteps) - 1:
                sample = pred_x0
            else:
                prev_t = timesteps[i + 1]
                prev_alpha_bar = self.alphas_cumprod[prev_t].to(sample)
                sample = prev_alpha_bar.sqrt() * pred_x0 + (1.0 - prev_alpha_bar).sqrt() * pred_noise
        return self._denormalize_target(sample)

    def forward(self, input, compute_aux_loss=False, **kwargs):
        if not hasattr(input, "__getitem__"):
            raise TypeError("EncoderRgbDiffusionPolicy expects an obs_dict-like input")
        cond = self._encode_condition(input, update_state_stats=compute_aux_loss)
        target = self._extract_target(kwargs)
        if compute_aux_loss:
            if target is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires {self.diffusion_target_key} "
                    "when compute_aux_loss=True"
                )
            target = target.to(device=cond.device, dtype=cond.dtype)
            if target.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Diffusion target dim mismatch: got {target.shape[-1]}, expected {self.action_dim}"
                )
            timesteps = torch.randint(
                0,
                self.num_train_timesteps,
                target.shape[:-1],
                device=target.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                self._update_target_stats(target)
            normalized_target = self._normalize_target(target)

            if self.diffusion_objective == "flow_matching":
                noise = torch.randn_like(normalized_target)
                t = self._sample_flow_time(
                    normalized_target.shape[:-1],
                    device=normalized_target.device,
                    dtype=normalized_target.dtype,
                )
                t_broadcast = t
                while t_broadcast.ndim < normalized_target.ndim:
                    t_broadcast = t_broadcast.unsqueeze(-1)
                noisy_action = (1.0 - t_broadcast) * noise + t_broadcast * normalized_target
                velocity = normalized_target - noise
                t_discretized = (t * self.num_timestep_buckets).long()
                pred_velocity = self._predict_noise(noisy_action, t_discretized, cond)
                diffusion_loss = F.mse_loss(pred_velocity, velocity)
                loss_name = "diffusion_flow"
            elif self.diffusion_objective == "ddpm_noise":
                noise = torch.randn_like(normalized_target)
                noisy_action = self._q_sample(normalized_target, noise, timesteps)
                pred_noise = self._predict_noise(noisy_action, timesteps, cond)
                diffusion_loss = F.mse_loss(pred_noise, noise)
                loss_name = "diffusion_noise"
            else:
                raise ValueError(f"Unsupported diffusion_objective={self.diffusion_objective}")

            target_flat = target.detach().float().reshape(-1, self.action_dim)
            norm_flat = normalized_target.detach().float().reshape(-1, self.action_dim)
            return {
                "action_mean": target.detach(),
                "aux_losses": {
                    loss_name: diffusion_loss,
                    "diffusion_target/raw_abs_mean": target_flat.abs().mean(),
                    "diffusion_target/raw_std_mean": target_flat.std(dim=0, unbiased=False).mean(),
                    "diffusion_target/norm_abs_mean": norm_flat.abs().mean(),
                    "diffusion_target/norm_std_mean": norm_flat.std(dim=0, unbiased=False).mean(),
                },
                "aux_loss_coef": {loss_name: self.diffusion_loss_coef},
            }

        return self._sample(cond)


class EncoderVectorDiffusionPolicy(EncoderRgbDiffusionPolicy):
    """Flow-matching policy with separate proprioceptive and privileged encoders.

    The two inputs have independent running normalization and are encoded
    separately before their features are concatenated. Diffusion target
    normalization, flow matching, and sampling are inherited unchanged.
    """

    def __init__(
        self,
        obs_dim_dict=None,
        module_config_dict=None,
        module_dim_dict=None,
        env_config=None,
        algo_config=None,
        process_output_dim=False,
        **kwargs,
    ):
        config = dict(module_config_dict or {})
        self.proprio_key = config.get("proprio_key", "proprio_obs")
        self.privileged_key = config.get("privileged_key", "privileged_obs")
        self.proprio_input_dim = int(config.get("proprio_input_dim", 805))
        self.privileged_input_dim = int(config.get("privileged_input_dim", 48))
        self.proprio_feature_dim = int(config.get("proprio_feature_dim", 512))
        self.privileged_feature_dim = int(config.get("privileged_feature_dim", 512))

        for key, expected_dim in (
            (self.proprio_key, self.proprio_input_dim),
            (self.privileged_key, self.privileged_input_dim),
        ):
            if obs_dim_dict is not None and key in obs_dim_dict:
                configured_dim = obs_dim_dict[key]
                if not isinstance(configured_dim, int):
                    configured_dim = int(torch.tensor(configured_dim).prod().item())
                if configured_dim != expected_dim:
                    raise ValueError(
                        f"{key} configured dim mismatch: observation manager reports "
                        f"{configured_dim}, policy expects {expected_dim}"
                    )

        fused_dim = self.proprio_feature_dim + self.privileged_feature_dim
        configured_cond_dim = int(config.get("cond_dim", fused_dim))
        if configured_cond_dim != fused_dim:
            raise ValueError(
                f"cond_dim must equal proprio_feature_dim + privileged_feature_dim "
                f"({fused_dim}), got {configured_cond_dim}"
            )
        config["cond_dim"] = fused_dim

        # Strip inherited visual-adaptor keys before constructing the parent;
        # the composed experiment may inherit the historical RGB config, but
        # this policy must not retain or instantiate any visual submodule.
        for visual_key in (
            "image_key",
            "image_shape",
            "image_encoder_type",
            "resnet_type",
            "pretrained",
            "trainable",
            "normalize_image",
            "image_mean",
            "image_std",
            "fusion_hidden_dims",
            "replace_terms",
        ):
            config.pop(visual_key, None)
        config["_skip_image_encoder"] = True
        # The temporary parent condition layer is replaced after construction;
        # this width only keeps parent initialization internally consistent.
        config["image_feature_dim"] = fused_dim
        config["cond_hidden_dims"] = []
        config["state_key"] = None
        config["normalize_image"] = False

        super().__init__(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=config,
            module_dim_dict=module_dim_dict,
            env_config=env_config,
            algo_config=algo_config,
            process_output_dim=process_output_dim,
            **kwargs,
        )

        activation = config.get("activation", "SiLU")
        self.cond_encoder = nn.Identity()
        self.proprio_encoder = _build_mlp(
            self.proprio_input_dim,
            config.get("proprio_hidden_dims", [1024]),
            self.proprio_feature_dim,
            activation,
        )
        self.privileged_encoder = _build_mlp(
            self.privileged_input_dim,
            config.get("privileged_hidden_dims", [512]),
            self.privileged_feature_dim,
            activation,
        )

        default_normalization = config.get("observation_normalization", "standardize")
        self.proprio_normalization = config.get(
            "proprio_normalization", default_normalization
        )
        self.privileged_normalization = config.get(
            "privileged_normalization", default_normalization
        )
        for name, normalization in (
            ("proprio_normalization", self.proprio_normalization),
            ("privileged_normalization", self.privileged_normalization),
        ):
            if normalization not in ("none", "standardize"):
                raise ValueError(
                    f"Unsupported {name}={normalization}; expected 'none' or 'standardize'"
                )
        self.proprio_norm_momentum = float(
            config.get("proprio_norm_momentum", config.get("observation_norm_momentum", 0.05))
        )
        self.privileged_norm_momentum = float(
            config.get("privileged_norm_momentum", config.get("observation_norm_momentum", 0.05))
        )
        self.proprio_norm_clip = float(
            config.get("proprio_norm_clip", config.get("observation_norm_clip", 5.0))
        )
        self.privileged_norm_clip = float(
            config.get("privileged_norm_clip", config.get("observation_norm_clip", 5.0))
        )
        self.proprio_std_eps = float(
            config.get("proprio_std_eps", config.get("observation_std_eps", 1.0e-4))
        )
        self.privileged_std_eps = float(
            config.get("privileged_std_eps", config.get("observation_std_eps", 1.0e-4))
        )
        self.register_buffer("proprio_mean", torch.zeros(self.proprio_input_dim))
        self.register_buffer("proprio_var", torch.ones(self.proprio_input_dim))
        self.register_buffer("proprio_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("privileged_mean", torch.zeros(self.privileged_input_dim))
        self.register_buffer("privileged_var", torch.ones(self.privileged_input_dim))
        self.register_buffer("privileged_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update_observation_stats(self, observation, prefix):
        if getattr(self, f"{prefix}_normalization") != "standardize":
            return
        input_dim = getattr(self, f"{prefix}_input_dim")
        flat = observation.detach().reshape(-1, input_dim).float()
        if flat.numel() == 0:
            return
        eps = getattr(self, f"{prefix}_std_eps")
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False).clamp_min(eps**2)
        mean = getattr(self, f"{prefix}_mean")
        var = getattr(self, f"{prefix}_var")
        updates = getattr(self, f"{prefix}_updates")
        if int(updates.item()) == 0:
            mean.copy_(batch_mean.to(mean))
            var.copy_(batch_var.to(var))
        else:
            momentum = getattr(self, f"{prefix}_norm_momentum")
            mean.lerp_(batch_mean.to(mean), momentum)
            var.lerp_(batch_var.to(var), momentum)
        updates.add_(1)

    def _normalize_observation(self, observation, prefix):
        if getattr(self, f"{prefix}_normalization") == "none":
            return observation
        mean = getattr(self, f"{prefix}_mean").to(
            device=observation.device, dtype=observation.dtype
        )
        eps = getattr(self, f"{prefix}_std_eps")
        std = getattr(self, f"{prefix}_var").clamp_min(eps**2).sqrt().to(
            device=observation.device, dtype=observation.dtype
        )
        normalized = (observation - mean) / std
        clip = getattr(self, f"{prefix}_norm_clip")
        if clip > 0:
            normalized = normalized.clamp(-clip, clip)
        return normalized

    @staticmethod
    def _flatten_observation(observation, input_dim, key):
        observation = observation.float()
        if observation.shape[-1] != input_dim:
            suffix_size = 1
            suffix_start = observation.ndim
            while suffix_start > 0 and suffix_size < input_dim:
                suffix_start -= 1
                suffix_size *= observation.shape[suffix_start]
            if suffix_size == input_dim:
                observation = observation.reshape(*observation.shape[:suffix_start], input_dim)
        if observation.shape[-1] != input_dim:
            raise ValueError(
                f"{key} dim mismatch: got shape {tuple(observation.shape)}, "
                f"expected trailing dimension(s) with {input_dim} values"
            )
        return observation

    def _encode_condition(self, obs_dict, update_state_stats=False):
        for key in (self.proprio_key, self.privileged_key):
            if key not in obs_dict:
                raise KeyError(f"EncoderVectorDiffusionPolicy requires observation key '{key}'")
        proprio = self._flatten_observation(
            obs_dict[self.proprio_key], self.proprio_input_dim, self.proprio_key
        )
        privileged = self._flatten_observation(
            obs_dict[self.privileged_key], self.privileged_input_dim, self.privileged_key
        )
        if proprio.shape[:-1] != privileged.shape[:-1]:
            raise ValueError(
                f"Observation batch shape mismatch: {self.proprio_key} has "
                f"{proprio.shape[:-1]}, {self.privileged_key} has {privileged.shape[:-1]}"
            )
        if update_state_stats:
            self._update_observation_stats(proprio, "proprio")
            self._update_observation_stats(privileged, "privileged")
        proprio_feat = self.proprio_encoder(
            self._normalize_observation(proprio, "proprio")
        )
        privileged_feat = self.privileged_encoder(
            self._normalize_observation(privileged, "privileged")
        )
        return torch.cat([proprio_feat, privileged_feat], dim=-1)


class EncoderVectorMlpPolicy(EncoderVectorDiffusionPolicy):
    """Direct behavior-cloning policy for structured observations.

    This variant encodes the raw proprioceptive and privileged observations
    separately and replaces the diffusion path with an independent MLP action
    head. Optional empirical normalization applies only to the 64 latent
    outputs; the returned 66-D action mean is always in the raw action space.
    """

    def __init__(self, *args, **kwargs):
        module_config_dict = kwargs.get("module_config_dict") or {}
        if not module_config_dict and len(args) >= 2 and args[1] is not None:
            module_config_dict = args[1]
        module_config_dict = dict(module_config_dict)

        super().__init__(*args, **kwargs)

        activation = module_config_dict.get("activation", "SiLU")
        self.bc_loss_coef = float(module_config_dict.get("bc_loss_coef", 1.0))
        self.action_head = _build_mlp(
            self.cond_dim,
            module_config_dict.get("mlp_hidden_dims", [1024, 1024, 512]),
            self.action_dim,
            activation,
        )

        # The parent owns the shared condition encoders.  Its target-stat
        # buffers remain for checkpoint compatibility, but this direct-BC
        # policy does not use them.  It must not retain the diffusion-only
        # trainable branches in its optimizer or state dict.
        del self.time_encoder
        del self.denoiser

        self.latent_normalization = module_config_dict.get("latent_normalization", "none")
        if self.latent_normalization not in ("none", "empirical"):
            raise ValueError(f"Unsupported latent_normalization={self.latent_normalization}")
        self.latent_normalizer = None
        if self.latent_normalization == "empirical":
            algo_config = kwargs.get("algo_config")
            if algo_config is None and len(args) > 4:
                algo_config = args[4]
            self.latent_dim = int((algo_config or {}).get("diffusion_latent_dim", 64))
            if self.latent_dim != 64 or self.action_dim != self.latent_dim + 2:
                raise ValueError("Empirical latent normalization requires 64 latent + 2 hand outputs")
            if self.target_normalization != "none":
                raise ValueError("Set target_normalization=none when using empirical latent normalization")
            self.latent_normalizer = EmpiricalNormalizer(
                self.latent_dim,
                eps=float(module_config_dict.get("latent_norm_eps", 1e-2)),
                until=module_config_dict.get("latent_norm_until"),
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.latent_normalizer is None and any(
            key.startswith(prefix + "latent_normalizer.") for key in state_dict
        ):
            error_msgs.append("Checkpoint requires latent_normalization=empirical")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _encode_condition(self, obs_dict):
        for key in (self.proprio_key, self.privileged_key):
            if key not in obs_dict:
                raise KeyError(f"EncoderVectorMlpPolicy requires observation key '{key}'")
        proprio = self._flatten_observation(obs_dict[self.proprio_key], self.proprio_input_dim, self.proprio_key)
        privileged = self._flatten_observation(
            obs_dict[self.privileged_key], self.privileged_input_dim, self.privileged_key
        )
        if proprio.shape[:-1] != privileged.shape[:-1]:
            raise ValueError(
                f"Observation batch shape mismatch: {self.proprio_key} has "
                f"{proprio.shape[:-1]}, {self.privileged_key} has "
                f"{privileged.shape[:-1]}"
            )
        proprio_feat = self.proprio_encoder(proprio)
        privileged_feat = self.privileged_encoder(privileged)
        return torch.cat([proprio_feat, privileged_feat], dim=-1)

    def forward(self, input, compute_aux_loss=False, **kwargs):
        if not hasattr(input, "__getitem__"):
            raise TypeError("EncoderVectorMlpPolicy expects an obs_dict-like input")

        cond = self._encode_condition(input)
        prediction = self.action_head(cond)
        pred_action = prediction
        if self.latent_normalizer is not None:
            pred_action = torch.cat(
                [
                    self.latent_normalizer.inverse(prediction[..., : self.latent_dim]),
                    prediction[..., self.latent_dim :],
                ],
                dim=-1,
            )

        if not compute_aux_loss:
            return pred_action

        target = self._extract_target(kwargs)
        if target is None:
            raise ValueError(
                f"{self.__class__.__name__} requires {self.diffusion_target_key} when compute_aux_loss=True"
            )
        target_dtype = torch.float32 if self.latent_normalizer is not None else cond.dtype
        target = target.to(device=cond.device, dtype=target_dtype)
        if target.shape[-1] != self.action_dim:
            raise ValueError(f"BC target dim mismatch: got {target.shape[-1]}, expected {self.action_dim}")
        if target.shape[:-1] != cond.shape[:-1]:
            raise ValueError(
                "BC target batch shape must match encoded condition batch shape: "
                f"got {tuple(target.shape[:-1])}, expected {tuple(cond.shape[:-1])}"
            )

        loss_target = target
        if self.latent_normalizer is not None:
            loss_target = torch.cat(
                [self.latent_normalizer.normalize(target[..., : self.latent_dim]), target[..., self.latent_dim :]],
                dim=-1,
            )
        # Preserve the 66-D mean reduction, including the two raw hand targets.
        # The Trainer commits statistics after all PPO epochs, never in forward.
        bc_loss_per_sample = (
            F.mse_loss(prediction.float(), loss_target, reduction="none")
            if self.latent_normalizer is not None
            else F.mse_loss(prediction, loss_target, reduction="none")
        ).mean(dim=-1)
        bc_loss = bc_loss_per_sample.mean()

        target_flat = target.detach().float().reshape(-1, self.action_dim)
        pred_flat = pred_action.detach().float().reshape(-1, self.action_dim)
        diagnostics = {}
        if self.latent_normalizer is not None:
            diagnostics = {
                "bc_latent/raw_mse": F.mse_loss(
                    pred_action[..., : self.latent_dim].detach(), target[..., : self.latent_dim]
                ),
                "bc_latent/normalized_mse": F.mse_loss(
                    prediction[..., : self.latent_dim].detach().float(), loss_target[..., : self.latent_dim]
                ),
                "bc_hand/raw_mse": F.mse_loss(
                    pred_action[..., self.latent_dim :].detach(), target[..., self.latent_dim :]
                ),
            }
        return {
            "action_mean": pred_action,
            "aux_losses": {
                "latent_bc_mse": bc_loss,
                "bc_target/raw_abs_mean": target_flat.abs().mean(),
                "bc_target/raw_std_mean": target_flat.std(dim=0, unbiased=False).mean(),
                "bc_pred/raw_abs_mean": pred_flat.abs().mean(),
                **diagnostics,
            },
            "aux_loss_coef": {"latent_bc_mse": self.bc_loss_coef},
            "aux_losses_per_sample": {"latent_bc_mse": bc_loss_per_sample},
        }


class EncoderJointVectorMlpPolicy(EncoderVectorMlpPolicy):
    """Direct-BC MLP over concatenated raw structured observations.

    Unlike :class:`EncoderVectorMlpPolicy`, this policy does not encode the
    proprioceptive and privileged groups independently.  It flattens both
    groups, concatenates them, and feeds the raw vector directly to the action
    MLP.
    """

    def __init__(self, *args, **kwargs):
        module_config_dict = kwargs.get("module_config_dict") or {}
        if not module_config_dict and len(args) >= 2 and args[1] is not None:
            module_config_dict = args[1]
        config = dict(module_config_dict)

        super().__init__(*args, **kwargs)

        # Remove the split feature extractors constructed by the parent.  The
        # joint policy has no trainable branch before concatenation.
        del self.proprio_encoder
        del self.privileged_encoder
        del self.action_head

        self.joint_input_dim = self.proprio_input_dim + self.privileged_input_dim
        activation = config.get("activation", "SiLU")
        self.action_head = _build_mlp(
            self.joint_input_dim,
            config.get("mlp_hidden_dims", [2048, 1800, 512]),
            self.action_dim,
            activation,
        )

    def _joint_observation(self, obs_dict):
        for key in (self.proprio_key, self.privileged_key):
            if key not in obs_dict:
                raise KeyError(
                    f"EncoderJointVectorMlpPolicy requires observation key '{key}'"
                )
        proprio = self._flatten_observation(
            obs_dict[self.proprio_key], self.proprio_input_dim, self.proprio_key
        )
        privileged = self._flatten_observation(
            obs_dict[self.privileged_key], self.privileged_input_dim, self.privileged_key
        )
        if proprio.shape[:-1] != privileged.shape[:-1]:
            raise ValueError(
                f"Observation batch shape mismatch: {self.proprio_key} has "
                f"{proprio.shape[:-1]}, {self.privileged_key} has {privileged.shape[:-1]}"
            )
        return torch.cat([proprio, privileged], dim=-1)

    def _encode_condition(self, obs_dict):
        return self._joint_observation(obs_dict)


class EncoderVectorTransformerFlowPolicy(EncoderVectorDiffusionPolicy):
    """Old-SUGAR-style RDT trained with conditional flow matching on action chunks.

    The structured observation encoders and their normalizers intentionally match
    :class:`EncoderVectorDiffusionPolicy`.  Their two 512-D outputs become distinct
    condition tokens, while the action horizon remains an explicit tensor axis.
    """

    is_action_chunk_policy = True

    def __init__(self, *args, **kwargs):
        module_config_dict = kwargs.get("module_config_dict") or {}
        if not module_config_dict and len(args) >= 2 and args[1] is not None:
            module_config_dict = args[1]
        config = dict(module_config_dict)
        super().__init__(*args, **kwargs)

        self.action_horizon = int(config.get("action_horizon", 40))
        hidden_dim = int(config.get("transformer_hidden_dim", 512))
        num_layers = int(config.get("transformer_num_layers", 14))
        num_heads = int(config.get("transformer_num_heads", 8))
        attention_dropout = float(config.get("transformer_attention_dropout", 0.1))
        gradient_checkpointing = bool(
            config.get("transformer_gradient_checkpointing", False)
        )
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.proprio_feature_dim != hidden_dim or self.privileged_feature_dim != hidden_dim:
            raise ValueError(
                "Transformer condition tokens must match transformer_hidden_dim; got "
                f"proprio={self.proprio_feature_dim}, privileged={self.privileged_feature_dim}, "
                f"hidden={hidden_dim}"
            )
        if self.diffusion_objective != "flow_matching":
            raise ValueError(
                "EncoderVectorTransformerFlowPolicy only supports flow_matching"
            )

        # Replace the one-step parent's parameterized time/MLP denoiser.  The
        # inherited normalization buffers and observation MLPs remain unchanged.
        self.time_encoder = nn.Identity()
        self.denoiser = nn.Identity()
        self.rdt = _ActionChunkRdt(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _encode_condition_tokens(self, obs_dict):
        for key in (self.proprio_key, self.privileged_key):
            if key not in obs_dict:
                raise KeyError(
                    f"EncoderVectorTransformerFlowPolicy requires observation key '{key}'"
                )
        proprio = self._flatten_observation(
            obs_dict[self.proprio_key], self.proprio_input_dim, self.proprio_key
        )
        privileged = self._flatten_observation(
            obs_dict[self.privileged_key], self.privileged_input_dim, self.privileged_key
        )
        if proprio.shape[:-1] != privileged.shape[:-1]:
            raise ValueError(
                f"Observation batch shape mismatch: {proprio.shape[:-1]} vs "
                f"{privileged.shape[:-1]}"
            )
        proprio_token = self.proprio_encoder(
            self._normalize_observation(proprio, "proprio")
        )
        privileged_token = self.privileged_encoder(
            self._normalize_observation(privileged, "privileged")
        )
        # Match SUGAR's [object, robot] token convention.
        return torch.stack([privileged_token, proprio_token], dim=-2)

    def _predict_velocity(self, noisy_action, timestep_bucket, condition_tokens):
        return self.rdt(noisy_action, timestep_bucket, condition_tokens)

    def _sample_flow_time(self, prefix_shape, device, dtype):
        """Sample GR00T N1.7 flow time, biased toward the noisy endpoint."""
        prefix_shape = tuple(prefix_shape)
        flat_count = max(int(math.prod(prefix_shape)), 1)
        sample = self.beta_dist.sample((flat_count,)).to(device=device, dtype=dtype)
        return ((1.0 - sample) * self.noise_s).reshape(prefix_shape)

    @staticmethod
    def _distributed_moments(value, feature_dim, valid_rows=None):
        flat = value.detach().reshape(-1, feature_dim).float()
        if valid_rows is not None:
            mask = valid_rows.detach().reshape(-1).bool()
            if mask.numel() != flat.shape[0]:
                raise ValueError(
                    f"valid mask has {mask.numel()} rows for tensor with {flat.shape[0]} rows"
                )
            flat = flat[mask]
        if flat.numel() == 0:
            total = torch.zeros(feature_dim, device=value.device, dtype=torch.float32)
            square_total = torch.zeros_like(total)
            count = torch.zeros((), device=value.device, dtype=torch.float64)
        else:
            total = flat.sum(dim=0)
            square_total = flat.square().sum(dim=0)
            count = torch.tensor(float(flat.shape[0]), device=value.device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total)
            torch.distributed.all_reduce(square_total)
            torch.distributed.all_reduce(count)
        if count.item() == 0:
            return None
        count_float = count.to(dtype=total.dtype)
        mean = total / count_float
        variance = (square_total / count_float - mean.square()).clamp_min(0.0)
        return mean, variance

    @staticmethod
    def _apply_ema_moments(mean_buffer, var_buffer, updates_buffer, moments, momentum, eps):
        if moments is None:
            return
        batch_mean, batch_var = moments
        batch_var = batch_var.clamp_min(eps**2)
        if int(updates_buffer.item()) == 0:
            mean_buffer.copy_(batch_mean.to(mean_buffer))
            var_buffer.copy_(batch_var.to(var_buffer))
        else:
            mean_buffer.lerp_(batch_mean.to(mean_buffer), momentum)
            var_buffer.lerp_(batch_var.to(var_buffer), momentum)
        updates_buffer.add_(1)

    @torch.no_grad()
    def update_chunk_normalizers(self, obs_dict, target, valid):
        """Update observation/target EMA stats once from globally masked moments."""
        valid = valid.bool()
        proprio = self._flatten_observation(
            obs_dict[self.proprio_key], self.proprio_input_dim, self.proprio_key
        )
        privileged = self._flatten_observation(
            obs_dict[self.privileged_key], self.privileged_input_dim, self.privileged_key
        )
        self._apply_ema_moments(
            self.proprio_mean,
            self.proprio_var,
            self.proprio_updates,
            self._distributed_moments(proprio, self.proprio_input_dim, valid),
            self.proprio_norm_momentum,
            self.proprio_std_eps,
        )
        self._apply_ema_moments(
            self.privileged_mean,
            self.privileged_var,
            self.privileged_updates,
            self._distributed_moments(privileged, self.privileged_input_dim, valid),
            self.privileged_norm_momentum,
            self.privileged_std_eps,
        )
        target_valid = valid.unsqueeze(-1).expand(*valid.shape, self.action_horizon)
        self._apply_ema_moments(
            self.target_mean,
            self.target_var,
            self.target_updates,
            self._distributed_moments(target, self.action_dim, target_valid),
            self.target_norm_momentum,
            self.target_std_eps,
        )

    def sample_action_chunk(self, obs_dict):
        condition_tokens = self._encode_condition_tokens(obs_dict)
        prefix = condition_tokens.shape[:-2]
        sample = torch.randn(
            *prefix,
            self.action_horizon,
            self.action_dim,
            device=condition_tokens.device,
            dtype=condition_tokens.dtype,
        )
        step_count = max(int(self.num_inference_steps), 1)
        dt = 1.0 / step_count
        for step in range(step_count):
            bucket = min(
                int(step / float(step_count) * self.num_timestep_buckets),
                self.num_timestep_buckets - 1,
            )
            timestep = torch.full(prefix, bucket, device=sample.device, dtype=torch.long)
            sample = sample + dt * self._predict_velocity(
                sample, timestep, condition_tokens
            )
        return self._denormalize_target(sample)

    def forward(self, input, compute_aux_loss=False, **kwargs):
        if not hasattr(input, "__getitem__"):
            raise TypeError(
                "EncoderVectorTransformerFlowPolicy expects an obs_dict-like input"
            )
        if not compute_aux_loss:
            return self.sample_action_chunk(input)

        target = self._extract_target(kwargs)
        if target is None:
            raise ValueError(
                f"{self.__class__.__name__} requires {self.diffusion_target_key}"
            )
        condition_tokens = self._encode_condition_tokens(input)
        target = target.to(device=condition_tokens.device, dtype=condition_tokens.dtype)
        expected_suffix = (self.action_horizon, self.action_dim)
        if target.shape[-2:] != expected_suffix:
            raise ValueError(
                f"diffusion target must end in {expected_suffix}, got {tuple(target.shape)}"
            )
        if target.shape[:-2] != condition_tokens.shape[:-2]:
            raise ValueError(
                f"target prefix {target.shape[:-2]} does not match observations "
                f"{condition_tokens.shape[:-2]}"
            )
        valid = kwargs.get("diffusion_target_valid")
        if valid is None:
            valid = torch.ones(target.shape[:-2], device=target.device, dtype=torch.bool)
        else:
            valid = valid.to(device=target.device, dtype=torch.bool)
        if valid.shape != target.shape[:-2]:
            raise ValueError(
                f"diffusion_target_valid shape {tuple(valid.shape)} must equal "
                f"{tuple(target.shape[:-2])}"
            )
        if kwargs.get("update_running_stats", True):
            self.update_chunk_normalizers(input, target, valid)

        sanitized_target = torch.nan_to_num(target)
        normalized_target = self._normalize_target(sanitized_target)
        noise = torch.randn_like(normalized_target)
        # One flow time per complete trajectory, shared by all 40 action tokens.
        flow_time = self._sample_flow_time(
            normalized_target.shape[:-2],
            device=target.device,
            dtype=target.dtype,
        )
        broadcast_time = flow_time.unsqueeze(-1).unsqueeze(-1)
        noisy_action = (
            (1.0 - broadcast_time) * noise + broadcast_time * normalized_target
        )
        velocity = normalized_target - noise
        timestep = (flow_time * self.num_timestep_buckets).long().clamp(
            0, self.num_timestep_buckets - 1
        )
        predicted = self._predict_velocity(noisy_action, timestep, condition_tokens)
        per_chunk_sse = (predicted - velocity).square().sum(dim=(-2, -1))
        valid_float = valid.to(per_chunk_sse.dtype)
        denominator = valid_float.sum().clamp_min(1.0) * self.action_horizon * self.action_dim
        flow_loss = (per_chunk_sse * valid_float).sum() / denominator

        valid_targets = sanitized_target[valid]
        valid_normalized = normalized_target[valid]
        if valid_targets.numel() == 0:
            raw_abs_mean = sanitized_target.sum() * 0.0
            raw_std_mean = raw_abs_mean
            norm_abs_mean = raw_abs_mean
            norm_std_mean = raw_abs_mean
        else:
            raw_flat = valid_targets.float().reshape(-1, self.action_dim)
            norm_flat = valid_normalized.float().reshape(-1, self.action_dim)
            raw_abs_mean = raw_flat.abs().mean()
            raw_std_mean = raw_flat.std(dim=0, unbiased=False).mean()
            norm_abs_mean = norm_flat.abs().mean()
            norm_std_mean = norm_flat.std(dim=0, unbiased=False).mean()
        return {
            # Keep the Actor auxiliary-loss protocol 66-D without exposing the
            # horizon to its Gaussian distribution.
            "action_mean": sanitized_target[..., 0, :].detach(),
            "aux_losses": {
                "diffusion_flow": flow_loss,
                "diffusion_target/raw_abs_mean": raw_abs_mean,
                "diffusion_target/raw_std_mean": raw_std_mean,
                "diffusion_target/norm_abs_mean": norm_abs_mean,
                "diffusion_target/norm_std_mean": norm_std_mean,
            },
            "aux_loss_coef": {"diffusion_flow": self.diffusion_loss_coef},
        }


class EncoderVectorSingleStepTransformerFlowPolicy(
    EncoderVectorTransformerFlowPolicy
):
    """One-step RDT/flow policy trained by the standard DAgger path.

    The action-chunk trainer needs a delayed target buffer when the horizon is
    greater than one.  At horizon one there is no delayed label, so this
    adapter exposes the transformer as an ordinary 66-D policy: rollout
    samples have their singleton horizon removed and regular DAgger targets
    gain that axis only while the flow-matching loss is evaluated.
    """

    is_action_chunk_policy = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.action_horizon != 1:
            raise ValueError(
                "EncoderVectorSingleStepTransformerFlowPolicy requires "
                f"action_horizon=1, got {self.action_horizon}"
            )

    def forward(self, input, compute_aux_loss=False, **kwargs):
        if not compute_aux_loss:
            return super().forward(input, compute_aux_loss=False, **kwargs).squeeze(-2)

        target = self._extract_target(kwargs)
        if target is not None:
            kwargs = dict(kwargs)
            kwargs[self.diffusion_target_key] = target.unsqueeze(-2)
        return super().forward(input, compute_aux_loss=True, **kwargs)


class EncoderRgbMlpPolicy(EncoderRgbDiffusionPolicy):
    """RGB/state latent policy trained with direct MSE, not diffusion loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        module_config_dict = kwargs.get("module_config_dict") or {}
        if not module_config_dict and len(args) >= 2 and args[1] is not None:
            module_config_dict = args[1]
        activation = module_config_dict.get("activation", "SiLU")
        self.mlp_loss_coef = float(module_config_dict.get("mlp_loss_coef", 1.0))
        self.action_head = _build_mlp(
            self.cond_dim,
            module_config_dict.get("mlp_hidden_dims", [1024, 1024, 512]),
            self.action_dim,
            activation,
        )

    def forward(self, input, compute_aux_loss=False, **kwargs):
        if not hasattr(input, "__getitem__"):
            raise TypeError("EncoderRgbMlpPolicy expects an obs_dict-like input")
        cond = self._encode_condition(input, update_state_stats=compute_aux_loss)
        pred_normalized = self.action_head(cond)
        pred_action = self._denormalize_target(pred_normalized)

        if compute_aux_loss:
            target = self._extract_target(kwargs)
            if target is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires {self.diffusion_target_key} "
                    "when compute_aux_loss=True"
                )
            target = target.to(device=cond.device, dtype=cond.dtype)
            if target.shape[-1] != self.action_dim:
                raise ValueError(
                    f"MLP target dim mismatch: got {target.shape[-1]}, expected {self.action_dim}"
                )
            with torch.no_grad():
                self._update_target_stats(target)
            normalized_target = self._normalize_target(target)
            mlp_loss = F.mse_loss(pred_normalized, normalized_target)

            target_flat = target.detach().float().reshape(-1, self.action_dim)
            norm_flat = normalized_target.detach().float().reshape(-1, self.action_dim)
            pred_flat = pred_action.detach().float().reshape(-1, self.action_dim)
            return {
                "action_mean": pred_action,
                "aux_losses": {
                    "latent_mlp_mse": mlp_loss,
                    "mlp_target/raw_abs_mean": target_flat.abs().mean(),
                    "mlp_target/raw_std_mean": target_flat.std(dim=0, unbiased=False).mean(),
                    "mlp_target/norm_abs_mean": norm_flat.abs().mean(),
                    "mlp_target/norm_std_mean": norm_flat.std(dim=0, unbiased=False).mean(),
                    "mlp_pred/raw_abs_mean": pred_flat.abs().mean(),
                },
                "aux_loss_coef": {"latent_mlp_mse": self.mlp_loss_coef},
            }

        return pred_action
