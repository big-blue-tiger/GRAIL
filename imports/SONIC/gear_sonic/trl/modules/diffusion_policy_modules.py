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
from torchvision import models


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
            nn.Linear(timestep_dim, timestep_dim * 4),
            getattr(nn, activation)(),
            nn.Linear(timestep_dim * 4, timestep_dim),
        )
        self.denoiser = _build_mlp(
            self.action_dim + cond_dim + timestep_dim,
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
    """Flow-matching latent policy conditioned on one structured vector.

    This class intentionally reuses ``EncoderRgbDiffusionPolicy``'s diffusion
    target, normalization, flow-matching, DDPM fallback, and sampling logic.
    The RGB encoder is disabled before the parent is constructed, so no
    ResNet or image buffers are created.  The condition path is a direct
    ``observation_input_dim -> cond_dim`` MLP, matching the RGB policy's
    post-encoder condition head.
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
        self.observation_key = config.get("observation_key", "student_obs")
        self.observation_input_dim = int(config.get("observation_input_dim", 138))
        if obs_dim_dict is not None and self.observation_key in obs_dim_dict:
            configured_dim = obs_dim_dict[self.observation_key]
            if not isinstance(configured_dim, int):
                configured_dim = int(torch.tensor(configured_dim).prod().item())
            if configured_dim != self.observation_input_dim:
                raise ValueError(
                    f"{self.observation_key} configured dim mismatch: "
                    f"observation manager reports {configured_dim}, "
                    f"policy expects {self.observation_input_dim}"
                )
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
            "proprio_key",
            "proprio_feature_dim",
            "proprio_hidden_dims",
            "fusion_hidden_dims",
            "replace_terms",
        ):
            config.pop(visual_key, None)
        config["_skip_image_encoder"] = True
        # The parent uses image_feature_dim as the input width of cond_encoder;
        # repurpose that width for the structured vector without constructing an
        # image encoder.  state_key is deliberately disabled for this policy.
        config["image_feature_dim"] = self.observation_input_dim
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

        normalization = config.get("observation_normalization", "standardize")
        if normalization not in ("none", "standardize"):
            raise ValueError(
                "Unsupported observation_normalization="
                f"{normalization}; expected 'none' or 'standardize'"
            )
        self.observation_normalization = normalization
        self.observation_norm_momentum = float(
            config.get("observation_norm_momentum", 0.05)
        )
        self.observation_norm_clip = float(config.get("observation_norm_clip", 5.0))
        self.observation_std_eps = float(config.get("observation_std_eps", 1.0e-4))
        self.register_buffer("observation_mean", torch.zeros(self.observation_input_dim))
        self.register_buffer("observation_var", torch.ones(self.observation_input_dim))
        self.register_buffer("observation_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update_observation_stats(self, observation):
        if self.observation_normalization != "standardize":
            return
        flat = observation.detach().reshape(-1, self.observation_input_dim).float()
        if flat.numel() == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False).clamp_min(self.observation_std_eps**2)
        if int(self.observation_updates.item()) == 0:
            self.observation_mean.copy_(batch_mean.to(self.observation_mean))
            self.observation_var.copy_(batch_var.to(self.observation_var))
        else:
            momentum = self.observation_norm_momentum
            self.observation_mean.lerp_(batch_mean.to(self.observation_mean), momentum)
            self.observation_var.lerp_(batch_var.to(self.observation_var), momentum)
        self.observation_updates += 1

    def _normalize_observation(self, observation):
        if self.observation_normalization == "none":
            return observation
        mean = self.observation_mean.to(device=observation.device, dtype=observation.dtype)
        std = self.observation_var.clamp_min(self.observation_std_eps**2).sqrt().to(
            device=observation.device, dtype=observation.dtype
        )
        normalized = (observation - mean) / std
        if self.observation_norm_clip > 0:
            normalized = normalized.clamp(-self.observation_norm_clip, self.observation_norm_clip)
        return normalized

    def _encode_condition(self, obs_dict, update_state_stats=False):
        if self.observation_key not in obs_dict:
            raise KeyError(
                f"EncoderVectorDiffusionPolicy requires observation key "
                f"'{self.observation_key}'"
            )
        observation = obs_dict[self.observation_key].float()
        if observation.shape[-1] != self.observation_input_dim:
            observation = observation.reshape(*observation.shape[:-1], -1)
        if observation.shape[-1] != self.observation_input_dim:
            raise ValueError(
                f"{self.observation_key} dim mismatch: got {observation.shape[-1]}, "
                f"expected {self.observation_input_dim}"
            )
        if update_state_stats:
            self._update_observation_stats(observation)
        return self.cond_encoder(self._normalize_observation(observation))


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
