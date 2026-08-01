from __future__ import annotations

import math

import torch
import torch.nn as nn

from sugar_il.model.common.module_attr_mixin import ModuleAttrMixin


class FlowTimeEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        if frequency_embedding_size <= 0 or frequency_embedding_size % 2:
            raise ValueError("frequency_embedding_size must be a positive even integer")
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time_bucket: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half,
                device=time_bucket.device,
                dtype=torch.float32,
            )
            / half
        )
        args = time_bucket.float()[:, None] * frequencies[None]
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        return self.mlp(embedding)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(hidden_size, inner_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(inner_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(self.dropout(value * torch.nn.functional.silu(gate)))


class RDTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        ffn_ratio: int = 4,
        ffn_activation: str = "gelu",
        use_time_adaln: bool = False,
    ):
        super().__init__()
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive")
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm3 = nn.RMSNorm(hidden_size, eps=1e-6)
        inner_size = hidden_size * ffn_ratio
        if ffn_activation == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, inner_size),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(inner_size, hidden_size),
            )
        elif ffn_activation == "swiglu":
            self.ffn = SwiGLUFFN(hidden_size, inner_size, dropout)
        else:
            raise ValueError("ffn_activation must be 'gelu' or 'swiglu'")
        self.time_modulation = (
            nn.Linear(hidden_size, hidden_size * 2) if use_time_adaln else None
        )
        if self.time_modulation is not None:
            nn.init.zeros_(self.time_modulation.weight)
            nn.init.zeros_(self.time_modulation.bias)

    def _normalize(
        self,
        norm: nn.Module,
        x: torch.Tensor,
        time_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        x = norm(x)
        if self.time_modulation is None:
            return x
        if time_condition is None:
            raise ValueError("time_condition is required for AdaLN time conditioning")
        shift, scale = self.time_modulation(time_condition).chunk(2, dim=-1)
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        time_condition: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        normalized = self._normalize(self.norm1, x, time_condition)
        x = x + self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        normalized = self._normalize(self.norm2, x, time_condition)
        cross, weights = self.cross_attention(
            normalized,
            condition,
            condition,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + cross
        x = x + self.ffn(self._normalize(self.norm3, x, time_condition))
        return x, weights


class TransformerForActionFlowMatching(ModuleAttrMixin):
    """Predict a velocity field for an object-aware latent action trajectory."""

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 64,
        action_horizon: int = 40,
        n_layer: int = 12,
        n_head: int = 8,
        n_emb: int = 256,
        max_cond_tokens: int = 5,
        p_drop_attn: float = 0.1,
        frequency_embedding_size: int = 256,
        ffn_ratio: int = 4,
        ffn_activation: str = "gelu",
        time_conditioning: str = "token",
        use_action_time_encoder: bool = False,
        velocity_head_layers: int = 1,
        zero_init_velocity_head: bool = False,
    ):
        super().__init__()
        if action_horizon != 40 or input_dim != 64 or output_dim != 64:
            raise ValueError(
                "Object-aware flow model requires horizon=40 and latent_dim=64"
            )
        if n_layer <= 0 or n_head <= 0 or n_emb <= 0:
            raise ValueError("n_layer, n_head, and n_emb must be positive")
        self.action_horizon = action_horizon
        if n_emb % n_head != 0:
            raise ValueError("n_emb must be divisible by n_head")
        if time_conditioning not in {"token", "adaln"}:
            raise ValueError("time_conditioning must be 'token' or 'adaln'")
        if velocity_head_layers not in {1, 2}:
            raise ValueError("velocity_head_layers must be 1 or 2")
        self.time_conditioning = time_conditioning
        self.use_action_time_encoder = use_action_time_encoder
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.empty(1, action_horizon, n_emb))
        self.time_emb = FlowTimeEmbedder(n_emb, frequency_embedding_size)
        self.action_time_encoder = (
            nn.Sequential(
                nn.Linear(n_emb * 2, n_emb),
                nn.SiLU(),
                nn.Linear(n_emb, n_emb),
            )
            if use_action_time_encoder
            else None
        )
        self.cond_pos_emb = nn.Parameter(torch.empty(1, max_cond_tokens, n_emb))
        self.blocks = nn.ModuleList(
            RDTBlock(
                n_emb,
                n_head,
                p_drop_attn,
                ffn_ratio,
                ffn_activation,
                time_conditioning == "adaln",
            )
            for _ in range(n_layer)
        )
        self.final_norm = nn.RMSNorm(n_emb, eps=1e-6)
        if velocity_head_layers == 1:
            self.head = nn.Linear(n_emb, output_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(n_emb, n_emb),
                nn.GELU(approximate="tanh"),
                nn.Linear(n_emb, output_dim),
            )
        if zero_init_velocity_head:
            output_layer = self.head if isinstance(self.head, nn.Linear) else self.head[-1]
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cond_pos_emb, std=0.02)

    def forward(
        self,
        trajectory: torch.Tensor,
        time_bucket: torch.Tensor,
        condition: torch.Tensor,
        gen_attn_map: bool = False,
    ):
        time_bucket = time_bucket.to(trajectory.device).reshape(-1)
        if time_bucket.numel() != trajectory.shape[0]:
            raise ValueError("time_bucket must contain one value per batch item")
        time_embedding = self.time_emb(time_bucket)
        if self.time_conditioning == "token":
            condition = torch.cat((condition, time_embedding.unsqueeze(1)), dim=1)
        condition = condition + self.cond_pos_emb[:, : condition.shape[1]]
        x = self.input_emb(trajectory)
        if self.action_time_encoder is not None:
            repeated_time = time_embedding[:, None].expand(-1, x.shape[1], -1)
            x = self.action_time_encoder(torch.cat((x, repeated_time), dim=-1))
        x = x + self.pos_emb[:, : trajectory.shape[1]]
        attention_maps = [] if gen_attn_map else None
        for block in self.blocks:
            x, weights = block(
                x,
                condition,
                time_condition=(
                    time_embedding if self.time_conditioning == "adaln" else None
                ),
                need_weights=gen_attn_map,
            )
            if gen_attn_map:
                attention_maps.append(weights.detach().cpu())
        return self.head(self.final_norm(x)), attention_maps


class HandPrimitiveHead(nn.Module):
    def __init__(
        self,
        horizon: int = 40,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, horizon, hidden_size))
        layer = nn.TransformerDecoderLayer(
            hidden_size,
            num_heads,
            hidden_size * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers,
            norm=nn.LayerNorm(hidden_size),
        )
        self.output = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(condition.shape[0], -1, -1)
        return self.output(self.decoder(queries, condition))
