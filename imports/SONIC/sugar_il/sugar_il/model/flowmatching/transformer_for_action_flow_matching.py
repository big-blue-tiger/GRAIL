from __future__ import annotations

import math

import torch
import torch.nn as nn

from sugar_il.model.common.module_attr_mixin import ModuleAttrMixin


class FlowTimeEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
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


class RDTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
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
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        need_weights: bool = False,
    ):
        normalized = self.norm1(x)
        x = x + self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        normalized = self.norm2(x)
        cross, weights = self.cross_attention(
            normalized,
            condition,
            condition,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + cross
        x = x + self.ffn(self.norm3(x))
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
    ):
        super().__init__()
        if action_horizon != 40 or input_dim != 64 or output_dim != 64:
            raise ValueError(
                "Object-aware flow model requires horizon=40 and latent_dim=64"
            )
        self.action_horizon = action_horizon
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.empty(1, action_horizon, n_emb))
        self.time_emb = FlowTimeEmbedder(n_emb)
        self.cond_pos_emb = nn.Parameter(torch.empty(1, max_cond_tokens, n_emb))
        self.blocks = nn.ModuleList(
            RDTBlock(n_emb, n_head, p_drop_attn) for _ in range(n_layer)
        )
        self.final_norm = nn.RMSNorm(n_emb, eps=1e-6)
        self.head = nn.Linear(n_emb, output_dim)
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
        condition = torch.cat(
            (condition, self.time_emb(time_bucket).unsqueeze(1)),
            dim=1,
        )
        condition = condition + self.cond_pos_emb[:, : condition.shape[1]]
        x = self.input_emb(trajectory) + self.pos_emb[:, : trajectory.shape[1]]
        attention_maps = [] if gen_attn_map else None
        for block in self.blocks:
            x, weights = block(x, condition, gen_attn_map)
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
