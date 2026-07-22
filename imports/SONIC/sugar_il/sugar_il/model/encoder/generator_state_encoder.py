from __future__ import annotations

import torch
import torch.nn as nn

from sugar_il.model.common.module_attr_mixin import ModuleAttrMixin


def _token_mlp(input_dim: int, feature_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, feature_dim),
        nn.LayerNorm(feature_dim),
    )


class GeneratorStateObsEncoder(ModuleAttrMixin):
    """Encode the four object-aware observation groups as four tokens."""

    def __init__(self, shape_meta: dict | None = None, feature_dim: int = 256, dropout: float = 0.1, last_action_dropout: float = 0.1, **_: object):
        super().__init__()
        self.feature_dim = feature_dim
        self.last_action_dropout = last_action_dropout
        self.bps_net = _token_mlp(10, feature_dim, dropout)
        self.object_net = _token_mlp(18, feature_dim, dropout)
        self.target_net = _token_mlp(18, feature_dim, dropout)
        self.last_action_net = _token_mlp(66, feature_dim, dropout)

    @staticmethod
    def _require_single_step(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,D], got {tuple(value.shape)}")
        return value[:, 0]

    def forward(self, obs_dict: dict[str, torch.Tensor], training: bool = True) -> torch.Tensor:
        get = lambda key: self._require_single_step(obs_dict[key], key)
        bps = self.bps_net(get("object_bps"))
        current = self.object_net(torch.cat((get("object_pos_b"), get("object_ori_b_6d"), get("hand_object_transform_6d")), dim=-1))
        target = self.target_net(torch.cat((get("target_object_pos_b"), get("target_object_ori_b_6d"), get("target_hand_object_transform_6d")), dim=-1))
        last_action = self.last_action_net(torch.cat((get("last_latent"), get("last_hand_primitive")), dim=-1))
        if training and self.last_action_dropout > 0:
            keep = torch.rand(last_action.shape[0], 1, device=last_action.device) >= self.last_action_dropout
            last_action = last_action * keep.to(last_action.dtype)
        return torch.stack((bps, current, target, last_action), dim=1)

    @torch.no_grad()
    def output_shape(self):
        return (1, 4, self.feature_dim), [1, 1, 1, 1]
