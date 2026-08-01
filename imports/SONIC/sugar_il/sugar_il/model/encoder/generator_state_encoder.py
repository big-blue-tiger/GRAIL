from __future__ import annotations

import torch
import torch.nn as nn

from sugar_il.model.common.module_attr_mixin import ModuleAttrMixin


def _token_mlp(
    input_dim: int,
    hidden_dim: int,
    feature_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, feature_dim),
        nn.LayerNorm(feature_dim),
    )


class GeneratorStateObsEncoder(ModuleAttrMixin):
    """Encode scene geometry, current object state, and proprioception as three tokens."""

    def __init__(
        self,
        shape_meta: dict | None = None,
        feature_dim: int = 256,
        token_hidden_dim: int = 256,
        dropout: float = 0.1,
        proprioception_dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if feature_dim <= 0 or token_hidden_dim <= 0:
            raise ValueError("feature_dim and token_hidden_dim must be positive")
        self.feature_dim = feature_dim
        self.token_hidden_dim = token_hidden_dim
        self.proprioception_dropout = proprioception_dropout
        self.scene_geometry_net = _token_mlp(
            14, token_hidden_dim, feature_dim, dropout
        )
        self.object_net = _token_mlp(26, token_hidden_dim, feature_dim, dropout)
        self.proprioception_net = _token_mlp(
            92, token_hidden_dim, feature_dim, dropout
        )

    @staticmethod
    def _require_single_step(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,D], got {tuple(value.shape)}")
        return value[:, 0]

    def forward(self, obs_dict: dict[str, torch.Tensor], training: bool = True) -> torch.Tensor:
        get = lambda key: self._require_single_step(obs_dict[key], key)
        scene_geometry = self.scene_geometry_net(
            torch.cat((get("object_bps"), get("table_geometry")), dim=-1)
        )
        current = self.object_net(torch.cat((
            get("object_pos_b"),
            get("object_ori_b_6d"),
            get("hand_object_transform_6d"),
            get("hand_object_contact_force_magnitude"),
        ), dim=-1))
        proprioception = self.proprioception_net(torch.cat(
            tuple(get(key) for key in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")),
            dim=-1,
        ))
        if training and self.proprioception_dropout > 0:
            keep = torch.rand(proprioception.shape[0], 1, device=proprioception.device) >= self.proprioception_dropout
            proprioception = proprioception * keep.to(proprioception.dtype)
        return torch.stack((scene_geometry, current, proprioception), dim=1)

    @torch.no_grad()
    def output_shape(self):
        return (1, 3, self.feature_dim), [1, 1, 1]
