"""
Transformer bottleneck for BatVision U-Net (C-line).

T8: global self-attention on deep encoder features (B, 512, 8, 8) -> 64 tokens.
Residual form y = x + gamma * T(x) with gamma initialized to 0 for identity start.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBottleneckT8(nn.Module):
    """Transformer encoder on 512x8x8 feature maps (64 spatial tokens)."""

    SPATIAL_SIZE = 8
    NUM_TOKENS = SPATIAL_SIZE * SPATIAL_SIZE  # 64
    CHANNELS = 512

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model != self.CHANNELS:
            raise ValueError(f"TransformerBottleneckT8 expects d_model={self.CHANNELS}, got {d_model}")

        # Learnable 2D position encoding for 8x8 grid -> 64 tokens, each dim 512.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.NUM_TOKENS, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Identity-safe residual scale; init_net does not reset nn.Parameter scalars.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.CHANNELS or h != self.SPATIAL_SIZE or w != self.SPATIAL_SIZE:
            raise ValueError(
                f"TransformerBottleneckT8 expects (B,{self.CHANNELS},{self.SPATIAL_SIZE},{self.SPATIAL_SIZE}), "
                f"got {tuple(x.shape)}"
            )

        tokens = x.flatten(2).transpose(1, 2)  # (B, 64, 512)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        tx = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * tx


def reapply_transformer_bottleneck_gamma_zero(net: nn.Module) -> None:
    """Ensure gamma=0 after define_G init_net (safety; gamma is not Conv/Linear)."""
    core = net.module if hasattr(net, "module") else net
    for m in core.modules():
        if isinstance(m, TransformerBottleneckT8):
            nn.init.constant_(m.gamma, 0.0)
