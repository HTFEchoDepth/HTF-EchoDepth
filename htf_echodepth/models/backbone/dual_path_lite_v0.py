"""DualPathLite v0 — DTTNet-inspired latent dual-axis RNN module for BatVision."""
from __future__ import annotations

import torch
import torch.nn as nn

EXTERNAL_LAYOUT = "[B,C,F,T]"
INTERNAL_LAYOUT = "[B,C,T,F]"
DEFAULT_N_HEADS = 4
DEFAULT_NUM_LAYERS = 1
DEFAULT_HIDDEN_RATIO = 2
DEFAULT_BETA_DP_INIT = 0.0


class DualPathRNNBlock(nn.Module):
    """
    Norm → BLSTM → FC → residual add along one axis.

    DTTNet RNNModule adaptation for BatVision [B, K, T, N] tensors where
    K indexes the fixed axis (F for time-path, T for freq-path) and T is the
    sequence axis (T for time-path, F for freq-path).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        group_num: int | None = None,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        g = group_num if group_num is not None else max(feature_dim // 16, 1)
        self.groupnorm = nn.GroupNorm(g, feature_dim)
        self.rnn = nn.LSTM(
            feature_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        rnn_out = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_out, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, K, T, N] — sequence along dim T."""
        b, k, t, n = x.shape
        out = x.reshape(b * k, t, n)
        out = self.groupnorm(out.transpose(-1, -2)).transpose(-1, -2)
        out = self.rnn(out)[0]
        out = self.fc(out)
        out = out.view(b, k, t, n) + x
        return out


class DualPathLiteV0(nn.Module):
    """
    Minimal DTTNet IDPM for BatVision latent features.

    External I/O: [B,C,F,T]. Internal: [B,C,T,F] with split heads, time-path
    then freq-path BLSTM blocks (BandSequenceModelModule-style).
    Returns dual-path residual only (caller applies beta_dp gate).
    """

    STAGE_MASK = ("enc4",)

    def __init__(
        self,
        channels: int,
        *,
        n_heads: int = DEFAULT_N_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        hidden_ratio: int = DEFAULT_HIDDEN_RATIO,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(f"channels {channels} must be divisible by n_heads {n_heads}")
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.num_layers = num_layers
        self.hidden_ratio = hidden_ratio
        self.external_layout = EXTERNAL_LAYOUT
        self.internal_layout = INTERNAL_LAYOUT

        head_dim = self.head_dim
        hidden_dim = head_dim * hidden_ratio
        group_num = max(head_dim // 16, 1)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "time": DualPathRNNBlock(head_dim, hidden_dim, group_num=group_num),
                        "freq": DualPathRNNBlock(head_dim, hidden_dim, group_num=group_num),
                    }
                )
            )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _split_heads(self, x_tf: torch.Tensor) -> torch.Tensor:
        """[B,C,T,F] -> [B*n_heads, head_dim, T, F]."""
        b, c, t, f = x_tf.shape
        return x_tf.view(b * self.n_heads, self.head_dim, t, f)

    def _merge_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """[B*n_heads, head_dim, T, F] -> [B,C,T,F]."""
        _, head_dim, t, f = x.shape
        return x.view(batch_size, self.channels, t, f)

    def _dual_path_layer(self, x: torch.Tensor, layer: nn.ModuleDict) -> torch.Tensor:
        """
        One IDPM layer on [B*n_heads, head_dim, T, F].

        Permute to DTTNet layout [BK, F, T, head_dim], time-path along T,
        freq-path along F, then permute back.
        """
        x = x.permute(0, 3, 2, 1).contiguous()  # [BK, F, T, head_dim]
        x = layer["time"](x)  # sequence along T, K=F fixed
        x = x.permute(0, 2, 1, 3).contiguous()  # [BK, T, F, head_dim]
        x = layer["freq"](x)  # sequence along F, K=T fixed
        x = x.permute(0, 3, 2, 1).contiguous()  # [BK, head_dim, T, F]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B,C,F,T] BatVision layout
        Returns:
            residual: [B,C,F,T] dual-path delta (not including input)
        """
        b = x.shape[0]
        x_tf = x.transpose(2, 3).contiguous()  # [B,C,T,F]
        x_in = self._split_heads(x_tf)

        y = x_in
        for layer in self.layers:
            y = self._dual_path_layer(y, layer)
        y = self.dropout(y)

        residual_tf = y - x_in
        residual_tf = self._merge_heads(residual_tf, b)
        return residual_tf.transpose(2, 3).contiguous()
