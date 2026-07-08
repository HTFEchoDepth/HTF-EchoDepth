"""ResTFC-TDF block for BatVision [B,C,F,T] layout (H=F, W=T)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _make_activation(name: str) -> type[nn.Module]:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported activation: {name!r}")


class TDFModule(nn.Module):
    """Frequency MLP on [B, C, T, F]; Linear maps along F only."""

    def __init__(
        self,
        channels: int,
        f_bins: int,
        *,
        bn_factor: int = 16,
        min_bn_units: int = 16,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.f_bins = f_bins
        bn_units = max(f_bins // bn_factor, min_bn_units)
        self.bn_units = bn_units
        act = _make_activation(activation)()
        self.mlp = nn.Sequential(
            nn.Linear(f_bins, bn_units),
            nn.BatchNorm2d(channels),
            act,
            nn.Linear(bn_units, f_bins),
            nn.BatchNorm2d(channels),
            act,
        )
        self._tdf_axis = "F"
        self._layout = "[B,C,T,F]"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        return self.mlp(x)

    def tdf_linear_specs(self) -> list[tuple[str, int, int]]:
        specs: list[tuple[str, int, int]] = []
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                specs.append(("Linear", layer.in_features, layer.out_features))
        return specs


class ResTFCTDFBlock(nn.Module):
    """
    v3-like residual block: same-resolution [B,C,F,T] -> [B,C,F,T].
    TDF runs after transpose to [B,C,T,F] along frequency axis F.
    """

    def __init__(
        self,
        channels: int,
        f_bins: int,
        *,
        bn_factor: int = 16,
        min_bn_units: int = 16,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation: str = "gelu",
        block_name: str = "res_tfc_tdf",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.f_bins = f_bins
        self.block_name = block_name
        act_cls = _make_activation(activation)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(channels)
        self.tdf = TDFModule(
            channels,
            f_bins,
            bn_factor=bn_factor,
            min_bn_units=min_bn_units,
            activation=activation,
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(channels)
        self.act = act_cls()
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, x: torch.Tensor, *, return_debug: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        # BatVision layout: [B, C, F, T]
        s = x
        z = self.act(self.norm1(self.conv1(x)))
        z_tf = z.transpose(2, 3)
        z_tdf = self.tdf(z_tf).transpose(2, 3)
        z = z + z_tdf
        z = self.act(self.norm2(self.conv2(z)))
        y = s + self.res_scale * z
        if not return_debug:
            return y
        debug = {
            "block_name": self.block_name,
            "input_shape": tuple(x.shape),
            "internal_tf_shape": tuple(z_tf.shape),
            "tdf_f_bins": self.f_bins,
            "tdf_axis": self.tdf._tdf_axis,
            "tdf_layout": self.tdf._layout,
            "tdf_linear": self.tdf.tdf_linear_specs(),
            "res_scale": float(self.res_scale.detach().cpu()),
        }
        return y, debug
