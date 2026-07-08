"""TFC-TDF v3 block-aligned stage block for BatVision [B,C,F,T] layout."""
from __future__ import annotations

import torch
import torch.nn as nn


def _get_norm(norm_type: str):
    """Match official v3 get_norm() factory (tfc_tdf_v3.py L39-50)."""

    def norm(c: int) -> nn.Module:
        if norm_type in ("BatchNorm", "batch"):
            return nn.BatchNorm2d(c)
        if norm_type in ("InstanceNorm", "instance"):
            return nn.InstanceNorm2d(c, affine=True)
        if "GroupNorm" in norm_type:
            g = int(norm_type.replace("GroupNorm", ""))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        raise ValueError(f"Unsupported norm: {norm_type!r}")

    return norm


def _get_act(act_type: str) -> nn.Module:
    """Match official v3 get_act() (tfc_tdf_v3.py L53-62)."""
    act_type = str(act_type).lower()
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "relu":
        return nn.ReLU()
    if act_type.startswith("elu") and act_type != "gelu":
        alpha = float(act_type.replace("elu", ""))
        return nn.ELU(alpha)
    raise ValueError(f"Unsupported activation: {act_type!r}")


class TDFModuleV3Aligned(nn.Module):
    """Official v3 TDF on [B,C,T,F]: Norm→Act→Linear(F→F/bn)→Norm→Act→Linear."""

    def __init__(
        self,
        channels: int,
        f_bins: int,
        *,
        bn_factor: int = 16,
        norm_type: str = "BatchNorm",
        act_type: str = "gelu",
    ) -> None:
        super().__init__()
        norm = _get_norm(norm_type)
        self.channels = channels
        self.f_bins = f_bins
        self.bn_units = f_bins // bn_factor
        self.net = nn.Sequential(
            norm(channels),
            _get_act(act_type),
            nn.Linear(f_bins, self.bn_units, bias=False),
            norm(channels),
            _get_act(act_type),
            nn.Linear(self.bn_units, f_bins, bias=False),
        )
        self._tdf_axis = "F"
        self._layout = "[B,C,T,F]"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def tdf_linear_specs(self) -> list[tuple[str, int, int]]:
        specs: list[tuple[str, int, int]] = []
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                specs.append(("Linear", layer.in_features, layer.out_features))
        return specs


class ResTFCTDFBlockV3Aligned(nn.Module):
    """
    Single inner block from official TFC_TDF (kuielab/sdx23 tfc_tdf_v3.py L96-128).

    External layout: [B,C,F,T] (BatVision).
    Internal layout: [B,C,T,F] (official v3 after net-level transpose).
    """

    def __init__(
        self,
        channels: int,
        f_bins: int,
        *,
        in_channels: int | None = None,
        bn_factor: int = 16,
        norm_type: str = "BatchNorm",
        act_type: str = "gelu",
        block_name: str = "res_tfc_tdf_v3_aligned",
    ) -> None:
        super().__init__()
        in_c = channels if in_channels is None else in_channels
        c = channels
        norm = _get_norm(norm_type)

        self.in_channels = in_c
        self.channels = c
        self.f_bins = f_bins
        self.block_name = block_name
        self.bn_factor = bn_factor
        self.norm_type = norm_type
        self.act_type = act_type

        self.tfc1 = nn.Sequential(
            norm(in_c),
            _get_act(act_type),
            nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
        )
        self.tdf = nn.Sequential(
            norm(c),
            _get_act(act_type),
            nn.Linear(f_bins, f_bins // bn_factor, bias=False),
            norm(c),
            _get_act(act_type),
            nn.Linear(f_bins // bn_factor, f_bins, bias=False),
        )
        self.tfc2 = nn.Sequential(
            norm(c),
            _get_act(act_type),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
        )
        self.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

    def forward(
        self, x: torch.Tensor, *, return_debug: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        x_tf = x.transpose(2, 3)
        s = self.shortcut(x_tf)
        x_tf = self.tfc1(x_tf)
        x_tf = x_tf + self.tdf(x_tf)
        x_tf = self.tfc2(x_tf)
        y_tf = x_tf + s
        y = y_tf.transpose(2, 3)

        if not return_debug:
            return y

        debug = {
            "block_name": self.block_name,
            "input_shape": tuple(x.shape),
            "internal_tf_shape": tuple(x_tf.shape),
            "tdf_f_bins": self.f_bins,
            "tdf_axis": "F",
            "tdf_layout": "[B,C,T,F]",
            "tdf_linear": self.tdf_linear_specs(),
            "has_res_scale": False,
            "shortcut": "Conv2d 1x1",
            "forward_order": "shortcut; tfc1; +tdf; tfc2; +shortcut",
        }
        return y, debug

    def tdf_linear_specs(self) -> list[tuple[str, int, int]]:
        specs: list[tuple[str, int, int]] = []
        for layer in self.tdf:
            if isinstance(layer, nn.Linear):
                specs.append(("Linear", layer.in_features, layer.out_features))
        return specs
