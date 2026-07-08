"""FA-DTF-TDF v3 V1 block: DTF-TDF v3 V0 + FAC-core-lite before DTFSpatialConv2d."""
from __future__ import annotations

import torch
import torch.nn as nn

from htf_echodepth.models.backbone.dtf_spatial_conv_v0 import (
    DTFSpatialConv2d,
    INTERNAL_LAYOUT,
    KERNEL_DTFAT_FAITHFUL_FREQ,
    KERNEL_DTFAT_FAITHFUL_TIME,
    KERNEL_PRACTICAL_ODD_FREQ,
    KERNEL_PRACTICAL_ODD_TIME,
)
from htf_echodepth.models.backbone.fac_core_lite_v1 import FACCoreLiteV1
from htf_echodepth.models.backbone.res_tfc_tdf_block_v3_aligned import _get_act, _get_norm


class ResFADTFTDFBlockV3V1(nn.Module):
    """
    TFC-TDF v3 aligned block with FAC-core-lite + DTF faithful replacement at tfc1.

    External layout: [B,C,F,T] (BatVision).
    Internal layout: [B,C,T,F] (official v3).

    Forward:
        s = shortcut(X)
        U = Norm→Act(X)
        U_f = FACCoreLiteV1(U)
        Z = DTFSpatialConv2d(U_f)
        Z = Z + TDF(Z)
        Z = Norm→Act→Conv3x3(Z)
        Y = Z + s
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
        block_name: str = "res_fa_dtf_tdf_v3_v1",
        time_kernel: tuple[int, int] = KERNEL_DTFAT_FAITHFUL_TIME,
        freq_kernel: tuple[int, int] = KERNEL_DTFAT_FAITHFUL_FREQ,
        alpha_init: float = 0.5,
        padding_mode: str = "same",
        fac_reduction: int = 4,
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
        self.time_kernel = time_kernel
        self.freq_kernel = freq_kernel

        self.tfc1_norm = norm(in_c)
        self.tfc1_act = _get_act(act_type)
        self.fac = FACCoreLiteV1(in_c, f_bins, reduction=fac_reduction)
        self.tfc1_dtf = DTFSpatialConv2d(
            in_c,
            c,
            time_kernel=time_kernel,
            freq_kernel=freq_kernel,
            alpha_init=alpha_init,
            padding_mode=padding_mode,
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
        u = self.tfc1_norm(x_tf)
        u = self.tfc1_act(u)
        if return_debug:
            u_f, fac_dbg = self.fac(u, return_debug=True)
        else:
            u_f = self.fac(u)
            fac_dbg = None
        x_tf = self.tfc1_dtf(u_f)
        x_tf = x_tf + self.tdf(x_tf)
        x_tf = self.tfc2(x_tf)
        y_tf = x_tf + s
        y = y_tf.transpose(2, 3)

        if not return_debug:
            return y

        with torch.no_grad():
            _, dtf_dbg = self.tfc1_dtf(u_f, return_debug=True)

        debug = {
            "block_name": self.block_name,
            "input_shape": tuple(x.shape),
            "internal_tf_shape": tuple(x_tf.shape),
            "external_layout": "[B,C,F,T]",
            "internal_layout": INTERNAL_LAYOUT,
            "tdf_f_bins": self.f_bins,
            "tdf_axis": "F",
            "tdf_layout": INTERNAL_LAYOUT,
            "tdf_linear": self.tdf_linear_specs(),
            "has_res_scale": False,
            "shortcut": "Conv2d 1x1",
            "tfc1_replacement": "FACCoreLiteV1 -> DTFSpatialConv2d",
            "tfc2_type": type(self.tfc2[2]).__name__,
            "forward_order": (
                "shortcut; Norm→Act→FACCoreLiteV1→DTFSpatialConv2d; "
                "+tdf; Norm→Act→Conv3x3; +shortcut"
            ),
            "fac_core": fac_dbg,
            "dtf_spatial": dtf_dbg,
        }
        return y, debug

    def tdf_linear_specs(self) -> list[tuple[str, int, int]]:
        specs: list[tuple[str, int, int]] = []
        for layer in self.tdf:
            if isinstance(layer, nn.Linear):
                specs.append(("Linear", layer.in_features, layer.out_features))
        return specs


def resolve_dtf_kernels(kernel_mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if kernel_mode == "practical_odd":
        return KERNEL_PRACTICAL_ODD_TIME, KERNEL_PRACTICAL_ODD_FREQ
    return KERNEL_DTFAT_FAITHFUL_TIME, KERNEL_DTFAT_FAITHFUL_FREQ
