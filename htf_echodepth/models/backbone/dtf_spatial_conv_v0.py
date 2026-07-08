"""DTF-AT faithful spatial conv primitive for TFC-TDF v3 internal [B,C,T,F] layout."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

VARIANT = "dtf_spatial_conv_faithful"

# Internal TFC-TDF v3 block layout after transpose: [B,C,T,F] -> Conv2d H=T, W=F
INTERNAL_LAYOUT = "[B,C,T,F]"
TIME_AXIS = "T (dim=2 / height)"
FREQ_AXIS = "F (dim=3 / width)"

# DTF-AT kernels on [B,C,T,F]: time elongated along T -> (6,3); freq along F -> (3,6)
KERNEL_DTFAT_FAITHFUL_TIME = (6, 3)
KERNEL_DTFAT_FAITHFUL_FREQ = (3, 6)
KERNEL_PRACTICAL_ODD_TIME = (7, 3)
KERNEL_PRACTICAL_ODD_FREQ = (3, 7)


class DTFSpatialConv2d(nn.Module):
    """
    Conv3x3 replacement: depthwise time/freq branches + alpha fusion + 1x1 channel mix.

    Runs on internal block layout [B,C,T,F]. No shortcut / no MBConv expand-project skeleton.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_kernel: tuple[int, int] = KERNEL_DTFAT_FAITHFUL_TIME,
        freq_kernel: tuple[int, int] = KERNEL_DTFAT_FAITHFUL_FREQ,
        alpha_init: float = 0.5,
        padding_mode: str = "same",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_kernel = time_kernel
        self.freq_kernel = freq_kernel
        self.padding_mode = padding_mode
        self.dw_time = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=time_kernel,
            stride=1,
            padding=padding_mode,
            groups=in_channels,
            bias=False,
        )
        self.dw_freq = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=freq_kernel,
            stride=1,
            padding=padding_mode,
            groups=in_channels,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(
        self, x: torch.Tensor, *, return_debug: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        z_t = self.dw_time(x)
        z_f = self.dw_freq(x)
        a = self.alpha
        z = a * z_t + (1.0 - a) * z_f
        y = self.pw(z)
        if not return_debug:
            return y
        return y, {
            "layout": INTERNAL_LAYOUT,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "time_kernel": self.time_kernel,
            "freq_kernel": self.freq_kernel,
            "groups_time": self.dw_time.groups,
            "groups_freq": self.dw_freq.groups,
            "alpha": float(a.detach().cpu()),
            "z_t_rms": float(z_t.detach().pow(2).mean().sqrt().cpu()),
            "z_f_rms": float(z_f.detach().pow(2).mean().sqrt().cpu()),
        }
