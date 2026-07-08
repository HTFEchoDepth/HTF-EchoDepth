"""
FADC full-single layer for BatVision U-Net (unet_fadc_full_single).

FreqSelect (FFT + lowfreq_att) + AdaKern (mean/res kernel + dual OmniAttention)
+ spatially adaptive AdaDR via torchvision.ops.deform_conv2d (no mmcv).

See FADC_FULL_SINGLE_IMPLEMENTATION_SPEC.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from htf_echodepth.models.backbone.fadc_core import (
    AdaKernCoupledDeform,
    FrequencySelectionFull,
    OmniAttentionLite,
    SpatialAdaDRDeform,
)

# Re-export for backward compatibility (scripts import from this module).
__all__ = [
    "FrequencySelectionFull",
    "OmniAttentionLite",
    "SpatialAdaDRDeform",
    "AdaKernCoupledDeform",
    "FreqRFBlockFullSingle",
    "reapply_fadc_full_single_zero_init",
]


class FreqRFBlockFullSingle(nn.Module):
    """Full FADC block @ (B, 256, 32, 32): FreqSelect -> AdaKern+AdaDR -> residual."""

    C = 256
    H = 32
    W = 32

    def __init__(self, max_dilation: float = 4.0) -> None:
        super().__init__()
        self.fs = FrequencySelectionFull(self.C, lowfreq_att=True)
        self.adakern_adadr = AdaKernCoupledDeform(self.C, self.C, max_dilation=max_dilation)
        self.proj = nn.Conv2d(self.C, self.C, kernel_size=1, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def zero_init_freq_weight_convs(self) -> None:
        self.fs._zero_init_freq_convs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == self.C and h == self.H and w == self.W, (
            f"FreqRFBlockFullSingle expects (B,{self.C},{self.H},{self.W}), got {(b, c, h, w)}"
        )
        x_freq = self.fs(x)
        y_core, _ = self.adakern_adadr(x_freq)
        return x + self.beta * self.proj(y_core)


def reapply_fadc_full_single_zero_init(net: nn.Module) -> None:
    """Re-apply FreqSelect zero init after init_net overwrites conv weights."""
    root = net.module if hasattr(net, "module") else net
    for m in root.modules():
        if isinstance(m, FreqRFBlockFullSingle):
            m.zero_init_freq_weight_convs()
