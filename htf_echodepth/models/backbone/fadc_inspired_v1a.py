"""
FADC-inspired v1a: official-semantics FrequencySelection (freq path) + shared-kernel discrete AdaDR.
Frozen spec — see project FADC_v1a implementation spec. Pure PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySelectionV1a(nn.Module):
    """FreqSelect-v1a: matches Linwei-Chen/FADC FADC_only/conv_custom.py FrequencySelection lp_type='freq' branch."""

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.k_list = [2, 4, 8]
        self.spatial_group = 1
        self.lowfreq_att = False
        self.freq_weight_conv_list = nn.ModuleList()
        for _ in range(len(self.k_list)):
            self.freq_weight_conv_list.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.spatial_group,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=self.spatial_group,
                    bias=True,
                )
            )
        self._zero_init_freq_convs()

    def _zero_init_freq_convs(self) -> None:
        for m in self.freq_weight_conv_list:
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def sp_act(freq_weight: torch.Tensor) -> torch.Tensor:
        return freq_weight.sigmoid() * 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_feat = x
        b, _, h, w = x.shape
        pre_x = x.clone()
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
        x_list = []
        for idx, freq in enumerate(self.k_list):
            mask = torch.zeros((b, 1, h, w), device=x.device, dtype=torch.float32)
            h0 = round(h / 2 - h / (2 * freq))
            h1 = round(h / 2 + h / (2 * freq))
            w0 = round(w / 2 - w / (2 * freq))
            w1 = round(w / 2 + w / (2 * freq))
            mask[:, :, h0:h1, w0:w1] = 1.0
            low_part = torch.fft.ifft2(
                torch.fft.ifftshift(x_fft * mask.to(dtype=x_fft.dtype)), norm="ortho"
            ).real
            high_part = pre_x - low_part
            pre_x = low_part
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(
                b, self.spatial_group, -1, h, w
            )
            x_list.append(tmp.reshape(b, -1, h, w))
        if self.lowfreq_att:
            raise RuntimeError("FrequencySelectionV1a: lowfreq_att must stay False per frozen spec.")
        x_list.append(pre_x)
        return sum(x_list)


class FreqRFBlockV1a(nn.Module):
    """Insert after encoder down at 32×32 with 256 channels (B,256,32,32)."""

    C = 256
    H = 32
    W = 32

    def __init__(self) -> None:
        super().__init__()
        self.fs = FrequencySelectionV1a(self.C)
        self.shared_conv = nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.logits_head = nn.Conv2d(self.C, 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(self.C, self.C, kernel_size=1, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def zero_init_freq_weight_convs(self) -> None:
        """Re-apply zero init after init_net (normal) overwrites freq_weight_conv."""
        self.fs._zero_init_freq_convs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == self.C and h == self.H and w == self.W, (
            f"FreqRFBlockV1a expects (B,{self.C},{self.H},{self.W}), got {(b, c, h, w)}"
        )
        x_freq = self.fs(x)
        weight = self.shared_conv.weight
        bias = self.shared_conv.bias
        y1 = F.conv2d(x_freq, weight, bias, stride=1, padding=1, dilation=1)
        y2 = F.conv2d(x_freq, weight, bias, stride=1, padding=2, dilation=2)
        y4 = F.conv2d(x_freq, weight, bias, stride=1, padding=4, dilation=4)
        logits = self.logits_head(x_freq)
        a = F.softmax(logits, dim=1)
        a1 = a[:, 0:1, :, :]
        a2 = a[:, 1:2, :, :]
        a4 = a[:, 2:3, :, :]
        y_fused = a1 * y1 + a2 * y2 + a4 * y4
        return x + self.beta * self.proj(y_fused)


def reapply_freqselect_v1a_zero_init(net: nn.Module) -> None:
    """Walk module tree (including DataParallel .module) and zero freq_weight convs in v1a blocks."""
    root = net.module if hasattr(net, "module") else net
    for m in root.modules():
        if isinstance(m, FreqRFBlockV1a):
            m.zero_init_freq_weight_convs()
