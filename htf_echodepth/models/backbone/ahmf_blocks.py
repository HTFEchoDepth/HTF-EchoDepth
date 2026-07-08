"""
AHMF-Net building blocks (ported from official AD_UNet.py, structure-aligned).

See external_repos/AHMF-Net-official/models/AD_UNet.py and AHMF_PAPER_METHOD_SPEC.md.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class SDI(nn.Module):
    """Inter-layer multi-scale fusion (IMDF in paper; SDI in official code)."""

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(5)]
        )

    def forward(self, xs: list[torch.Tensor], anchor: torch.Tensor) -> torch.Tensor:
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(
                    x,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=True,
                )
            ans = ans * self.convs[i](x)
        return ans


class AmplitudeAwareProductNormSDI(nn.Module):
    """Amplitude-Aware Product-Norm SDI (AAPN-SDI v0.1-A).

    Preserves signed Hadamard product semantics; calibrates output amplitude via
    direction normalization + amplitude-aware weight + learnable injection scale:
    f = g_i * w(A) * D,  g_i = g_ref * exp(delta_i),  D = f_raw / RMS(f_raw).
    """

    def __init__(
        self,
        channel: int,
        a_ref: float,
        g_ref: float | None = None,
        *,
        lambda_amp: float = 0.5,
        tau: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(5)]
        )
        g_ref_val = float(g_ref if g_ref is not None else a_ref)
        self.register_buffer("A_ref", torch.tensor(float(a_ref), dtype=torch.float32))
        self.register_buffer("g_ref", torch.tensor(g_ref_val, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.lambda_amp = float(lambda_amp)
        self.tau = float(tau)
        self.eps = float(eps)

    @staticmethod
    def _global_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt((x**2).mean(dim=(1, 2, 3), keepdim=True) + eps)

    def _align_and_smooth(
        self, xs: list[torch.Tensor], anchor: torch.Tensor
    ) -> list[torch.Tensor]:
        target_size = anchor.shape[-1]
        hs: list[torch.Tensor] = []
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(
                    x,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=True,
                )
            hs.append(self.convs[i](x))
        return hs

    def _hadamard_product(self, hs: list[torch.Tensor]) -> torch.Tensor:
        f_raw = hs[0]
        for h in hs[1:]:
            f_raw = f_raw * h
        return f_raw

    def forward_with_diagnostics(
        self, xs: list[torch.Tensor], anchor: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        hs = self._align_and_smooth(xs, anchor)
        f_raw = self._hadamard_product(hs)
        amplitude = self._global_rms(f_raw, self.eps).detach()
        direction = f_raw / (amplitude + self.eps)
        ratio = amplitude / (self.A_ref + self.eps)
        weight = 1.0 + self.lambda_amp * torch.tanh(torch.log(ratio + self.eps) / self.tau)
        g_i = self.g_ref * torch.exp(self.delta)
        f_out = g_i * weight * direction
        return {
            "hs": hs,
            "f_raw": f_raw,
            "f_raw_rms": amplitude,
            "A": amplitude,
            "A_ref": self.A_ref,
            "D": direction,
            "r": ratio,
            "w": weight,
            "g_ref": self.g_ref,
            "delta_i": self.delta,
            "g_i": g_i,
            "f_out": f_out,
        }

    def forward(self, xs: list[torch.Tensor], anchor: torch.Tensor) -> torch.Tensor:
        return self.forward_with_diagnostics(xs, anchor)["f_out"]


class EfficientChannelAttention(nn.Module):
    def __init__(self, c: int, b: int = 1, gamma: int = 2) -> None:
        super().__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )
        self.channel = EfficientChannelAttention(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.residual_function(x)
        eca_out = self.channel(out)
        out = out * eca_out
        return nn.ReLU(inplace=True)(out + self.shortcut(x))


class ASPP(nn.Module):
    """Official AHMF ASPP: dilations 1, 6, 12, 18 + global pooling."""

    def __init__(self, in_channel: int, depth: int) -> None:
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode="bilinear", align_corners=False)
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        return self.conv_1x1_output(
            torch.cat(
                [image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18],
                dim=1,
            )
        )


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first(x)
        return self.second(out)
