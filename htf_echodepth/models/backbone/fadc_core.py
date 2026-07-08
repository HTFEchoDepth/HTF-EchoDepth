"""
Reusable FADC mechanism modules (FreqSelect, AdaDR, AdaKern).

Used by U-Net full_single (outer residual) and AHMF FADCContextCore (ASPP replacement).
See FADC_FULL_SINGLE_IMPLEMENTATION_SPEC.md and AHMF_FADC_V0_IMPLEMENTATION_SPEC.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.ops import deform_conv2d as _tv_deform_conv2d
except ImportError:  # pragma: no cover
    _tv_deform_conv2d = None


_KERNEL_3X3_OFFSET_TEMPLATE = torch.tensor(
    [
        [-1.0, -1.0],
        [-1.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)


class FrequencySelectionFull(nn.Module):
    """Feature-space FreqSelect: FFT band split + spatial reweighting."""

    def __init__(
        self,
        in_channels: int,
        k_list: list[int] | None = None,
        lowfreq_att: bool = True,
        spatial_group: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.k_list = list(k_list or [2, 4, 8])
        self.lowfreq_att = lowfreq_att
        self.spatial_group = spatial_group
        n_convs = len(self.k_list) + (1 if self.lowfreq_att else 0)
        self.freq_weight_conv_list = nn.ModuleList()
        for _ in range(n_convs):
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
        x_list: list[torch.Tensor] = []
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
            freq_weight = self.sp_act(self.freq_weight_conv_list[idx](att_feat))
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(
                b, self.spatial_group, -1, h, w
            )
            x_list.append(tmp.reshape(b, -1, h, w))
        if self.lowfreq_att:
            freq_weight = self.sp_act(self.freq_weight_conv_list[len(self.k_list)](att_feat))
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(
                b, self.spatial_group, -1, h, w
            )
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)
        return sum(x_list)


class OmniAttentionLite(nn.Module):
    """AdaKern attention: global pool -> channel + filter gates."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        reduction: float = 0.0625,
        min_channel: int = 16,
    ) -> None:
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.relu(self.bn(self.fc(self.avgpool(x))))
        c_att = torch.sigmoid(self.channel_fc(t))
        f_att = torch.sigmoid(self.filter_fc(t))
        return c_att, f_att


class SpatialAdaDRDeform(nn.Module):
    """Per-pixel dilation map -> 3x3 deformable conv (torchvision backend)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        max_dilation: float = 4.0,
        padding: int = 1,
        backend: str = "torchvision",
    ) -> None:
        super().__init__()
        if backend != "torchvision":
            raise ValueError(f"Unsupported AdaDR backend: {backend}")
        if _tv_deform_conv2d is None:
            raise ImportError("torchvision.ops.deform_conv2d is required for SpatialAdaDRDeform")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_dilation = float(max_dilation)
        self.padding = padding
        self.conv_offset = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=True)
        self.conv_mask = nn.Conv2d(
            in_channels, kernel_size * kernel_size, kernel_size=3, padding=1, bias=True
        )
        self.register_buffer(
            "_offset_template",
            _KERNEL_3X3_OFFSET_TEMPLATE.clone(),
            persistent=False,
        )
        self._init_offset_mask()

    def _init_offset_mask(self) -> None:
        nn.init.constant_(self.conv_offset.weight, 0.0)
        nn.init.constant_(self.conv_offset.bias, -6.0)
        nn.init.constant_(self.conv_mask.weight, 0.0)
        nn.init.constant_(self.conv_mask.bias, 0.0)

    def dilation_map(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.conv_offset(x)
        return 1.0 + (self.max_dilation - 1.0) * torch.sigmoid(raw)

    def build_offset_and_mask(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        d_map = self.dilation_map(x)
        tmpl = self._offset_template.to(device=x.device, dtype=x.dtype)
        offsets = []
        for k in range(tmpl.shape[0]):
            dy = tmpl[k, 0].view(1, 1, 1, 1)
            dx = tmpl[k, 1].view(1, 1, 1, 1)
            offsets.append(d_map * dy)
            offsets.append(d_map * dx)
        offset = torch.cat(offsets, dim=1)
        mask = torch.sigmoid(self.conv_mask(x))
        return offset, mask, d_map

    def forward_single(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        offset: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if offset is None or mask is None:
            offset, mask, _ = self.build_offset_and_mask(x)
        assert _tv_deform_conv2d is not None
        return _tv_deform_conv2d(
            x,
            offset,
            weight,
            bias,
            stride=1,
            padding=self.padding,
            dilation=1,
            mask=mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        offset, mask, _ = self.build_offset_and_mask(x)
        return self.forward_single(x, weight, bias, offset, mask)


class AdaKernCoupledDeform(nn.Module):
    """AdaKern + AdaDR: Y = AdaDR(X, W_low) + AdaDR(X, W_high) with shared offset/mask."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_dilation: float = 4.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.omni_low = OmniAttentionLite(in_channels, out_channels)
        self.omni_high = OmniAttentionLite(in_channels, out_channels)
        self.adadr = SpatialAdaDRDeform(
            in_channels, out_channels, kernel_size=3, max_dilation=max_dilation, padding=1
        )

    @staticmethod
    def _modulate_weight(
        w: torch.Tensor,
        c_att: torch.Tensor,
        f_att: torch.Tensor,
        use_residual: bool,
    ) -> torch.Tensor:
        b = c_att.shape[0]
        w_exp = w.unsqueeze(0).expand(b, -1, -1, -1, -1)
        w_mean = w_exp.mean(dim=(-1, -2), keepdim=True)
        w_res = w_exp - w_mean
        base = w_res if use_residual else w_mean
        scale = (2.0 * c_att.unsqueeze(1)) * (2.0 * f_att.unsqueeze(2))
        return base * scale

    def effective_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_l, f_l = self.omni_low(x)
        c_h, f_h = self.omni_high(x)
        w_low = self._modulate_weight(self.weight, c_l, f_l, use_residual=False)
        w_high = self._modulate_weight(self.weight, c_h, f_h, use_residual=True)
        w_eff = w_low + w_high
        offset, mask, d_map = self.adadr.build_offset_and_mask(x)
        return w_eff, offset, mask, d_map

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_eff, offset, mask, d_map = self.effective_weights(x)
        b = x.shape[0]
        outs = []
        for i in range(b):
            outs.append(
                self.adadr.forward_single(
                    x[i : i + 1],
                    w_eff[i],
                    self.bias,
                    offset[i : i + 1],
                    mask[i : i + 1],
                )
            )
        return torch.cat(outs, dim=0), d_map


class FADCContextCore(nn.Module):
    """
    ASPP functional replacement for AHMF IMAF stages.

    (B,C,H,W) -> FreqSelect -> AdaKern+AdaDR -> 1x1 fuse; no outer beta residual.
    """

    def __init__(
        self,
        channels: int,
        max_dilation: float = 4.0,
        lowfreq_att: bool = True,
        use_freqselect: bool = True,
        use_adakern: bool = True,
        use_adadr: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_freqselect = use_freqselect
        self.use_adakern = use_adakern
        self.use_adadr = use_adadr

        if use_freqselect:
            self.fs = FrequencySelectionFull(channels, lowfreq_att=lowfreq_att)
        else:
            self.fs = None

        if use_adakern and use_adadr:
            self.adakern_adadr = AdaKernCoupledDeform(channels, channels, max_dilation=max_dilation)
        elif use_adadr:
            self.adakern_adadr = None
            self._adadr = SpatialAdaDRDeform(channels, channels, max_dilation=max_dilation)
            self._weight = nn.Parameter(torch.empty(channels, channels, 3, 3))
            self._bias = nn.Parameter(torch.zeros(channels))
            nn.init.kaiming_normal_(self._weight, mode="fan_out", nonlinearity="relu")
        else:
            self.adakern_adadr = None
            self._adadr = None

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def zero_init_freq_weight_convs(self) -> None:
        if self.fs is not None:
            self.fs._zero_init_freq_convs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.fs(x) if self.fs is not None else x
        if self.adakern_adadr is not None:
            y_core, _ = self.adakern_adadr(x_in)
        elif self._adadr is not None:
            y_core = self._adadr(x_in, self._weight, self._bias)
        else:
            y_core = x_in
        return self.fuse(y_core)


def reapply_fadc_context_core_zero_init(net: nn.Module) -> None:
    """Re-apply FreqSelect zero init after init_net overwrites conv weights."""
    root = net.module if hasattr(net, "module") else net
    for m in root.modules():
        if isinstance(m, FADCContextCore):
            m.zero_init_freq_weight_convs()
