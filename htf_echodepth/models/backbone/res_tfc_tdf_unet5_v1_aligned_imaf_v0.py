"""V5.1 aligned encoder + IMAF-style ECA/ASPP enhancement + concat-skip decoder."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from htf_echodepth.models.backbone.ahmf_blocks import EfficientChannelAttention
from htf_echodepth.models.backbone.res_tfc_tdf_block_v3_aligned import ResTFCTDFBlockV3Aligned, _get_act
from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v0 import Downsample, Upsample, _norm_layer
from htf_echodepth.models.backbone.unetbaseline_model import init_net

STAGE_ASPP_DILATIONS: tuple[tuple[int, ...], ...] = (
    (1, 2, 4),
    (1, 2, 4),
    (1, 2, 3),
    (1, 2, 3),
    (1, 2),
)


def _block_cfg(cfg: DictConfig) -> dict[str, Any]:
    blk = getattr(cfg.model, "res_tfc_tdf_v3_aligned", None)
    if blk is None:
        return {"bn_factor": 16, "activation": "gelu", "norm": "BatchNorm"}
    norm = str(getattr(blk, "norm", "BatchNorm"))
    if norm == "batch":
        norm = "BatchNorm"
    return {
        "bn_factor": int(getattr(blk, "bn_factor", 16)),
        "activation": str(getattr(blk, "activation", "gelu")),
        "norm": norm,
    }


IMAF_STAGE_NAMES: tuple[str, ...] = ("enc0", "enc1", "enc2", "enc3", "enc4")


def _parse_imaf_stage_mask(raw: Any) -> dict[str, bool] | None:
    if raw is None:
        return None
    return {s: bool(getattr(raw, s, False)) for s in IMAF_STAGE_NAMES}


def _imaf_cfg(cfg: DictConfig) -> dict[str, Any]:
    icfg = getattr(cfg.model, "imaf_v0", None)
    if icfg is None:
        return {
            "gamma_aspp_init": 0.1,
            "gamma_aspp_init_by_stage": None,
            "imaf_stage_mask": None,
        }
    by_stage = getattr(icfg, "gamma_aspp_init_by_stage", None)
    stage_mask = _parse_imaf_stage_mask(getattr(icfg, "imaf_stage_mask", None))
    if by_stage is not None:
        if isinstance(by_stage, dict) or hasattr(by_stage, "enc0") or hasattr(by_stage, "enc3"):
            vals = tuple(float(getattr(by_stage, s, by_stage.get(s, 0.1) if isinstance(by_stage, dict) else 0.1)) for s in IMAF_STAGE_NAMES)
        else:
            vals = tuple(float(v) for v in by_stage)
        if len(vals) != 5:
            raise ValueError(
                f"gamma_aspp_init_by_stage must have 5 entries (enc0..enc4), got {len(vals)}"
            )
        return {
            "gamma_aspp_init": vals[0],
            "gamma_aspp_init_by_stage": vals,
            "imaf_stage_mask": stage_mask,
        }
    return {
        "gamma_aspp_init": float(getattr(icfg, "gamma_aspp_init", 0.1)),
        "gamma_aspp_init_by_stage": None,
        "imaf_stage_mask": stage_mask,
    }


class StageAwareASPP(nn.Module):
    """Parallel dilated branches -> concat -> 1x1 projection (channel-preserving)."""

    def __init__(self, channels: int, dilation_rates: tuple[int, ...]) -> None:
        super().__init__()
        self.dilation_rates = dilation_rates
        branches: list[nn.Module] = []
        for rate in dilation_rates:
            if rate == 1:
                branches.append(nn.Conv2d(channels, channels, kernel_size=1, bias=False))
            else:
                branches.append(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        stride=1,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.proj = nn.Sequential(
            nn.Conv2d(channels * len(dilation_rates), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, *, return_branch_rms: bool = False):
        branch_outs = [branch(x) for branch in self.branches]
        out = self.proj(torch.cat(branch_outs, dim=1))
        if return_branch_rms:
            rms = [float(b.detach().pow(2).mean().sqrt().cpu()) for b in branch_outs]
            return out, rms
        return out


class ECAChannelReweight(nn.Module):
    """GAP channel attention; output shape matches input."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.eca = EfficientChannelAttention(channels)

    def forward(self, x: torch.Tensor, *, return_attn: bool = False):
        attn = self.eca(x)
        out = x * attn
        if return_attn:
            return out, attn
        return out


class IMAFStageEnhancement(nn.Module):
    """ResTFC block output -> ECA -> ASPP with residual gamma scaling."""

    def __init__(
        self,
        channels: int,
        dilation_rates: tuple[int, ...],
        *,
        gamma_aspp_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.eca = ECAChannelReweight(channels)
        self.aspp = StageAwareASPP(channels, dilation_rates)
        self.gamma_aspp = nn.Parameter(torch.tensor(gamma_aspp_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor, *, return_diag: bool = False):
        residual = x
        x_eca, attn = self.eca(x, return_attn=True)
        aspp_delta, branch_rms = self.aspp(x_eca, return_branch_rms=True)
        out = residual + self.gamma_aspp * aspp_delta
        if return_diag:
            return out, {
                "input": x,
                "eca_out": x_eca,
                "attn": attn,
                "aspp_delta": aspp_delta,
                "aspp_branch_rms": branch_rms,
                "gamma_aspp": float(self.gamma_aspp.detach().cpu()),
            }
        return out


class ResTFCTDFUNet5V1AlignedIMAFV0(nn.Module):
    """
    V5.1 aligned 5-stage U-Net with IMAF-style encoder enhancement:
    each stage: ResTFCTDFBlockV3Aligned -> ECA -> ASPP (residual).
    Decoder: original V5.1 aligned concat-skip path.
    """

    STAGE_CHANNELS = (32, 64, 128, 256, 512)
    STAGE_SPATIAL = (256, 128, 64, 32, 16)

    def __init__(
        self,
        input_nc: int = 2,
        output_nc: int = 1,
        *,
        depth_norm: bool = True,
        bn_factor: int = 16,
        activation: str = "gelu",
        norm: str = "BatchNorm",
        gamma_aspp_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.depth_norm = depth_norm
        self.activation_name = activation
        self.norm_name = norm
        norm_layer = _norm_layer("batch" if norm == "BatchNorm" else norm)
        act = _get_act(activation)

        def blk(ch: int, f: int, name: str) -> ResTFCTDFBlockV3Aligned:
            return ResTFCTDFBlockV3Aligned(
                ch,
                f,
                bn_factor=bn_factor,
                norm_type=norm,
                act_type=activation,
                block_name=name,
            )

        c0, c1, c2, c3, c4 = self.STAGE_CHANNELS
        f0, f1, f2, f3, f4 = self.STAGE_SPATIAL

        self.enc0_stem = nn.Sequential(
            nn.Conv2d(input_nc, c0, kernel_size=3, padding=1, bias=False),
            norm_layer(c0),
            act,
        )
        self.enc0 = blk(c0, f0, "enc0")
        self.imaf0 = IMAFStageEnhancement(c0, STAGE_ASPP_DILATIONS[0], gamma_aspp_init=gamma_aspp_init)
        self.down0 = Downsample(c0, c1, norm_layer)
        self.enc1 = blk(c1, f1, "enc1")
        self.imaf1 = IMAFStageEnhancement(c1, STAGE_ASPP_DILATIONS[1], gamma_aspp_init=gamma_aspp_init)
        self.down1 = Downsample(c1, c2, norm_layer)
        self.enc2 = blk(c2, f2, "enc2")
        self.imaf2 = IMAFStageEnhancement(c2, STAGE_ASPP_DILATIONS[2], gamma_aspp_init=gamma_aspp_init)
        self.down2 = Downsample(c2, c3, norm_layer)
        self.enc3 = blk(c3, f3, "enc3")
        self.imaf3 = IMAFStageEnhancement(c3, STAGE_ASPP_DILATIONS[3], gamma_aspp_init=gamma_aspp_init)
        self.down3 = Downsample(c3, c4, norm_layer)
        self.enc4 = blk(c4, f4, "enc4")
        self.imaf4 = IMAFStageEnhancement(c4, STAGE_ASPP_DILATIONS[4], gamma_aspp_init=gamma_aspp_init)

        self.up3 = Upsample(c4, c3, norm_layer)
        self.fuse3 = nn.Conv2d(c3 + c3, c3, kernel_size=1, bias=False)
        self.dec3 = blk(c3, f3, "up3")

        self.up2 = Upsample(c3, c2, norm_layer)
        self.fuse2 = nn.Conv2d(c2 + c2, c2, kernel_size=1, bias=False)
        self.dec2 = blk(c2, f2, "up2")

        self.up1 = Upsample(c2, c1, norm_layer)
        self.fuse1 = nn.Conv2d(c1 + c1, c1, kernel_size=1, bias=False)
        self.dec1 = blk(c1, f1, "up1")

        self.up0 = Upsample(c1, c0, norm_layer)
        self.fuse0 = nn.Conv2d(c0 + c0, c0, kernel_size=1, bias=False)
        self.dec0 = blk(c0, f0, "up0")

        head_layers: list[nn.Module] = [
            nn.Conv2d(c0, output_nc, kernel_size=3, padding=1),
        ]
        if depth_norm:
            head_layers.append(nn.Sigmoid())
        else:
            head_layers.append(nn.ReLU(inplace=True))
        self.head = nn.Sequential(*head_layers)

        self._encoder_blocks = [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]
        self._imaf_stages = [self.imaf0, self.imaf1, self.imaf2, self.imaf3, self.imaf4]
        self._stage_blocks = {
            "enc0": self.enc0,
            "enc1": self.enc1,
            "enc2": self.enc2,
            "enc3": self.enc3,
            "enc4": self.enc4,
            "up3": self.dec3,
            "up2": self.dec2,
            "up1": self.dec1,
            "up0": self.dec0,
        }

    def _run_encoder_stage(
        self,
        x: torch.Tensor,
        enc: ResTFCTDFBlockV3Aligned,
        imaf: IMAFStageEnhancement,
        *,
        return_diag: bool = False,
    ):
        x = enc(x)
        if return_diag:
            return imaf(x, return_diag=True)
        return imaf(x)

    def _encode(self, x: torch.Tensor, *, return_diag: bool = False):
        shapes: dict[str, tuple] = {}
        stage_diags: dict[str, Any] = {}

        x = self.enc0_stem(x)
        shapes["enc0_stem"] = tuple(x.shape)
        if return_diag:
            e0, d0 = self._run_encoder_stage(x, self.enc0, self.imaf0, return_diag=True)
            stage_diags["enc0"] = d0
        else:
            e0 = self._run_encoder_stage(x, self.enc0, self.imaf0)
        shapes["enc0"] = tuple(e0.shape)

        x = self.down0(e0)
        shapes["down0"] = tuple(x.shape)
        if return_diag:
            e1, d1 = self._run_encoder_stage(x, self.enc1, self.imaf1, return_diag=True)
            stage_diags["enc1"] = d1
        else:
            e1 = self._run_encoder_stage(x, self.enc1, self.imaf1)
        shapes["enc1"] = tuple(e1.shape)

        x = self.down1(e1)
        shapes["down1"] = tuple(x.shape)
        if return_diag:
            e2, d2 = self._run_encoder_stage(x, self.enc2, self.imaf2, return_diag=True)
            stage_diags["enc2"] = d2
        else:
            e2 = self._run_encoder_stage(x, self.enc2, self.imaf2)
        shapes["enc2"] = tuple(e2.shape)

        x = self.down2(e2)
        shapes["down2"] = tuple(x.shape)
        if return_diag:
            e3, d3 = self._run_encoder_stage(x, self.enc3, self.imaf3, return_diag=True)
            stage_diags["enc3"] = d3
        else:
            e3 = self._run_encoder_stage(x, self.enc3, self.imaf3)
        shapes["enc3"] = tuple(e3.shape)

        x = self.down3(e3)
        shapes["down3"] = tuple(x.shape)
        if return_diag:
            e4, d4 = self._run_encoder_stage(x, self.enc4, self.imaf4, return_diag=True)
            stage_diags["enc4"] = d4
        else:
            e4 = self._run_encoder_stage(x, self.enc4, self.imaf4)
        shapes["enc4"] = tuple(e4.shape)

        if return_diag:
            return (e0, e1, e2, e3, e4), shapes, stage_diags
        return (e0, e1, e2, e3, e4), shapes

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_shapes: bool = False,
        return_diag: bool = False,
    ):
        if return_diag:
            (e0, e1, e2, e3, e4), shapes, stage_diags = self._encode(x, return_diag=True)
        else:
            (e0, e1, e2, e3, e4), shapes = self._encode(x)

        x = self.up3(e4)
        shapes["up3_up"] = tuple(x.shape)
        x = torch.cat([x, e3], dim=1)
        shapes["up3_cat"] = tuple(x.shape)
        x = self.fuse3(x)
        shapes["up3_fuse"] = tuple(x.shape)
        x = self.dec3(x)
        shapes["up3"] = tuple(x.shape)

        x = self.up2(x)
        shapes["up2_up"] = tuple(x.shape)
        x = torch.cat([x, e2], dim=1)
        x = self.fuse2(x)
        x = self.dec2(x)
        shapes["up2"] = tuple(x.shape)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.fuse1(x)
        x = self.dec1(x)
        shapes["up1"] = tuple(x.shape)

        x = self.up0(x)
        x = torch.cat([x, e0], dim=1)
        x = self.fuse0(x)
        x = self.dec0(x)
        shapes["up0"] = tuple(x.shape)

        logits = self.head[0](x)
        shapes["logits"] = tuple(logits.shape)
        out = self.head[1](logits) if len(self.head) > 1 else logits
        shapes["head"] = tuple(out.shape)

        if return_diag:
            return out, shapes, stage_diags, logits
        if return_shapes:
            return out, shapes
        return out


def define_res_tfc_tdf_unet5_v1_aligned_imaf_v0(
    cfg: DictConfig,
    input_nc: int,
    output_nc: int,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gpu_ids: list[int] | None = None,
):
    gpu_ids = gpu_ids or []
    bcfg = _block_cfg(cfg)
    icfg = _imaf_cfg(cfg)
    depth_norm = bool(getattr(cfg.dataset, "depth_norm", False))
    net = ResTFCTDFUNet5V1AlignedIMAFV0(
        input_nc=input_nc,
        output_nc=output_nc,
        depth_norm=depth_norm,
        bn_factor=bcfg["bn_factor"],
        activation=bcfg["activation"],
        norm=bcfg["norm"],
        gamma_aspp_init=icfg["gamma_aspp_init"],
    )
    return init_net(net, init_type, init_gain, gpu_ids)
