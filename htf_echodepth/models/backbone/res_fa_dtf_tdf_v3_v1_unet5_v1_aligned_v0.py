"""5-stage U-Net v1 aligned with FA-DTF-TDF v3 V1 stage blocks (FAC + tfc1 DTF)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from htf_echodepth.models.backbone.dtf_spatial_conv_v0 import VARIANT
from htf_echodepth.models.backbone.res_fa_dtf_tdf_block_v3_v1 import ResFADTFTDFBlockV3V1, resolve_dtf_kernels
from htf_echodepth.models.backbone.res_tfc_tdf_block_v3_aligned import _get_act
from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v0 import Downsample, Upsample, _norm_layer
from htf_echodepth.models.backbone.unetbaseline_model import init_net


def _block_cfg(cfg: DictConfig) -> dict[str, Any]:
    blk = getattr(cfg.model, "fa_dtf_tdf_v3_v1", None)
    if blk is None:
        tk, fk = resolve_dtf_kernels("dtfat_faithful_even")
        return {
            "bn_factor": 16,
            "activation": "gelu",
            "norm": "BatchNorm",
            "time_kernel": tk,
            "freq_kernel": fk,
            "kernel_mode": "dtfat_faithful_even",
            "alpha_init": 0.5,
            "fac_reduction": 4,
        }
    mode = str(getattr(blk, "kernel_mode", "dtfat_faithful_even"))
    tk, fk = resolve_dtf_kernels(mode)
    if mode != "practical_odd":
        tk = tuple(getattr(blk, "time_kernel", tk))
        fk = tuple(getattr(blk, "freq_kernel", fk))
    norm = str(getattr(blk, "norm", "BatchNorm"))
    if norm == "batch":
        norm = "BatchNorm"
    return {
        "bn_factor": int(getattr(blk, "bn_factor", 16)),
        "activation": str(getattr(blk, "activation", "gelu")),
        "norm": norm,
        "time_kernel": tk,
        "freq_kernel": fk,
        "kernel_mode": mode,
        "alpha_init": float(getattr(blk, "alpha_init", 0.5)),
        "fac_reduction": int(getattr(blk, "fac_reduction", 4)),
    }


class ResFADTFTDFV3V1UNet5V1AlignedV0(nn.Module):
    """
    Same skeleton as ResDTFTDFV3V0UNet5V1AlignedV0; stage blocks use ResFADTFTDFBlockV3V1.
    Input / output layout: [B,C,F,T].
    """

    STAGE_CHANNELS = (32, 64, 128, 256, 512)
    STAGE_SPATIAL = (256, 128, 64, 32, 16)
    FEATURE_LAYOUT = "[B,C,F,T]"
    VARIANT = VARIANT

    def __init__(
        self,
        input_nc: int = 2,
        output_nc: int = 1,
        *,
        depth_norm: bool = True,
        bn_factor: int = 16,
        activation: str = "gelu",
        norm: str = "BatchNorm",
        time_kernel: tuple[int, int] = (6, 3),
        freq_kernel: tuple[int, int] = (3, 6),
        alpha_init: float = 0.5,
        fac_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.depth_norm = depth_norm
        self.activation_name = activation
        self.norm_name = norm
        self.time_kernel = time_kernel
        self.freq_kernel = freq_kernel
        self.fac_reduction = fac_reduction
        norm_layer = _norm_layer("batch" if norm == "BatchNorm" else norm)
        act = _get_act(activation)

        def blk(ch: int, f: int, name: str) -> ResFADTFTDFBlockV3V1:
            return ResFADTFTDFBlockV3V1(
                ch,
                f,
                bn_factor=bn_factor,
                norm_type=norm,
                act_type=activation,
                block_name=name,
                time_kernel=time_kernel,
                freq_kernel=freq_kernel,
                alpha_init=alpha_init,
                fac_reduction=fac_reduction,
            )

        c0, c1, c2, c3, c4 = self.STAGE_CHANNELS
        f0, f1, f2, f3, f4 = self.STAGE_SPATIAL

        self.enc0_stem = nn.Sequential(
            nn.Conv2d(input_nc, c0, kernel_size=3, padding=1, bias=False),
            norm_layer(c0),
            act,
        )
        self.enc0 = blk(c0, f0, "enc0")
        self.down0 = Downsample(c0, c1, norm_layer)
        self.enc1 = blk(c1, f1, "enc1")
        self.down1 = Downsample(c1, c2, norm_layer)
        self.enc2 = blk(c2, f2, "enc2")
        self.down2 = Downsample(c2, c3, norm_layer)
        self.enc3 = blk(c3, f3, "enc3")
        self.down3 = Downsample(c3, c4, norm_layer)
        self.enc4 = blk(c4, f4, "enc4")

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

        head_layers: list[nn.Module] = [nn.Conv2d(c0, output_nc, kernel_size=3, padding=1)]
        if depth_norm:
            head_layers.append(nn.Sigmoid())
        else:
            head_layers.append(nn.ReLU(inplace=True))
        self.head = nn.Sequential(*head_layers)

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

    def alpha_snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, blk in self._stage_blocks.items():
            out[f"{name}.alpha"] = float(blk.tfc1_dtf.alpha.detach().cpu())
        return out

    def fac_snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, blk in self._stage_blocks.items():
            out[f"{name}.E_f_norm"] = float(blk.fac.E_f.norm().detach().cpu())
        return out

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_shapes: bool = False,
        return_diag: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, tuple]] | tuple[torch.Tensor, dict, dict, torch.Tensor]:
        shapes: dict[str, tuple] = {}
        stage_diags: dict[str, dict] = {}

        x = self.enc0_stem(x)
        shapes["enc0_stem"] = tuple(x.shape)
        if return_diag:
            e0, dbg = self.enc0(x, return_debug=True)
            stage_diags["enc0"] = dbg
        else:
            e0 = self.enc0(x)
        shapes["enc0"] = tuple(e0.shape)

        x = self.down0(e0)
        shapes["down0"] = tuple(x.shape)
        e1 = self.enc1(x)
        shapes["enc1"] = tuple(e1.shape)

        x = self.down1(e1)
        shapes["down1"] = tuple(x.shape)
        e2 = self.enc2(x)
        shapes["enc2"] = tuple(e2.shape)

        x = self.down2(e2)
        shapes["down2"] = tuple(x.shape)
        e3 = self.enc3(x)
        shapes["enc3"] = tuple(e3.shape)

        x = self.down3(e3)
        shapes["down3"] = tuple(x.shape)
        e4 = self.enc4(x)
        shapes["enc4"] = tuple(e4.shape)

        x = self.up3(e4)
        shapes["up3_up"] = tuple(x.shape)
        x = torch.cat([x, e3], dim=1)
        shapes["up3_cat"] = tuple(x.shape)
        x = self.fuse3(x)
        shapes["up3_fuse"] = tuple(x.shape)
        x = self.dec3(x)
        shapes["up3"] = tuple(x.shape)

        x = self.up2(x)
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
        shapes["head_logits"] = tuple(logits.shape)
        out = self.head(x)
        shapes["head"] = tuple(out.shape)

        if return_diag:
            return out, shapes, stage_diags, logits
        if return_shapes:
            return out, shapes
        return out


def define_res_fa_dtf_tdf_v3_v1_unet5_v1_aligned_v0(
    cfg: DictConfig,
    input_nc: int,
    output_nc: int,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gpu_ids: list[int] | None = None,
):
    gpu_ids = gpu_ids or []
    bcfg = _block_cfg(cfg)
    depth_norm = bool(getattr(cfg.dataset, "depth_norm", False))
    net = ResFADTFTDFV3V1UNet5V1AlignedV0(
        input_nc=input_nc,
        output_nc=output_nc,
        depth_norm=depth_norm,
        bn_factor=bcfg["bn_factor"],
        activation=bcfg["activation"],
        norm=bcfg["norm"],
        time_kernel=bcfg["time_kernel"],
        freq_kernel=bcfg["freq_kernel"],
        alpha_init=bcfg["alpha_init"],
        fac_reduction=bcfg["fac_reduction"],
    )
    return init_net(net, init_type, init_gain, gpu_ids)
