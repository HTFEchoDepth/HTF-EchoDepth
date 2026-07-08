"""5-stage / 4-down / 4-up ResTFC-TDF U-Net v0 for BatVision depth baseline."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from htf_echodepth.models.backbone.res_tfc_tdf_block import ResTFCTDFBlock, _make_activation
from htf_echodepth.models.backbone.unetbaseline_model import init_net


def _block_cfg(cfg: DictConfig) -> dict[str, Any]:
    blk = getattr(cfg.model, "res_tfc_tdf", None)
    if blk is None:
        return {"bn_factor": 16, "min_bn_units": 16, "activation": "gelu", "norm": "batch"}
    return {
        "bn_factor": int(getattr(blk, "bn_factor", 16)),
        "min_bn_units": int(getattr(blk, "min_bn_units", 16)),
        "activation": str(getattr(blk, "activation", "gelu")),
        "norm": str(getattr(blk, "norm", "batch")),
    }


def _norm_layer(name: str) -> type[nn.Module]:
    if name == "batch":
        return nn.BatchNorm2d
    if name == "group":
        return lambda c: nn.GroupNorm(min(32, c), c)
    raise ValueError(f"Unsupported norm: {name!r}")


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResTFCTDFUNet5V0(nn.Module):
    """
    Explicit 5-stage encoder + 4-skip U-Net decoder.
    Input / output layout: [B,C,F,T] with H=F, W=T.
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
        min_bn_units: int = 16,
        activation: str = "gelu",
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.depth_norm = depth_norm
        self.activation_name = activation
        self.norm_name = norm
        norm_layer = _norm_layer(norm)
        act_cls = _make_activation(activation)

        def blk(ch: int, f: int, name: str) -> ResTFCTDFBlock:
            return ResTFCTDFBlock(
                ch,
                f,
                bn_factor=bn_factor,
                min_bn_units=min_bn_units,
                norm_layer=norm_layer,
                activation=activation,
                block_name=name,
            )

        c0, c1, c2, c3, c4 = self.STAGE_CHANNELS
        f0, f1, f2, f3, f4 = self.STAGE_SPATIAL

        self.enc0_stem = nn.Sequential(
            nn.Conv2d(input_nc, c0, kernel_size=3, padding=1, bias=False),
            norm_layer(c0),
            act_cls(),
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

        head_layers: list[nn.Module] = [
            nn.Conv2d(c0, output_nc, kernel_size=3, padding=1),
        ]
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

    def forward(self, x: torch.Tensor, *, return_shapes: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, tuple]]:
        shapes: dict[str, tuple] = {}

        x = self.enc0_stem(x)
        shapes["enc0_stem"] = tuple(x.shape)
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

        out = self.head(x)
        shapes["head"] = tuple(out.shape)

        if return_shapes:
            return out, shapes
        return out

    def res_scale_snapshot(self) -> dict[str, float]:
        return {name: float(blk.res_scale.detach().cpu()) for name, blk in self._stage_blocks.items()}


def define_res_tfc_tdf_unet5_v0(
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
    net = ResTFCTDFUNet5V0(
        input_nc=input_nc,
        output_nc=output_nc,
        depth_norm=depth_norm,
        bn_factor=bcfg["bn_factor"],
        min_bn_units=bcfg["min_bn_units"],
        activation=bcfg["activation"],
        norm=bcfg["norm"],
    )
    return init_net(net, init_type, init_gain, gpu_ids)
