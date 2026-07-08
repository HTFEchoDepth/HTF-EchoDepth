"""5-stage U-Net v1 aligned with FA-DTF-TDF-DP v3 V2 (DualPathLite, configurable stage_mask)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from htf_echodepth.models.backbone.dual_path_lite_v0 import (
    DEFAULT_BETA_DP_INIT,
    DEFAULT_HIDDEN_RATIO,
    DEFAULT_N_HEADS,
    DEFAULT_NUM_LAYERS,
    DualPathLiteV0,
    EXTERNAL_LAYOUT,
)
from htf_echodepth.models.backbone.res_fa_dtf_tdf_v3_v1_unet5_v1_aligned_v0 import (
    ResFADTFTDFV3V1UNet5V1AlignedV0,
    _block_cfg,
)
from htf_echodepth.models.backbone.unetbaseline_model import init_net

STAGE_CHANNEL_MAP = {
    "enc2": 128,
    "enc3": 256,
    "enc4": 512,
}
VALID_STAGE_MASK = frozenset(STAGE_CHANNEL_MAP)


def _dp_cfg(cfg: DictConfig) -> dict[str, Any]:
    blk = getattr(cfg.model, "fa_dtf_tdf_dp_v3_v2", None)
    if blk is None:
        return {
            "n_heads": DEFAULT_N_HEADS,
            "num_layers": DEFAULT_NUM_LAYERS,
            "hidden_ratio": DEFAULT_HIDDEN_RATIO,
            "beta_dp_init": DEFAULT_BETA_DP_INIT,
            "dropout": 0.0,
            "stage_mask": ("enc4",),
        }
    stage_mask = tuple(getattr(blk, "stage_mask", ("enc4",)))
    return {
        "n_heads": int(getattr(blk, "n_heads", DEFAULT_N_HEADS)),
        "num_layers": int(getattr(blk, "num_layers", DEFAULT_NUM_LAYERS)),
        "hidden_ratio": int(getattr(blk, "hidden_ratio", DEFAULT_HIDDEN_RATIO)),
        "beta_dp_init": float(getattr(blk, "beta_dp_init", DEFAULT_BETA_DP_INIT)),
        "dropout": float(getattr(blk, "dropout", 0.0)),
        "stage_mask": stage_mask,
    }


class ResFADTFTDFDPV3V2UNet5V1AlignedV0(ResFADTFTDFV3V1UNet5V1AlignedV0):
    """
    FA-DTF-TDF v3 V1 backbone + zero-gated DualPathLite on configurable latent stages.

    Supported stage_mask subsets of (enc2, enc3, enc4). Input / output: [B,C,F,T].
    """

    STAGE_CHANNEL_MAP = STAGE_CHANNEL_MAP
    FEATURE_LAYOUT = EXTERNAL_LAYOUT

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
        n_heads: int = DEFAULT_N_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        hidden_ratio: int = DEFAULT_HIDDEN_RATIO,
        beta_dp_init: float = DEFAULT_BETA_DP_INIT,
        dropout: float = 0.0,
        stage_mask: tuple[str, ...] = ("enc4",),
        variant: str = "fa_dtf_tdf_dp_v3_v2",
    ) -> None:
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            depth_norm=depth_norm,
            bn_factor=bn_factor,
            activation=activation,
            norm=norm,
            time_kernel=time_kernel,
            freq_kernel=freq_kernel,
            alpha_init=alpha_init,
            fac_reduction=fac_reduction,
        )
        self.VARIANT = variant
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.hidden_ratio = hidden_ratio
        self.stage_mask = tuple(stage_mask)
        self.head_dims = {
            stage: STAGE_CHANNEL_MAP[stage] // n_heads
            for stage in self.stage_mask
            if stage in STAGE_CHANNEL_MAP
        }

        for stage in self.stage_mask:
            if stage not in VALID_STAGE_MASK:
                raise ValueError(f"Invalid stage_mask entry: {stage}")
            channels = STAGE_CHANNEL_MAP[stage]
            if channels % n_heads != 0:
                raise ValueError(f"{stage} channels {channels} not divisible by n_heads {n_heads}")

        self.dual_path: nn.ModuleDict = nn.ModuleDict()
        self.beta_dp: nn.ParameterDict = nn.ParameterDict()
        for stage in self.stage_mask:
            channels = STAGE_CHANNEL_MAP[stage]
            self.dual_path[stage] = DualPathLiteV0(
                channels,
                n_heads=n_heads,
                num_layers=num_layers,
                hidden_ratio=hidden_ratio,
                dropout=dropout,
            )
            self.beta_dp[stage] = nn.Parameter(torch.tensor(float(beta_dp_init)))

    def dual_path_snapshot(self) -> dict[str, float]:
        return {k: float(v.detach().cpu()) for k, v in self.beta_dp.items()}

    def _apply_dual_path(
        self,
        stage: str,
        feat: torch.Tensor,
        *,
        shapes: dict[str, tuple],
        stage_diags: dict[str, dict],
        return_diag: bool,
    ) -> torch.Tensor:
        if stage not in self.stage_mask:
            return feat
        feat_dp = self.dual_path[stage](feat)
        shapes[f"{stage}_dp"] = tuple(feat_dp.shape)
        beta = self.beta_dp[stage]
        feat = feat + beta * feat_dp
        shapes[f"{stage}_gated"] = tuple(feat.shape)
        if return_diag:
            stage_diags[f"{stage}_dp"] = {
                "beta_dp": float(beta.detach().cpu()),
                f"{stage}_dp_rms": float(feat_dp.detach().pow(2).mean().sqrt().cpu()),
            }
        return feat

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
        e2 = self._apply_dual_path("enc2", e2, shapes=shapes, stage_diags=stage_diags, return_diag=return_diag)

        x = self.down2(e2)
        shapes["down2"] = tuple(x.shape)
        e3 = self.enc3(x)
        shapes["enc3"] = tuple(e3.shape)
        e3 = self._apply_dual_path("enc3", e3, shapes=shapes, stage_diags=stage_diags, return_diag=return_diag)

        x = self.down3(e3)
        shapes["down3"] = tuple(x.shape)
        e4 = self.enc4(x)
        shapes["enc4"] = tuple(e4.shape)
        e4 = self._apply_dual_path("enc4", e4, shapes=shapes, stage_diags=stage_diags, return_diag=return_diag)

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


def define_res_fa_dtf_tdf_dp_v3_v2_unet5_v1_aligned_v0(
    cfg: DictConfig,
    input_nc: int,
    output_nc: int,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gpu_ids: list[int] | None = None,
):
    gpu_ids = gpu_ids or []
    bcfg = _block_cfg(cfg)
    dcfg = _dp_cfg(cfg)
    depth_norm = bool(getattr(cfg.dataset, "depth_norm", False))
    variant = str(getattr(cfg.model, "variant", "fa_dtf_tdf_dp_v3_v2"))
    net = ResFADTFTDFDPV3V2UNet5V1AlignedV0(
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
        n_heads=dcfg["n_heads"],
        num_layers=dcfg["num_layers"],
        hidden_ratio=dcfg["hidden_ratio"],
        beta_dp_init=dcfg["beta_dp_init"],
        dropout=dcfg["dropout"],
        stage_mask=dcfg["stage_mask"],
        variant=variant,
    )
    return init_net(net, init_type, init_gain, gpu_ids)


# Backward-compatible alias for H4-L1 gate script.
define_res_fa_dtf_tdf_dp_v3_v2_h4_l1_unet5_v1_aligned_v0 = define_res_fa_dtf_tdf_dp_v3_v2_unet5_v1_aligned_v0
