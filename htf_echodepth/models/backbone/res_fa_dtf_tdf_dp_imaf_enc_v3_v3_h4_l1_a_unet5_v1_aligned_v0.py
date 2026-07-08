"""FA-DTF-TDF-DP-IMAF-Enc v3 V3-H4-L1-A: H4-L1 encoder + Hybrid-IMAF V0 integrator (5 stages)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_v3_v2_unet5_v1_aligned_v0 import (
    ResFADTFTDFDPV3V2UNet5V1AlignedV0,
    _block_cfg,
    _dp_cfg,
)
from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_v0 import STAGE_ASPP_DILATIONS
from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_v0 import IMAFStageEnhancement, _imaf_cfg
from htf_echodepth.models.backbone.unetbaseline_model import init_net

STAGE_CHANNELS = (32, 64, 128, 256, 512)


class ResFADTFTDFDPIMAFEncV3V3H4L1AUNet5V1AlignedV0(ResFADTFTDFDPV3V2UNet5V1AlignedV0):
    """
    H4-L1 (DualPathLite @ enc4) + code-faithful Hybrid-IMAF V0 on enc0..enc4.

    Encoder order per stage: block -> IMAF -> (enc4 only) DualPathLite gate.
    Decoder unchanged from H4-L1.
    """

    STAGE_ASPP_DILATIONS = STAGE_ASPP_DILATIONS

    def __init__(
        self,
        *args: Any,
        gamma_aspp_init: float = 0.1,
        gamma_aspp_init_by_stage: tuple[float, ...] | list[float] | None = None,
        variant: str = "fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a",
        **kwargs: Any,
    ) -> None:
        kwargs["variant"] = variant
        kwargs.setdefault("stage_mask", ("enc4",))
        super().__init__(*args, **kwargs)
        self.gamma_aspp_init = gamma_aspp_init
        if gamma_aspp_init_by_stage is not None:
            gammas = tuple(float(v) for v in gamma_aspp_init_by_stage)
            if len(gammas) != 5:
                raise ValueError(
                    f"gamma_aspp_init_by_stage must have 5 entries (enc0..enc4), got {len(gammas)}"
                )
            self.gamma_aspp_init_by_stage = gammas
        else:
            gammas = (float(gamma_aspp_init),) * 5
            self.gamma_aspp_init_by_stage = None

        c0, c1, c2, c3, c4 = STAGE_CHANNELS
        dil = self.STAGE_ASPP_DILATIONS
        self.imaf0 = IMAFStageEnhancement(c0, dil[0], gamma_aspp_init=gammas[0])
        self.imaf1 = IMAFStageEnhancement(c1, dil[1], gamma_aspp_init=gammas[1])
        self.imaf2 = IMAFStageEnhancement(c2, dil[2], gamma_aspp_init=gammas[2])
        self.imaf3 = IMAFStageEnhancement(c3, dil[3], gamma_aspp_init=gammas[3])
        self.imaf4 = IMAFStageEnhancement(c4, dil[4], gamma_aspp_init=gammas[4])
        self._imaf_stages = [self.imaf0, self.imaf1, self.imaf2, self.imaf3, self.imaf4]

    def _run_encoder_stage(
        self,
        x: torch.Tensor,
        enc: nn.Module,
        imaf: IMAFStageEnhancement,
        *,
        return_diag: bool = False,
    ):
        if return_diag:
            e, dbg = enc(x, return_debug=True)
            return imaf(e, return_diag=True)
        e = enc(x)
        return imaf(e)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_shapes: bool = False,
        return_diag: bool = False,
    ):
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
        shapes["enc4_pre_dp"] = tuple(e4.shape)
        e4 = self._apply_dual_path("enc4", e4, shapes=shapes, stage_diags=stage_diags, return_diag=return_diag)
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


def define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0(
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
    icfg = _imaf_cfg(cfg)
    depth_norm = bool(getattr(cfg.dataset, "depth_norm", False))
    variant = str(getattr(cfg.model, "variant", "fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a"))
    net = ResFADTFTDFDPIMAFEncV3V3H4L1AUNet5V1AlignedV0(
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
        gamma_aspp_init=icfg["gamma_aspp_init"],
        gamma_aspp_init_by_stage=icfg.get("gamma_aspp_init_by_stage"),
        variant=variant,
    )
    return init_net(net, init_type, init_gain, gpu_ids)
