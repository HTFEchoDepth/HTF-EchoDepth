"""HTF-EchoDepth — Hierarchical Time-Frequency Echo Depth Estimation."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from htf_echodepth.models.backbone.dual_path_lite_v0 import (
    DEFAULT_BETA_DP_INIT,
    DEFAULT_HIDDEN_RATIO,
    DEFAULT_N_HEADS,
    DEFAULT_NUM_LAYERS,
)
from htf_echodepth.models.backbone.res_fa_dtf_tdf_block_v3_v1 import resolve_dtf_kernels
from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0 import (
    ResFADTFTDFDPIMAFEncV3V3H4L1AUNet5V1AlignedV0,
)


class HTFEchoDepth(ResFADTFTDFDPIMAFEncV3V3H4L1AUNet5V1AlignedV0):
    """Paper HTF-EchoDepth architecture on BV2 (2×256×256 echo input)."""

    def __init__(
        self,
        input_nc: int = 2,
        output_nc: int = 1,
        *,
        depth_norm: bool = True,
        gamma_aspp_init: float = 0.1,
        **kwargs: Any,
    ) -> None:
        time_kernel, freq_kernel = resolve_dtf_kernels("dtfat_faithful_even")
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            depth_norm=depth_norm,
            bn_factor=16,
            activation="gelu",
            norm="BatchNorm",
            time_kernel=time_kernel,
            freq_kernel=freq_kernel,
            alpha_init=0.5,
            fac_reduction=4,
            n_heads=DEFAULT_N_HEADS,
            num_layers=DEFAULT_NUM_LAYERS,
            hidden_ratio=DEFAULT_HIDDEN_RATIO,
            beta_dp_init=DEFAULT_BETA_DP_INIT,
            dropout=0.0,
            stage_mask=("enc4",),
            gamma_aspp_init=gamma_aspp_init,
            variant="htf_echodepth",
            **kwargs,
        )


def build_htf_echodepth(**kwargs: Any) -> HTFEchoDepth:
    """Factory for HTF-EchoDepth with paper-default hyperparameters."""
    return HTFEchoDepth(**kwargs)
