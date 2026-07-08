"""BV2 training loss: L1 + thresholded log-ratio hinge (δ1-aware)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def log_ratio_hinge_threshold() -> float:
    """Threshold τ = log(1.25) corresponding to δ1."""
    return math.log(1.25)


def masked_mean_log_ratio_hinge(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    eps: float = 1e-6,
    tau: float | None = None,
    min_gt: float | None = None,
) -> torch.Tensor:
    """Mean over valid pixels of relu(|ln pred − ln gt| − τ)."""
    if tau is None:
        tau = log_ratio_hinge_threshold()
    valid = gt != 0
    if min_gt is not None:
        valid = valid & (gt >= float(min_gt))
    if not valid.any():
        return pred.sum() * 0.0
    p = torch.clamp(pred[valid], min=eps)
    g = torch.clamp(gt[valid], min=eps)
    ratio_err = torch.abs(torch.log(p) - torch.log(g))
    return torch.relu(ratio_err - float(tau)).mean()


@dataclass
class BV2LossResult:
    total: torch.Tensor
    l1: torch.Tensor
    log_ratio_hinge: torch.Tensor
    lambda_lrh: float


class BV2Loss(nn.Module):
    """L1 + λ · thresholded log-ratio hinge for normalized BV2 depth targets."""

    def __init__(self, lambda_lrh: float = 0.003) -> None:
        super().__init__()
        self.lambda_lrh = float(lambda_lrh)
        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        min_gt: float | None = None,
    ) -> BV2LossResult:
        loss_l1 = self.l1(pred, gt)
        loss_lrh = masked_mean_log_ratio_hinge(pred, gt, min_gt=min_gt)
        total = loss_l1 + self.lambda_lrh * loss_lrh
        return BV2LossResult(
            total=total,
            l1=loss_l1,
            log_ratio_hinge=loss_lrh,
            lambda_lrh=self.lambda_lrh,
        )
