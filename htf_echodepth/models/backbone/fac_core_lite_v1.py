"""FAC-core-lite v1 (FDY-derived): frequency positional encoding + adaptive scaling."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

INTERNAL_LAYOUT = "[B,C,T,F]"


class FACCoreLiteV1(nn.Module):
    """
    FAC core mechanism (BatVision adaptation, not full FDY dynamic conv).

    U_f = U + sigmoid(ScaleNet(mean_time(U))) * E_f

    - U layout: [B,C,T,F] (internal TFC-TDF v3 block layout)
    - mean_time pools along dim=2 (T), preserving frequency axis F
    - E_f: trained-zero frequency positional encoding [1,1,1,F]
    - ScaleNet: Conv1d bottleneck on [B,C,F] -> [B,C,F]
    """

    def __init__(
        self,
        channels: int,
        freq_bins: int,
        *,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if freq_bins < 1:
            raise ValueError(f"freq_bins must be >= 1, got {freq_bins}")
        c_reduced = max(channels // reduction, 1)
        self.channels = channels
        self.freq_bins = freq_bins
        self.c_reduced = c_reduced
        self.E_f = nn.Parameter(torch.zeros(1, 1, 1, freq_bins))
        self.scale_down = nn.Conv1d(channels, c_reduced, kernel_size=1, bias=True)
        self.scale_up = nn.Conv1d(c_reduced, channels, kernel_size=1, bias=True)

    def forward(
        self, u: torch.Tensor, *, return_debug: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        # u: [B,C,T,F]
        q = u.mean(dim=2)
        h = F.gelu(self.scale_down(q))
        s = torch.sigmoid(self.scale_up(h))
        u_f = u + s.unsqueeze(2) * self.E_f

        if not return_debug:
            return u_f

        with torch.no_grad():
            perturbation = float((u_f - u).abs().mean().cpu())
            debug: dict[str, Any] = {
                "layout": INTERNAL_LAYOUT,
                "q_shape": tuple(q.shape),
                "s_shape": tuple(s.shape),
                "E_f_shape": tuple(self.E_f.shape),
                "scale_mean": float(s.mean().cpu()),
                "scale_min": float(s.min().cpu()),
                "scale_max": float(s.max().cpu()),
                "E_f_norm": float(self.E_f.norm().cpu()),
                "fac_perturbation": perturbation,
            }
        return u_f, debug
