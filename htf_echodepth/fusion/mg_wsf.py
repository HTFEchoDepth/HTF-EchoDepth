"""MG-WSF — Metric-Guided Weight-Space Fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from htf_echodepth.fusion.candidate_retention import retain_metric_role_candidates
from htf_echodepth.fusion.checkpoint_fusion import mix_state_dicts
from htf_echodepth.fusion.fusion_selection import select_best_fusion_weights
from htf_echodepth.fusion.wbrs import compute_wbrs, normalize_metrics

__all__ = [
    "retain_metric_role_candidates",
    "select_best_fusion_weights",
    "mix_state_dicts",
    "compute_wbrs",
    "normalize_metrics",
    "run_mg_wsf",
]


def run_mg_wsf(
    validation_records: list[dict[str, Any]],
    per_sample_preds: list[list[Any]],
    gts: list[Any],
    checkpoint_paths: list[str | Path],
    *,
    wbrs_reference: dict[str, float],
    valid_min_depth_m: float = 0.5,
    output_path: str | Path,
    grid_step: float = 0.05,
) -> dict[str, Any]:
    """End-to-end MG-WSF: MRCR → VGFS → weight-space fusion → single checkpoint."""
    pool_rows, specialists = retain_metric_role_candidates(validation_records)
    ordered_paths = [specialists[k]["checkpoint_path"] for k in sorted(specialists) if k in specialists]
    if len(ordered_paths) != len(checkpoint_paths):
        ordered_paths = [str(p) for p in checkpoint_paths]

    weights, val_metrics = select_best_fusion_weights(
        per_sample_preds,
        gts,
        valid_min_depth_m=valid_min_depth_m,
        wbrs_reference=wbrs_reference,
        grid_step=grid_step,
    )
    fused = mix_state_dicts(ordered_paths, weights)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(fused, out)
    return {
        "pool_rows": pool_rows,
        "fusion_weights": weights,
        "validation_metrics": val_metrics,
        "output_checkpoint": str(out),
    }
