"""VGFS — Validation-Guided Fusion Selection (weight search on validation)."""

from __future__ import annotations

from typing import Any

import numpy as np

from htf_echodepth.fusion.wbrs import compute_wbrs, normalize_metrics


def weight_grid(k: int, step: float = 0.05) -> list[list[float]]:
    """Simplex grid for K-way fusion weights."""
    if k == 1:
        return [[1.0]]
    if k == 2:
        alphas = np.arange(0.0, 1.0 + step / 2, step)
        return [[float(a), float(1.0 - a)] for a in alphas]
    if k > 3:
        # Release simplification: equal-weight fallback for k>3 donor pools.
        # Full simplex grid search is used for k<=3. Not a performance claim.
        return [[1.0 / k] * k]
    grids: list[list[float]] = []
    steps = np.arange(0.0, 1.0 + step / 2, step)
    for w1 in steps:
        for w2 in steps:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            if w3 < 0:
                w3 = 0.0
            if abs(w1 + w2 + w3 - 1.0) > 1e-6:
                continue
            grids.append([float(w1), float(w2), float(w3)])
    known = [0.60, 0.20, 0.20]
    if k == 3 and not any(np.allclose(g[:3], known, atol=1e-9) for g in grids):
        grids.append(known)
    return grids


def blend_prediction_arrays(
    arrays: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for w, arr in zip(weights, arrays):
        out = out + w * arr.astype(np.float64)
    return out


def select_best_fusion_weights(
    per_sample_preds: list[list[np.ndarray]],
    gts: list[np.ndarray],
    *,
    valid_min_depth_m: float,
    wbrs_reference: dict[str, float],
    primary_metric: str = "WBRS",
    grid_step: float = 0.05,
) -> tuple[list[float], dict[str, float]]:
    """VGFS: search fusion weights minimizing primary validation metric."""
    from htf_echodepth.metrics.depth_metrics import aggregate_samples, metrics_to_wbrs_input

    k = len(per_sample_preds)
    grids = weight_grid(k, step=grid_step)
    best_w: list[float] | None = None
    best_m: dict[str, float] | None = None
    minimize = primary_metric in ("WBRS", "RMSE", "REL", "LOG10", "log10")

    for weights in grids:
        fused_samples: list[tuple[np.ndarray, np.ndarray]] = []
        for sample_preds, gt in zip(per_sample_preds, gts):
            fused = blend_prediction_arrays(sample_preds, weights)
            fused_samples.append((gt, fused))
        m = aggregate_samples(fused_samples, valid_min_depth_m=valid_min_depth_m)
        canon = metrics_to_wbrs_input(m)
        canon["WBRS"] = compute_wbrs(canon, wbrs_reference)
        pm = canon.get(primary_metric.upper(), canon.get("WBRS", float("nan")))
        if not np.isfinite(pm):
            continue
        if best_m is None:
            best_w, best_m = weights, canon
            continue
        cur = best_m.get(primary_metric.upper(), best_m["WBRS"])
        if minimize and pm < cur - 1e-12:
            best_w, best_m = weights, canon
        elif not minimize and pm > cur + 1e-12:
            best_w, best_m = weights, canon

    if best_w is None or best_m is None:
        best_w = [1.0 / k] * k
        fused_samples = []
        for sample_preds, gt in zip(per_sample_preds, gts):
            fused = blend_prediction_arrays(sample_preds, weights=best_w)
            fused_samples.append((gt, fused))
        m = aggregate_samples(fused_samples, valid_min_depth_m=valid_min_depth_m)
        best_m = metrics_to_wbrs_input(m)
        best_m["WBRS"] = compute_wbrs(best_m, wbrs_reference)
    return best_w, normalize_metrics(best_m)
