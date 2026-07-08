"""BV2 valid-depth evaluation metrics (Protocol B: gt ≥ valid_min_depth)."""

from __future__ import annotations

from typing import Any

import numpy as np

METRIC_EPS = 1e-6
BV2_VALID_MIN_DEPTH_M = 0.5
BV2_MAX_DEPTH_M = 30.0


def valid_depth_mask(
    gt: np.ndarray,
    *,
    valid_min_depth_m: float = BV2_VALID_MIN_DEPTH_M,
) -> np.ndarray:
    """Boolean mask for BV2 valid-depth protocol."""
    gt = np.asarray(gt, dtype=np.float64).squeeze()
    return gt >= float(valid_min_depth_m)


def compute_masked_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """RMSE, REL, log10, δ1/δ2/δ3 on masked pixels."""
    g = np.asarray(gt, dtype=np.float64).squeeze()[mask]
    p = np.asarray(pred, dtype=np.float64).squeeze()[mask]
    if g.size == 0:
        return {k: 0.0 for k in ("rmse", "rel", "log10", "delta1", "delta2", "delta3")}

    g = np.maximum(g, METRIC_EPS)
    p = np.maximum(p, METRIC_EPS)

    thresh = np.maximum(g / p, p / g)
    delta1 = float((thresh < 1.25).mean())
    delta2 = float((thresh < 1.25**2).mean())
    delta3 = float((thresh < 1.25**3).mean())
    rmse = float(np.sqrt(np.mean((g - p) ** 2)))
    rel = float(np.mean(np.abs(g - p) / g))
    log10 = float(np.nanmean(np.abs(np.log10(g) - np.log10(p))))

    def _safe(v: float) -> float:
        return v if np.isfinite(v) else 0.0

    return {
        "rmse": _safe(rmse),
        "rel": _safe(rel),
        "log10": _safe(log10),
        "delta1": _safe(delta1),
        "delta2": _safe(delta2),
        "delta3": _safe(delta3),
    }


def evaluate_sample(
    gt_m: np.ndarray,
    pred_m: np.ndarray,
    *,
    valid_min_depth_m: float = BV2_VALID_MIN_DEPTH_M,
) -> dict[str, float]:
    """Per-sample BV2 metrics in meters."""
    mask = valid_depth_mask(gt_m, valid_min_depth_m=valid_min_depth_m)
    return compute_masked_metrics(gt_m, pred_m, mask)


def aggregate_samples(samples: list[tuple[np.ndarray, np.ndarray]], **kwargs: Any) -> dict[str, float]:
    """Mean metrics over (gt, pred) sample pairs."""
    if not samples:
        z = 0.0
        return {k: z for k in ("rmse", "rel", "log10", "delta1", "delta2", "delta3", "n_samples")}
    rows = [evaluate_sample(gt, pred, **kwargs) for gt, pred in samples]
    keys = ("rmse", "rel", "log10", "delta1", "delta2", "delta3")
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["n_samples"] = float(len(rows))
    return out


def metrics_to_wbrs_input(metrics: dict[str, float]) -> dict[str, float]:
    """Map lowercase eval keys to WBRS canonical names."""
    return {
        "RMSE": float(metrics["rmse"]),
        "REL": float(metrics["rel"]),
        "LOG10": float(metrics["log10"]),
        "DELTA1": float(metrics["delta1"]),
        "DELTA2": float(metrics["delta2"]),
        "DELTA3": float(metrics["delta3"]),
    }
