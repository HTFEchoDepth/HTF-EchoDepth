"""WBRS — Weighted Balanced Relative Score for validation-guided fusion."""

from __future__ import annotations

import math
from typing import Any, Mapping

BRS_METRIC_KEYS = ("RMSE", "REL", "LOG10", "DELTA1", "DELTA2", "DELTA3")

DEFAULT_WBRS_WEIGHTS: dict[str, float] = {
    "RMSE": 0.25,
    "REL": 0.20,
    "LOG10": 0.10,
    "DELTA1": 0.30,
    "DELTA2": 0.10,
    "DELTA3": 0.05,
}

_KEY_ALIASES = {
    "abs_rel": "REL",
    "rel": "REL",
    "rmse": "RMSE",
    "log10": "LOG10",
    "delta1": "DELTA1",
    "delta2": "DELTA2",
    "delta3": "DELTA3",
}


def _canonical_key(key: str) -> str:
    k = key.strip()
    if k in BRS_METRIC_KEYS:
        return k
    if k in _KEY_ALIASES:
        return _KEY_ALIASES[k]
    upper = k.upper()
    if upper in BRS_METRIC_KEYS:
        return upper
    if upper in _KEY_ALIASES:
        return _KEY_ALIASES[upper]
    return upper


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        canon = _canonical_key(str(key))
        if canon in BRS_METRIC_KEYS:
            out[canon] = float(value)
    return out


def normalize_wbrs_weights(weights: Mapping[str, Any] | None = None) -> dict[str, float]:
    if weights is None:
        return dict(DEFAULT_WBRS_WEIGHTS)
    out: dict[str, float] = {}
    for key, value in weights.items():
        canon = _canonical_key(str(key))
        if canon in BRS_METRIC_KEYS:
            out[canon] = float(value)
    missing = [k for k in BRS_METRIC_KEYS if k not in out]
    if missing:
        raise ValueError(f"WBRS weights missing keys: {missing}")
    total = sum(out[k] for k in BRS_METRIC_KEYS)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"WBRS weights must sum to 1.0, got {total:.8f}")
    return out


def compute_wbrs(
    metrics: dict[str, Any],
    reference: dict[str, float],
    weights: Mapping[str, Any] | None = None,
    eps: float = 1e-6,
) -> float:
    """Weighted BRS v1.1; lower is better (~1 ≈ reference-level)."""
    cur = normalize_metrics(metrics)
    ref = normalize_metrics(reference)
    w = normalize_wbrs_weights(weights)
    for key in BRS_METRIC_KEYS:
        if key not in cur or key not in ref:
            raise ValueError(f"WBRS requires {key} in current and reference metrics")
    score = 0.0
    for key in ("RMSE", "REL", "LOG10"):
        score += w[key] * (cur[key] / max(ref[key], eps))
    for key in ("DELTA1", "DELTA2", "DELTA3"):
        score += w[key] * (ref[key] / max(cur[key], eps))
    return float(score)
