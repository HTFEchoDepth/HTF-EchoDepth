"""MRCR — Metric-Role Candidate Retention for MG-WSF donor pool."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METRIC_KEYS = ("RMSE", "REL", "LOG10", "DELTA1", "DELTA2", "DELTA3", "WBRS")

DONOR_ROLE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("C_WBRS", "WBRS", "min"),
    ("C_REL", "REL", "min"),
    ("C_RMSE", "RMSE", "min"),
    ("C_d2", "DELTA2", "max"),
    ("C_d3", "DELTA3", "max"),
    ("C_d1", "DELTA1", "max"),
)

ROLE_PRIORITY = {
    "best_wbrs": 0,
    "best_rmse": 1,
    "best_rel": 1,
    "best_log10": 1,
    "best_delta1": 1,
    "best_delta2": 1,
    "best_delta3": 1,
    "last": 3,
}


def role_from_filename(name: str) -> str:
    if name.startswith("best_log10"):
        return "best_log10"
    if name.startswith("best_wbrs"):
        return "best_wbrs"
    if name.startswith("best_rmse"):
        return "best_rmse"
    if name.startswith("best_rel"):
        return "best_rel"
    if name.startswith("best_delta1"):
        return "best_delta1"
    if name.startswith("best_delta2"):
        return "best_delta2"
    if name.startswith("best_delta3"):
        return "best_delta3"
    if name == "last.pth":
        return "last"
    return "other"


def is_pool_candidate(row: dict[str, Any]) -> bool:
    """Checkpoints eligible for MRCR specialist search."""
    path = str(row.get("checkpoint_path", ""))
    name = row.get("checkpoint_name") or Path(path).name
    path_norm = path.replace("\\", "/")
    if name.startswith("best_") and name.endswith(".pth"):
        return True
    if name == "last.pth":
        return True
    if name.startswith("epoch_") and name.endswith(".pth") and "/validation_epochs/" in path_norm:
        return True
    return False


def _row_sort_key(row: dict[str, Any]) -> tuple:
    path = str(row.get("checkpoint_path", ""))
    name = row.get("checkpoint_name") or Path(path).name
    role = row.get("role_from_filename") or role_from_filename(name)
    return (ROLE_PRIORITY.get(role, 99), row.get("epoch", 9999), name)


def retain_metric_role_candidates(
    validation_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """MRCR: pick best checkpoint per metric role on validation."""
    pool_input = [r for r in validation_records if is_pool_candidate(r)]
    if not pool_input:
        raise RuntimeError("No MRCR-eligible checkpoints in validation records")

    by_path = {r["checkpoint_path"]: r for r in pool_input}
    specialists: dict[str, dict[str, Any]] = {}

    for donor_type, metric, direction in DONOR_ROLE_SPECS:
        valid = [r for r in pool_input if np.isfinite(r.get(metric, float("nan")))]
        if not valid:
            continue
        if direction == "min":
            best_val = min(r[metric] for r in valid)
            tied = [r for r in valid if abs(r[metric] - best_val) < 1e-9]
            best = min(tied, key=_row_sort_key)
        else:
            best_val = max(r[metric] for r in valid)
            tied = [r for r in valid if abs(r[metric] - best_val) < 1e-9]
            best = min(tied, key=_row_sort_key)
        specialists[donor_type] = best

    path_to_types: dict[str, list[str]] = defaultdict(list)
    spec_metric = {dt: met for dt, met, _ in DONOR_ROLE_SPECS}
    for donor_type, row in specialists.items():
        path_to_types[row["checkpoint_path"]].append(donor_type)

    pool_rows: list[dict[str, Any]] = []
    for path, types in sorted(path_to_types.items(), key=lambda x: x[0]):
        row = by_path[path]
        pool_rows.append(
            {
                "donor_type": ";".join(sorted(types)),
                "checkpoint_path": path,
                "checkpoint_name": row["checkpoint_name"],
                "epoch": row.get("epoch", ""),
                "metric_value": {t: specialists[t][spec_metric[t]] for t in types},
            }
        )
    return pool_rows, specialists
