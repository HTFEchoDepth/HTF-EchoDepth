"""CandidateManager — validation-based metric-role checkpoint retention for MG-WSF."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch.nn as nn

from htf_echodepth.fusion.candidate_retention import METRIC_KEYS, retain_metric_role_candidates, role_from_filename
from htf_echodepth.utils.checkpoint import save_checkpoint


class CandidateManager:
    """Track validation metrics and retain metric-role candidate checkpoints (MRCR)."""

    METRIC_ROLES: dict[str, tuple[str, str]] = {
        "best_rmse": ("rmse", "min"),
        "best_rel": ("rel", "min"),
        "best_log10": ("log10", "min"),
        "best_delta1": ("delta1", "max"),
        "best_delta2": ("delta2", "max"),
        "best_delta3": ("delta3", "max"),
        "best_wbrs": ("WBRS", "min"),
    }

    def __init__(self, output_dir: str | Path, *, enabled_roles: dict[str, bool] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled_roles = enabled_roles or {role: True for role in self.METRIC_ROLES}
        self.best_values: dict[str, float] = {}
        self.validation_records: list[dict[str, Any]] = []
        self.epoch_rows: list[dict[str, Any]] = []

    def _is_better(self, role: str, value: float) -> bool:
        _, direction = self.METRIC_ROLES[role]
        if role not in self.best_values:
            return True
        prev = self.best_values[role]
        if direction == "min":
            return value < prev - 1e-12
        return value > prev + 1e-12

    def _metric_value(self, role: str, val_metrics: dict[str, float], canon: dict[str, float], wbrs: float) -> float:
        key, _ = self.METRIC_ROLES[role]
        if key == "WBRS":
            return float(wbrs)
        if key in val_metrics:
            return float(val_metrics[key])
        return float(canon.get(key.upper(), float("nan")))

    def update(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Any,
        *,
        val_metrics: dict[str, float],
        canon: dict[str, float],
        wbrs: float,
        save_last: bool = False,
    ) -> list[str]:
        """Update metric-role bests; return list of newly saved checkpoint filenames."""
        saved: list[str] = []
        row = {
            "epoch": epoch,
            **{k: float(val_metrics[k]) for k in ("rmse", "rel", "log10", "delta1", "delta2", "delta3")},
            "WBRS": float(wbrs),
        }
        self.epoch_rows.append(row)

        for role, enabled in self.enabled_roles.items():
            if not enabled:
                continue
            value = self._metric_value(role, val_metrics, canon, wbrs)
            if not self._is_better(role, value):
                continue
            self.best_values[role] = value
            ckpt_name = f"{role}.pth"
            ckpt_path = self.output_dir / ckpt_name
            metrics_payload = {**canon, "WBRS": wbrs}
            save_checkpoint(ckpt_path, model, epoch=epoch, optimizer=optimizer, metrics=metrics_payload, tag=role)
            saved.append(ckpt_name)
            self._register_record(epoch, ckpt_path, metrics_payload)

        if save_last:
            last_path = self.output_dir / "last.pth"
            save_checkpoint(last_path, model, epoch=epoch, optimizer=optimizer, metrics={**canon, "WBRS": wbrs}, tag="last")
            saved.append("last.pth")
            self._register_record(epoch, last_path, {**canon, "WBRS": wbrs})

        return saved

    def _register_record(self, epoch: int, ckpt_path: Path, metrics: dict[str, float]) -> None:
        rel_path = ckpt_path.relative_to(self.output_dir).as_posix()
        name = ckpt_path.name
        record = {
            "epoch": epoch,
            "checkpoint_path": rel_path,
            "checkpoint_name": name,
            "role_from_filename": role_from_filename(name),
            "RMSE": float(metrics.get("RMSE", metrics.get("rmse", float("nan")))),
            "REL": float(metrics.get("REL", metrics.get("rel", float("nan")))),
            "LOG10": float(metrics.get("LOG10", metrics.get("log10", float("nan")))),
            "DELTA1": float(metrics.get("DELTA1", metrics.get("delta1", float("nan")))),
            "DELTA2": float(metrics.get("DELTA2", metrics.get("delta2", float("nan")))),
            "DELTA3": float(metrics.get("DELTA3", metrics.get("delta3", float("nan")))),
            "WBRS": float(metrics.get("WBRS", float("nan"))),
        }
        self.validation_records = [r for r in self.validation_records if r["checkpoint_name"] != name]
        self.validation_records.append(record)

    def write_val_metrics_csv(self) -> Path:
        path = self.output_dir / "val_metrics.csv"
        fieldnames = ["epoch", "rmse", "rel", "log10", "delta1", "delta2", "delta3", "WBRS"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.epoch_rows)
        return path

    def _role_metric_name(self, role: str) -> str:
        key, _ = self.METRIC_ROLES.get(role, ("", ""))
        if key == "WBRS":
            return "WBRS"
        return key

    def write_candidate_registry(self) -> Path:
        path = self.output_dir / "candidate_registry.csv"
        rows: list[dict[str, Any]] = []
        for record in sorted(self.validation_records, key=lambda r: r["checkpoint_name"]):
            role = record["role_from_filename"]
            metric_name = self._role_metric_name(role) if role in self.METRIC_ROLES else role
            rows.append(
                {
                    "role": role,
                    "metric_name": metric_name,
                    "checkpoint_path": record["checkpoint_path"],
                    "epoch": record["epoch"],
                    "RMSE": record["RMSE"],
                    "REL": record["REL"],
                    "LOG10": record["LOG10"],
                    "DELTA1": record["DELTA1"],
                    "DELTA2": record["DELTA2"],
                    "DELTA3": record["DELTA3"],
                    "WBRS": record["WBRS"],
                }
            )
        fieldnames = ["role", "metric_name", "checkpoint_path", "epoch"] + list(METRIC_KEYS)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def finalize(self) -> tuple[Path, Path]:
        val_path = self.write_val_metrics_csv()
        reg_path = self.write_candidate_registry()
        return val_path, reg_path

    def mrcr_pool(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        return retain_metric_role_candidates(self.validation_records)
