"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from htf_echodepth.models.compatibility import load_model_state


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, Any] | None = None,
    tag: str = "",
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "tag": tag,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, out)


def load_checkpoint(model: nn.Module, path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return load_model_state(model, path, **kwargs)
