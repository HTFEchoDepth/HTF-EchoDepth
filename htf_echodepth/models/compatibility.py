"""Checkpoint loading helpers preserving legacy state_dict key compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def extract_state_dict(payload: dict[str, Any] | Any) -> dict[str, torch.Tensor]:
    """Return a model state_dict from a checkpoint payload or raw dict."""
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Checkpoint payload must be a dict with 'state_dict' or a raw state_dict.")


def normalize_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip DataParallel ``module.`` prefixes for checkpoint compatibility."""
    if not state:
        return state
    if not any(k.startswith("module.") for k in state):
        return state
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}


def load_model_state(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights into ``model`` from a ``.pth`` checkpoint file."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = normalize_state_dict_keys(extract_state_dict(payload))
    model.load_state_dict(state, strict=strict)
    meta = payload if isinstance(payload, dict) else {"state_dict": payload}
    return meta
