"""Weight-space checkpoint fusion for MG-WSF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def mix_state_dicts(
    checkpoint_paths: list[str | Path],
    weights: list[float],
    *,
    anchor_index: int = 0,
) -> dict[str, Any]:
    """Linearly mix floating-point tensors; copy non-float tensors from anchor."""
    paths = [Path(p) for p in checkpoint_paths]
    payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    state_dicts = [p["state_dict"] for p in payloads]
    keys0 = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != keys0:
            raise RuntimeError(f"state_dict key mismatch between checkpoint 0 and {i}")

    anchor_sd = state_dicts[anchor_index]
    mixed: dict[str, torch.Tensor] = {}
    for k in sorted(keys0):
        tensors = [sd[k] for sd in state_dicts]
        if any(t.shape != tensors[0].shape for t in tensors):
            raise RuntimeError(f"shape mismatch on key {k}")
        if tensors[0].dtype.is_floating_point:
            acc = torch.zeros_like(tensors[0], dtype=torch.float32)
            for w, t in zip(weights, tensors):
                acc = acc + w * t.float()
            mixed[k] = acc.to(tensors[0].dtype)
        else:
            mixed[k] = anchor_sd[k].clone()

    formula = " + ".join(f"{w:.4f}*{p.name}" for w, p in zip(weights, paths))
    return {
        "epoch": -1,
        "fusion_tag": "mg_wsf_fused",
        "fusion_formula": formula,
        "source_checkpoints": [
            {"path": str(p), "epoch": payloads[i].get("epoch")} for i, p in enumerate(paths)
        ],
        "state_dict": mixed,
        "optimizer": None,
        "note": "MG-WSF weight-space fusion; optimizer not fused",
    }
