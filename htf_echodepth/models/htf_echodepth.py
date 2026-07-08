"""HTF-EchoDepth public model entry point."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


_RELEASE_MESSAGE = "HTF-EchoDepth model implementation will be released soon."


class HTFEchoDepth(nn.Module):
    """Public placeholder for the HTF-EchoDepth architecture."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(_RELEASE_MESSAGE)


def build_htf_echodepth(**kwargs: Any) -> HTFEchoDepth:
    """Factory for the public HTF-EchoDepth placeholder."""
    return HTFEchoDepth(**kwargs)
