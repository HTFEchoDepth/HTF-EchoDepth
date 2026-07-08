"""Image / spectrogram / depth resizing and normalization."""

from __future__ import annotations

import torch
import torchvision.transforms as transforms


class MinMaxNorm(torch.nn.Module):
    """Min-max normalize tensor to [0, 1] using scalar or per-channel bounds."""

    def __init__(self, min_val: float, max_val: float) -> None:
        super().__init__()
        self.min_val = torch.tensor(float(min_val))
        self.max_val = torch.tensor(float(max_val))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] == 2:
            out = []
            for c in range(2):
                out.append((tensor[c] - self.min_val) / (self.max_val - self.min_val))
            return torch.stack(out, dim=0)
        return (tensor - self.min_val) / (self.max_val - self.min_val)


def build_resize_transform(size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize((size, size))])


def build_depth_transform(
    *,
    image_size: int,
    max_depth_m: float,
    normalize: bool = True,
) -> transforms.Compose:
    steps: list = [transforms.ToTensor(), transforms.Resize((image_size, image_size))]
    if normalize:
        steps.append(MinMaxNorm(min_val=0.0, max_val=max_depth_m))
    return transforms.Compose(steps)
