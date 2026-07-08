"""
FADC-inspired v0: frequency reweighting + multi-dilation branches + spatial softmax fusion.
Pure PyTorch; fixed depthwise Gaussian blur (non-learnable).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqRFBlockV0(nn.Module):
    """Insert after encoder down at 32×32 with 256 channels (B,256,32,32)."""

    C = 256
    H = 32
    W = 32

    def __init__(self):
        super().__init__()
        # Fixed depthwise 3×3 Gaussian kernel [[1,2,1],[2,4,2],[1,2,1]] / 16 — buffer only (not a Conv2d submodule) so init_net does not overwrite it.
        blur_w = torch.tensor(
            [[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]
        ) / 16.0
        blur_w = blur_w.repeat(self.C, 1, 1, 1).contiguous()  # (256, 1, 3, 3)
        self.register_buffer("blur_weight", blur_w)

        self.gate_low = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gate_high = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.branch_d1 = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
        )
        self.branch_d4 = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=4, dilation=4, bias=True),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
        )

        self.fuse_logits = nn.Conv2d(1024, 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(self.C, self.C, kernel_size=1, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 256, 32, 32)
        assert x.dim() == 4, f"expected 4D tensor, got {x.dim()}D"
        b, c, h, w = x.shape
        assert c == self.C and h == self.H and w == self.W, (
            f"FreqRFBlockV0 expects (B,{self.C},{self.H},{self.W}), got {(b, c, h, w)}"
        )

        low = F.conv2d(x, self.blur_weight, bias=None, stride=1, padding=1, groups=self.C)
        high = x - low

        x_low = torch.cat([x, low], dim=1)
        x_high = torch.cat([x, high], dim=1)
        w_low = self.gate_low(x_low)
        w_high = self.gate_high(x_high)
        assert w_low.shape == (b, 1, h, w) and w_high.shape == (b, 1, h, w)

        x_freq = x + self.alpha * (w_low * low + w_high * high)

        z_d1 = self.branch_d1(x_freq)
        z_d2 = self.branch_d2(x_freq)
        z_d4 = self.branch_d4(x_freq)

        logits = self.fuse_logits(torch.cat([x_freq, z_d1, z_d2, z_d4], dim=1))
        attn = F.softmax(logits, dim=1)
        a1 = attn[:, 0:1, :, :]
        a2 = attn[:, 1:2, :, :]
        a4 = attn[:, 2:3, :, :]
        z = a1 * z_d1 + a2 * z_d2 + a4 * z_d4

        y = x + self.beta * self.proj(z)
        return y
