"""Compact four-plane phase predictor; the Fresnel operator lives outside the net."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = math.gcd(4, out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class PhaseUNet(nn.Module):
    """U-Net: [B, 4, H, W] targets -> [B, 1, H, W] radians in [-pi, pi].

    Production inputs are 128x128. Multiples of eight also work for small
    integration tests. GroupNorm supports the single-example overfit experiment.
    """

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if not isinstance(base_channels, int) or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        self.base_channels = base_channels
        c = base_channels
        self.encoder1 = _ConvBlock(4, c)
        self.encoder2 = _ConvBlock(c, 2 * c)
        self.encoder3 = _ConvBlock(2 * c, 4 * c)
        self.bottleneck = _ConvBlock(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.decoder3 = _ConvBlock(12 * c, 4 * c)
        self.decoder2 = _ConvBlock(6 * c, 2 * c)
        self.decoder1 = _ConvBlock(3 * c, c)
        self.head = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _join(deep: Tensor, skip: Tensor) -> Tensor:
        up = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((up, skip), dim=1)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 4:
            raise ValueError("images must have shape [B, 4, H, W]")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point targets in [0, 1]")
        if min(images.shape[-2:]) < 8 or any(n % 8 for n in images.shape[-2:]):
            raise ValueError("image height and width must be positive multiples of eight")
        e1 = self.encoder1(images)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        deep = self.bottleneck(self.pool(e3))
        d3 = self.decoder3(self._join(deep, e3))
        d2 = self.decoder2(self._join(d3, e2))
        d1 = self.decoder1(self._join(d2, e1))
        return math.pi * torch.tanh(self.head(d1))
