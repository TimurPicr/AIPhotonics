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

    Production inputs are 128x128. Spatial sizes are multiples of
    8 * interleave_factor. Space/depth permutations preserve all input pixels;
    GroupNorm supports the single-example overfit experiment.
    """

    def __init__(self, base_channels: int = 16, interleave_factor: int = 1) -> None:
        super().__init__()
        if not isinstance(base_channels, int) or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        self.base_channels = base_channels
        if interleave_factor not in (1, 2, 4, 8, 16):
            raise ValueError("interleave_factor must be one of 1, 2, 4, 8, 16")
        self.interleave_factor = interleave_factor
        c = base_channels
        self.encoder1 = _ConvBlock(4 * interleave_factor**2, c)
        self.encoder2 = _ConvBlock(c, 2 * c)
        self.encoder3 = _ConvBlock(2 * c, 4 * c)
        self.bottleneck = _ConvBlock(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.decoder3 = _ConvBlock(12 * c, 4 * c)
        self.decoder2 = _ConvBlock(6 * c, 2 * c)
        self.decoder1 = _ConvBlock(3 * c, c)
        self.head = nn.Conv2d(c, interleave_factor**2, 1)

    @staticmethod
    def _join(deep: Tensor, skip: Tensor) -> Tensor:
        up = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((up, skip), dim=1)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 4:
            raise ValueError("images must have shape [B, 4, H, W]")
        if not images.is_floating_point():
            raise TypeError("images must be floating-point targets in [0, 1]")
        divisor = 8 * self.interleave_factor
        if min(images.shape[-2:]) < divisor or any(n % divisor for n in images.shape[-2:]):
            raise ValueError(f"image height and width must be positive multiples of {divisor}")
        packed = F.pixel_unshuffle(images, self.interleave_factor) if self.interleave_factor > 1 else images
        e1 = self.encoder1(packed)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        deep = self.bottleneck(self.pool(e3))
        d3 = self.decoder3(self._join(deep, e3))
        d2 = self.decoder2(self._join(d3, e2))
        d1 = self.decoder1(self._join(d2, e1))
        raw = self.head(d1)
        if self.interleave_factor > 1:
            raw = F.pixel_shuffle(raw, self.interleave_factor)
        return math.pi * torch.tanh(raw)
