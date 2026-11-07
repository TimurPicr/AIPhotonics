"""Predict target-plane phases and refine one phase-only mask with Fresnel propagation."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import DEFAULT_CONFIG, OpticalConfig
from src.physics import backpropagate_four, propagate_four
from src.reconstruction import normalize_recons, reconstruction_values


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = math.gcd(out_channels, 4)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.ReLU(),
    )


def _join(deep: Tensor, skip: Tensor) -> Tensor:
    up = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat((up, skip), dim=1)


def _phase(field: Tensor) -> Tensor:
    # A vanishing field has no phase. Replace it before angle to keep gradients finite.
    return torch.where(field.abs() > 1e-8, field, torch.ones_like(field)).angle()


class HologramUNet(nn.Module):
    """Map four targets in distance order to an SLM mask and four reconstructions."""

    def __init__(
        self,
        base_channels: int = 16,
        *,
        plane_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        rounds: int = 2,
        config: OpticalConfig = DEFAULT_CONFIG,
    ) -> None:
        super().__init__()
        if not isinstance(base_channels, int) or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if len(config.distances_m) != 4 or len(plane_weights) != 4:
            raise ValueError("Four optical distances and four plane weights are required")
        total = sum(plane_weights)
        if any(not math.isfinite(w) or w < 0 for w in plane_weights) or not math.isfinite(total) or total <= 0:
            raise ValueError("Plane weights must be finite, nonnegative and have a positive sum")
        self.register_buffer("plane_weights", torch.tensor([w / total for w in plane_weights]).view(1, 4, 1, 1))
        self.config = config
        self.rounds = rounds
        c = base_channels
        self.encoder1 = _conv_block(4, c)
        self.encoder2 = _conv_block(c, 2 * c)
        self.encoder3 = _conv_block(2 * c, 4 * c)
        self.bottleneck = _conv_block(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.decoder3 = _conv_block(12 * c, 4 * c)
        self.decoder2 = _conv_block(6 * c, 2 * c)
        self.decoder1 = _conv_block(3 * c, c)
        self.head = nn.Conv2d(c, 4, 1)

    def predict_plane_phases(self, targets: Tensor) -> Tensor:
        """Return four phase maps in radians, ordered like the input amplitudes."""
        if (
            targets.ndim != 4 or targets.shape[1] != 4 or targets.shape[-2] != targets.shape[-1]
            or targets.shape[-1] < 8 or targets.shape[-1] % 8
        ):
            raise ValueError("Targets must have shape [B, 4, N, N] with N a positive multiple of 8")
        if targets.dtype not in (torch.float32, torch.float64):
            raise TypeError("Targets must be float32 or float64")
        if not ((targets >= 0) & (targets <= 1)).all():
            raise ValueError("Targets must contain finite values in [0, 1]")
        e1 = self.encoder1(targets)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        deep = self.bottleneck(self.pool(e3))
        d3 = self.decoder3(_join(deep, e3))
        d2 = self.decoder2(_join(d3, e2))
        d1 = self.decoder1(_join(d2, e1))
        return math.pi * self.head(d1).tanh()

    def forward(self, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Return [B, 1, N, N] mask radians and [B, 4, N, N] normalized images."""
        plane_phases = self.predict_plane_phases(targets)
        amplitudes = targets.sqrt() if self.config.reconstruction_value == "intensity" else targets
        for index in range(self.rounds):
            target_fields = torch.polar(amplitudes, plane_phases)
            mask_fields = backpropagate_four(target_fields, config=self.config)
            mask_phase = _phase((mask_fields * self.plane_weights).sum(dim=1, keepdim=True))
            fields = propagate_four(mask_phase, config=self.config)
            if index + 1 < self.rounds:
                plane_phases = _phase(fields)
        recons = reconstruction_values(fields, config=self.config)
        return mask_phase, normalize_recons(recons, config=self.config)
