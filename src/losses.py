"""Differentiable SSIM and Pearson, averaged over examples and planes."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _ssim(recons: Tensor, targets: Tensor) -> Tensor:
    # Uniform 7 x 7 windows, valid centers and sample covariance.
    mx = F.avg_pool2d(recons, 7, stride=1)
    my = F.avg_pool2d(targets, 7, stride=1)
    correction = 49 / 48
    vx = correction * (F.avg_pool2d(recons.square(), 7, stride=1) - mx.square()).clamp_min(0)
    vy = correction * (F.avg_pool2d(targets.square(), 7, stride=1) - my.square()).clamp_min(0)
    cov = correction * (F.avg_pool2d(recons * targets, 7, stride=1) - mx * my)
    luminance = (2 * mx * my + 0.01**2) / (mx.square() + my.square() + 0.01**2)
    structure = (2 * cov + 0.03**2) / (vx + vy + 0.03**2)
    return (luminance * structure).clamp(-1, 1).mean(dim=(-2, -1))


def _pearson(recons: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    x, y = recons.flatten(2), targets.flatten(2)
    valid = (x.amax(dim=-1) > x.amin(dim=-1)) & (y.amax(dim=-1) > y.amin(dim=-1))
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    x = F.normalize(x, dim=-1, eps=1e-8)
    y = F.normalize(y, dim=-1, eps=1e-8)
    correlation = (x * y).sum(dim=-1).clamp(-1, 1)
    # Use zero correlation for a constant plane and keep its SSIM contribution.
    return torch.where(valid, correlation, 0.0), valid


def plane_metrics(recons: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-plane SSIM, Pearson and a mask of defined Pearson values."""
    if recons.shape != targets.shape or recons.ndim != 4 or recons.shape[1] != 4 or min(recons.shape[-2:]) < 7:
        raise ValueError("Expected matching [B, 4, H, W] images with H and W at least 7")
    pearson, valid = _pearson(recons, targets)
    return _ssim(recons, targets), pearson, valid


def reconstruction_loss(
    recons: Tensor, targets: Tensor, *, alpha: float = 0.7, beta: float = 0.3,
) -> Tensor:
    """Compare equally weighted pairs of normalized images in [0, 1]."""
    if not math.isfinite(alpha) or not math.isfinite(beta) or min(alpha, beta) < 0 or max(alpha, beta) <= 0:
        raise ValueError("Loss weights must be finite, nonnegative and at least one must be positive")
    ssim, pearson, _ = plane_metrics(recons, targets)
    return alpha * (1 - ssim.mean()) + beta * (1 - pearson.mean())
