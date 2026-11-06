"""Physics-driven, equally weighted four-plane reconstruction objective."""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor
from torch.nn import functional as F

TensorOperator = Callable[[Tensor], Tensor]


def reconstruction_mse(
    values: Tensor,
    targets: Tensor,
    *,
    normalize_fn: TensorOperator | None = None,
) -> Tensor:
    """Mean squared error after A's shared reconstruction normalization.

    ``normalize_fn`` is an explicit injection point for interface tests. Production
    callers use src.reconstruction.normalize_recons, never a copied formula.
    """
    return reconstruction_loss(values, targets, normalize_fn=normalize_fn)


def differentiable_ssim(predictions: Tensor, targets: Tensor) -> Tensor:
    """C's SSIM: uniform 7x7 window, sample covariance, data_range=1, valid crop."""
    if predictions.shape != targets.shape or predictions.ndim != 4 or min(predictions.shape[-2:]) < 7:
        raise ValueError("SSIM needs matching [B, C, H, W] tensors with H,W >= 7")
    mean_x = F.avg_pool2d(predictions, 7, stride=1)
    mean_y = F.avg_pool2d(targets, 7, stride=1)
    correction = 49 / 48
    variance_x = correction * (F.avg_pool2d(predictions.square(), 7, stride=1) - mean_x.square())
    variance_y = correction * (F.avg_pool2d(targets.square(), 7, stride=1) - mean_y.square())
    covariance = correction * (F.avg_pool2d(predictions * targets, 7, stride=1) - mean_x * mean_y)
    numerator = (2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_x.square() + mean_y.square() + 0.01**2) * (variance_x + variance_y + 0.03**2)
    return (numerator / denominator).mean()


def reconstruction_loss(
    values: Tensor,
    targets: Tensor,
    *,
    normalize_fn: TensorOperator | None = None,
    ssim_weight: float = 0.0,
) -> Tensor:
    if not math.isfinite(ssim_weight) or ssim_weight < 0:
        raise ValueError("ssim_weight must be finite and nonnegative")
    if values.shape != targets.shape or values.ndim != 4 or values.shape[1] != 4:
        raise ValueError("values and targets must have the same shape [B, 4, H, W]")
    if not values.is_floating_point() or not targets.is_floating_point():
        raise TypeError("reconstructions and targets must be floating point")
    if normalize_fn is None:
        from src.reconstruction import normalize_recons

        normalize_fn = normalize_recons
    predictions = normalize_fn(values)
    if predictions.shape != targets.shape:
        raise ValueError("normalization must preserve reconstruction shape")
    loss = torch.mean((predictions - targets).square())
    if ssim_weight:
        loss = loss + ssim_weight * (1 - differentiable_ssim(predictions, targets))
    return loss


def physics_loss(
    phase: Tensor,
    targets: Tensor,
    *,
    reconstruct_fn: TensorOperator | None = None,
    normalize_fn: TensorOperator | None = None,
    ssim_weight: float = 0.0,
) -> Tensor:
    """Backpropagate through all four Fresnel fields to the predicted phase."""
    if reconstruct_fn is None:
        from src.reconstruction import reconstruct_four

        reconstruct_fn = reconstruct_four
    return reconstruction_loss(
        reconstruct_fn(phase), targets, normalize_fn=normalize_fn, ssim_weight=ssim_weight,
    )
