"""Physics-driven, equally weighted four-plane reconstruction objective."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

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
    return torch.mean((predictions - targets).square())


def physics_loss(
    phase: Tensor,
    targets: Tensor,
    *,
    reconstruct_fn: TensorOperator | None = None,
    normalize_fn: TensorOperator | None = None,
) -> Tensor:
    """Backpropagate through all four Fresnel fields to the predicted phase."""
    if reconstruct_fn is None:
        from src.reconstruction import reconstruct_four

        reconstruct_fn = reconstruct_four
    return reconstruction_mse(reconstruct_fn(phase), targets, normalize_fn=normalize_fn)
