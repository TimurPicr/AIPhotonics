"""Adam baseline that optimizes one phase-only SLM mask per target quartet."""

import math

import torch
from torch import Tensor

from src.losses import reconstruction_loss
from src.reconstruction import normalize_recons, reconstruct_four


@torch.enable_grad()
def optimize_phase(
    targets: Tensor, *, steps: int = 100, seed: int = 42, lr: float = 0.1,
    alpha: float = 0.7, beta: float = 0.3,
) -> Tensor:
    if targets.ndim != 4 or targets.shape[:2] != (1, 4) or targets.shape[-1] != targets.shape[-2]:
        raise ValueError("Adam expects one target quartet of shape [1, 4, N, N]")
    if targets.dtype not in (torch.float32, torch.float64):
        raise TypeError("Targets must be float32 or float64")
    if not ((targets >= 0) & (targets <= 1)).all():
        raise ValueError("Targets must contain finite values in [0, 1]")
    if steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Adam steps and learning rate must be positive")
    generator = torch.Generator().manual_seed(seed)
    phase = (torch.rand(targets[:, :1].shape, generator=generator, dtype=targets.dtype) * (2 * math.pi) - math.pi)
    phase = phase.to(targets).requires_grad_()
    optimizer = torch.optim.Adam([phase], lr=lr)
    targets = targets.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        recons = normalize_recons(reconstruct_four(phase))
        loss = reconstruction_loss(recons, targets, alpha=alpha, beta=beta)
        loss.backward()
        optimizer.step()
    return (phase.detach() + math.pi).remainder(2 * math.pi) - math.pi
