"""Optional quantization-aware Adam refinement of a predicted phase mask."""

import math

import torch
from torch import Tensor

from src.io import quantize_phase
from src.losses import reconstruction_loss
from src.reconstruction import normalize_recons, reconstruct_four


@torch.inference_mode(False)
@torch.enable_grad()
def refine_phase(
    initial_phase: Tensor, targets: Tensor, *, steps: int = 0, lr: float = 0.1,
    alpha: float = 0.7, beta: float = 0.3,
) -> Tensor:
    """Optimize only the mask, preserving the input tensors and network weights.

    A zero budget returns the original values. Positive budgets use periodic
    8-bit phase quantization with a straight-through gradient. Works under
    no_grad/inference_mode and supports independent quartets in one batch.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("Refinement steps must be a nonnegative integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Refinement learning rate must be finite and positive")
    if steps == 0:
        return initial_phase.detach()
    if (targets.ndim != 4 or targets.shape[0] < 1 or targets.shape[1] != 4
            or initial_phase.shape != targets[:, :1].shape):
        raise ValueError("Expected phase [B, 1, 128, 128] and targets [B, 4, 128, 128]")
    if targets.dtype not in (torch.float32, torch.float64) or initial_phase.dtype != targets.dtype:
        raise TypeError("Phase and targets must have the same float32 or float64 dtype")
    if initial_phase.device != targets.device:
        raise ValueError("Phase and targets must be on the same device")
    if not ((targets >= 0) & (targets <= 1)).all():
        raise ValueError("Targets must contain finite values in [0, 1]")
    # Cloning also makes tensors created under inference_mode safe for autograd.
    phase = initial_phase.detach().clone().requires_grad_(True)
    targets = targets.detach().clone()
    optimizer = torch.optim.Adam([phase], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        wrapped = (phase + math.pi).remainder(2 * math.pi) - math.pi
        quantized = quantize_phase(wrapped, keep_device=True)
        phase_ste = phase + (quantized - phase).detach()
        recons = normalize_recons(reconstruct_four(phase_ste))
        # Sum per-quartet losses so Adam's effective epsilon is batch-independent.
        loss = len(targets) * reconstruction_loss(recons, targets, alpha=alpha, beta=beta)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite phase refinement loss")
        loss.backward()
        optimizer.step()
    return (phase.detach() + math.pi).remainder(2 * math.pi) - math.pi
