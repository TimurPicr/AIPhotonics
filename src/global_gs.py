"""Global multi-plane Gerchberg–Saxton using the shared Fresnel operator."""

import math

import torch
from torch import Tensor

from config import CONFIG
from src.physics import UnitaryFresnel


PHASE_EPS = 1e-8
TARGET_NORM_EPS = 1e-8


def _target_amplitudes(targets: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(targets, dim=(-2, -1), keepdim=True)
    if (norm <= TARGET_NORM_EPS).any():
        raise ValueError("A black target plane is incompatible with nonzero conserved power")
    return targets * (targets.shape[-1] / norm)


def _replace_amplitude(field: Tensor, amplitude: Tensor) -> Tensor:
    return amplitude * torch.exp(1j * field.angle())


def _gs_step(phase: Tensor, amplitudes: Tensor, operators: tuple[UnitaryFresnel, ...]) -> Tensor:
    field = torch.exp(1j * phase)
    back_fields = []
    for index, operator in enumerate(operators):
        propagated = operator.forward(field)
        projected = _replace_amplitude(propagated, amplitudes[:, index:index + 1])
        back_fields.append(operator.adjoint(projected))
    mean_field = torch.stack(back_fields).mean(dim=0)
    return torch.where(mean_field.abs() <= PHASE_EPS, phase, mean_field.angle())


@torch.no_grad()
def global_mp_gs(
    targets: Tensor,
    *,
    iterations: int = 100,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> Tensor:
    """Return the phase-only mask after a fixed number of projections."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if CONFIG.reconstruction_value != "amplitude":
        raise ValueError("Global MP-GS requires the amplitude reconstruction protocol")
    if targets.ndim == 3:
        targets = targets.unsqueeze(0)
    size = CONFIG.image_size
    if targets.shape != (1, 4, size, size):
        raise ValueError(f"Expected targets of shape [1, 4, {size}, {size}] or [4, {size}, {size}]")
    if targets.dtype not in (torch.float32, torch.float64):
        raise ValueError("Targets must be float32 or float64 in [0, 1]")
    if not torch.isfinite(targets).all() or (targets < 0).any() or (targets > 1).any():
        raise ValueError("Targets must contain finite amplitudes in [0, 1]")

    targets = targets.to(device=device)
    amplitudes = _target_amplitudes(targets)
    rng = torch.Generator().manual_seed(seed)
    phase = (torch.rand((1, 1, size, size), generator=rng, dtype=targets.dtype) * (2 * math.pi) - math.pi)
    phase = phase.to(targets)
    field = torch.exp(1j * phase)
    operators = tuple(UnitaryFresnel(field, z) for z in CONFIG.z_m)
    for _ in range(iterations):
        phase = _gs_step(phase, amplitudes, operators)
    return phase
