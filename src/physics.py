"""Differentiable single-FFT Fresnel propagation on a centered square grid."""

import cmath
import math

import torch
from torch import Tensor

from config import DEFAULT_CONFIG, OpticalConfig


def output_pitch_m(z_m: float, size: int, *, config: OpticalConfig = DEFAULT_CONFIG) -> float:
    """Return the output pixel pitch on the native Fresnel grid at z."""
    if size < 1 or not math.isfinite(z_m) or z_m <= 0:
        raise ValueError("Fresnel propagation needs a positive grid size and finite positive z")
    return config.wavelength_m * z_m / (size * config.slm_pitch_m)


def _radius_squared(size: int, pitch_m: float, device: torch.device) -> Tensor:
    # Double precision keeps the argument of the quadratic phase accurate at the corners.
    axis = (torch.arange(size, device=device, dtype=torch.float64) - size // 2) * pitch_m
    return axis[:, None].square() + axis[None, :].square()


def fresnel_1fft(phase: Tensor, z_m: float, *, config: OpticalConfig = DEFAULT_CONFIG) -> Tensor:
    """Propagate [B, 1, N, N] radians to a complex field of the same shape."""
    if phase.ndim != 4 or phase.shape[1] != 1 or phase.shape[-2] != phase.shape[-1]:
        raise ValueError("phase must have shape [B, 1, N, N]")
    if phase.dtype not in (torch.float32, torch.float64):
        raise TypeError("phase must have dtype float32 or float64")

    size = phase.shape[-1]
    output_pitch = output_pitch_m(z_m, size, config=config)
    wavelength_z = config.wavelength_m * z_m
    phase_scale = math.pi / wavelength_z
    incident = torch.exp(1j * phase)
    # Compute the static coefficients in double precision before casting to the FFT dtype.
    coefficient_device = "cpu" if phase.device.type == "mps" else phase.device
    input_chirp = torch.exp(1j * phase_scale * _radius_squared(size, config.slm_pitch_m, coefficient_device))
    output_chirp = torch.exp(1j * phase_scale * _radius_squared(size, output_pitch, coefficient_device))
    input_chirp = input_chirp.to(incident)
    output_chirp = output_chirp.to(incident)

    dims = (-2, -1)
    centered_input = torch.fft.ifftshift(incident * input_chirp, dim=dims)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered_input, dim=dims, norm="backward"), dim=dims)
    scale = cmath.exp(2j * math.pi * z_m / config.wavelength_m) * config.slm_pitch_m**2 / (1j * wavelength_z)
    return spectrum * output_chirp * scale


def propagate_four(phase: Tensor, *, config: OpticalConfig = DEFAULT_CONFIG) -> Tensor:
    """Propagate in configuration order; output has shape [B, 4, N, N]."""
    return torch.cat([fresnel_1fft(phase, z, config=config) for z in config.distances_m], dim=1)
