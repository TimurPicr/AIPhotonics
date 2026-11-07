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


_SPATIAL_DIMS = (-2, -1)


def _chirps(field: Tensor, z_m: float, config: OpticalConfig) -> tuple[Tensor, Tensor]:
    size = field.shape[-1]
    output_pitch = output_pitch_m(z_m, size, config=config)
    phase_scale = math.pi / (config.wavelength_m * z_m)
    # Compute the static coefficients in double precision before casting to the FFT dtype.
    device = torch.device("cpu") if field.device.type == "mps" else field.device
    input_chirp = torch.exp(1j * phase_scale * _radius_squared(size, config.slm_pitch_m, device))
    output_chirp = torch.exp(1j * phase_scale * _radius_squared(size, output_pitch, device))
    return input_chirp.to(field), output_chirp.to(field)


def _centered_fft(field: Tensor, *, norm: str) -> Tensor:
    centered_input = torch.fft.ifftshift(field, dim=_SPATIAL_DIMS)
    spectrum = torch.fft.fft2(centered_input, dim=_SPATIAL_DIMS, norm=norm)
    return torch.fft.fftshift(spectrum, dim=_SPATIAL_DIMS)


class UnitaryFresnel:
    """Fresnel operator scaled by dx_out/dx_in, with its discrete adjoint.

    The complex field supplied at construction sets the grid, device and dtype.
    Reuse the coefficients across iterations on that grid.
    """

    def __init__(self, field: Tensor, z_m: float, *, config: OpticalConfig = DEFAULT_CONFIG):
        if not field.is_complex() or field.ndim < 2 or field.shape[-2] != field.shape[-1]:
            raise ValueError("Fresnel needs a complex field on a square grid")
        self.input_chirp, output_chirp = _chirps(field, z_m, config)
        self.output_factor = output_chirp * (cmath.exp(2j * math.pi * z_m / config.wavelength_m) / 1j)

    def forward(self, field: Tensor) -> Tensor:
        return _centered_fft(field * self.input_chirp, norm="ortho") * self.output_factor

    def adjoint(self, field: Tensor) -> Tensor:
        centered_input = torch.fft.ifftshift(field * self.output_factor.conj(), dim=_SPATIAL_DIMS)
        spatial = torch.fft.ifft2(centered_input, dim=_SPATIAL_DIMS, norm="ortho")
        return torch.fft.fftshift(spatial, dim=_SPATIAL_DIMS) * self.input_chirp.conj()


def _fresnel_field_1fft(incident: Tensor, z_m: float, config: OpticalConfig) -> Tensor:
    wavelength_z = config.wavelength_m * z_m
    input_chirp, output_chirp = _chirps(incident, z_m, config)
    spectrum = _centered_fft(incident * input_chirp, norm="backward")
    scale = cmath.exp(2j * math.pi * z_m / config.wavelength_m) * config.slm_pitch_m**2 / (1j * wavelength_z)
    return spectrum * output_chirp * scale


def fresnel_1fft(phase: Tensor, z_m: float, *, config: OpticalConfig = DEFAULT_CONFIG) -> Tensor:
    """Propagate [B, 1, N, N] radians to a complex field of the same shape."""
    if phase.ndim != 4 or phase.shape[1] != 1 or phase.shape[-2] != phase.shape[-1]:
        raise ValueError("phase must have shape [B, 1, N, N]")
    if phase.dtype not in (torch.float32, torch.float64):
        raise TypeError("phase must have dtype float32 or float64")
    return _fresnel_field_1fft(torch.exp(1j * phase), z_m, config)


def propagate_four(phase: Tensor, *, config: OpticalConfig = DEFAULT_CONFIG) -> Tensor:
    """Propagate in configuration order; output has shape [B, 4, N, N]."""
    return torch.cat([fresnel_1fft(phase, z, config=config) for z in config.distances_m], dim=1)
