"""Shared field conversion and normalization for training, baseline and export."""

from torch import Tensor

from config import DEFAULT_CONFIG, OpticalConfig
from src.physics import propagate_four


def reconstruction_values(field: Tensor, *, config: OpticalConfig = DEFAULT_CONFIG) -> Tensor:
    amplitude = field.abs()
    if config.reconstruction_value == "amplitude":
        return amplitude
    if config.reconstruction_value == "intensity":
        return amplitude.square()
    raise ValueError(f"Unknown reconstruction mode: {config.reconstruction_value}")


def reconstruct_four(phase: Tensor, *, config: OpticalConfig = DEFAULT_CONFIG) -> Tensor:
    """Return four unnormalized, nonnegative reconstructions in distance order."""
    return reconstruction_values(propagate_four(phase, config=config), config=config)


def normalize_recons(values: Tensor, *, config: OpticalConfig = DEFAULT_CONFIG) -> Tensor:
    """Scale each sample and plane by its own maximum, with a floor for zeros."""
    if config.normalization != "per_plane_max":
        raise ValueError(f"Unknown normalization: {config.normalization}")
    peak = values.amax(dim=(-2, -1), keepdim=True).clamp_min(config.normalization_eps)
    return values / peak
