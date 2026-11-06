"""Shared optical and reconstruction protocol for training and evaluation."""

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class OpticalConfig:
    wavelength_m: float = 532e-9
    slm_pitch_m: float = 10e-6
    distances_m: tuple[float, ...] = (0.0241, 0.0253, 0.0265, 0.0277)
    image_size: int = 128
    # Working assumption until the full task statement is confirmed.
    reconstruction_value: Literal["amplitude", "intensity"] = "amplitude"
    normalization: Literal["per_plane_max"] = "per_plane_max"
    normalization_eps: float = 1e-8

    @property
    def z_m(self) -> tuple[float, ...]:
        """Distance order used by the fixed-dataset generator."""
        return self.distances_m

    def snapshot(self) -> dict:
        """Return a JSON-compatible protocol for checkpoints and metrics."""
        snapshot = asdict(self)
        snapshot["distances_m"] = list(self.distances_m)
        return snapshot


DEFAULT_CONFIG = OpticalConfig()
CONFIG = DEFAULT_CONFIG
