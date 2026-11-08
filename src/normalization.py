"""Target-plane normalization for multi-plane hologram datasets.

DeepCGH (UNC-optics) equalizes the L2 energy of every depth plane against
plane 0 when ``normalize=True``:

    plane_p *= sqrt(sum(plane_0 ** 2) / sum(plane_p ** 2))

and follows it with one global min-max over the sample's plane stack.  Its
targets are binary masks, for which a per-plane min-max is a no-op.  This
module adds that per-plane min-max for continuous grayscale targets: every
plane starts in [0, 1] with the same black level, so energy equalization acts
on contrast-normalized planes and the shared global scaling preserves equal
energies exactly.  That extra step is a continuous-domain adaptation, not a
line-for-line copy of DeepCGH.

This module is kept separate from ``src.reconstruction.normalize_recons``
(prediction-side scaling inside the loss) so dataset generation can switch
target conventions without touching the training loop.
"""

import torch
from torch import Tensor

TARGET_MODES = ("per_plane_max", "per_plane_l2")
DEFAULT_TARGET_MODE = "per_plane_l2"


def _validate(targets: Tensor, eps: float) -> None:
    if eps <= 0:
        raise ValueError("eps must be positive")
    if targets.ndim < 3:
        raise ValueError("Expected targets of shape [..., planes, H, W]")
    if not torch.isfinite(targets).all():
        raise ValueError("Targets must be finite")
    if (targets < 0).any():
        raise ValueError("Targets must be nonnegative amplitudes")


def per_plane_minmax(targets: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Map every plane to [0, 1]; constant planes are returned unchanged."""
    _validate(targets, eps)
    low = targets.amin(dim=(-2, -1), keepdim=True)
    high = targets.amax(dim=(-2, -1), keepdim=True)
    span = high - low
    scaled = (targets - low) / span.clamp_min(eps)
    return torch.where(span > eps, scaled, targets)


def equalize_plane_energy(targets: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Scale each plane to plane 0's L2 energy (DeepCGH ``normalize=True``).

    Planes with zero energy stay zero.  If plane 0 itself has no energy the
    stack is returned unchanged instead of being zeroed out.
    """
    _validate(targets, eps)
    energy = targets.square().sum(dim=(-2, -1), keepdim=True)
    reference = energy.narrow(-3, 0, 1)
    scale = (reference / energy.clamp_min(eps)).sqrt()
    scale = torch.where(reference > eps, scale, torch.ones_like(scale))
    return targets * scale


def global_minmax(targets: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Rescale each sample's plane stack by one shared factor to [0, 1].

    The reduction runs over planes, height and width of every sample
    independently, so samples in a batch never share a scale.  Unlike
    per-plane scaling this preserves relative plane brightness; the
    brightest plane sets the scale.
    """
    _validate(targets, eps)
    low = targets.amin(dim=(-3, -2, -1), keepdim=True)
    high = targets.amax(dim=(-3, -2, -1), keepdim=True)
    span = high - low
    scaled = (targets - low) / span.clamp_min(eps)
    return torch.where(span > eps, scaled, targets / high.clamp_min(eps))


def normalize_targets(targets: Tensor, *, mode: str = DEFAULT_TARGET_MODE, eps: float = 1e-8) -> Tensor:
    """Normalize a target stack of shape [..., planes, H, W].

    ``per_plane_l2`` (default): per-plane min-max, then L2 energy equalization
    against plane 0, then one per-sample global min-max.  Every plane
    contributes equally to the loss.  DeepCGH with ``normalize=True`` applies
    the same L2 equalization and global min-max to binary masks, where the
    per-plane min-max changes nothing; for grayscale it is an added
    contrast-normalization step.

    ``per_plane_max``: current pipeline behaviour -- each plane divided by its
    own maximum.  Kept for A/B experiments.

    Use :func:`global_minmax` directly when true relative plane brightness
    must be preserved instead of equalized.
    """
    _validate(targets, eps)
    if mode == "per_plane_max":
        peak = targets.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return targets / peak
    if mode == "per_plane_l2":
        # DeepCGH equalizes raw plane energies and then applies one global
        # min-max; it has no per-plane min-max because its targets are binary.
        # Here the extra per-plane min-max normalizes grayscale contrast.
        return global_minmax(
            equalize_plane_energy(per_plane_minmax(targets, eps=eps), eps=eps), eps=eps
        )
    raise ValueError(f"Unknown target normalization: {mode}")
