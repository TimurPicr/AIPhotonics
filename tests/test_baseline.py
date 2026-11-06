import math

import pytest
import torch

from config import CONFIG
from src.baseline import optimize_phase
from src.physics import fresnel_1fft
from src.reconstruction import normalize_recons, reconstruct_four, reconstruction_values


@pytest.mark.parametrize("plane", [None, 3])
def test_adam_reduces_loss_from_a_supplied_start(plane):
    generator = torch.Generator().manual_seed(42)
    reference = torch.randn(1, 1, 8, 8, generator=generator)
    start = torch.randn(reference.shape, generator=generator)
    untouched = start.clone()

    def reconstruct(phase):
        if plane is None:
            return normalize_recons(reconstruct_four(phase))
        return normalize_recons(reconstruction_values(fresnel_1fft(phase, CONFIG.z_m[plane - 1])))

    targets = reconstruct(reference)
    result = optimize_phase(targets, initial_phase=start, steps=60, seed=42, plane=plane)
    initial_loss = (reconstruct(start) - targets).square().mean()
    final_loss = (reconstruct(result) - targets).square().mean()
    assert final_loss < 0.3 * initial_loss
    assert result.shape == reference.shape and not result.requires_grad
    assert result.min() >= -math.pi and result.max() <= math.pi
    torch.testing.assert_close(start, untouched, rtol=0, atol=0)


def test_seeded_random_start_is_reproducible():
    targets = torch.full((1, 4, 8, 8), 0.5)
    first = optimize_phase(targets, initial_phase=None, steps=4, seed=42)
    second = optimize_phase(targets, initial_phase=None, steps=4, seed=42)
    other = optimize_phase(targets, initial_phase=None, steps=4, seed=43)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, other)
