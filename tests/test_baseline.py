import math

import torch

from src.baseline import optimize_phase
from src.losses import reconstruction_loss
from src.reconstruction import normalize_recons, reconstruct_four


def test_adam_improves_the_common_loss_and_preserves_rng():
    rng = torch.Generator().manual_seed(42)
    phase = torch.rand(1, 1, 16, 16, generator=rng) * (2 * math.pi) - math.pi
    targets = normalize_recons(reconstruct_four(phase.roll(1, dims=-1)))
    initial_loss = reconstruction_loss(normalize_recons(reconstruct_four(phase)), targets)
    before = torch.random.get_rng_state()
    fitted = optimize_phase(targets, steps=12, seed=42)
    fitted_loss = reconstruction_loss(normalize_recons(reconstruct_four(fitted)), targets)
    assert fitted_loss < initial_loss
    assert fitted.abs().max() <= math.pi and not fitted.requires_grad
    assert torch.equal(torch.random.get_rng_state(), before)
