import math

import pytest
import torch

from src.io import quantize_phase
from src.losses import reconstruction_loss
from src.reconstruction import normalize_recons, reconstruct_four
from src.refinement import refine_phase


def phase_problem(batch_size=1):
    rng = torch.Generator().manual_seed(7)
    desired = torch.rand((batch_size, 1, 128, 128), generator=rng) * (2 * math.pi) - math.pi
    initial = desired + 0.2 * torch.randn(desired.shape, generator=rng)
    initial = (initial + math.pi).remainder(2 * math.pi) - math.pi
    targets = normalize_recons(reconstruct_four(quantize_phase(desired, keep_device=True)))
    return initial, targets


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_refinement_improves_exported_phase_under_inference_contexts(context):
    torch.set_num_threads(2)
    with context():
        initial, targets = phase_problem()
        original_phase, original_targets = initial.clone(), targets.clone()
        before = reconstruction_loss(normalize_recons(reconstruct_four(quantize_phase(initial, keep_device=True))), targets)
        rng_state = torch.random.get_rng_state()
        fitted = refine_phase(initial, targets, steps=4)
        after = reconstruction_loss(normalize_recons(reconstruct_four(quantize_phase(fitted, keep_device=True))), targets)
    assert after < before
    assert fitted.abs().max() <= math.pi and not fitted.requires_grad
    torch.testing.assert_close(initial, original_phase, rtol=0, atol=0)
    torch.testing.assert_close(targets, original_targets, rtol=0, atol=0)
    assert torch.equal(rng_state, torch.random.get_rng_state())


def test_zero_steps_preserves_values_without_constructing_optimizer(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A disabled refinement must not create an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", unexpected)
    initial = torch.full((1, 1, 128, 128), 4.0, requires_grad=True)
    actual = refine_phase(initial, torch.ones(1, 4, 128, 128), steps=0)
    torch.testing.assert_close(actual, initial, rtol=0, atol=0)
    assert not actual.requires_grad


def test_batched_refinement_matches_independent_quartets():
    torch.set_num_threads(2)
    initial, targets = phase_problem(batch_size=2)
    together = refine_phase(initial, targets, steps=3)
    separate = torch.cat([refine_phase(p, t, steps=3) for p, t in zip(initial.split(1), targets.split(1))])
    torch.testing.assert_close(torch.exp(1j * together), torch.exp(1j * separate), rtol=0, atol=2e-5)


def test_refinement_detaches_from_network_parameters():
    from src.model import HologramUNet

    torch.set_num_threads(2)
    _, targets = phase_problem()
    model = HologramUNet(base_channels=2).eval()
    initial, _ = model(targets)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    fitted = refine_phase(initial, targets, steps=2)
    assert not fitted.requires_grad
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


@pytest.mark.parametrize("steps,lr", [(-1, 0.1), (1.5, 0.1), (True, 0.1), (1, 0), (1, -0.1), (1, float("nan"))])
def test_invalid_refinement_options_are_rejected(steps, lr):
    with pytest.raises(ValueError):
        refine_phase(torch.zeros(1, 1, 128, 128), torch.ones(1, 4, 128, 128), steps=steps, lr=lr)
