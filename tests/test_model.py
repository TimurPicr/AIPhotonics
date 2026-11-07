import math

import torch

from src.losses import reconstruction_loss
from src.model import HologramUNet
from src.physics import backpropagate_four, propagate_four
from src.reconstruction import normalize_recons, reconstruct_four


def test_physical_loop_reapplies_target_amplitudes_and_weights(monkeypatch):
    rng = torch.Generator().manual_seed(42)
    targets = torch.rand(1, 4, 16, 16, generator=rng)
    predicted = torch.rand(targets.shape, generator=rng)
    model = HologramUNet(base_channels=2, plane_weights=(1, 2, 0, 3), rounds=2)
    calls = []

    def initial_phases(images):
        calls.append(images)
        return predicted

    monkeypatch.setattr(model, "predict_plane_phases", initial_phases)
    mask, recons = model(targets)
    first = backpropagate_four(targets * torch.exp(1j * predicted))
    first_mask = (first[:, :1] + 2 * first[:, 1:2] + 3 * first[:, 3:4]).angle()
    transferred = propagate_four(first_mask)
    second = backpropagate_four(targets * torch.exp(1j * transferred.angle()))
    expected = (second[:, :1] + 2 * second[:, 1:2] + 3 * second[:, 3:4]).angle()
    assert len(calls) == 1 and calls[0] is targets
    torch.testing.assert_close(torch.exp(1j * mask), torch.exp(1j * expected), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(recons, normalize_recons(reconstruct_four(expected)), atol=2e-5, rtol=2e-5)


def test_gradients_reach_all_four_phase_maps_through_both_transfers(monkeypatch):
    torch.manual_seed(42)
    targets = torch.rand(2, 4, 32, 32)
    model = HologramUNet(base_channels=4)
    predict = model.predict_plane_phases
    predicted = []

    def capture(images):
        phases = predict(images)
        phases.retain_grad()
        predicted.append(phases)
        return phases

    monkeypatch.setattr(model, "predict_plane_phases", capture)
    mask, recons = model(targets)
    reconstruction_loss(recons, targets).backward()
    assert predicted[0].shape == targets.shape and predicted[0].abs().max() <= math.pi
    assert mask.shape == (2, 1, 32, 32) and mask.abs().max() <= math.pi
    assert recons.shape == targets.shape
    grad = predicted[0].grad
    assert torch.isfinite(grad).all() and (grad.abs().sum(dim=(0, 2, 3)) > 0).all()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())


def test_black_targets_produce_finite_masks_and_gradients():
    model = HologramUNet(base_channels=2)
    targets = torch.zeros(1, 4, 16, 16)
    mask, recons = model(targets)
    reconstruction_loss(recons, targets).backward()
    assert torch.equal(mask, torch.zeros_like(mask))
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())
