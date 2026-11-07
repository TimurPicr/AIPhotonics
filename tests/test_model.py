import math

import torch

from src.losses import plane_metrics, reconstruction_loss
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


def test_channels_last_preserves_loss_and_gradients():
    import copy

    torch.manual_seed(42)
    targets = torch.rand(2, 4, 32, 32)
    reference = HologramUNet(base_channels=4)
    optimized = copy.deepcopy(reference).to(memory_format=torch.channels_last)
    _, expected = reference(targets)
    expected_loss = reconstruction_loss(expected, targets)
    expected_loss.backward()
    _, actual = optimized(targets.contiguous(memory_format=torch.channels_last))
    actual_loss = reconstruction_loss(actual, targets)
    actual_loss.backward()
    # CPU convolution layouts accumulate float32 sums in different orders.
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-4)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    actual_ssim, actual_cc, _ = plane_metrics(actual, targets)
    expected_ssim, expected_cc, _ = plane_metrics(expected, targets)
    torch.testing.assert_close(actual_ssim, expected_ssim, rtol=0, atol=1e-5)
    torch.testing.assert_close(actual_cc, expected_cc, rtol=0, atol=1e-5)
    gradient = torch.cat([parameter.grad.flatten() for parameter in optimized.parameters()])
    expected_gradient = torch.cat([parameter.grad.flatten() for parameter in reference.parameters()])
    assert (gradient - expected_gradient).norm() / expected_gradient.norm() < 5e-3
