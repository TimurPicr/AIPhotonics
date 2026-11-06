import pytest
import torch

from src.losses import physics_loss, reconstruction_mse


def test_all_planes_have_equal_weight():
    values = torch.zeros(1, 4, 8, 8, requires_grad=True)
    targets = torch.zeros_like(values)
    targets[:, 2] = 1
    loss = reconstruction_mse(values, targets, normalize_fn=lambda x: x)
    torch.testing.assert_close(loss, torch.tensor(0.25))
    loss.backward()
    assert torch.count_nonzero(values.grad[:, 2]) == 64
    assert torch.count_nonzero(values.grad[:, [0, 1, 3]]) == 0


def test_delegates_normalization_without_detaching_gradients():
    values = torch.full((2, 4, 8, 8), 0.5, requires_grad=True)
    loss = reconstruction_mse(values, torch.zeros_like(values), normalize_fn=lambda x: 2 * x)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert torch.isfinite(values.grad).all()
    assert (values.grad > 0).all()


def test_complex_interface_fixture_keeps_phase_gradient():
    # Explicit interface fixture, not the production Fresnel propagator.
    phase = torch.randn(1, 1, 16, 16, requires_grad=True)
    targets = torch.rand(1, 4, 16, 16)

    def fixture(phi):
        amplitude = torch.fft.fft2(torch.exp(1j * phi)).abs()
        return amplitude.repeat(1, 4, 1, 1)

    loss = physics_loss(
        phase, targets, reconstruct_fn=fixture,
        normalize_fn=lambda x: x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8),
    )
    loss.backward()
    assert loss.ndim == 0 and loss.is_floating_point()
    assert torch.isfinite(phase.grad).all()
    assert phase.grad.abs().sum() > 0


def test_rejects_accidental_broadcasting():
    with pytest.raises(ValueError, match="same shape"):
        reconstruction_mse(torch.ones(1, 4, 8, 8), torch.ones(4, 8, 8))
