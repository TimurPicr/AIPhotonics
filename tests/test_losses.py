import pytest
import torch

from src.losses import reconstruction_loss


def test_loss_combines_ssim_and_pearson_with_requested_weights():
    x = torch.linspace(0, 1, 64, dtype=torch.float64).reshape(1, 1, 8, 8).repeat(1, 4, 1, 1)
    y = 1 - x
    ssim_term = reconstruction_loss(y, x, alpha=1, beta=0)
    pearson_term = reconstruction_loss(y, x, alpha=0, beta=1)
    assert float(pearson_term) == pytest.approx(2.0)
    torch.testing.assert_close(reconstruction_loss(y, x), 0.7 * ssim_term + 0.3 * pearson_term)
    assert float(reconstruction_loss(x, x)) == pytest.approx(0, abs=1e-12)


def test_pearson_averages_plane_correlations_with_equal_weight():
    x = torch.linspace(0, 1, 64, dtype=torch.float64).reshape(1, 1, 8, 8)
    targets = x.repeat(1, 4, 1, 1)
    recons = torch.cat((0.1 * x, 0.3 + 0.1 * x, 1 - x, 1 - x), dim=1)
    assert float(reconstruction_loss(recons, targets, alpha=0, beta=1)) == pytest.approx(1)


def test_constant_planes_use_analytic_ssim_and_zero_pearson():
    targets = torch.full((1, 4, 8, 8), 0.1, dtype=torch.float64)
    recons = torch.full_like(targets, 0.3, requires_grad=True)
    expected_ssim = (2 * 0.1 * 0.3 + 0.01**2) / (0.1**2 + 0.3**2 + 0.01**2)
    loss = reconstruction_loss(recons, targets)
    assert float(loss.detach()) == pytest.approx(0.7 * (1 - expected_ssim) + 0.3)
    loss.backward()
    assert torch.isfinite(recons.grad).all()
