import math

import pytest
import torch

from src.model import PhaseUNet


def test_phase_model_shape_bounds_and_all_four_input_gradients():
    torch.manual_seed(42)
    model = PhaseUNet()
    images = torch.rand(1, 4, 128, 128, requires_grad=True)
    phase = model(images)
    assert phase.shape == (1, 1, 128, 128)
    assert phase.dtype == torch.float32
    assert torch.isfinite(phase).all()
    assert phase.abs().max() <= math.pi
    phase.square().mean().backward()
    assert torch.isfinite(images.grad).all()
    assert (images.grad.abs().sum(dim=(0, 2, 3)) > 0).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_batch_members_do_not_change_each_others_predictions():
    torch.manual_seed(43)
    model = PhaseUNet(base_channels=4).eval()
    images = torch.rand(2, 4, 32, 32)
    with torch.no_grad():
        together = model(images)
        separate = torch.cat([model(x[None]) for x in images])
    torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("shape", [(4, 128, 128), (1, 3, 128, 128), (1, 4, 17, 16)])
def test_model_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        PhaseUNet(base_channels=4)(torch.zeros(shape))


def test_model_rejects_integer_images():
    with pytest.raises(TypeError, match="floating"):
        PhaseUNet()(torch.zeros(1, 4, 128, 128, dtype=torch.uint8))
