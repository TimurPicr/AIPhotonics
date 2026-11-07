"""Independent Fresnel reference, physical invariants, and gradient checks."""

import cmath
import math
from dataclasses import replace

import pytest
import torch

from config import DEFAULT_CONFIG
from src.physics import UnitaryFresnel, _fresnel_field_1fft, backpropagate_four, fresnel_1fft, output_pitch_m, propagate_four
from src.reconstruction import normalize_recons, reconstruct_four


def test_plane_inverse_returns_physical_fields_to_the_slm():
    phase = torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(42))
    actual = backpropagate_four(propagate_four(phase))
    torch.testing.assert_close(actual, torch.exp(1j * phase).expand_as(actual), atol=1e-6, rtol=1e-6)


def _direct_fresnel(phase: torch.Tensor, z: float) -> torch.Tensor:
    wavelength = 532e-9
    dx_in = 10e-6
    size = phase.shape[-1]
    dx_out = wavelength * z / (size * dx_in)
    coordinates = range(-(size // 2), size - size // 2)
    scale = cmath.exp(2j * math.pi * z / wavelength) * dx_in**2 / (1j * wavelength * z)
    result = torch.empty_like(phase, dtype=torch.complex128)
    for row, y_idx in enumerate(coordinates):
        for col, x_idx in enumerate(coordinates):
            x, y = x_idx * dx_out, y_idx * dx_out
            field = 0j
            for in_row, v_idx in enumerate(coordinates):
                for in_col, u_idx in enumerate(coordinates):
                    u, v = u_idx * dx_in, v_idx * dx_in
                    argument = math.pi * ((x - u)**2 + (y - v)**2) / (wavelength * z)
                    field += cmath.exp(1j * (argument + phase[0, 0, in_row, in_col].item()))
            result[0, 0, row, col] = scale * field
    return result


def test_output_pitch():
    pitches_um = [1e6 * output_pitch_m(z, 128) for z in DEFAULT_CONFIG.distances_m]
    assert pitches_um == pytest.approx([10.0165625, 10.5153125, 11.0140625, 11.5128125])


@pytest.mark.parametrize("size", [5, 6])
def test_fft_matches_direct_sum(size):
    phase = torch.randn(1, 1, size, size, generator=torch.Generator().manual_seed(42))
    for z in DEFAULT_CONFIG.distances_m:
        field = fresnel_1fft(phase, z)
        expected = _direct_fresnel(phase, z)
        relative_error = (field - expected).norm() / expected.norm()
        assert field.shape == phase.shape and field.dtype == torch.complex64
        assert relative_error < 1e-4


def test_power_and_plane_order_at_full_resolution():
    phase = torch.randn(2, 1, 128, 128, generator=torch.Generator().manual_seed(42))
    fields = propagate_four(phase)
    assert fields.shape == (2, 4, 128, 128)
    input_power = 128**2 * DEFAULT_CONFIG.slm_pitch_m**2
    for index, z in enumerate(DEFAULT_CONFIG.distances_m):
        field = fields[:, index:index + 1]
        torch.testing.assert_close(field, fresnel_1fft(phase, z))
        output_power = field.abs().square().sum(dim=(-2, -1)) * output_pitch_m(z, 128)**2
        torch.testing.assert_close(output_power, torch.full_like(output_power, input_power), rtol=1e-4, atol=0)
    assert not torch.allclose(fields[:, 0].abs(), fields[:, -1].abs())


def test_reconstruction_modes_and_independent_normalization():
    phase = torch.zeros(1, 1, 8, 8)
    amplitude = reconstruct_four(phase)
    intensity = reconstruct_four(phase, config=replace(DEFAULT_CONFIG, reconstruction_value="intensity"))
    torch.testing.assert_close(amplitude, propagate_four(phase).abs())
    torch.testing.assert_close(intensity, amplitude.square())

    values = torch.tensor([[[[0.0, 0.0]], [[1.0, 2.0]]], [[[6.0, 2.0]], [[0.0, 1e-10]]]])
    expected = torch.tensor([[[[0.0, 0.0]], [[0.5, 1.0]]], [[[1.0, 1.0 / 3.0]], [[0.0, 0.01]]]])
    torch.testing.assert_close(normalize_recons(values), expected)


def test_four_plane_gradient_matches_finite_difference():
    generator = torch.Generator().manual_seed(42)
    phase = torch.randn(1, 1, 8, 8, generator=generator).requires_grad_()
    direction = torch.randn(phase.shape, generator=generator)
    direction /= direction.norm()
    weights = torch.rand(1, 4, 8, 8, generator=generator)

    def probe(phi):
        return (normalize_recons(reconstruct_four(phi)) * weights).mean()

    (gradient,) = torch.autograd.grad(probe(phase), phase)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    with torch.no_grad():
        step = 1e-2
        finite_difference = (probe(phase + step * direction) - probe(phase - step * direction)) / (2 * step)
    torch.testing.assert_close((gradient * direction).sum(), finite_difference, rtol=1e-2, atol=0)


@pytest.mark.parametrize("z", DEFAULT_CONFIG.distances_m)
@pytest.mark.parametrize("size", [5, 128])
def test_unitary_fresnel_matches_physical_operator_and_its_adjoint(z, size):
    rng = torch.Generator().manual_seed(123)
    shape = (1, 1, size, size)
    u = torch.randn(shape, generator=rng, dtype=torch.complex64)
    v = torch.randn(shape, generator=rng, dtype=torch.complex64)
    operator = UnitaryFresnel(u, z)
    propagated = operator.forward(u)
    lhs = torch.vdot(propagated.flatten(), v.flatten())
    rhs = torch.vdot(u.flatten(), operator.adjoint(v).flatten())
    assert (lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-8) < 1e-5
    assert (operator.adjoint(propagated) - u).norm() / u.norm() < 1e-5
    assert (propagated.norm() - u.norm()).abs() / u.norm() < 1e-5

    expected = _fresnel_field_1fft(u, z, DEFAULT_CONFIG) * (output_pitch_m(z, size) / DEFAULT_CONFIG.slm_pitch_m)
    assert (propagated - expected).norm() / expected.norm() < 1e-5
