import math

import pytest
import torch

from config import CONFIG
from src.global_gs import PHASE_EPS, _gs_step, _replace_amplitude, _target_amplitudes, global_mp_gs
from src.physics import UnitaryFresnel
from src.reconstruction import normalize_recons, reconstruct_four


def test_targets_are_amplitudes_and_projection_matches_their_norm():
    targets = torch.linspace(0, 1, 8**2).reshape(1, 1, 8, 8).repeat(1, 4, 1, 1)
    expected = targets * (8 / targets.square().sum(dim=(-2, -1), keepdim=True).sqrt())
    amplitudes = _target_amplitudes(targets)
    torch.testing.assert_close(amplitudes, expected)
    torch.testing.assert_close(amplitudes.flatten(2).norm(dim=-1), torch.full((1, 4), 8.0))
    field = torch.randn(amplitudes.shape, dtype=torch.complex64, generator=torch.Generator().manual_seed(42))
    field[..., 1, 1] = 0
    torch.testing.assert_close(_replace_amplitude(field, amplitudes).abs(), amplitudes)


def test_one_step_projects_the_sum_of_all_four_back_fields():
    rng = torch.Generator().manual_seed(37)
    phase = torch.randn(1, 1, 8, 8, generator=rng)
    amplitudes = _target_amplitudes(torch.rand(1, 4, 8, 8, generator=rng))
    incident = torch.polar(torch.ones_like(phase), phase)
    operators = tuple(UnitaryFresnel(incident, z) for z in CONFIG.z_m)
    accumulated = torch.zeros_like(incident)
    for k, operator in enumerate(operators):
        field = operator.forward(incident)
        projected = torch.polar(amplitudes[:, k:k + 1], field.angle())
        accumulated += operator.adjoint(projected) / 4
    expected = torch.where(accumulated.abs() <= PHASE_EPS, phase, accumulated.angle())
    torch.testing.assert_close(torch.exp(1j * _gs_step(phase, amplitudes, operators)), torch.exp(1j * expected))


@pytest.mark.parametrize("value, message", [(0, "black target"), (1.01, "finite amplitudes"),
                                             (float("nan"), "finite amplitudes")])
def test_invalid_target_fails_before_solving(value, message):
    targets = torch.ones(1, 4, 128, 128)
    targets[:, 1] = value
    with pytest.raises(ValueError, match=message):
        global_mp_gs(targets, iterations=3)


def test_solver_returns_repeatable_phase_for_reachable_targets():
    rng = torch.Generator().manual_seed(17)
    known_phase = torch.rand((1, 1, 128, 128), generator=rng) * (2 * math.pi) - math.pi
    targets = normalize_recons(reconstruct_four(known_phase))
    before = torch.random.get_rng_state()
    phase = global_mp_gs(targets, iterations=3, seed=42)
    repeat = global_mp_gs(targets[0], iterations=3, seed=42)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert phase.shape == (1, 1, 128, 128) and torch.isfinite(phase).all()
    assert phase.min() >= -math.pi and phase.max() <= math.pi
    torch.testing.assert_close(phase, repeat, rtol=0, atol=0)
