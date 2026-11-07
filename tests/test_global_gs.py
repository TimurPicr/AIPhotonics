import json
import math

import pytest
import torch
from PIL import Image

from config import CONFIG
from src.evaluate import evaluate_four
from src.global_gs import PHASE_EPS, _gs_step, _replace_amplitude, _target_amplitudes, global_mp_gs, run_global_gs
from src.io import export_phase_png, export_result, load_four_png, load_phase_png, write_png
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


def test_reachable_png_targets_and_repeatable_gs_export(tmp_path):
    rng = torch.Generator().manual_seed(17)
    codes = torch.randint(0, 256, (1, 1, 128, 128), generator=rng)
    known_path = tmp_path / "known_phase.png"
    export_phase_png(codes.float() / 255 * (2 * math.pi) - math.pi, str(known_path))
    known_phase = load_phase_png(str(known_path))
    values = normalize_recons(reconstruct_four(known_phase))
    inputs = [tmp_path / f"target_{i}.png" for i in range(4)]
    for image, path in zip(values[0], inputs):
        write_png(image, path)
    targets = load_four_png(inputs)
    oracle = export_result(targets, known_phase, tmp_path / "oracle")
    assert oracle["mean"] == {"pearson": pytest.approx(1.0), "ssim": pytest.approx(1.0)}

    before = torch.random.get_rng_state()
    out = tmp_path / "gs"
    report = run_global_gs(targets, out, iterations=3, eval_every=2, seed=42)
    repeat = run_global_gs(targets[0], tmp_path / "repeat", iterations=3, eval_every=2, seed=42)
    assert torch.equal(torch.random.get_rng_state(), before)

    files = [out / "phase.png", *(out / f"recon_{i}.png" for i in range(1, 5))]
    for path in files:
        with Image.open(path) as image:
            assert (image.format, image.mode, image.size) == ("PNG", "L", (128, 128))
        assert path.read_bytes() == (tmp_path / "repeat" / path.name).read_bytes()
    phase = load_phase_png(str(files[0]))
    assert phase.shape == (1, 1, 128, 128) and torch.isfinite(phase).all()
    assert phase.min() >= -math.pi and phase.max() <= math.pi
    recons = load_four_png(files[1:])
    expected = normalize_recons(reconstruct_four(phase)).mul(255).round().div(255)
    torch.testing.assert_close(recons, expected, rtol=0, atol=0)
    measured = evaluate_four(targets, recons)
    assert measured == {key: report[key] for key in measured} == {key: repeat[key] for key in measured}
    assert report == json.loads((out / "metrics.json").read_text())

    history = json.loads((out / "history.json").read_text())
    assert history == json.loads((tmp_path / "repeat" / "history.json").read_text())
    assert [row["iteration"] for row in history["history"]] == [0, 2, 3]
    assert all(math.isfinite(row[key]) for row in history["history"]
               for key in ("mean_ssim", "min_ssim", "mean_pearson"))
    best = max(history["history"], key=lambda row: (row["mean_ssim"], row["min_ssim"]))
    assert best["iteration"] == history["best_iteration"] == report["metadata"]["best_iteration"]
    assert best["mean_ssim"] == report["mean"]["ssim"]
    assert best["min_ssim"] == report["worst_plane"]["ssim"]
    timing = report["metadata"]["timing"]
    assert 0 < timing["solve_seconds"] <= timing["total_seconds"]
