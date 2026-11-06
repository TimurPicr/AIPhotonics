import itertools

import pytest
import torch

from src.model import PhaseUNet
from src.train import build_parser, fit, seed_everything, select_device, train_step


def interface_objective(phase, targets):
    # Test fixture, deliberately not a physical propagator.
    return (torch.sigmoid(phase).expand_as(targets) - targets).square().mean()


def test_train_loop_learns_and_saves_last_state(tmp_path):
    seed_everything(42)
    model = PhaseUNet(base_channels=2)
    targets = torch.full((1, 4, 16, 16), 0.8)
    records = []
    history = fit(
        model, itertools.repeat(targets), steps=12, objective=interface_objective,
        checkpoint_dir=tmp_path, on_step=records.append,
    )
    assert history[-1]["loss"] < history[0]["loss"]
    assert all(record["gradient_norm"] > 0 for record in history)
    assert len(records) == 12
    saved = torch.load(tmp_path / "last.pt", weights_only=True)
    assert saved["step"] == 12
    assert saved["samples_seen"] == history[-1]["samples_seen"] == 12
    assert saved["optimizer_state_dict"]["state"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved["model_state_dict"][key])
    assert not list(tmp_path.glob("*.tmp"))


def test_nonfinite_objective_does_not_update_parameters():
    model = PhaseUNet(base_channels=2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(FloatingPointError, match="finite real scalar"):
        train_step(model, torch.rand(1, 4, 16, 16), optimizer, objective=lambda p, t: p.sum() * float("nan"))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_interrupted_callback_saves_completed_step(tmp_path):
    model = PhaseUNet(base_channels=2)

    def interrupt(record):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fit(
            model, itertools.repeat(torch.rand(1, 4, 16, 16)), steps=5,
            objective=interface_objective, checkpoint_dir=tmp_path, on_step=interrupt,
        )
    assert torch.load(tmp_path / "last.pt", weights_only=True)["step"] == 1


def test_cli_arguments_and_cpu_fft_backward():
    assert build_parser().parse_args(["--steps", "20"]).steps == 20
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--steps", "0"])
    assert select_device("cpu").type == "cpu"
