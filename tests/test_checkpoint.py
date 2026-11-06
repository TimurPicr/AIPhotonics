import itertools
import random

import numpy as np
import pytest
import torch

from src.model import PhaseUNet
from src.train import fit, load_checkpoint, restore_rng_state, save_checkpoint, seed_everything


def objective(phase, targets):
    return (torch.sigmoid(phase).expand_as(targets) - targets).square().mean()


def test_checkpoint_restores_exact_predictions_and_checks_protocol(tmp_path):
    model = PhaseUNet(base_channels=4).eval()
    images = torch.rand(1, 4, 32, 32)
    protocol = {"reconstruction_value": "amplitude", "wavelength_m": 532e-9}
    save_checkpoint(tmp_path / "initial.pt", model, protocol=protocol)
    loaded, metadata = load_checkpoint(tmp_path / "initial.pt", expected_protocol=protocol)
    assert not loaded.training and metadata["step"] == 0
    with torch.no_grad():
        torch.testing.assert_close(model(images), loaded(images), rtol=0, atol=0)
    with pytest.raises(ValueError, match="protocol"):
        load_checkpoint(tmp_path / "initial.pt", expected_protocol={"reconstruction_value": "intensity"})


def test_rng_roundtrip(tmp_path):
    seed_everything(45)
    save_checkpoint(tmp_path / "initial.pt", PhaseUNet(base_channels=2))
    expected = random.random(), np.random.random(), torch.rand(5)
    seed_everything(99)
    _, checkpoint = load_checkpoint(tmp_path / "initial.pt")
    restore_rng_state(checkpoint)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(5), expected[2], rtol=0, atol=0)


def test_resume_matches_uninterrupted_adam(tmp_path):
    targets = torch.full((1, 4, 16, 16), 0.8)
    seed_everything(42)
    full = PhaseUNet(base_channels=2)
    fit(full, itertools.repeat(targets), steps=10, objective=objective)

    seed_everything(42)
    first = PhaseUNet(base_channels=2)
    fit(first, itertools.repeat(targets), steps=4, objective=objective, checkpoint_dir=tmp_path)
    resumed, checkpoint = load_checkpoint(tmp_path / "last.pt")
    optimizer = torch.optim.Adam(resumed.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_rng_state(checkpoint)
    fit(resumed, itertools.repeat(targets), steps=6, objective=objective, optimizer=optimizer, start_step=4)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)


def test_bad_checkpoint_fails_clearly(tmp_path):
    torch.save({"format_version": 999}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(tmp_path / "bad.pt")
