from pathlib import Path

import pytest
import torch

from src.evaluate import evaluate_four
from src.model import PhaseUNet
from src.train import evaluate_validation, validation_score


def test_selection_uses_worst_plane_ssim_and_rejects_failed_pairs():
    result = {"aggregate": {
        "worst_plane_mean": {"ssim": 0.4}, "mean": {"pearson": 0.7},
        "failure_count": 0, "valid_pearson_pairs": 8, "total_pairs": 8,
    }}
    assert validation_score(result) == (0.4, 0.7)
    result["aggregate"]["valid_pearson_pairs"] = 7
    assert validation_score(result) is None


def test_validation_uses_export_hook_restores_mode_and_removes_temporary_files():
    model = PhaseUNet(base_channels=2).train()
    sample = {"id": "val_fixture", "category": "geometry"}
    targets = torch.rand(1, 4, 128, 128)
    folders = []

    def interface_export(target, phase, out, *, metadata):
        # Explicit interface fixture; production uses C's PNG/Fresnel exporter.
        folders.append(Path(out))
        (out / "temporary.txt").write_text("test")
        assert metadata["split"] == "val"
        assert not phase.requires_grad
        return evaluate_four(target, torch.sigmoid(phase).expand_as(target).cpu())

    result = evaluate_validation(model, [(sample, targets)], export_fn=interface_export)
    assert result["aggregate"]["samples"] == 1
    assert result["aggregate"]["total_pairs"] == 4
    assert model.training
    assert all(not folder.exists() for folder in folders)


def test_validation_restores_model_mode_when_export_fails():
    model = PhaseUNet(base_channels=2).train()

    def fail(*args, **kwargs):
        raise ValueError("export failed")

    with pytest.raises(ValueError, match="export failed"):
        evaluate_validation(
            model, [({"id": "val_fixture", "category": "geometry"}, torch.rand(1, 4, 128, 128))],
            export_fn=fail,
        )
    assert model.training
