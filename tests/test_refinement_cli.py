import json

import pytest
import torch

from config import CONFIG
from src.data import ManifestDataset, prepare_split
from src.evaluate import _score, main as evaluate
from src.model import HologramUNet
from src.refinement import refine_phase
from src.train import load_checkpoint, main as train


@pytest.mark.parametrize("saved,arguments,expected_steps,expected_lr", [
    (None, [], 0, 0.1),
    (None, ["--refine-steps", "2", "--refine-lr", "0.05"], 2, 0.05),
    ({"steps": 2, "lr": 0.05}, [], 2, 0.05),
    ({"steps": 2, "lr": 0.05}, ["--refine-steps", "0"], 0, 0.05),
])
def test_evaluate_refines_only_network_and_resolves_checkpoint_settings(
    tmp_path, saved, arguments, expected_steps, expected_lr,
):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    prepare_split(tmp_path, "val", 1)
    manifest = tmp_path / "val/manifest.json"
    model = HologramUNet(base_channels=2).eval()
    checkpoint = {"version": 1, "protocol": CONFIG.snapshot(), "model_config": {"base_channels": 2},
                  "model_state_dict": model.state_dict(), "epochs": 1,
                  "baseline_config": {"adam_steps": 1, "adam_lr": 0.1, "gs_iterations": 1, "seed": 42},
                  "loss_config": {"alpha": 0.6, "beta": 0.4}}
    if saved is not None:
        checkpoint["refinement_config"] = saved
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    out = tmp_path / "comparison"
    evaluate(["--checkpoint", str(path), "--eval-manifest", str(manifest), "--out", str(out),
              "--device", "cpu", *arguments])
    report = json.loads((out / "summary.json").read_text())
    assert report["refinement_config"] == {"steps": expected_steps, "lr": expected_lr}
    assert report["baseline_parameters"]["adam_steps"] == 1
    loaded, _ = load_checkpoint(path, torch.device("cpu"))
    targets = ManifestDataset(manifest)[0][None]
    with torch.no_grad():
        initial, _ = loaded(targets)
    expected = refine_phase(initial, targets, steps=expected_steps, lr=expected_lr, alpha=0.6, beta=0.4)
    assert report["samples"][0]["metrics"]["network"] == _score(expected, targets)
    if expected_steps:
        assert f"U-Net + {expected_steps} Adam" in (out / "report.md").read_text()


@pytest.mark.parametrize("validation", [False, True])
def test_training_saves_and_applies_refinement_settings(tmp_path, validation):
    prepare_split(tmp_path, "train", 1, seed=1)
    manifest = tmp_path / "train/manifest.json"
    run = tmp_path / "run"
    arguments = ["--train-manifest", str(manifest), "--evaluate-train", "--epochs", "1",
                 "--batch-size", "1", "--base-channels", "2", "--device", "cpu", "--out", str(run),
                 "--adam-steps", "1", "--gs-iterations", "1", "--refine-steps", "2", "--refine-lr", "0.05"]
    if validation:
        arguments.append("--validate-every-epoch")
    train(arguments)
    model, checkpoint = load_checkpoint(run / "last.pt", torch.device("cpu"))
    # Training constructs contiguous weights; load_checkpoint uses channels_last
    # on CPU. Match the training layout for exact quantized-score comparison.
    model.to(memory_format=torch.contiguous_format)
    assert checkpoint["refinement_config"] == {"steps": 2, "lr": 0.05}
    report_path = run / ("validation.json" if validation else "comparison/summary.json")
    report = json.loads(report_path.read_text())
    assert report["refinement_config"] == checkpoint["refinement_config"]
    targets = ManifestDataset(manifest)[0][None]
    with torch.no_grad():
        phase, _ = model(targets)
    expected = refine_phase(phase, targets, steps=2, lr=0.05)
    assert report["samples"][0]["metrics"]["network"] == _score(expected, targets)
    if validation:
        arguments[arguments.index("--epochs") + 1] = "2"
        train([*arguments, "--resume", str(run / "last.pt")])
        assert json.loads(report_path.read_text())["epoch"] == 2


@pytest.mark.parametrize("arguments", [["--refine-steps", "-1"], ["--refine-lr", "0"], ["--refine-lr", "nan"]])
def test_evaluate_rejects_invalid_refinement_before_loading_checkpoint(tmp_path, arguments):
    with pytest.raises(SystemExit):
        evaluate(["--checkpoint", str(tmp_path / "missing.pt"), "--eval-manifest", "missing.json",
                  "--out", str(tmp_path / "out"), *arguments])
    assert not (tmp_path / "out").exists()
