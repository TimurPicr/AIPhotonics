import json

import pytest
import torch
from PIL import Image

import src.train as training
from src.data import prepare_split
from src.evaluate import _summary, main as evaluate
from src.train import load_checkpoint, main as train


def test_training_writes_checkpoint_and_reproducible_comparison(tmp_path):
    prepare_split(tmp_path, "train", 2, seed=1)
    prepare_split(tmp_path, "val", 1)
    manifest = str(tmp_path / "val/manifest.json")
    run = tmp_path / "run"
    train(["--train-manifest", str(tmp_path / "train/manifest.json"), "--eval-manifest", manifest,
           "--epochs", "2", "--batch-size", "3", "--base-channels", "2",
           "--adam-steps", "2", "--gs-iterations", "2", "--device", "cpu", "--out", str(run)])
    path = run / "last.pt"
    _, checkpoint = load_checkpoint(path, torch.device("cpu"))
    assert checkpoint["epochs"] == 2
    assert list(run.glob("*.pt")) == [path]
    log = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in log] == [1, 2] and all(row["quartets"] == 2 for row in log)
    report = json.loads((run / "comparison/summary.json").read_text(encoding="utf-8"))
    assert report["shared_pairs"] == {"ssim": 4, "pearson": 4}
    assert set(report["methods"]) == {"network", "adam", "global_mp_gs"}
    assert all(-1 <= value <= 1 for result in report["methods"].values() for value in result["mean"].values())
    assert report["epoch"] == 2 and "epoch_metrics" not in report
    markdown = (run / "comparison/report.md").read_text(encoding="utf-8")
    assert "![Training loss by epoch](loss.png)" in markdown
    assert report["loss_plot"] == "loss.png"
    with Image.open(run / "comparison/loss.png") as plot:
        assert plot.format == "PNG" and plot.width > 500 and plot.height > 200
    assert not (run / "comparison/metrics.png").exists()
    (run / "epoch_0001.pt").write_bytes(b"unused archive from a previous run")
    replay = tmp_path / "replay"
    evaluate(["--checkpoint", str(path), "--eval-manifest", manifest, "--device", "cpu", "--out", str(replay)])
    assert json.loads((replay / "summary.json").read_text(encoding="utf-8")) == report
    assert (replay / "loss.png").is_file()
    assert not (replay / "metrics.png").exists()


def test_interruption_during_next_epoch_keeps_previous_weights(tmp_path, monkeypatch):
    prepare_split(tmp_path, "train", 1, seed=1)
    manifest = str(tmp_path / "train/manifest.json")
    run = tmp_path / "run"
    original_loss = training.reconstruction_loss
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(training, "reconstruction_loss", interrupt_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        train(["--train-manifest", manifest, "--evaluate-train", "--epochs", "2", "--base-channels", "2",
               "--adam-steps", "1", "--gs-iterations", "1", "--device", "cpu", "--out", str(run)])
    path = run / "last.pt"
    _, checkpoint = load_checkpoint(path, torch.device("cpu"))
    assert checkpoint["epochs"] == 1
    assert list(run.glob("*.pt")) == [path]
    assert not (run / "comparison").exists()
    with (run / "training.jsonl").open("a") as log:
        log.write('{"epoch": 2, "mean_lo')
    out = tmp_path / "partial_report"
    evaluate(["--checkpoint", str(run / "last.pt"), "--eval-manifest", manifest, "--device", "cpu", "--out", str(out)])
    report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert report["epoch"] == 1
    assert (out / "loss.png").is_file()


def test_summary_uses_shared_pairs_when_one_method_fails_or_pearson_is_undefined():
    full = {"ssim": [0.1, 0.2, 0.3, 0.4], "pearson": [0.3, 0.4, 0.6, 0.7]}
    partial = {"ssim": [0.4] * 4, "pearson": [0.0, None, 0.0, 0.0]}
    methods, shared = _summary([
        {"metrics": {"network": full, "adam": full, "global_mp_gs": partial}},
        {"metrics": {"network": full, "adam": full}},
    ])
    assert shared == {"ssim": 4, "pearson": 3}
    assert methods["network"]["mean"]["pearson"] == pytest.approx((0.3 + 0.6 + 0.7) / 3)
    assert methods["global_mp_gs"]["solved"] == 1
