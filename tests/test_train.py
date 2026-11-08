import json
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import src.train as training
from src.data import prepare_split
from src.evaluate import _summary, main as evaluate
from src.train import load_checkpoint, main as train


@pytest.mark.parametrize("validate_every_epoch", [False, True])
def test_mlx_numerical_validation_is_opt_in(tmp_path, monkeypatch, capsys, validate_every_epoch):
    class StubMLXBackend:
        def __init__(self, reference, **kwargs):
            self.model = reference

        def step(self, targets):
            return 0.25

        def __call__(self, targets):
            return self.model(targets)

        def state_dict(self):
            return self.model.state_dict()

        def optimizer_state_dict(self):
            return {}

    monkeypatch.setitem(sys.modules, "src.mlx_backend", SimpleNamespace(MLXBackend=StubMLXBackend))

    def unexpected_comparison(*args, **kwargs):
        pytest.fail("MLX must not automatically run the final numerical comparison")

    monkeypatch.setattr(training, "compare", unexpected_comparison)
    if not validate_every_epoch:
        monkeypatch.setattr(training, "validate_epoch", unexpected_comparison)
    prepare_split(tmp_path, "train", 1, seed=1)
    run = tmp_path / "run"
    arguments = ["--train-manifest", str(tmp_path / "train/manifest.json"), "--evaluate-train",
                 "--backend", "mlx", "--device", "cpu", "--epochs", "1", "--batch-size", "1",
                 "--base-channels", "2", "--gs-iterations", "1", "--preload", "--out", str(run)]
    if validate_every_epoch:
        arguments.append("--validate-every-epoch")
    train(arguments)
    _, checkpoint = load_checkpoint(run / "last.pt", torch.device("cpu"))
    assert checkpoint["backend"] == "mlx" and checkpoint["epochs"] == 1
    rows = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    epoch = next(row for row in rows if row["event"] == "epoch")
    assert epoch["mean_loss"] == 0.25
    assert ("validation" in epoch) == validate_every_epoch
    assert (run / "validation.json").exists() == validate_every_epoch
    assert (run / "best.pt").exists() == validate_every_epoch
    assert rows[-1]["event"] == "complete"
    expected_output = run / ("validation.json" if validate_every_epoch else "last.pt")
    assert capsys.readouterr().out.splitlines()[-1] == str(expected_output)


def test_training_writes_checkpoint_and_reproducible_comparison(tmp_path):
    prepare_split(tmp_path, "train", 2, seed=1)
    prepare_split(tmp_path, "val", 1)
    manifest = str(tmp_path / "val/manifest.json")
    run = tmp_path / "run"
    train(["--train-manifest", str(tmp_path / "train/manifest.json"), "--eval-manifest", manifest,
           "--epochs", "2", "--batch-size", "3", "--base-channels", "2",
           "--adam-steps", "2", "--gs-iterations", "2", "--device", "cpu", "--threads", "2", "--out", str(run)])
    path = run / "best.pt"
    last_path = run / "last.pt"
    _, checkpoint = load_checkpoint(path, torch.device("cpu"))
    _, last = load_checkpoint(last_path, torch.device("cpu"))
    assert 1 <= checkpoint["epochs"] <= 2 and last["epochs"] == 2
    assert set(run.glob("*.pt")) == {path, last_path}
    log = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in log] == [1, 2] and all(row["quartets"] == 2 for row in log)
    assert checkpoint["epochs"] == log[-1]["best_epoch"]
    report = json.loads((run / "comparison/summary.json").read_text(encoding="utf-8"))
    assert report["shared_pairs"] == {"ssim": 4, "pearson": 4}
    assert set(report["methods"]) == {"network", "adam", "global_mp_gs"}
    assert all(-1 <= value <= 1 for result in report["methods"].values() for value in result["mean"].values())
    assert report["epoch"] == checkpoint["epochs"] and "epoch_metrics" not in report
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
        if torch.is_grad_enabled():
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(training, "reconstruction_loss", interrupt_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        train(["--train-manifest", manifest, "--evaluate-train", "--epochs", "2", "--base-channels", "2",
               "--adam-steps", "1", "--gs-iterations", "1", "--device", "cpu", "--out", str(run)])
    path = run / "best.pt"
    last_path = run / "last.pt"
    _, checkpoint = load_checkpoint(path, torch.device("cpu"))
    _, last = load_checkpoint(last_path, torch.device("cpu"))
    assert checkpoint["epochs"] == 1 and last["epochs"] == 1
    assert set(run.glob("*.pt")) == {path, last_path}
    assert not (run / "comparison").exists()
    with (run / "training.jsonl").open("a") as log:
        log.write('{"epoch": 2, "mean_lo')
    out = tmp_path / "partial_report"
    evaluate(["--checkpoint", str(path), "--eval-manifest", manifest, "--device", "cpu", "--out", str(out)])
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


def test_monitor_scores_checkpoints_and_updates_existing_history(tmp_path, monkeypatch):
    import time

    from config import CONFIG
    from src.data import ManifestDataset
    from src.io import file_hash
    from src.losses import plane_metrics, reconstruction_loss
    from src.model import HologramUNet
    from src.monitor import main as monitor

    torch.set_num_threads(2)
    for split, size, seed in (("train", 3, 1), ("val", 1, 2), ("test", 1, 3)):
        prepare_split(tmp_path, split, size, seed=seed)
    # Constant targets must produce explicit undefined CC, not a misleading zero.
    test_path = tmp_path / "test/manifest.json"
    manifest = json.loads(test_path.read_text())
    for source in manifest["samples"][0]["sources"]:
        path = test_path.parent / source["path"]
        Image.new("L", (128, 128)).save(path)
        source["sha256"] = file_hash(path)
    test_path.write_text(json.dumps(manifest))
    model = HologramUNet(base_channels=2).to(memory_format=torch.channels_last).eval()
    checkpoint = {
        "version": 1, "protocol": CONFIG.snapshot(), "model_config": {"base_channels": 2},
        "model_state_dict": model.state_dict(), "epochs": 1, "batch_size": 2,
        "train_manifest": str(tmp_path / "train/manifest.json"),
        "train_manifest_sha256": file_hash(tmp_path / "train/manifest.json"),
        "loss_config": {"alpha": 0.6, "beta": 0.4},
    }
    torch.save(checkpoint, tmp_path / "best.pt")
    arguments = ["--run", str(tmp_path), "--val-manifest", str(tmp_path / "val/manifest.json"),
                 "--test-manifest", str(test_path), "--device", "cpu"]
    monitor(arguments + ["--once"])
    journal = tmp_path / "monitor.jsonl"
    row = json.loads(journal.read_text())
    targets = torch.stack(list(ManifestDataset(checkpoint["train_manifest"])))
    with torch.inference_mode():
        recons = torch.cat([model(batch)[1] for batch in targets.split(checkpoint["batch_size"])])
        ssim, cc, valid = plane_metrics(recons, targets)
        expected_loss = reconstruction_loss(recons, targets, **checkpoint["loss_config"]).item()
    assert row["train"]["loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert row["train"]["ssim"] == pytest.approx(ssim.mean().item(), abs=1e-6)
    assert row["train"]["cc"] == pytest.approx(cc[valid].mean().item(), abs=1e-6)
    assert row["test"]["cc"] is None and row["test"]["valid_cc_pairs"] == 0
    with Image.open(tmp_path / "monitor.png") as plot:
        assert plot.width > 1000 and plot.height > 300
    with journal.open("a") as stream:
        stream.write('{"epoch":')
    monitor(arguments + ["--once"])
    assert len(journal.read_text().splitlines()) == 1

    import matplotlib.pyplot as plt

    deadline = time.monotonic() + 20

    def advance_checkpoint_and_close_when_updated(_):
        if checkpoint["epochs"] == 1:
            checkpoint["epochs"] = 2
            torch.save(checkpoint, tmp_path / "next.pt")
            (tmp_path / "next.pt").replace(tmp_path / "best.pt")
        if len(journal.read_text().splitlines()) == 2:
            figure = plt.gcf()
            assert all(list(line.get_xdata()) == [1, 2] for axis in figure.axes for line in axis.lines)
            plt.close("all")
        elif time.monotonic() > deadline:
            pytest.fail("Monitor did not observe the new checkpoint")
        time.sleep(0.01)

    monkeypatch.setattr(plt, "pause", advance_checkpoint_and_close_when_updated)
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monitor(arguments + ["--interval", "0.01"])
    assert [json.loads(line)["epoch"] for line in journal.read_text().splitlines()] == [1, 2]


def test_best_checkpoint_keeps_better_earlier_epoch_and_breaks_ties_by_pearson(tmp_path, monkeypatch):
    prepare_split(tmp_path, "train", 1, seed=1)
    prepare_split(tmp_path, "val", 1, seed=2)
    scores = iter([(0.3, 0.4), (0.3, 0.5), (0.2, 0.9), (0.3, 0.5)])
    expected = {}

    def validation(model, *args):
        ssim, cc = next(scores)
        if cc == 0.5 and not expected:
            expected.update({key: value.detach().clone() for key, value in model.state_dict().items()})
        return {"loss": 1 - ssim, "worst_plane_ssim": ssim, "mean_pearson": cc}

    def comparison(model, dataset, out, **kwargs):
        assert kwargs["epoch"] == 2
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    monkeypatch.setattr(training, "validate", validation)
    monkeypatch.setattr(training, "compare", comparison)
    run = tmp_path / "run"
    train(["--train-manifest", str(tmp_path / "train/manifest.json"),
           "--eval-manifest", str(tmp_path / "val/manifest.json"), "--epochs", "4",
           "--base-channels", "2", "--device", "cpu", "--out", str(run)])
    _, checkpoint = load_checkpoint(run / "best.pt", torch.device("cpu"))
    _, last = load_checkpoint(run / "last.pt", torch.device("cpu"))
    assert checkpoint["epochs"] == 2 and last["epochs"] == 4
    assert set(run.glob("*.pt")) == {run / "best.pt", run / "last.pt"}
    assert [json.loads(line)["best_epoch"] for line in (run / "training.jsonl").read_text().splitlines()] == [1, 2, 2, 2]
