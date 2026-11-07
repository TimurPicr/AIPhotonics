"""Watch best.pt in a separate process and plot checkpoint loss, SSIM and CC."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import torch

from src.data import ManifestDataset, verify_disjoint
from src.evaluate import add_runtime_arguments, runtime
from src.io import file_hash
from src.losses import plane_metrics, reconstruction_loss
from src.train import load_checkpoint


@torch.inference_mode()
def evaluate_checkpoint(path, val_manifest, test_manifest, device):
    model, checkpoint = load_checkpoint(path, device)
    datasets = {name: ManifestDataset(manifest) for name, manifest in (
        ("train", checkpoint["train_manifest"]), ("val", val_manifest), ("test", test_manifest),
    )}
    if file_hash(datasets["train"].path) != checkpoint["train_manifest_sha256"]:
        raise ValueError("Training manifest differs from the checkpoint")
    verify_disjoint([dataset.manifest for dataset in datasets.values()])
    row = {"epoch": checkpoint["epochs"], "checkpoint_sha256": checkpoint["checkpoint_sha256"]}
    for name, dataset in datasets.items():
        loss_sum = ssim_sum = cc_sum = valid_pairs = 0
        for targets in torch.utils.data.DataLoader(dataset, batch_size=checkpoint["batch_size"]):
            targets = targets.to(device)
            _, recons = model(targets)
            loss = reconstruction_loss(recons, targets, **checkpoint["loss_config"])
            ssim, cc, valid = plane_metrics(recons, targets)
            if not torch.isfinite(loss) or not torch.isfinite(ssim).all() or not torch.isfinite(cc).all():
                raise FloatingPointError(f"Non-finite loss or metrics in {name}")
            loss_sum += loss.item() * len(targets)
            ssim_sum += ssim.sum().item()
            cc_sum += cc[valid].sum().item()
            valid_pairs += valid.sum().item()
        row[name] = {
            "loss": loss_sum / len(dataset), "ssim": ssim_sum / (4 * len(dataset)),
            "cc": cc_sum / valid_pairs if valid_pairs else None,
            "valid_cc_pairs": valid_pairs, "total_pairs": 4 * len(dataset),
        }
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5, help="Checkpoint polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Evaluate once and save a plot without opening a window")
    add_runtime_arguments(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    device = runtime(args)
    if args.once:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Refuse to mix histories from different validation/test selections.
    history_path = args.run / "monitor.jsonl"
    manifests = {name: file_hash(path) for name, path in (
        ("val", args.val_manifest), ("test", args.test_manifest),
    )}
    history = []
    if history_path.exists():
        contents = history_path.read_text()
        complete = contents[:contents.rfind("\n") + 1]
        for line in complete.splitlines():
            if line.strip():
                row = json.loads(line)
                if row["manifests"] != manifests:
                    parser.error("monitor.jsonl belongs to different validation/test manifests")
                history.append(row)
        if contents != complete:
            history_path.write_text(complete)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    lines = {}
    for axis, metric in zip(axes, ("loss", "ssim", "cc")):
        for name in ("train", "val", "test"):
            lines[name, metric], = axis.plot([], [], ".-", label=f"{name}_{metric}")
        axis.set(xlabel="Epoch", ylabel=metric.upper())
        axis.grid(alpha=0.25)
        axis.legend()
    checkpoint = args.run / "best.pt"
    if not checkpoint.exists() and (args.run / "last.pt").exists():
        checkpoint = args.run / "last.pt"  # Compatibility with earlier runs.
    signature = None
    pending = None

    def update():
        nonlocal signature, pending
        if pending is not None and pending.done():
            row = pending.result()
            pending = None
            if not history or row["checkpoint_sha256"] != history[-1]["checkpoint_sha256"]:
                row["manifests"] = manifests
                with history_path.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                history.append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
        if pending is None and (not args.once or signature is None) and checkpoint.exists():
            stat = checkpoint.stat()
            current = (stat.st_mtime_ns, stat.st_size)
            if current != signature:
                signature = current
                pending = worker.submit(evaluate_checkpoint, checkpoint, args.val_manifest, args.test_manifest, device)
        for (name, metric), line in lines.items():
            line.set_data([row["epoch"] for row in history],
                          [row[name][metric] if row[name][metric] is not None else float("nan") for row in history])
        for axis in axes:
            axis.relim()
            axis.autoscale_view()
        status = "Evaluating checkpoint…" if pending is not None else "Waiting for next checkpoint…"
        if history:
            row = history[-1]
            counts = ", ".join(f"{name}: {row[name]['valid_cc_pairs']}/{row[name]['total_pairs']}"
                               for name in ("train", "val", "test"))
            status += f"  Epoch {row['epoch']}; valid CC pairs — {counts}"
        figure.suptitle("Float checkpoint evaluation\n" + status, fontsize=10)
        figure.canvas.draw_idle()

    with ThreadPoolExecutor(max_workers=1) as worker:
        if args.once:
            if not checkpoint.is_file():
                parser.error(f"Checkpoint does not exist: {checkpoint}")
            update()
            pending.result()
            update()
            figure.savefig(args.run / "monitor.png", dpi=150)
        else:
            plt.show(block=False)
            while plt.fignum_exists(figure.number):
                update()
                plt.pause(args.interval)
        plt.close(figure)


if __name__ == "__main__":
    main()
