"""Train the hologram U-Net and write a comparison against Adam and Global MP-GS."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path

import torch

from config import CONFIG
from src.data import ManifestDataset, verify_disjoint
from src.evaluate import add_runtime_arguments, compare, runtime
from src.io import file_hash, quantize_phase, quantize_png
from src.losses import plane_metrics, reconstruction_loss
from src.model import HologramUNet
from src.reconstruction import normalize_recons, reconstruct_four


def load_checkpoint(path: Path, device: torch.device) -> tuple[HologramUNet, dict]:
    contents = path.read_bytes()
    checkpoint = torch.load(BytesIO(contents), weights_only=True, map_location="cpu")
    if checkpoint.get("version") != 1 or checkpoint.get("protocol") != CONFIG.snapshot():
        raise ValueError("Unsupported checkpoint version or optical protocol")
    model = HologramUNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["checkpoint_sha256"] = sha256(contents).hexdigest()
    layout = torch.channels_last if device.type == "cpu" else torch.contiguous_format
    return model.to(device, memory_format=layout).eval(), checkpoint


@torch.inference_mode()
def validate(model, batches, device, loss_config):
    model.eval()
    loss_sum = 0.0
    ssim_sum = torch.zeros(4, dtype=torch.float64)
    cc_sum = torch.zeros(4, dtype=torch.float64)
    valid_pairs = torch.zeros(4, dtype=torch.int64)
    for targets in batches:
        targets = targets.to(device)
        phase, recons = model(targets)
        loss = reconstruction_loss(recons, targets, **loss_config)
        quantized = torch.cat([quantize_phase(item) for item in phase.split(1)]).to(device)
        images = quantize_png(normalize_recons(reconstruct_four(quantized)))
        ssim, cc, valid = plane_metrics(images.double(), targets.cpu().double())
        if not torch.isfinite(loss) or not torch.isfinite(ssim).all() or not torch.isfinite(cc).all():
            raise FloatingPointError("Non-finite validation loss or metrics")
        loss_sum += loss.item() * len(targets)
        ssim_sum += ssim.sum(dim=0)
        cc_sum += cc.sum(dim=0)
        valid_pairs += valid.sum(dim=0)
    size = len(batches.dataset)
    return {"loss": loss_sum / size, "ssim": (ssim_sum / size).tolist(),
            "worst_plane_ssim": (ssim_sum / size).min().item(),
            "mean_pearson": cc_sum.sum().item() / (4 * size),
            "valid_cc_pairs": valid_pairs.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    evaluation = parser.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--eval-manifest", type=Path)
    evaluation.add_argument("--evaluate-train", action="store_true", help="compare on training inputs for an overfit run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--plane-weights", type=float, nargs=4, default=(1., 1., 1., 1.))
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adam-steps", type=int, default=100)
    parser.add_argument("--adam-lr", type=float, default=0.1)
    parser.add_argument("--gs-iterations", type=int, default=100)
    add_runtime_arguments(parser)
    parser.set_defaults(threads=min(6, torch.get_num_threads()))
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.adam_steps, args.gs_iterations) < 1:
        parser.error("Epochs, batch size and baseline iteration counts must be positive")
    if any(not math.isfinite(rate) or rate <= 0 for rate in (args.lr, args.adam_lr)):
        parser.error("Learning rates must be finite and positive")
    if any(not math.isfinite(w) or w < 0 for w in (args.alpha, args.beta)) or args.alpha == args.beta == 0:
        parser.error("Loss weights must be finite, nonnegative and at least one must be positive")

    device = runtime(args)
    torch.manual_seed(args.seed)
    model_config = {
        "base_channels": args.base_channels, "rounds": args.rounds, "plane_weights": tuple(args.plane_weights),
    }
    loss_config = {"alpha": args.alpha, "beta": args.beta}
    baseline_config = {
        "adam_steps": args.adam_steps, "adam_lr": args.adam_lr, "gs_iterations": args.gs_iterations, "seed": args.seed,
    }
    train_set = ManifestDataset(args.train_manifest)
    if train_set.manifest["split"] != "train":
        parser.error("--train-manifest must identify a training split")
    if args.evaluate_train:
        eval_set = train_set
    else:
        eval_set = ManifestDataset(args.eval_manifest)
        verify_disjoint([train_set.manifest, eval_set.manifest])
    batches = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    validation_batches = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size)
    layout = torch.channels_last if device.type == "cpu" else torch.contiguous_format
    model = HologramUNet(**model_config).to(device, memory_format=layout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.out / "best.pt"
    temporary = checkpoint_path.with_suffix(".tmp")
    checkpoint_metadata = {
        "version": 1, "protocol": CONFIG.snapshot(), "model_config": model_config,
        "loss_config": loss_config, "baseline_config": baseline_config, "batch_size": args.batch_size,
        "learning_rate": args.lr, "train_manifest": str(train_set.path.resolve()),
        "train_manifest_sha256": file_hash(train_set.path),
        "validation_manifest": str(eval_set.path.resolve()), "validation_manifest_sha256": file_hash(eval_set.path),
        "selection": "max worst-plane mean PNG SSIM, then mean PNG Pearson (undefined pairs count as zero)",
    }
    best_score = (-math.inf, -math.inf)
    best_epoch = None
    with (args.out / "training.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for targets in batches:
                targets = targets.to(device, memory_format=layout)
                optimizer.zero_grad(set_to_none=True)
                _, recons = model(targets)
                loss = reconstruction_loss(recons, targets, **loss_config)
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                loss_sum += value * len(targets)
            validation = validate(model, validation_batches, device, loss_config)
            score = (validation["worst_plane_ssim"], validation["mean_pearson"])
            if score > best_score:
                torch.save({**checkpoint_metadata, "epochs": epoch, "model_state_dict": model.state_dict(),
                            "validation": validation}, temporary)
                temporary.replace(checkpoint_path)
                best_score, best_epoch = score, epoch

            line = json.dumps({"epoch": epoch, "mean_loss": loss_sum / len(train_set), "quartets": len(train_set),
                               "validation": validation, "best_epoch": best_epoch}, allow_nan=False)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    compare(model, eval_set, args.out / "comparison", checkpoint=checkpoint_path, epoch=checkpoint["epochs"],
            checkpoint_hash=checkpoint["checkpoint_sha256"], **baseline_config, **loss_config)
    print(args.out / "comparison" / "report.md")


if __name__ == "__main__":
    main()
