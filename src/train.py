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
from src.io import file_hash
from src.losses import reconstruction_loss
from src.model import HologramUNet


def load_checkpoint(path: Path, device: torch.device) -> tuple[HologramUNet, dict]:
    contents = path.read_bytes()
    checkpoint = torch.load(BytesIO(contents), weights_only=True, map_location="cpu")
    if checkpoint.get("version") != 1 or checkpoint.get("protocol") != CONFIG.snapshot():
        raise ValueError("Unsupported checkpoint version or optical protocol")
    model = HologramUNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["checkpoint_sha256"] = sha256(contents).hexdigest()
    return model.to(device).eval(), checkpoint


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
    model = HologramUNet(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.out / "last.pt"
    temporary = checkpoint_path.with_suffix(".tmp")
    checkpoint_metadata = {
        "version": 1, "protocol": CONFIG.snapshot(), "model_config": model_config,
        "loss_config": loss_config, "baseline_config": baseline_config, "batch_size": args.batch_size,
        "learning_rate": args.lr, "train_manifest": str(train_set.path.resolve()),
        "train_manifest_sha256": file_hash(train_set.path),
    }
    with (args.out / "training.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            loss_sum = 0.0
            for targets in batches:
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                _, recons = model(targets)
                loss = reconstruction_loss(recons, targets, **loss_config)
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                loss_sum += value * len(targets)
            torch.save({**checkpoint_metadata, "epochs": epoch, "model_state_dict": model.state_dict()}, temporary)
            temporary.replace(checkpoint_path)

            line = json.dumps({"epoch": epoch, "mean_loss": loss_sum / len(train_set), "quartets": len(train_set)})
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

    compare(model, eval_set, args.out / "comparison", checkpoint=checkpoint_path, epoch=args.epochs,
            checkpoint_hash=file_hash(checkpoint_path), **baseline_config, **loss_config)
    print(args.out / "comparison" / "report.md")


if __name__ == "__main__":
    main()
