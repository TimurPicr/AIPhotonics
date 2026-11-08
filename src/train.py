"""Train the hologram U-Net and write a comparison against Adam and Global MP-GS."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time

import torch

from config import CONFIG
from src.data import ManifestDataset, verify_disjoint
from src.evaluate import add_refinement_arguments, add_runtime_arguments, compare, runtime, validate_epoch
from src.io import file_hash
from src.losses import reconstruction_loss
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


@torch.no_grad()
def _preload(dataset, device, batch_size, record):
    parts = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        parts.append(torch.stack([dataset[index] for index in range(start, stop)]).to(device))
        if start == 0 or stop % 1024 == 0 or stop == len(dataset):
            record({'event': 'data_progress', 'split': dataset.manifest['split'],
                    'quartets': stop, 'total_quartets': len(dataset), 'device': str(device)})
    return torch.cat(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    evaluation = parser.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--eval-manifest", type=Path)
    evaluation.add_argument("--evaluate-train", action="store_true", help="compare on training inputs for an overfit run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume", type=Path, help="continue this run's last.pt; --epochs is the total epoch count")
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
    add_refinement_arguments(parser)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    parser.add_argument("--validate-every-epoch", action="store_true",
                        help="validate against Global MP-GS after every epoch (opt-in for MLX)")
    parser.add_argument("--preload", action="store_true", help="verify targets once and keep them on the selected device")
    add_runtime_arguments(parser)
    parser.set_defaults(threads=min(6, torch.get_num_threads()))
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.adam_steps, args.gs_iterations) < 1:
        parser.error("Epochs, batch size and baseline iteration counts must be positive")
    if any(not math.isfinite(rate) or rate <= 0 for rate in (args.lr, args.adam_lr)):
        parser.error("Learning rates must be finite and positive")
    if args.refine_steps < 0:
        parser.error("--refine-steps must be nonnegative")
    if not math.isfinite(args.refine_lr) or args.refine_lr <= 0:
        parser.error("--refine-lr must be finite and positive")
    if any(not math.isfinite(w) or w < 0 for w in (args.alpha, args.beta)) or args.alpha == args.beta == 0:
        parser.error("Loss weights must be finite, nonnegative and at least one must be positive")
    if args.resume and (args.backend != 'torch' or not args.validate_every_epoch):
        parser.error('Resume requires the torch backend and --validate-every-epoch')

    device = runtime(args)
    torch.manual_seed(args.seed)
    model_config = {
        "base_channels": args.base_channels, "rounds": args.rounds, "plane_weights": tuple(args.plane_weights),
    }
    loss_config = {"alpha": args.alpha, "beta": args.beta}
    refinement_config = {"steps": args.refine_steps, "lr": args.refine_lr}
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
    batch_generator = torch.Generator().manual_seed(args.seed)
    batches = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=batch_generator,
    )
    resumed = None
    first_epoch = 1
    baseline_cache = {}
    if args.resume:
        if args.resume.resolve() != (args.out / 'last.pt').resolve():
            parser.error('--resume must name --out/last.pt so the existing journal is preserved')
        model, resumed = load_checkpoint(args.resume, device)
        if resumed.get('refinement_config', {'steps': 0, 'lr': 0.1}) != refinement_config:
            parser.error('Resume configuration mismatch: refinement_config')
        required = {
            'backend': 'torch', 'model_config': model_config, 'loss_config': loss_config,
            'baseline_config': baseline_config, 'batch_size': args.batch_size, 'learning_rate': args.lr,
            'seed': args.seed, 'train_manifest_sha256': file_hash(train_set.path),
            'eval_manifest_sha256': file_hash(eval_set.path),
        }
        for key, expected in required.items():
            if resumed.get(key) != expected:
                parser.error(f'Resume configuration or data mismatch: {key}')
        if not resumed.get('optimizer_state_dict') or args.epochs <= resumed['epochs']:
            parser.error('Resume needs optimizer state and a larger total --epochs value')
        previous_rows = [json.loads(line) for line in (args.out / 'training.jsonl').read_text().splitlines()]
        completed = [row['epoch'] for row in previous_rows if row.get('event') == 'epoch']
        if completed != list(range(1, resumed['epochs'] + 1)):
            parser.error('Checkpoint must match the completed, validated epochs in the existing journal')
        previous_report = json.loads((args.out / 'validation.json').read_text())
        report_checks = {
            'epoch': resumed['epochs'], 'checkpoint_sha256': resumed['checkpoint_sha256'],
            'manifest_sha256': required['eval_manifest_sha256'], 'protocol': CONFIG.snapshot(),
            'seed': args.seed, 'gs_iterations': args.gs_iterations,
            'quantization': 'phase and reconstructions, 8-bit',
        }
        if any(previous_report.get(key) != value for key, value in report_checks.items()):
            parser.error('Previous validation does not match this checkpoint, dataset or GS configuration')
        if previous_report.get('refinement_config', {'steps': 0, 'lr': 0.1}) != refinement_config:
            parser.error('Previous validation uses different refinement settings')
        if [row['id'] for row in previous_report['samples']] != [s['id'] for s in eval_set.samples]:
            parser.error('Previous validation sample order differs from the evaluation manifest')
        baseline_cache = {
            row['id']: {'metrics': row['metrics'].get('global_mp_gs'),
                        'error': row['errors'].get('global_mp_gs')}
            for row in previous_report['samples']
        }
        first_epoch = resumed['epochs'] + 1
        if 'data_generator_state' in resumed:
            batch_generator.set_state(resumed['data_generator_state'].cpu())
        else:
            # Older checkpoints predate sampler-state storage. Replay only the
            # index sampler to recover its exact state without reading PNGs.
            replay = torch.utils.data.DataLoader(
                range(len(train_set)), batch_size=args.batch_size, shuffle=True, generator=batch_generator)
            for _ in range(resumed['epochs']):
                for _ in replay:
                    pass
    else:
        model = HologramUNet(**model_config).to(device)
    mlx_backend = None
    if args.backend == 'mlx':
        if device.type != 'cpu':
            parser.error('MLX uses its Metal GPU; choose --device cpu for reference scoring')
        from src.mlx_backend import MLXBackend
        mlx_backend = MLXBackend(model, lr=args.lr, seed=args.seed, loss_config=loss_config)
    optimizer = None if mlx_backend else torch.optim.Adam(model.parameters(), lr=args.lr)
    if resumed:
        optimizer.load_state_dict(resumed['optimizer_state_dict'])
    minimal = mlx_backend is not None or args.validate_every_epoch
    if not resumed:
        args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.out / "last.pt"
    temporary = checkpoint_path.with_suffix(".tmp")
    checkpoint_metadata = {
        "version": 1, "protocol": CONFIG.snapshot(), "model_config": model_config,
        "loss_config": loss_config, "baseline_config": baseline_config, "batch_size": args.batch_size,
        "refinement_config": refinement_config,
        "learning_rate": args.lr, "train_manifest": str(train_set.path.resolve()),
        "train_manifest_sha256": file_hash(train_set.path),
        "validation_manifest": str(eval_set.path.resolve()), "validation_manifest_sha256": file_hash(eval_set.path),
        "selection": "max worst-plane mean PNG SSIM, then mean PNG Pearson (undefined pairs count as zero)",
    }
    if minimal:
        checkpoint_metadata.update({
            'backend': args.backend, 'device': 'mlx:gpu' if mlx_backend else str(device), 'seed': args.seed,
            'eval_manifest': str(eval_set.path.resolve()), 'eval_manifest_sha256': file_hash(eval_set.path),
            'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        })
    if resumed:
        checkpoint_metadata.update({'resumed_from_epoch': resumed['epochs'],
                                    'resume_checkpoint_sha256': resumed['checkpoint_sha256']})
    started = time.monotonic()
    with (args.out / "training.jsonl").open("a" if resumed else "w", encoding="utf-8") as log:
        def record(row):
            line = json.dumps(row, allow_nan=False)
            log.write(line + '\n')
            log.flush()
            print(line, flush=True)

        if minimal:
            record({'event': 'resume' if resumed else 'start', 'pid': os.getpid(), 'backend': args.backend,
                    'device': 'mlx:gpu' if mlx_backend else str(device),
                    'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.lr,
                    'train_quartets': len(train_set), 'val_quartets': len(eval_set),
                    'gs_iterations': args.gs_iterations, 'seed': args.seed,
                    'first_epoch': first_epoch, 'cached_gs_quartets': len(baseline_cache)})
        preloaded_val = None
        if args.preload:
            preloaded_train = _preload(train_set, device, args.batch_size, record)
            if args.validate_every_epoch:
                preloaded_val = preloaded_train if args.evaluate_train else _preload(eval_set, device, args.batch_size, record)
            batches = torch.utils.data.DataLoader(
                preloaded_train, batch_size=args.batch_size, shuffle=True,
                generator=batch_generator)
        for epoch in range(first_epoch, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            epoch_started = time.monotonic()
            seen = 0
            for batch_index, targets in enumerate(batches, 1):
                if mlx_backend:
                    value = mlx_backend.step(targets)
                else:
                    targets = targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    _, recons = model(targets)
                    loss = reconstruction_loss(recons, targets, **loss_config)
                    value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                if not mlx_backend:
                    loss.backward()
                    optimizer.step()
                loss_sum += value * len(targets)
                seen += len(targets)
                if minimal and (batch_index == 1 or batch_index % 25 == 0 or batch_index == len(batches)):
                    record({'event': 'train_progress', 'epoch': epoch, 'batch': batch_index,
                            'total_batches': len(batches), 'quartets': seen, 'mean_loss': loss_sum / seen,
                            'seconds': time.monotonic() - epoch_started})
            state = {'model_state_dict': mlx_backend.state_dict() if mlx_backend else model.state_dict()}
            if mlx_backend:
                state['optimizer_state_dict'] = mlx_backend.optimizer_state_dict()
            elif minimal:
                state['optimizer_state_dict'] = optimizer.state_dict()
                state['data_generator_state'] = batch_generator.get_state()
            torch.save({**checkpoint_metadata, "epochs": epoch, **state}, temporary)
            temporary.replace(checkpoint_path)

            row = {"epoch": epoch, "mean_loss": loss_sum / len(train_set), "quartets": len(train_set)}
            if minimal:
                row.update({'event': 'epoch', 'seconds': time.monotonic() - epoch_started})
            if args.validate_every_epoch:
                model.eval()
                report = validate_epoch(
                    mlx_backend if mlx_backend else model, eval_set, args.out / 'validation.json',
                    epoch=epoch, batch_size=args.batch_size, device=device, preloaded_targets=preloaded_val,
                    gs_iterations=args.gs_iterations, seed=args.seed, checkpoint=checkpoint_path,
                    refine_steps=args.refine_steps, refine_lr=args.refine_lr,
                    baseline_cache=baseline_cache, progress=record, **loss_config)
                row.update({'event': 'epoch', 'val_loss': report['mean_loss'],
                            'validation': report['methods'], 'shared_pairs': report['shared_pairs'],
                            'seconds': time.monotonic() - epoch_started})
            record(row)

        if minimal:
            record({'event': 'complete', 'epochs': args.epochs, 'seconds': time.monotonic() - started})

    if not minimal:
        compare(model, eval_set, args.out / "comparison", checkpoint=checkpoint_path, epoch=args.epochs,
                refine_steps=args.refine_steps, refine_lr=args.refine_lr,
                checkpoint_hash=file_hash(checkpoint_path), **baseline_config, **loss_config)
        print(args.out / "comparison" / "report.md")
    elif args.validate_every_epoch:
        print(args.out / 'validation.json', flush=True)
    else:
        print(checkpoint_path, flush=True)


if __name__ == "__main__":
    main()
