"""Compare a trained network and two phase solvers on a fixed manifest."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch

from config import CONFIG
from src.baseline import optimize_phase
from src.data import ManifestDataset
from src.global_gs import global_mp_gs
from src.io import file_hash, quantize_phase, quantize_png
from src.losses import plane_metrics, reconstruction_loss
from src.reconstruction import normalize_recons, reconstruct_four
from src.refinement import refine_phase


METHODS = ("network", "adam", "global_mp_gs")
LABELS = {"network": "U-Net", "adam": "Adam", "global_mp_gs": "Global MP-GS"}


def _plot_loss(log: Path, out: Path, *, through_epoch: int | None = None) -> bool:
    if not log.is_file():
        return False
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith("\n") and line.strip()]
    rows = [row for row in rows if row.get("event", "epoch") == "epoch"]
    if through_epoch is not None:
        rows = [row for row in rows if row["epoch"] <= through_epoch]
    if not rows:
        return False

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(7, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot([row["epoch"] for row in rows], [row["mean_loss"] for row in rows], ".-")
    axes.set(title="Training loss", xlabel="Epoch", ylabel="Mean loss")
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.grid(alpha=0.25)
    figure.savefig(out, dpi=150)
    return True


@torch.no_grad()
def _batch_scores(phases, targets):
    quantized = quantize_phase(phases, keep_device=True)
    images = quantize_png(normalize_recons(reconstruct_four(quantized)), keep_device=True)
    ssim, pearson, valid = plane_metrics(images.double(), targets.to(images.device).double())
    ssim, pearson, valid = ssim.cpu().tolist(), pearson.cpu().tolist(), valid.cpu().tolist()
    return [{'ssim': s, 'pearson': [r if ok else None for r, ok in zip(p, v)]}
            for s, p, v in zip(ssim, pearson, valid)]


@torch.no_grad()
def _score(phase, targets):
    return _batch_scores(phase, targets)[0]


def _summary(samples, method_names=METHODS):
    methods = {
        method: {"solved": sum(method in row["metrics"] for row in samples), "mean": {}}
        for method in method_names
    }
    shared = {}
    for metric in ("ssim", "pearson"):
        pairs = [
            values for row in samples if all(method in row["metrics"] for method in method_names)
            for values in zip(*(row["metrics"][method][metric] for method in method_names))
            if all(value is not None for value in values)
        ]
        shared[metric] = len(pairs)
        for index, method in enumerate(method_names):
            methods[method]["mean"][metric] = mean(pair[index] for pair in pairs) if pairs else None
    return methods, shared


@torch.no_grad()
def validate_epoch(
    infer, dataset, out, *, epoch, batch_size, gs_iterations, seed, alpha, beta,
    checkpoint, baseline_cache, progress=None, device='cpu', preloaded_targets=None,
    refine_steps=0, refine_lr=0.1,
):
    """Score all validation quartets; reuse the unchanged Global MP-GS results.

    Both methods use _score's reference optics and identical 8-bit quantization.
    Keep one JSON report, overwriting it after each complete validation pass.
    """
    rows, loss_sum = [], 0.0
    device = torch.device(device)
    loader = torch.utils.data.DataLoader(dataset if preloaded_targets is None else preloaded_targets,
                                        batch_size=batch_size, shuffle=False)
    for targets in loader:
        targets = targets.to(device)
        phases, recons = infer(targets)
        if refine_steps:
            phases = refine_phase(phases, targets, steps=refine_steps, lr=refine_lr, alpha=alpha, beta=beta)
            recons = normalize_recons(reconstruct_four(phases))
        value = float(reconstruction_loss(recons, targets, alpha=alpha, beta=beta))
        if not math.isfinite(value):
            raise FloatingPointError('Non-finite validation loss')
        loss_sum += value * len(targets)
        batch_samples = dataset.samples[len(rows):len(rows) + len(targets)]
        network_scores = _batch_scores(phases, targets)
        black = (targets.amax(dim=(-2, -1)) == 0).any(dim=1).cpu().tolist()
        missing = []
        for index, (sample, is_black) in enumerate(zip(batch_samples, black)):
            if sample['id'] in baseline_cache:
                continue
            if is_black:
                baseline_cache[sample['id']] = {
                    'metrics': None, 'error': 'A black target plane is incompatible with nonzero conserved power'}
            else:
                missing.append(index)
        if missing:
            selected = targets[missing]
            gs = global_mp_gs(selected, iterations=gs_iterations, seed=seed, device=device)
            for index, score in zip(missing, _batch_scores(gs, selected)):
                baseline_cache[batch_samples[index]['id']] = {'metrics': score, 'error': None}
        for sample, score in zip(batch_samples, network_scores):
            baseline = baseline_cache[sample['id']]
            scores = {'network': score}
            errors = {}
            if baseline['metrics'] is not None:
                scores['global_mp_gs'] = baseline['metrics']
            else:
                errors['global_mp_gs'] = baseline['error']
            rows.append({'id': sample['id'], 'category': sample.get('category'), 'metrics': scores, 'errors': errors})
            if len(rows) % 50 == 0 or len(rows) == len(dataset):
                message = {'event': 'validation_progress', 'epoch': epoch,
                           'quartets': len(rows), 'total_quartets': len(dataset)}
                if progress:
                    progress(message)
                else:
                    print(json.dumps(message), flush=True)
    methods, shared = _summary(rows, ('network', 'global_mp_gs'))
    report = {
        'epoch': epoch, 'split': dataset.manifest['split'], 'quartets': len(dataset),
        'total_pairs': 4 * len(dataset), 'mean_loss': loss_sum / len(dataset),
        'manifest': str(dataset.path.resolve()), 'manifest_sha256': file_hash(dataset.path),
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': file_hash(checkpoint),
        'protocol': CONFIG.snapshot(), 'seed': seed, 'gs_iterations': gs_iterations,
        'refinement_config': {'steps': refine_steps, 'lr': refine_lr},
        'quantization': 'phase and reconstructions, 8-bit',
        'inference_device': str(device), 'scoring_device': str(device), 'gs_device': str(device),
        'shared_pairs': shared, 'methods': methods, 'samples': rows,
    }
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(out)
    return report


@torch.no_grad()
def compare(
    model, dataset: ManifestDataset, out: Path, *, adam_steps: int, adam_lr: float,
    gs_iterations: int, seed: int, alpha: float, beta: float, checkpoint: Path, epoch: int, checkpoint_hash: str,
    refine_steps: int = 0, refine_lr: float = 0.1,
) -> dict:
    """Evaluate the same quantized phase and image pipeline for every solver."""
    if min(adam_steps, gs_iterations) < 1:
        raise ValueError("Baseline iteration counts must be positive")
    if not math.isfinite(adam_lr) or adam_lr <= 0:
        raise ValueError("Adam learning rate must be finite and positive")
    if refine_steps < 0 or not math.isfinite(refine_lr) or refine_lr <= 0:
        raise ValueError("Refinement steps must be nonnegative and learning rate finite and positive")
    out.mkdir(parents=True, exist_ok=False)
    has_loss_plot = _plot_loss(checkpoint.with_name("training.jsonl"), out / "loss.png", through_epoch=epoch)
    device = next(model.parameters()).device
    model.eval()
    samples = []
    for index, sample in enumerate(dataset.samples):
        targets = dataset[index][None].to(device)
        mask, _ = model(targets)
        if refine_steps:
            mask = refine_phase(mask, targets, steps=refine_steps, lr=refine_lr, alpha=alpha, beta=beta)
        adam = optimize_phase(targets, steps=adam_steps, lr=adam_lr, seed=seed, alpha=alpha, beta=beta)
        scores = {"network": _score(mask, targets), "adam": _score(adam, targets)}
        errors = {}
        if (targets.amax(dim=(-2, -1)) == 0).any():
            errors["global_mp_gs"] = "A black target plane is incompatible with nonzero conserved power"
        else:
            gs = global_mp_gs(targets, iterations=gs_iterations, seed=seed)
            scores["global_mp_gs"] = _score(gs, targets)
        samples.append({"id": sample["id"], "category": sample.get("category"), "metrics": scores, "errors": errors})
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"Comparison {index + 1}/{len(dataset)}", flush=True)

    methods, shared = _summary(samples)
    report = {
        "manifest": str(dataset.path.resolve()), "manifest_sha256": file_hash(dataset.path),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "loss_plot": "loss.png" if has_loss_plot else None,
        "epoch": epoch,
        "split": dataset.manifest["split"], "quartets": len(dataset), "total_pairs": 4 * len(dataset),
        "protocol": CONFIG.snapshot(), "device": str(device), "seed": seed,
        "quantization": "phase and reconstructions, 8-bit",
        "baseline_parameters": {"adam_steps": adam_steps, "adam_lr": adam_lr, "gs_iterations": gs_iterations,
                                "alpha": alpha, "beta": beta},
        "refinement_config": {"steps": refine_steps, "lr": refine_lr},
        "shared_pairs": shared, "methods": methods, "samples": samples,
    }
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    lines = [
        "# Сравнение реконструкций",
        "",
        f"Набор `{report['split']}`, {len(dataset)} четвёрок. Веса U-Net после эпохи {epoch}. "
        f"Adam — {adam_steps} шагов, "
        f"Global MP-GS — {gs_iterations} итераций на четвёрку.",
        f"Доводка выхода U-Net: {refine_steps} шагов Adam, lr={refine_lr}; "
        "при ненулевом числе шагов фаза в цикле квантуется в 8 бит через STE.",
        "",
        "| Метод | Средний SSIM | Средний Pearson | Решено четвёрок |",
        "|---|---:|---:|---:|",
    ]
    for method, result in methods.items():
        ssim, pearson = [
            f"{result['mean'][metric]:.6f}" if result["mean"][metric] is not None else "—"
            for metric in ("ssim", "pearson")
        ]
        label = f"U-Net + {refine_steps} Adam" if method == "network" and refine_steps else LABELS[method]
        lines.append(f"| {label} | {ssim} | {pearson} | {result['solved']}/{len(dataset)} |")
    lines += [
        "",
        f"Общие пары для среднего SSIM — {shared['ssim']}/{report['total_pairs']}, "
        f"для Pearson — {shared['pearson']}/{report['total_pairs']}.",
        "Для каждой метрики набор пар одинаков для всех методов. "
        "Пары с неопределённым Pearson исключаются из среднего Pearson. "
        "Метрики каждой плоскости и причины пропусков сохранены в `summary.json`.",
    ]
    if has_loss_plot:
        lines += ["", "![Training loss by epoch](loss.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def add_runtime_arguments(parser):
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)


def add_refinement_arguments(parser, *, inherit=False):
    parser.add_argument("--refine-steps", type=int, default=None if inherit else 0,
                        help="Additional Adam steps on the network mask; 0 disables refinement")
    parser.add_argument("--refine-lr", type=float, default=None if inherit else 0.1,
                        help="Learning rate for network mask refinement (default: 0.1)")


def runtime(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    return torch.device("cpu" if device == "auto" else device)


def main(argv=None):
    from src.train import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--adam-lr", type=float)
    parser.add_argument("--gs-iterations", type=int)
    add_refinement_arguments(parser, inherit=True)
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    if args.refine_steps is not None and args.refine_steps < 0:
        parser.error("--refine-steps must be nonnegative")
    if args.refine_lr is not None and (not math.isfinite(args.refine_lr) or args.refine_lr <= 0):
        parser.error("--refine-lr must be finite and positive")
    model, checkpoint = load_checkpoint(args.checkpoint, runtime(args))
    settings = checkpoint["baseline_config"].copy()
    settings.update({name: getattr(args, name) for name in settings if getattr(args, name, None) is not None})
    refinement = {"steps": 10, "lr": 0.1, **checkpoint.get("refinement_config", {})}
    if args.refine_steps is not None:
        refinement["steps"] = args.refine_steps
    if args.refine_lr is not None:
        refinement["lr"] = args.refine_lr
    compare(model, ManifestDataset(args.eval_manifest), args.out, checkpoint=args.checkpoint, epoch=checkpoint["epochs"],
            checkpoint_hash=checkpoint["checkpoint_sha256"],
            refine_steps=refinement["steps"], refine_lr=refinement["lr"],
            **settings, **checkpoint["loss_config"])
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
