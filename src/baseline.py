"""Optimize a phase mask directly with Adam and export it using the PNG protocol."""

import argparse
import math
import platform
import time
from pathlib import Path

import torch
from torch import Tensor

from config import CONFIG
from src.physics import fresnel_1fft, output_pitch_m
from src.reconstruction import normalize_recons, reconstruct_four, reconstruction_values


def _random_phase(targets: Tensor, seed: int) -> Tensor:
    shape = (targets.shape[0], 1, *targets.shape[-2:])
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=generator, dtype=targets.dtype) * (2 * math.pi) - math.pi).to(targets)


def _loss(phase: Tensor, targets: Tensor, plane: int | None) -> Tensor:
    if plane is None:
        values = reconstruct_four(phase)
    else:
        values = reconstruction_values(fresnel_1fft(phase, CONFIG.z_m[plane - 1]))
    return (normalize_recons(values) - targets).square().mean()


@torch.enable_grad()
def optimize_phase(
    targets: Tensor,
    *,
    initial_phase: Tensor | None,
    steps: int,
    seed: int,
    plane: int | None = None,
    lr: float = 0.1,
) -> Tensor:
    """Fit four ordered targets, or one target at the given one-based plane index."""
    if plane is not None and plane not in range(1, len(CONFIG.z_m) + 1):
        raise ValueError("plane must be one of 1, 2, 3, 4")
    channels = len(CONFIG.z_m) if plane is None else 1
    if targets.ndim != 4 or targets.shape[1] != channels or targets.shape[-2] != targets.shape[-1]:
        raise ValueError(f"targets must have shape [B, {channels}, N, N]")
    if targets.dtype not in (torch.float32, torch.float64):
        raise TypeError("targets must have dtype float32 or float64")
    if steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps must be nonnegative and lr must be finite and positive")
    targets = targets.detach()
    if initial_phase is None:
        phase = _random_phase(targets, seed)
    else:
        if initial_phase.shape != targets[:, :1].shape or not initial_phase.is_floating_point():
            raise ValueError("initial_phase must be a floating tensor of shape [B, 1, N, N]")
        phase = initial_phase.detach().to(targets).clone()

    phase.requires_grad_()
    optimizer = torch.optim.Adam([phase], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _loss(phase, targets, plane).backward()
        optimizer.step()

    # The field is periodic in phase; wrap only for the PNG encoding range.
    return (phase.detach() + math.pi).remainder(2 * math.pi) - math.pi


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_baseline(
    targets: Tensor,
    out: str | Path,
    *,
    steps: int,
    seed: int,
    plane: int | None = None,
    lr: float = 0.1,
    metadata: dict | None = None,
) -> dict:
    """Run a seeded baseline on a four-PNG set and score the saved PNGs."""
    from src.io import export_result, write_json

    fit_targets = targets if plane is None else targets[:, plane - 1:plane]
    start_phase = _random_phase(fit_targets, seed)
    warmup_steps = 2
    optimize_phase(fit_targets, initial_phase=start_phase, steps=warmup_steps, seed=seed, plane=plane, lr=lr)
    with torch.no_grad():
        initial_loss = _loss(start_phase, fit_targets, plane).item()
    _sync(targets.device)
    started = time.perf_counter()
    phase = optimize_phase(fit_targets, initial_phase=start_phase, steps=steps, seed=seed, plane=plane, lr=lr)
    _sync(targets.device)
    optimization_seconds = time.perf_counter() - started
    with torch.no_grad():
        final_loss = _loss(phase, fit_targets, plane).item()
    if not math.isfinite(final_loss):
        raise FloatingPointError("Adam produced a non-finite reconstruction loss")

    result = export_result(targets, phase, out, metadata={
        **(metadata or {}),
        "method": "adam_phase",
        "seed": seed,
        "steps": steps,
        "learning_rate": lr,
        "optimized_planes": list(range(1, 5)) if plane is None else [plane],
        "float_mse": {"initial": initial_loss, "final": final_loss},
        "protocol": CONFIG.snapshot(),
        "dx_out_m": [output_pitch_m(z, CONFIG.image_size) for z in CONFIG.z_m],
        "device": str(targets.device),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "timing": {"optimization_seconds": optimization_seconds, "warmup_steps": warmup_steps},
    })
    result["metadata"]["timing"]["pipeline_seconds"] = time.perf_counter() - started
    write_json(Path(out) / "metrics.json", result)
    return result


def main() -> None:
    from src.io import file_hash, load_four_png

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs=4, required=True, help="four PNGs in distance order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plane", type=int, choices=range(1, 5), help="fit just this plane; report all four")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps, threads and lr must be positive")

    torch.set_num_threads(args.threads)
    targets = load_four_png(args.images).to(args.device)
    inputs = [{"path": path, "sha256": file_hash(path)} for path in args.images]
    result = run_baseline(targets, args.out, steps=args.steps, seed=args.seed, plane=args.plane, lr=args.lr,
                          metadata={"inputs": inputs, "threads": args.threads})
    print(result["metadata"]["float_mse"], result["mean"])


if __name__ == "__main__":
    main()
