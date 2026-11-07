"""Global multi-plane Gerchberg–Saxton with the shared Fresnel and PNG protocol."""

import argparse
import math
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch import Tensor

from config import CONFIG
from src.io import export_result, file_hash, load_four_png, write_json
from src.physics import UnitaryFresnel, output_pitch_m


PHASE_EPS = 1e-8
TARGET_NORM_EPS = 1e-8
WARMUP_STEPS = 2


@dataclass(frozen=True)
class GlobalGSResult:
    phase: Tensor
    history: list[dict]
    best_iteration: int
    solve_seconds: float
    elapsed_seconds: float


def _target_amplitudes(targets: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(targets, dim=(-2, -1), keepdim=True)
    if (norm <= TARGET_NORM_EPS).any():
        raise ValueError("A black target plane is incompatible with nonzero conserved power")
    return targets * (targets.shape[-1] / norm)


def _replace_amplitude(field: Tensor, amplitude: Tensor) -> Tensor:
    return amplitude * torch.exp(1j * field.angle())


def _gs_step(phase: Tensor, amplitudes: Tensor, operators: tuple[UnitaryFresnel, ...]) -> Tensor:
    field = torch.exp(1j * phase)
    back_fields = []
    for index, operator in enumerate(operators):
        propagated = operator.forward(field)
        projected = _replace_amplitude(propagated, amplitudes[:, index:index + 1])
        back_fields.append(operator.adjoint(projected))
    mean_field = torch.stack(back_fields).mean(dim=0)
    return torch.where(mean_field.abs() <= PHASE_EPS, phase, mean_field.angle())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def global_mp_gs(
    targets: Tensor,
    *,
    iterations: int = 100,
    seed: int = 42,
    eval_every: int = 10,
    device: str | torch.device | None = None,
) -> GlobalGSResult:
    """Fit one phase-only mask and select it by SSIM after the standard PNG round trip."""
    started = perf_counter()
    if iterations < 1 or eval_every < 1:
        raise ValueError("iterations and eval_every must be positive")
    if CONFIG.reconstruction_value != "amplitude":
        raise ValueError("Global MP-GS requires the amplitude reconstruction protocol")
    if targets.ndim == 3:
        targets = targets.unsqueeze(0)
    size = CONFIG.image_size
    if targets.shape != (1, 4, size, size):
        raise ValueError(f"Expected targets of shape [1, 4, {size}, {size}] or [4, {size}, {size}]")
    if targets.dtype not in (torch.float32, torch.float64):
        raise ValueError("Targets must be float32 or float64 in [0, 1]")
    if not torch.isfinite(targets).all() or (targets < 0).any() or (targets > 1).any():
        raise ValueError("Targets must contain finite amplitudes in [0, 1]")

    targets = targets.to(device=device)
    amplitudes = _target_amplitudes(targets)
    rng = torch.Generator().manual_seed(seed)
    phase = (torch.rand((1, 1, size, size), generator=rng, dtype=targets.dtype) * (2 * math.pi) - math.pi)
    phase = phase.to(targets)
    field = torch.exp(1j * phase)
    operators = tuple(UnitaryFresnel(field, z) for z in CONFIG.z_m)
    _synchronize(phase.device)
    solve_seconds = perf_counter() - started

    warmed_phase = phase
    for _ in range(WARMUP_STEPS):
        warmed_phase = _gs_step(warmed_phase, amplitudes, operators)
    _synchronize(phase.device)

    history = []
    best_phase, best_iteration = phase, 0
    best_score = (-math.inf, -math.inf)
    previous = 0
    with TemporaryDirectory(prefix="aiphotonics-gs-") as scratch:
        for iteration in chain(range(0, iterations, eval_every), (iterations,)):
            if iteration > previous:
                step_started = perf_counter()
                for _ in range(previous, iteration):
                    phase = _gs_step(phase, amplitudes, operators)
                _synchronize(phase.device)
                solve_seconds += perf_counter() - step_started
            previous = iteration
            metrics = export_result(targets, phase, scratch)
            score = (metrics["mean"]["ssim"], metrics["worst_plane"]["ssim"])
            history.append({"iteration": iteration, "mean_ssim": score[0], "min_ssim": score[1],
                            "mean_pearson": metrics["mean"]["pearson"]})
            if score > best_score:
                best_phase, best_iteration, best_score = phase, iteration, score
    return GlobalGSResult(best_phase, history, best_iteration, solve_seconds, perf_counter() - started)


def run_global_gs(
    targets: Tensor,
    out: str | Path,
    *,
    iterations: int = 100,
    seed: int = 42,
    eval_every: int = 10,
    device: str | torch.device | None = None,
    metadata: dict | None = None,
) -> dict:
    """Save the selected mask, its reconstructed PNGs and the checkpoint history."""
    started = perf_counter()
    out = Path(out)
    result = global_mp_gs(targets, iterations=iterations, seed=seed, eval_every=eval_every, device=device)
    if targets.ndim == 3:
        targets = targets.unsqueeze(0)
    metrics = export_result(targets, result.phase, out, metadata={
        **(metadata or {}),
        "method": "global_mp_gs",
        "seed": seed,
        "iterations": iterations,
        "eval_every": eval_every,
        "best_iteration": result.best_iteration,
        "phase_epsilon": PHASE_EPS,
        "target_norm_epsilon": TARGET_NORM_EPS,
        "protocol": CONFIG.snapshot(),
        "dx_out_m": [output_pitch_m(z, CONFIG.image_size) for z in CONFIG.z_m],
        "device": str(result.phase.device),
        "threads": torch.get_num_threads(),
        "timing": {"solve_seconds": result.solve_seconds, "warmup_steps": WARMUP_STEPS},
    })
    write_json(out / "history.json", {"best_iteration": result.best_iteration, "history": result.history})
    metrics["metadata"]["timing"]["total_seconds"] = perf_counter() - started
    write_json(out / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs=4, required=True, help="four grayscale PNGs in distance order")
    parser.add_argument("--out", "--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        targets = load_four_png(args.images)
        inputs = [{"path": path, "sha256": file_hash(path)} for path in args.images]
        metrics = run_global_gs(targets, args.out, iterations=args.iterations, eval_every=args.eval_every,
                                seed=args.seed, device=args.device, metadata={"inputs": inputs})
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(metrics["mean"], metrics["worst_plane"], metrics["metadata"]["best_iteration"])


if __name__ == "__main__":
    main()
