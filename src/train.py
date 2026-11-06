"""Train PhaseUNet through the shared Fresnel reconstruction (DeepCGH-style)."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch import Tensor

from src.losses import physics_loss
from src.model import PhaseUNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_protocol() -> dict:
    """A's canonical semantic protocol; source-only edits do not change optics."""
    from config import DEFAULT_CONFIG

    return DEFAULT_CONFIG.snapshot()


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        candidates = (["cuda"] if torch.cuda.is_available() else [])
        candidates += ["mps"] if torch.backends.mps.is_available() else []
        for name in candidates + ["cpu"]:
            try:
                return select_device(name)
            except ValueError:
                continue
        raise ValueError("no device can run complex FFT/backward")
    device = torch.device(requested)
    try:
        phase = torch.randn(1, 1, 16, 16, device=device, requires_grad=True)
        field = torch.fft.fft2(torch.exp(1j * phase))
        (field.abs() * torch.linspace(0, 1, 16, device=device)).mean().backward()
        if not torch.isfinite(phase.grad).all():
            raise RuntimeError("non-finite complex FFT gradient")
    except (RuntimeError, NotImplementedError, AssertionError) as error:
        raise ValueError(f"{device} cannot run complex FFT/backward; use --device cpu") from error
    return device


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    model: PhaseUNet,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    protocol: dict | None = None,
    training_config: dict | None = None,
    validation: dict | None = None,
    samples_seen: int = 0,
) -> None:
    """Save state dictionaries and plain metadata, never a pickled model object."""
    payload = {
        "format_version": 1,
        "model_config": {"base_channels": model.base_channels, "interleave_factor": model.interleave_factor},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "samples_seen": samples_seen,
        "protocol": protocol,
        "training_config": training_config or {},
        "torch_version": str(torch.__version__),
        "rng_state": capture_rng_state(),
        "validation": validation,
    }
    _atomic_save(payload, Path(path))


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(checkpoint: dict) -> None:
    state = checkpoint["rng_state"]
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_protocol: dict | None = None,
) -> tuple[PhaseUNet, dict]:
    """C's inference API: ``model, metadata = load_checkpoint(path)``.

    Model is returned in eval mode. Pass the current protocol to reject an
    incompatible trained checkpoint. A step-zero initial checkpoint is untrained.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format_version")
    if expected_protocol is not None and payload.get("protocol") != expected_protocol:
        raise ValueError("checkpoint optical/reconstruction protocol does not match")
    model_config = payload.get("model_config", {})
    if "base_channels" not in model_config or set(model_config) - {"base_channels", "interleave_factor"}:
        raise ValueError("checkpoint model configuration is unsupported")
    model = PhaseUNet(**model_config)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("checkpoint weights do not match the phase model") from error
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("checkpoint contains non-finite model parameters")
    return model.to(device).eval(), payload


def validation_score(result: dict) -> tuple[float, float] | None:
    """Prefer the worst plane's mean SSIM; use mean Pearson as a tie-breaker."""
    aggregate = result["aggregate"]
    ssim = aggregate["worst_plane_mean"]["ssim"]
    pearson = aggregate["mean"]["pearson"]
    if (
        aggregate["failure_count"]
        or aggregate["valid_pearson_pairs"] != aggregate["total_pairs"]
        or ssim is None or pearson is None
        or not math.isfinite(ssim) or not math.isfinite(pearson)
    ):
        return None
    return float(ssim), float(pearson)


def evaluate_validation(
    model: PhaseUNet,
    samples: list[tuple[dict, Tensor]],
    *,
    export_fn: Callable | None = None,
) -> dict:
    """Use C's real PNG roundtrip; reuse and remove one temporary export folder."""
    from src.evaluate import aggregate_results
    if export_fn is None:
        from src.io import export_result

        export_fn = export_result
    if not samples:
        raise ValueError("validation samples must not be empty")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    results, categories = [], {}
    try:
        with torch.no_grad(), tempfile.TemporaryDirectory(prefix="photonics-b-validation-") as temporary:
            for sample, targets in samples:
                phase = model(targets.to(device))
                result = export_fn(
                    targets, phase, Path(temporary),
                    metadata={"split": "val", "sample_id": sample["id"]},
                )
                results.append(result)
                categories.setdefault(sample["category"], []).append(result)
    finally:
        model.train(was_training)
    return {
        "aggregate": aggregate_results(results),
        "categories": {name: aggregate_results(group) for name, group in categories.items()},
    }


def train_step(
    model: PhaseUNet,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    objective: Callable[[Tensor, Tensor], Tensor] = physics_loss,
) -> dict[str, float]:
    """One update; injected objectives are for explicit interface tests only."""
    if not targets.is_floating_point():
        raise TypeError("training targets must be floats in [0, 1]")
    if not torch.isfinite(targets).all():
        raise FloatingPointError("training targets contain non-finite values")
    if (targets < 0).any() or (targets > 1).any():
        raise ValueError("training targets must lie in [0, 1]")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = objective(model(targets), targets)
    if loss.ndim != 0 or loss.is_complex() or not torch.isfinite(loss):
        raise FloatingPointError("training loss must be a finite real scalar")
    if not loss.requires_grad:
        raise RuntimeError("loss is disconnected from the model")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("model gradients are missing or non-finite")
    gradient_norm = math.sqrt(sum(float(g.detach().square().sum()) for g in gradients))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("optimizer produced non-finite parameters")
    return {"loss": float(loss.detach()), "gradient_norm": gradient_norm}


def fit(
    model: PhaseUNet,
    batches: Iterable[Tensor],
    *,
    steps: int,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    objective: Callable[[Tensor, Tensor], Tensor] = physics_loss,
    checkpoint_dir: str | Path | None = None,
    protocol: dict | None = None,
    training_config: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    start_samples: int = 0,
    on_step: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run additional steps, saving the last finite state on normal/error exit."""
    if steps < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps must be nonnegative and learning_rate positive/finite")
    model.to(device)
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=learning_rate)
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if directory is not None and start_step == 0:
        save_checkpoint(directory / "initial.pt", model, protocol=protocol, training_config=training_config)
    history = []
    completed = start_step
    samples_seen = start_samples
    iterator = iter(batches)
    try:
        for step in range(start_step + 1, start_step + steps + 1):
            try:
                targets = next(iterator)
            except StopIteration as error:
                raise ValueError("training data ended before the requested steps") from error
            started = time.perf_counter()
            result = train_step(model, targets.to(device), optimizer, objective=objective)
            completed = step
            samples_seen += targets.shape[0]
            record = {
                "step": step, "samples_seen": samples_seen, "batch_size": targets.shape[0],
                **result, "step_seconds": time.perf_counter() - started,
            }
            history.append(record)
            if on_step is not None:
                on_step(record)
    finally:
        if directory is not None and all(torch.isfinite(p).all() for p in model.parameters()):
            save_checkpoint(
                directory / "last.pt", model, optimizer=optimizer, step=completed,
                protocol=protocol, training_config=training_config,
                samples_seen=samples_seen,
            )
    return history


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=_positive_int, default=500)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--base-channels", type=_positive_int, default=16)
    parser.add_argument("--interleave-factor", type=int, choices=[1, 2, 4, 8, 16], default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=_positive_int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--overfit-samples", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--resume", type=Path, help="resume optimizer/RNG; --steps is additional updates")
    parser.add_argument("--train-manifest", type=Path, help="fixed training targets instead of the synthetic stream")
    parser.add_argument("--val-manifest", type=Path, default=Path("data/val/manifest.json"))
    parser.add_argument("--val-every", type=_positive_int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    if not math.isfinite(args.ssim_weight) or args.ssim_weight < 0:
        parser.error("--ssim-weight must be finite and nonnegative")
    try:
        from config import DEFAULT_CONFIG
        from src.data import load_manifest, make_manifest_loader, make_train_loader
        from src.io import file_hash
        from src.reconstruction import normalize_recons, reconstruct_four
    except ImportError as error:
        parser.error(f"A/C dependencies are not ready: {error}; no stub training is allowed")
    torch.set_num_threads(args.threads)
    try:
        device = select_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    seed_everything(args.seed)
    protocol = current_protocol()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["device"] = str(device)
    settings["train_manifest_sha256"] = file_hash(args.train_manifest) if args.train_manifest else None
    if args.train_manifest is not None and args.overfit_samples is not None:
        parser.error("--train-manifest and --overfit-samples cannot be combined")
    from importlib.metadata import version
    settings["versions"] = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "pillow": version("Pillow"), "scikit_image": version("scikit-image"),
    }
    settings["git_revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or None
    model = PhaseUNet(args.base_channels, args.interleave_factor)
    optimizer = None
    start_step = 0
    start_samples = 0
    resumed = None
    if args.resume is not None:
        model, resumed = load_checkpoint(args.resume, device=device, expected_protocol=protocol)
        for name in ["base_channels", "interleave_factor", "seed", "batch_size", "overfit_samples"]:
            previous = resumed["training_config"].get(name, 1 if name == "interleave_factor" else None)
            if previous != settings[name]:
                parser.error(f"resume requires the same {name}: saved={previous}, requested={settings[name]}")
        if resumed["training_config"].get("train_manifest_sha256") != settings["train_manifest_sha256"]:
            parser.error("resume requires the same training manifest")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if resumed.get("optimizer_state_dict") is None:
            parser.error("checkpoint has no optimizer state; resume from last.pt")
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_step = resumed["step"]
        start_samples = resumed.get("samples_seen", start_step * (args.overfit_samples or args.batch_size))
    batches = (make_manifest_loader(args.train_manifest, args.batch_size, args.seed) if args.train_manifest
               else make_train_loader(args.overfit_samples or args.batch_size, args.seed))
    if args.overfit_samples:
        batches = itertools.repeat(next(iter(batches)))
    else:
        batches = iter(batches)
        for _ in range(start_step):
            next(batches)
    validation_samples = None
    if args.overfit_samples is None:
        manifest, validation_samples = load_manifest(args.val_manifest)
        if manifest["split"] != "val":
            parser.error("checkpoint selection requires a validation manifest, never test/OOD")
        if list(DEFAULT_CONFIG.distances_m) != manifest["z_m"]:
            parser.error("validation distances do not match the current optical protocol")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(json.dumps({**settings, "protocol": protocol}, indent=2) + "\n")

    def objective(phase: Tensor, targets: Tensor) -> Tensor:
        return physics_loss(
            phase, targets, reconstruct_fn=reconstruct_four, normalize_fn=normalize_recons,
            ssim_weight=args.ssim_weight,
        )

    with (args.run_dir / "metrics.jsonl").open("a" if resumed is not None else "w") as log:
        best_score = None
        best_path = args.checkpoint_dir / "best.pt"
        if resumed is not None and best_path.exists():
            _, previous = load_checkpoint(best_path, expected_protocol=protocol)
            if previous.get("validation") is not None:
                best_score = validation_score(previous["validation"])
        if resumed is not None:
            restore_rng_state(resumed)

        def report_validation(step: int, *, select: bool, samples_seen: int) -> None:
            nonlocal best_score
            result = evaluate_validation(model, validation_samples)
            score = validation_score(result)
            row = {"kind": "validation", "step": step, "selection_score": score, **result}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps({"kind": "validation", "step": step, "score": score}), flush=True)
            if select and score is not None and (best_score is None or score > best_score):
                best_score = score
                save_checkpoint(
                    best_path, model, step=step, protocol=protocol,
                    training_config=settings, validation=result,
                    samples_seen=samples_seen,
                )

        def report(record: dict) -> None:
            log.write(json.dumps({"kind": "train", **record}, allow_nan=False) + "\n")
            log.flush()
            if record["step"] == 1 or record["step"] % 10 == 0:
                print(json.dumps(record, allow_nan=False), flush=True)
            if validation_samples is not None and (
                record["step"] % args.val_every == 0 or record["step"] == start_step + args.steps
            ):
                report_validation(record["step"], select=True, samples_seen=record["samples_seen"])

        try:
            if validation_samples is not None:
                report_validation(start_step, select=False, samples_seen=start_samples)
            fit(
                model, batches, steps=args.steps, learning_rate=args.lr, device=device,
                objective=objective, checkpoint_dir=args.checkpoint_dir,
                protocol=protocol, training_config=settings, on_step=report,
                optimizer=optimizer, start_step=start_step,
                start_samples=start_samples,
            )
        except KeyboardInterrupt:
            print("Interrupted: last finite checkpoint saved.", flush=True)
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
