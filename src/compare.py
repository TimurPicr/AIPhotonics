"""Compare the networks and phase baselines on eight validation sets."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

import torch

from config import CONFIG
from src.baseline import run_baseline
from src.data import CATEGORIES, load_manifest
from src.evaluate import aggregate_results
from src.global_gs import run_global_gs
from src.infer import infer_sample, load_predictor
from src.io import file_hash, write_json


def compare(
    manifest_path: Path,
    initial_path: Path,
    best_path: Path,
    out: Path,
    *,
    device: str,
    steps: int,
    seed: int,
    gs_iterations: int | None = None,
) -> dict:
    manifest, samples = load_manifest(manifest_path)
    if manifest["split"] != "val" or manifest["z_m"] != list(CONFIG.z_m):
        raise ValueError("comparison requires validation data in the configured plane order")
    if Counter(sample["category"] for sample, _ in samples) != dict.fromkeys(CATEGORIES, 2):
        raise ValueError("comparison needs two validation samples per category")
    initial, initial_meta = load_predictor(initial_path, device)
    best, best_meta = load_predictor(best_path, device)
    if initial_meta["checkpoint_step"] != 0 or best_meta["checkpoint_step"] <= 0:
        raise ValueError("comparison requires a step-zero initial checkpoint and a trained checkpoint")
    architecture = {"base_channels": best.base_channels, "interleave_factor": best.interleave_factor}
    if any(getattr(initial, key) != value for key, value in architecture.items()):
        raise ValueError("the initial and trained network architectures must match")

    selected_device = next(best.parameters()).device
    manifest_hash = file_hash(manifest_path)
    methods = {"initial_network": [], "network": [], "adam_phase": []}
    if gs_iterations is not None:
        methods["global_mp_gs"] = []
    for sample, targets in samples:
        sample_id = sample["id"]
        metadata = {"sample_id": sample_id, "split": "val", "category": sample["category"],
                    "manifest_sha256": manifest_hash, "threads": torch.get_num_threads(),
                    "git_revision": best_meta["git_revision"]}
        for name, model, model_meta in (("initial_network", initial, initial_meta), ("network", best, best_meta)):
            result = infer_sample(model, targets, out / name / sample_id, metadata={**model_meta, **metadata})
            methods[name].append(result)
        methods["adam_phase"].append(run_baseline(targets.to(selected_device), out / "adam_phase" / sample_id,
                                                  steps=steps, seed=seed, metadata=metadata))
        if gs_iterations is not None:
            inputs = [{"path": str(manifest_path.parent / source["path"]), "sha256": source["sha256"]}
                      for source in sample["sources"]]
            methods["global_mp_gs"].append(run_global_gs(
                targets, out / "global_mp_gs" / sample_id, iterations=gs_iterations, seed=seed,
                device=selected_device, metadata={**metadata, "inputs": inputs},
            ))

    summaries = {}
    timing_keys = {"adam_phase": ("optimization_seconds", "pipeline_seconds"),
                   "global_mp_gs": ("solve_seconds", "total_seconds")}
    for name, results in methods.items():
        timings = [result["metadata"]["timing"] for result in results]
        summaries[name] = {
            **aggregate_results(results),
            "median_seconds": {key: median(timing[key] for timing in timings)
                               for key in timing_keys.get(name, ("forward_seconds", "pipeline_seconds"))},
        }
        if gs_iterations is not None:
            summaries[name]["min_ssim"] = min(result["worst_plane"]["ssim"] for result in results)

    summary = {
        "split": "val",
        "manifest_sha256": manifest_hash,
        "samples": [{"id": sample["id"], "category": sample["category"]} for sample, _ in samples],
        "device": str(selected_device),
        "threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
        "protocol": CONFIG.snapshot(),
        "architecture": architecture,
        "checkpoints": {name: {key: metadata[key] for key in ("checkpoint_step", "checkpoint_sha256")}
                        for name, metadata in (("initial_network", initial_meta), ("network", best_meta))},
        "adam": {"seed": seed, "steps": steps, "starts": 1, "learning_rate": 0.1},
        "warmup": {"network_forward_passes": 2, "adam_steps": 2},
        "methods": summaries,
    }
    if gs_iterations is not None:
        gs_metadata = methods["global_mp_gs"][0]["metadata"]
        summary["global_mp_gs"] = {key: gs_metadata[key] for key in ("seed", "iterations", "eval_every")}
        summary["global_mp_gs"]["best_iterations"] = {
            result["metadata"]["sample_id"]: result["metadata"]["best_iteration"]
            for result in methods["global_mp_gs"]
        }
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/val/comparison_manifest.json"))
    parser.add_argument("--initial", type=Path, default=Path("checkpoints/initial.pt"))
    parser.add_argument("--best", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gs-iterations", type=int, help="include Global MP-GS with this many iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")
    if args.gs_iterations is not None and args.gs_iterations < 1:
        parser.error("gs-iterations must be positive")
    torch.set_num_threads(args.threads)
    summary = compare(args.manifest, args.initial, args.best, args.out,
                      device=args.device, steps=args.steps, seed=args.seed, gs_iterations=args.gs_iterations)
    for name, result in summary["methods"].items():
        print(name, result["mean"], result["median_seconds"])


if __name__ == "__main__":
    main()
