"""Generate one 8-bit grayscale phase DOE from each quartet of target images."""

import argparse
import math
from pathlib import Path

import torch

from config import CONFIG
from src.io import export_phase_png, load_four_images
from src.model import HologramUNet
from src.refinement import refine_phase


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "weights" / "h100.pt"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_checkpoint(path: Path) -> tuple[HologramUNet, dict]:
    """Restore the supplied trained model without importing training utilities."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1 or checkpoint.get("protocol") != CONFIG.snapshot():
        raise ValueError("Unsupported checkpoint version or optical protocol")
    model = HologramUNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(memory_format=torch.channels_last).eval(), checkpoint


def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                  and not path.name.startswith("."))


def discover_quartets(directory: Path) -> list[tuple[str, list[Path]]]:
    """Accept a single quartet directory or immediate subdirectories of quartets."""
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    images = _image_files(directory)
    if images:
        candidates = [(directory.name, images)]
    else:
        candidates = [(child.name, _image_files(child)) for child in sorted(directory.iterdir())
                      if child.is_dir() and not child.name.startswith(".")]
    if not candidates:
        raise ValueError(f"No image quartets found in {directory}")
    for name, paths in candidates:
        if len(paths) != 4:
            raise ValueError(f"{name}: expected exactly four images, found {len(paths)}")
    return candidates


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path, help="A quartet directory or a root of quartet directories")
    inputs.add_argument("--images", type=Path, nargs=4, metavar="IMAGE", help="Four images in optical distance order")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path, help="Output PNG path for a single quartet")
    outputs.add_argument("--output-dir", type=Path, help="Write <quartet_name>.png for each quartet")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--refine-steps", type=int, default=10, help="Adam phase refinement steps; 0 disables (default: 10)")
    parser.add_argument("--refine-lr", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads (default: 2)")
    args = parser.parse_args(argv)
    if args.refine_steps < 0 or args.threads < 1:
        parser.error("--refine-steps must be nonnegative and --threads must be positive")
    if not math.isfinite(args.refine_lr) or args.refine_lr <= 0:
        parser.error("--refine-lr must be finite and positive")
    if args.output and args.output.suffix.lower() != ".png":
        parser.error("--output must have the .png extension")

    try:
        quartets = [("phase", args.images)] if args.images else discover_quartets(args.input_dir)
        if args.output and len(quartets) != 1:
            raise ValueError("Use --output-dir when processing multiple quartets")
        destinations = [args.output or args.output_dir / f"{name}.png" for name, _ in quartets]
        source_paths = {path.resolve() for _, paths in quartets for path in paths}
        if any(path.resolve() in source_paths for path in destinations):
            raise ValueError("Output files must not overwrite input images")
        # Validate every quartet before generating any output files.
        targets = [load_four_images(paths) for _, paths in quartets]
        torch.set_num_threads(args.threads)
        model, checkpoint = load_checkpoint(args.checkpoint)
        for images, destination in zip(targets, destinations):
            with torch.inference_mode():
                phase, _ = model(images.contiguous(memory_format=torch.channels_last))
            phase = refine_phase(phase, images, steps=args.refine_steps, lr=args.refine_lr,
                                 **checkpoint.get("loss_config", {"alpha": 0.7, "beta": 0.3}))
            export_phase_png(phase, destination)
            print(destination, flush=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
