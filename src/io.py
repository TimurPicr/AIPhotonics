"""Strict grayscale PNG IO and the shared quantized export path."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from PIL import Image
import torch

SIZE = 128


def read_png(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'L' or image.size != (SIZE, SIZE):
            raise ValueError(f'{path}: expected 128x128 PNG in mode L')
        return torch.from_numpy(np.array(image, dtype=np.float32) / 255)


def load_four_png(paths: list[str]) -> torch.Tensor:
    if len(paths) != 4:
        raise ValueError('Exactly four PNG paths are required, in distance order')
    return torch.stack([read_png(path) for path in paths]).unsqueeze(0)


def _array(value: torch.Tensor, shape: tuple) -> np.ndarray:
    if tuple(value.shape) != shape or not torch.isfinite(value).all():
        raise ValueError(f'Expected finite tensor of shape {shape}')
    return value.detach().cpu().numpy()


def write_png(value: torch.Tensor, path: str | Path) -> None:
    array = _array(value, (SIZE, SIZE))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def export_phase_png(phi: torch.Tensor, path: str) -> None:
    array = _array(phi, (1, 1, SIZE, SIZE))[0, 0]
    write_png(torch.from_numpy((np.clip(array, -math.pi, math.pi) + math.pi) / (2 * math.pi)), path)


def load_phase_png(path: str) -> torch.Tensor:
    return (read_png(path) * (2 * math.pi) - math.pi)[None, None]


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export_result(targets: torch.Tensor, phase: torch.Tensor, out: str | Path,
                  *, metadata: dict | None = None) -> dict:
    from src.reconstruction import reconstruct_four, normalize_recons
    from src.evaluate import evaluate_four

    _array(targets, (1, 4, SIZE, SIZE))
    out = Path(out)
    export_phase_png(phase, str(out / 'phase.png'))
    quantized = load_phase_png(str(out / 'phase.png')).to(phase.device)
    with torch.no_grad():
        recons = normalize_recons(reconstruct_four(quantized)).cpu()
    for index in range(4):
        write_png(recons[0, index], out / f'recon_{index + 1}.png')
    saved = load_four_png([str(out / f'recon_{i}.png') for i in range(1, 5)])
    result = {'metadata': {**(metadata or {}), 'phase_png_quantized': True},
              **evaluate_four(targets.cpu(), saved)}
    write_json(out / 'metrics.json', result)
    return result
