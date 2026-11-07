"""Strict grayscale PNG and phase mask IO."""
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
    Image.fromarray(_png_codes(array)).save(path)


def _png_codes(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)


def _phase_values(phase: np.ndarray) -> np.ndarray:
    return (np.clip(phase, -math.pi, math.pi) + math.pi) / (2 * math.pi)


def export_phase_png(phi: torch.Tensor, path: str) -> None:
    array = _array(phi, (1, 1, SIZE, SIZE))[0, 0]
    write_png(torch.from_numpy(_phase_values(array)), path)


def load_phase_png(path: str) -> torch.Tensor:
    return (read_png(path) * (2 * math.pi) - math.pi)[None, None]


def quantize_png(values: torch.Tensor, *, keep_device: bool = False) -> torch.Tensor:
    """Apply the same 8-bit conversion as a PNG round trip, in memory."""
    if keep_device:
        codes = (values.detach().clamp(0, 1) * 255).round().to(torch.uint8)
        return codes.to(torch.float32) / 255
    codes = _png_codes(values.detach().cpu().numpy())
    return torch.from_numpy(codes.astype(np.float32) / 255)


def quantize_phase(phase: torch.Tensor, *, keep_device: bool = False) -> torch.Tensor:
    if keep_device:
        if phase.ndim != 4 or phase.shape[0] < 1 or phase.shape[1:] != (1, SIZE, SIZE) or not torch.isfinite(phase).all():
            raise ValueError('Expected finite phase masks of shape [B, 1, 128, 128]')
        values = (phase.detach().clamp(-math.pi, math.pi) + math.pi) / (2 * math.pi)
        return quantize_png(values, keep_device=True) * (2 * math.pi) - math.pi
    values = torch.from_numpy(_phase_values(_array(phase, (1, 1, SIZE, SIZE))))
    return quantize_png(values) * (2 * math.pi) - math.pi


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
