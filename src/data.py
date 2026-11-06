"""Seeded, independent 2D targets; no external datasets are required."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
import torch

from src.io import SIZE, file_hash, load_four_png, write_json, write_png

CATEGORIES = ('geometry', 'symbols', 'grayscale', 'mixed')
SEEDS = {'val': 42001, 'test': 42002, 'ood': 42003, 'examples': 42004, 'train_diagnostic': 42005}


def _target(rng: np.random.Generator, category: str, ood: bool = False) -> torch.Tensor:
    if category == 'mixed':
        category = str(rng.choice(CATEGORIES[:3]))
    image = Image.new('L', (SIZE, SIZE))
    draw = ImageDraw.Draw(image)
    if category == 'geometry':
        for _ in range(int(rng.integers(2, 9))):
            x, y = rng.integers(5, 95, size=2).tolist()
            w, h = rng.integers(4, 29, size=2).tolist()
            level = int(rng.integers(80, 256))
            if ood:
                draw.polygon([(x, y), (x + w, y + h), (x - 4, y + h)], fill=level)
            else:
                shape = rng.choice(['ellipse', 'rectangle', 'line'])
                getattr(draw, shape)((x, y, x + w, y + h), fill=level,
                                     **({'width': int(rng.integers(1, 5))} if shape == 'line' else {}))
    elif category == 'symbols':
        alphabet = 'XYZ789' if ood else 'ABCDEFGHJKLMNPQRSTUVW0123456'
        text = ''.join(rng.choice(list(alphabet), size=int(rng.integers(1, 4))))
        font = ImageFont.load_default(size=int(rng.integers(24, 42)))
        draw.text((int(rng.integers(2, 25)), int(rng.integers(10, 65))), text,
                  font=font, fill=int(rng.integers(100, 256)))
    elif category == 'grayscale':
        y, x = np.mgrid[:SIZE, :SIZE] / SIZE
        frequency = rng.uniform(12, 20) if ood else rng.uniform(1, 7)
        angle, offset = rng.uniform(0, 2 * np.pi, size=2)
        wave = (1 + np.cos(2 * np.pi * frequency * (x * np.cos(angle) + y * np.sin(angle)) + offset)) / 2
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        blob = np.exp(-((x - cx)**2 + (y - cy)**2) / rng.uniform(0.02, 0.2))
        array = (0.6 * wave + 0.4 * blob) * rng.uniform(0.3, 0.95)
        return torch.from_numpy(np.rint(array * 255).astype(np.float32) / 255)
    else:
        raise ValueError(f'Unknown category: {category}')
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255)


class TrainStream:
    def __init__(self, batch_size: int, seed: int):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.batch_size, self.seed = batch_size, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            yield torch.stack([torch.stack([_target(rng, str(rng.choice(CATEGORIES)))
                                            for _ in range(4)]) for _ in range(self.batch_size)])


def make_train_loader(batch_size: int, seed: int) -> TrainStream:
    return TrainStream(batch_size, seed)


def prepare_split(root: Path, split: str, size: int, *, seed: int | None = None,
                  z_m: list[float] | None = None) -> dict:
    if z_m is None:
        from config import CONFIG
        z_m = list(CONFIG.z_m)
    if len(z_m) != 4 or any(not np.isfinite(z) or z <= 0 for z in z_m):
        raise ValueError("Expected four positive finite distances")

    if size < 1:
        raise ValueError('Split size must be positive')
    seed = SEEDS[split] if seed is None else seed
    rng = np.random.default_rng(seed)
    samples = []
    for index in range(size):
        category = CATEGORIES[index % len(CATEGORIES)]
        sample_id = f'{split}_{index:04d}'
        sources = []
        for plane in range(4):
            source_seed = int(rng.integers(0, 2**63))
            image = _target(np.random.default_rng(source_seed), category, split == 'ood')
            path = root / split / sample_id / f'{plane + 1}.png'
            write_png(image, path)
            sources.append({'id': f'{sample_id}_{plane}', 'path': f'{sample_id}/{plane + 1}.png',
                            'seed': source_seed, 'sha256': file_hash(path)})
        samples.append({'id': sample_id, 'category': category, 'sources': sources})
    manifest = {'split': split, 'seed': seed, 'generator': 'synthetic-v1',
                'size': size, 'z_m': list(z_m), 'samples': samples,
                'versions': {'numpy': np.__version__, 'pillow': pillow_version}}
    write_json(root / split / 'manifest.json', manifest)
    return manifest


def load_manifest(path: str | Path, *, verify: bool = True) -> tuple[dict, list[tuple[dict, torch.Tensor]]]:
    import json
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest['size'] != len(manifest['samples']):
        raise ValueError('Manifest size mismatch')
    samples = []
    for sample in manifest['samples']:
        paths = [path.parent / source['path'] for source in sample['sources']]
        if verify:
            for source, png in zip(sample['sources'], paths):
                if file_hash(png) != source['sha256']:
                    raise ValueError(f'PNG hash mismatch: {png}')
        samples.append((sample, load_four_png([str(p) for p in paths])))
    return manifest, samples


def verify_disjoint(manifests: list[dict]) -> None:
    seen_ids, seen_hashes = set(), set()
    for manifest in manifests:
        ids = {s['id'] for sample in manifest['samples'] for s in sample['sources']}
        hashes = {s['sha256'] for sample in manifest['samples'] for s in sample['sources']}
        if ids & seen_ids or hashes & seen_hashes:
            raise ValueError('Shared sources or identical PNG between splits')
        seen_ids.update(ids)
        seen_hashes.update(hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-eval', action='store_true', required=True)
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--val-size', type=int, default=64)
    parser.add_argument('--test-size', type=int, default=64)
    parser.add_argument('--ood-size', type=int, default=16)
    args = parser.parse_args()
    manifests = [prepare_split(args.root, name, size) for name, size in
                 [('val', args.val_size), ('test', args.test_size), ('ood', args.ood_size),
                  ('train_diagnostic', 4)]]
    manifests.append(prepare_split(Path('.'), 'examples', 8))
    verify_disjoint(manifests)
    print('Prepared:', ', '.join(f"{m['split']}={m['size']}" for m in manifests))


if __name__ == '__main__':
    main()
