"""Generate reachable four-plane targets from saved predicted phase masks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from config import CONFIG
from src.data import verify_disjoint
from src.io import file_hash, load_phase_png, write_json, write_png
from src.reconstruction import normalize_recons, reconstruct_four

PHASE_ROOTS = ('outputs/test', 'outputs/ood', 'outputs/examples', 'outputs/comparison/final/network')


def prepare_forward_dataset(paths: list[Path], root: Path, *, train_size=64, val_size=16,
                            test_size=16, seed=42) -> dict:
    groups, seen_hashes, seen_parents = defaultdict(list), set(), set()
    for path in sorted(paths, key=lambda p: (not (p.parent / 'metrics.json').exists(), str(p))):
        metadata_path = path.parent / 'metrics.json'
        metadata = json.loads(metadata_path.read_text())['metadata'] if metadata_path.exists() else {}
        digest = file_hash(path)
        parent_id = metadata.get('sample_id', path.parent.name)
        if digest in seen_hashes or parent_id in seen_parents:
            continue
        seen_hashes.add(digest)
        seen_parents.add(parent_id)
        groups[metadata.get('category', 'mask_derived')].append((path, digest, parent_id, metadata))
    sizes = {'train': train_size, 'val': val_size, 'test': test_size}
    if not groups or any(size < 1 or size % len(groups) for size in sizes.values()):
        raise ValueError('Split sizes must be positive multiples of the source-category count')
    per_category = {split: size // len(groups) for split, size in sizes.items()}
    required = sum(per_category.values())
    if any(len(group) < required for group in groups.values()):
        raise ValueError('Not enough independent masks in each source category')
    rng = np.random.default_rng(seed)
    selected = {split: [] for split in sizes}
    for category, group in sorted(groups.items()):
        order = rng.permutation(len(group))
        offset = 0
        for split, count in per_category.items():
            selected[split].extend((category, *group[i]) for i in order[offset:offset+count])
            offset += count
    manifests = []
    for split, records in selected.items():
        samples = []
        for index, (category, source, digest, parent_id, metadata) in enumerate(records):
            sample_id = f'forward_{split}_{index:04d}'
            directory = root / split / sample_id
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, directory / 'phase.png')
            with torch.no_grad():
                phase = load_phase_png(str(directory / 'phase.png'))
                values = normalize_recons(reconstruct_four(phase, config=CONFIG), config=CONFIG)
            inputs = []
            for plane in range(4):
                target = directory / f'{plane+1}.png'
                write_png(values[0, plane], target)
                inputs.append({'id': f'{sample_id}_{plane}', 'path': f'{sample_id}/{plane+1}.png',
                               'sha256': file_hash(target)})
            samples.append({'id': sample_id, 'category': category, 'sources': inputs,
                            'phase_path': f'{sample_id}/phase.png', 'phase_sha256': digest,
                            'parent_sample_id': parent_id, 'parent_phase_path': str(source),
                            'parent_phase_sha256': digest,
                            'parent_checkpoint_sha256': metadata.get('checkpoint_sha256')})
        manifest = {'split': split, 'size': len(samples), 'seed': seed, 'generator': 'forward-v1',
                    'category_kind': 'source_target_category', 'z_m': list(CONFIG.z_m),
                    'protocol': CONFIG.snapshot(), 'samples': samples}
        write_json(root / split / 'manifest.json', manifest)
        manifests.append(manifest)
    verify_disjoint(manifests)
    phase_sets = [{sample['phase_sha256'] for sample in m['samples']} for m in manifests]
    assert sum(map(len, phase_sets)) == len(set.union(*phase_sets))
    summary = {'seed': seed, 'protocol': CONFIG.snapshot(), 'unique_parent_masks': len(seen_hashes),
               'splits': sizes, 'phase_families_disjoint': True,
               'manifests': {split: file_hash(root / split / 'manifest.json') for split in sizes}}
    write_json(root / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-roots', nargs='+', default=PHASE_ROOTS)
    parser.add_argument('--root', type=Path, default=Path('data/forward'))
    parser.add_argument('--train-size', type=int, default=64)
    parser.add_argument('--val-size', type=int, default=16)
    parser.add_argument('--test-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = [path for root in args.phase_roots for path in Path(root).glob('*/phase.png')]
    print(prepare_forward_dataset(paths, args.root, train_size=args.train_size,
                                  val_size=args.val_size, test_size=args.test_size, seed=args.seed))


if __name__ == '__main__':
    main()
