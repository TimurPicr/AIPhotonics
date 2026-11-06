import json
import math

import pytest
import torch

from src.data import load_manifest, make_manifest_loader
from src.forward_data import prepare_forward_dataset
from src.io import export_phase_png, export_result, file_hash, load_phase_png, write_json


def source_masks(tmp_path):
    rng = torch.Generator().manual_seed(7)
    paths = []
    for index in range(8):
        directory = tmp_path / 'sources' / f'source_{index}'
        phase = torch.rand((1,1,128,128), generator=rng) * (2*math.pi) - math.pi
        path = directory / 'phase.png'
        export_phase_png(phase, str(path))
        write_json(directory / 'metrics.json', {'metadata': {
            'sample_id': directory.name, 'category': 'geometry' if index % 2 else 'symbols',
            'method': 'network', 'checkpoint_sha256': 'fixture'}})
        paths.append(path)
    return paths


def test_forward_targets_are_reachable_and_source_families_disjoint(tmp_path):
    paths = source_masks(tmp_path)
    duplicate = tmp_path / 'duplicate' / 'phase.png'
    duplicate.parent.mkdir()
    duplicate.write_bytes(paths[0].read_bytes())
    root = tmp_path / 'forward'
    summary = prepare_forward_dataset(paths + [duplicate], root, train_size=4, val_size=2, test_size=2)
    assert summary['unique_parent_masks'] == 8
    hashes, parents = set(), set()
    for split, size in [('train',4), ('val',2), ('test',2)]:
        manifest, samples = load_manifest(root / split / 'manifest.json')
        assert len(samples) == size
        for sample, targets in samples:
            phase_path = root / split / sample['phase_path']
            assert file_hash(phase_path) == sample['phase_sha256']
            assert sample['phase_sha256'] not in hashes
            assert sample['parent_sample_id'] not in parents
            hashes.add(sample['phase_sha256'])
            parents.add(sample['parent_sample_id'])
            result = export_result(targets, load_phase_png(str(phase_path)), tmp_path / 'oracle')
            assert result['mean']['pearson'] == pytest.approx(1)
            assert result['mean']['ssim'] == pytest.approx(1)
    assert prepare_forward_dataset(paths, root, train_size=4, val_size=2, test_size=2) == summary


def test_manifest_stream_is_seeded_covers_each_epoch_and_rejects_validation(tmp_path):
    root = tmp_path / 'forward'
    prepare_forward_dataset(source_masks(tmp_path), root, train_size=4, val_size=2, test_size=2)
    path = root / 'train/manifest.json'
    a, b = make_manifest_loader(path, 2, 42), make_manifest_loader(path, 2, 42)
    first, second = next(a), next(a)
    assert torch.equal(first, next(b)) and torch.equal(second, next(b))
    assert first.shape == second.shape == (2,4,128,128)
    _, samples = load_manifest(path)
    expected = sorted(target.numpy().tobytes() for _,target in samples)
    actual = sorted(target.unsqueeze(0).numpy().tobytes() for batch in [first,second] for target in batch)
    assert actual == expected
    with pytest.raises(ValueError, match='train manifest'):
        make_manifest_loader(root / 'val/manifest.json', 2, 42)
    manifest = json.loads(path.read_text())
    manifest['protocol']['reconstruction_value'] = 'intensity'
    write_json(path, manifest)
    with pytest.raises(ValueError, match='protocol'):
        make_manifest_loader(path, 2, 42)
