import math
import numpy as np
from PIL import Image
import pytest
import torch

from src.io import export_phase_png, load_phase_png, load_four_png, write_png, write_json


def test_phase_roundtrip(tmp_path):
    phi = torch.linspace(-math.pi, math.pi, 128 * 128).reshape(1, 1, 128, 128)
    path = str(tmp_path / 'phase.png')
    export_phase_png(phi, path)
    restored = load_phase_png(path)
    assert (restored - phi).abs().max() <= math.pi / 255 + 1e-6
    with Image.open(path) as image:
        assert image.mode == 'L' and image.size == (128, 128)
        assert np.asarray(image).min() == 0 and np.asarray(image).max() == 255


def test_order_and_validation(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path / f'{i}.png'
        write_png(torch.full((128, 128), i / 3), path)
        paths.append(str(path))
    loaded = load_four_png(paths)
    assert loaded.shape == (1, 4, 128, 128)
    assert torch.allclose(loaded[0, :, 0, 0], torch.arange(4) / 3)
    with pytest.raises(ValueError):
        load_four_png(paths[:3])
    Image.new('RGB', (128, 128)).save(paths[0])
    with pytest.raises(ValueError, match='mode L'):
        load_four_png(paths)
    Image.new('L', (64, 64)).save(paths[0])
    with pytest.raises(ValueError):
        load_four_png(paths)


def test_finite_validation(tmp_path):
    with pytest.raises(ValueError):
        export_phase_png(torch.full((1, 1, 128, 128), float('nan')), str(tmp_path / 'bad.png'))
    with pytest.raises(ValueError):
        write_json(tmp_path / 'bad.json', {'x': float('nan')})


def test_manifests_reproducible_disjoint_and_verified(tmp_path):
    from src.data import prepare_split, load_manifest, verify_disjoint
    distances = [0.0241, 0.0253, 0.0265, 0.0277]
    val = prepare_split(tmp_path, 'val', 4, z_m=distances)
    assert val == prepare_split(tmp_path, 'val', 4, z_m=distances)
    test = prepare_split(tmp_path, 'test', 4, z_m=distances)
    verify_disjoint([val, test])
    with pytest.raises(ValueError, match='Shared sources'):
        verify_disjoint([val, val])
    manifest, samples = load_manifest(tmp_path / 'val/manifest.json')
    assert manifest['z_m'] == distances and len(samples) == 4
    assert {s['category'] for s, _ in samples} == {'geometry', 'symbols', 'grayscale', 'mixed'}
    path = tmp_path / 'val' / val['samples'][0]['sources'][0]['path']
    write_png(torch.zeros(128, 128), path)
    with pytest.raises(ValueError, match='hash mismatch'):
        load_manifest(tmp_path / 'val/manifest.json')
