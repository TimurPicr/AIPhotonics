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


def test_metrics_known_pairs():
    from src.evaluate import evaluate_four, aggregate_results
    target = torch.linspace(0, 1, 128 * 128).reshape(1, 1, 128, 128).repeat(1, 4, 1, 1)
    identical = evaluate_four(target, target)
    assert identical['mean']['pearson'] == pytest.approx(1)
    assert identical['mean']['ssim'] == pytest.approx(1)
    reversed_ = evaluate_four(target, 1 - target)
    assert reversed_['mean']['pearson'] == pytest.approx(-1)
    failure = evaluate_four(target, torch.zeros_like(target))
    assert failure['valid_pearson_pairs'] == 0 and failure['failure_count'] == 4
    diagnostic = evaluate_four(torch.zeros_like(target), torch.zeros_like(target))
    assert diagnostic['mean']['pearson'] is None and diagnostic['failure_count'] == 0
    aggregate = aggregate_results([identical, failure])
    assert aggregate['total_pairs'] == 8 and aggregate['failure_count'] == 4
    assert aggregate['valid_pearson_pairs'] == 4


def test_train_stream():
    from src.data import make_train_loader
    a, b = iter(make_train_loader(2, 42)), iter(make_train_loader(2, 42))
    first = next(a)
    assert first.shape == (2, 4, 128, 128) and first.dtype == torch.float32
    assert torch.equal(first, next(b))
    second = next(a)
    assert torch.equal(second, next(b)) and not torch.equal(first, second)
    assert not torch.equal(first[:, 0], first[:, 1])
    assert 0 <= first.min() <= first.max() <= 1


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


def test_export_uses_quantized_phase_and_saved_recons(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from src.evaluate import evaluate_four
    from src.io import export_result
    seen = []

    def interface_only_reconstruction(phase):
        seen.append(phase.clone())
        return phase.sigmoid().repeat(1, 4, 1, 1)

    monkeypatch.setitem(sys.modules, 'src.reconstruction', SimpleNamespace(
        reconstruct_four=interface_only_reconstruction, normalize_recons=lambda x: x))
    phase = torch.linspace(-math.pi, math.pi, 128 * 128).reshape(1, 1, 128, 128)
    target = torch.rand(1, 4, 128, 128)
    result = export_result(target, phase, tmp_path, metadata={'method': 'interface_test_only'})
    assert torch.equal(seen[0], load_phase_png(str(tmp_path / 'phase.png')))
    assert not torch.equal(seen[0], phase)
    recon = load_four_png([str(tmp_path / f'recon_{i}.png') for i in range(1, 5)])
    assert result['planes'] == evaluate_four(target, recon)['planes']
    assert json.loads((tmp_path / 'metrics.json').read_text()) == result
