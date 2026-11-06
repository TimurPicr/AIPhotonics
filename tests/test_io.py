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
