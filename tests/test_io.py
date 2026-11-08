import math
import numpy as np
from PIL import Image
import pytest
import torch

from src.io import export_phase_png, load_phase_png, load_four_png, load_four_images, write_png


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


def test_color_targets_are_converted_without_resizing(tmp_path):
    paths = [tmp_path / f'{index}.png' for index in range(4)]
    for index, path in enumerate(paths):
        Image.new('RGB', (128, 128), (index * 80,) * 3).save(path)
    loaded = load_four_images(paths)
    assert loaded.shape == (1, 4, 128, 128)
    torch.testing.assert_close(loaded[0, :, 0, 0], torch.arange(4) * 80 / 255)
    Image.new('L', (64, 128)).save(paths[0])
    with pytest.raises(ValueError, match='128x128'):
        load_four_images(paths)
