from pathlib import Path

import pytest
import torch

from src.evaluate import evaluate_four
from src.infer import infer_sample, load_predictor
from src.io import file_hash, load_four_png, load_phase_png
from src.reconstruction import normalize_recons, reconstruct_four

ROOT = Path(__file__).resolve().parents[1]


def test_initial_checkpoint_real_pipeline_is_repeatable(tmp_path):
    model, metadata = load_predictor(ROOT / 'checkpoints/initial.pt')
    assert metadata['method'] == 'initial_network' and len(metadata['dx_out_m']) == 4
    targets = load_four_png([str(ROOT / f'examples/examples_0000/{i}.png') for i in range(1, 5)])
    first = infer_sample(model, targets, tmp_path / 'first', metadata=metadata)
    second = infer_sample(model, targets, tmp_path / 'second', metadata=metadata)
    assert first['planes'] == second['planes']
    for name in ['phase.png', *(f'recon_{i}.png' for i in range(1, 5))]:
        assert file_hash(tmp_path / 'first' / name) == file_hash(tmp_path / 'second' / name)
    phase = load_phase_png(str(tmp_path / 'first/phase.png'))
    with torch.no_grad():
        expected = (normalize_recons(reconstruct_four(phase)) * 255).round() / 255
    saved = load_four_png([str(tmp_path / f'first/recon_{i}.png') for i in range(1, 5)])
    torch.testing.assert_close(saved, expected, rtol=0, atol=0)
    assert first['planes'] == evaluate_four(targets, saved)['planes']
    assert first['metadata']['timing']['forward_seconds'] > 0


def test_inference_rejects_incompatible_checkpoint(tmp_path):
    payload = torch.load(ROOT / 'checkpoints/initial.pt', map_location='cpu', weights_only=True)
    payload['protocol']['reconstruction_value'] = 'intensity'
    torch.save(payload, tmp_path / 'incompatible.pt')
    with pytest.raises(ValueError, match='protocol'):
        load_predictor(tmp_path / 'incompatible.pt')
