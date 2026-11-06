"""Four PNG targets and a compatible checkpoint to a quantized phase and reconstructions."""
import argparse
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import PIL
import torch

from src.io import export_result, file_hash, load_four_png, write_json
from src.train import load_checkpoint, select_device


def protocol_snapshot() -> dict:
    import config
    return {'config_sha256': file_hash(config.__file__)}


def load_predictor(checkpoint: str | Path, device: str = 'cpu') -> tuple:
    import config
    selected = select_device(device)
    model, payload = load_checkpoint(checkpoint, device=selected, expected_protocol=protocol_snapshot())
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    metadata = {'checkpoint_sha256': file_hash(checkpoint), 'checkpoint_step': payload['step'],
                'method': 'initial_network' if payload['step'] == 0 else 'network',
                'protocol': payload['protocol'], 'config_source': Path(config.__file__).read_text(),
                'device': str(selected), 'seed': payload['training_config'].get('seed'),
                'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                             'numpy': np.__version__, 'pillow': PIL.__version__},
                'git_revision': revision}
    return model, metadata


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def infer_sample(model, targets: torch.Tensor, out: str | Path, *, metadata: dict | None = None) -> dict:
    device = next(model.parameters()).device
    images = targets.to(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(2):
            model(images)
        _sync(device)
        started = time.perf_counter()
        phase = model(images)
        _sync(device)
        forward_seconds = time.perf_counter() - started
        result = export_result(targets, phase, out,
                               metadata={**(metadata or {}), 'timing': {'forward_seconds': forward_seconds,
                                          'warmup_passes': 2, 'refinement_seconds': 0.0}})
        result['metadata']['timing']['pipeline_seconds'] = time.perf_counter() - started
    write_json(Path(out) / 'metrics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs=4, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    targets = load_four_png(args.images)
    model, metadata = load_predictor(args.checkpoint, args.device)
    metadata['inputs'] = [{'path': path, 'sha256': file_hash(path)} for path in args.images]
    result = infer_sample(model, targets, args.out, metadata=metadata)
    print(result['mean'])


if __name__ == '__main__':
    main()
