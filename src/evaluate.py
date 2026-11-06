"""PNG metrics and fixed-manifest evaluation."""
import argparse
from pathlib import Path

import numpy as np
import skimage
from skimage.metrics import structural_similarity
import torch

SSIM_PARAMETERS = {'data_range': 1.0, 'win_size': 7, 'gaussian_weights': False,
                   'use_sample_covariance': True, 'K1': 0.01, 'K2': 0.03}


def _mean(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def evaluate_four(targets: torch.Tensor, recons: torch.Tensor) -> dict:
    for value in (targets, recons):
        if tuple(value.shape) != (1, 4, 128, 128):
            raise ValueError('Expected [1,4,128,128]')
        if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError('Metrics require finite PNG values in [0,1]')
    targets, recons = (v.detach().cpu().numpy().astype(np.float64)[0] for v in (targets, recons))
    planes, errors = [], []
    for index, (target, recon) in enumerate(zip(targets, recons), 1):
        tc, rc = target - target.mean(), recon - recon.mean()
        target_constant, recon_constant = not np.any(tc), not np.any(rc)
        reason = 'constant_target' if target_constant else 'constant_reconstruction' if recon_constant else None
        pearson = None if reason else float(np.clip(np.sum(tc * rc) / np.sqrt(np.sum(tc**2) * np.sum(rc**2)), -1, 1))
        if reason:
            errors.append({'plane': index, 'reason': reason, 'failure': not target_constant})
        planes.append({'plane': index, 'pearson': pearson,
                       'pearson_reason': reason,
                       'ssim': float(structural_similarity(target, recon, **SSIM_PARAMETERS))})
    return {'planes': planes, 'mean': {key: _mean([p[key] for p in planes]) for key in ('pearson', 'ssim')},
            'worst_plane': {key: min((p[key] for p in planes if p[key] is not None), default=None)
                            for key in ('pearson', 'ssim')},
            'valid_pearson_pairs': sum(p['pearson'] is not None for p in planes),
            'total_pairs': 4, 'failure_count': sum(e['failure'] for e in errors), 'errors': errors,
            'ssim_parameters': SSIM_PARAMETERS, 'skimage_version': skimage.__version__}


def aggregate_results(results: list[dict]) -> dict:
    if not results:
        raise ValueError('Cannot aggregate an empty set')
    planes = []
    for index in range(4):
        pairs = [r['planes'][index] for r in results]
        planes.append({'plane': index + 1,
                       **{key: _mean([p[key] for p in pairs]) for key in ('pearson', 'ssim')},
                       'valid_pearson_pairs': sum(p['pearson'] is not None for p in pairs),
                       'total_pairs': len(pairs)})
    all_pairs = [p for r in results for p in r['planes']]
    return {'samples': len(results), 'total_pairs': len(all_pairs), 'planes': planes,
            'mean': {key: _mean([p[key] for p in all_pairs]) for key in ('pearson', 'ssim')},
            'worst_plane_mean': {key: min((p[key] for p in planes if p[key] is not None), default=None)
                                 for key in ('pearson', 'ssim')},
            'valid_pearson_pairs': sum(r['valid_pearson_pairs'] for r in results),
            'failure_count': sum(r['failure_count'] for r in results),
            'diagnostic_pairs': sum(len(r['errors']) for r in results)}


def main():
    from src.data import load_manifest
    from src.infer import load_predictor, infer_sample
    from src.io import file_hash, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    model, metadata = load_predictor(args.checkpoint, args.device)
    manifest, samples = load_manifest(args.manifest)
    from config import CONFIG
    if list(CONFIG.z_m) != manifest['z_m']:
        raise ValueError('Manifest distances differ from active protocol')
    metadata.update(run_id=args.out.name, split=manifest['split'], dataset_seed=manifest['seed'],
                    manifest_sha256=file_hash(args.manifest))
    results, categories, errors = [], {}, []
    for sample, targets in samples:
        result = infer_sample(model, targets, args.out / sample['id'],
                              metadata={**metadata, 'sample_id': sample['id'],
                                        'split': manifest['split'], 'category': sample['category'],
                                        'inputs': sample['sources']})
        results.append(result)
        errors.extend({'sample_id': sample['id'], **error} for error in result['errors'])
        categories.setdefault(sample['category'], []).append(result)
    summary = {'metadata': metadata, 'split': manifest['split'], 'aggregate': aggregate_results(results), 'errors': errors,
               'categories': {name: aggregate_results(group) for name, group in categories.items()}}
    write_json(args.out / 'summary.json', summary)
    print(summary['aggregate'])


if __name__ == '__main__':
    main()
