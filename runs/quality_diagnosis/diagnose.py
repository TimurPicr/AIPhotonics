"""Validation-only diagnostics; the submitted checkpoint and protocol stay fixed."""
from pathlib import Path
import itertools
import math
import time

import numpy as np
import torch

from src import data

from src.baseline import optimize_phase
from src.data import load_manifest
from src.evaluate import aggregate_results, evaluate_four
from src.io import export_phase_png, file_hash, load_phase_png, write_json
from src.reconstruction import normalize_recons, reconstruct_four
from src.train import current_protocol, load_checkpoint

OUT = Path('runs/quality_diagnosis')
torch.set_num_threads(2)
model, checkpoint = load_checkpoint('checkpoints/best.pt', expected_protocol=current_protocol())
manifest, samples = load_manifest('data/val/manifest.json')
targets = torch.cat([target for _, target in samples])
with torch.no_grad():
    phases = torch.cat([model(batch) for batch in targets.split(8)])
    predictions = normalize_recons(reconstruct_four(phases))
    quantized = []
    for index in range(len(samples)):
        export_phase_png(phases[index:index+1], str(OUT / 'phase.png'))
        quantized.append(load_phase_png(str(OUT / 'phase.png')))
    (OUT / 'phase.png').unlink()
    quantized_predictions = normalize_recons(reconstruct_four(torch.cat(quantized)))
    reference_phase = torch.angle(torch.exp(1j * phases).mean(0, keepdim=True))
    reference = normalize_recons(reconstruct_four(reference_phase)).repeat(len(samples), 1, 1, 1)


def scores(values):
    results = [evaluate_four(targets[i:i+1], values[i:i+1]) for i in range(len(samples))]
    return {'aggregate': aggregate_results(results),
            'categories': {category: aggregate_results([r for r,(s,_) in zip(results,samples) if s['category']==category])
                           for category in ('geometry','symbols','grayscale','mixed')}}


def correlation(a, b):
    a, b = a.flatten(-2).double(), b.flatten(-2).double()
    a, b = a - a.mean(-1,keepdim=True), b - b.mean(-1,keepdim=True)
    return (a*b).sum(-1) / (a.square().sum(-1)*b.square().sum(-1)).sqrt().clamp_min(1e-12)


shifted = torch.roll(predictions, 4, 0)
fields = torch.exp(1j * phases)
phase_coherence = (fields * torch.roll(fields,4,0).conj()).mean((-2,-1)).abs().mean()
cosine = (predictions*targets).flatten(1).sum(1) / (predictions.flatten(1).norm(dim=1)*targets.flatten(1).norm(dim=1))
report = {'checkpoint_sha256':file_hash('checkpoints/best.pt'), 'protocol':current_protocol(),
          'manifest_sha256':file_hash('data/val/manifest.json'), 'split':'val', 'samples':len(samples),
          'metric_domain':'float reconstructions; quantized phase is labelled separately',
          'network_float':scores(predictions), 'network_png_phase_float_recons':scores(quantized_predictions),
          'diagnostic_only_squared_amplitude':scores(predictions.square()),
          'diagnostic_only_zero_reconstruction':scores(torch.zeros_like(predictions)),
          'diagnostic_only_same_phase_for_all_inputs':scores(reference),
          'diagnostic_only_wrong_target_within_category':scores(shifted),
          'volume_amplitude_cosine_mean':float(cosine.mean()),
          'phase_saturated_fraction':float((phases.abs() > .99*math.pi).float().mean()),
          'phase_coherence_between_inputs_same_category':float(phase_coherence),
          'recon_correlation_between_inputs_same_category':float(correlation(predictions,shifted).mean()),
          'target_correlation_between_inputs_same_category':float(correlation(targets,torch.roll(targets,4,0)).mean()),
          'recon_correlation_between_planes':float(torch.stack([correlation(predictions[:,a],predictions[:,b]).mean()
                                                               for a,b in itertools.combinations(range(4),2)]).mean()),
          'target_correlation_between_planes':float(torch.stack([correlation(targets[:,a],targets[:,b]).mean()
                                                                for a,b in itertools.combinations(range(4),2)]).mean())}
sparse = torch.tensor([s['category'] in ('geometry','symbols') for s,_ in samples])
support = targets[sparse] > .05
energy = predictions[sparse].square()
report['sparse_targets'] = {'support_fraction':float(support.float().mean()),
                          'reconstruction_energy_on_support':float(((energy*support).sum((-2,-1))/energy.sum((-2,-1))).mean()),
                          'background_amplitude_mean':float(predictions[sparse][~support].mean()),
                          'foreground_amplitude_mean':float(predictions[sparse][support].mean())}
gain = (predictions*targets).sum((-2,-1),keepdim=True) / predictions.square().sum((-2,-1),keepdim=True)
report['normalization_diagnostic'] = {'current_mse':float((predictions-targets).square().mean()),
    'target_fitted_gain_mse_not_official':float((gain*predictions-targets).square().mean()),
    'gain_mean':float(gain.mean()), 'gain_min':float(gain.min()), 'gain_max':float(gain.max())}
recorded_categories = []
original_target = data._target

def traced_target(rng, category, ood=False):
    category = str(rng.choice(data.CATEGORIES[:3])) if category == 'mixed' else category
    recorded_categories.append(category)
    return original_target(rng, category, ood)

data._target = traced_target
try:
    loader = iter(data.make_train_loader(8, 42))
    for _ in range(128):
        next(loader)
finally:
    data._target = original_target
train_categories = [recorded_categories[i:i+4] for i in range(0,len(recorded_categories),4)]
val_categories = [[str(np.random.default_rng(source['seed']).choice(data.CATEGORIES[:3]))
                   if sample['category']=='mixed' else sample['category'] for source in sample['sources']]
                  for sample,_ in samples]
report['category_mixture'] = {'train_quadruples':len(train_categories),
    'train_all_four_same_category':sum(len(set(group))==1 for group in train_categories),
    'validation_quadruples':len(val_categories),
    'validation_all_four_same_category':sum(len(set(group))==1 for group in val_categories),
    'expected_training_same_category_fraction':1/27}
write_json(OUT / 'summary.json', report)
print('Validation diagnostics saved', flush=True)

optimization = []
for index in (0,2):
    target = targets[index:index+1]
    for plane, seed, initialization in [(None,42,'random'),(None,43,'random'),(None,44,'random'),
                                        (1,42,'random'),(None,42,'network')]:
        fitting = target if plane is None else target[:,:1]
        started = time.perf_counter()
        phase = optimize_phase(fitting, initial_phase=phases[index:index+1] if initialization=='network' else None,
                               steps=1000,seed=seed,plane=plane)
        with torch.no_grad():
            recon = normalize_recons(reconstruct_four(phase))
        metrics = evaluate_four(target,recon)
        row = {'sample_id':samples[index][0]['id'], 'category':samples[index][0]['category'],
               'planes_fitted':[1,2,3,4] if plane is None else [plane], 'initialization':initialization,
               'seed':seed,'steps':1000,'seconds':time.perf_counter()-started,
               'float_mse':float((recon-target).square().mean()),'metrics':metrics}
        optimization.append(row)
        write_json(OUT / 'optimization.json', {'metric_domain':'float phase and reconstruction', 'experiments':optimization})
        print(row['sample_id'],plane,initialization,seed,metrics['mean'],flush=True)
assert file_hash('checkpoints/best.pt') == report['checkpoint_sha256']
