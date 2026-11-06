"""Evaluate the fixed forward-data experiment and its controls after model selection."""
from pathlib import Path

import numpy as np
import torch

from src.data import load_manifest
from src.evaluate import aggregate_results
from src.infer import infer_sample, load_predictor
from src.io import export_result, file_hash, load_phase_png, write_json
from src.train import current_protocol, load_checkpoint

ROOT = Path('data/forward')
OUT = Path('outputs/timur_forward')
CHECKPOINT = Path('checkpoints/timur_forward/best.pt')


def main():
    torch.set_num_threads(2)
    _, train_samples = load_manifest(ROOT / 'train/manifest.json')
    _, test_samples = load_manifest(ROOT / 'test/manifest.json')
    train_hashes = {sample['phase_sha256'] for sample,_ in train_samples}
    assert not train_hashes & {sample['phase_sha256'] for sample,_ in test_samples}
    _, payload = load_checkpoint(CHECKPOINT, expected_protocol=current_protocol())
    assert payload['training_config']['train_manifest_sha256'] == file_hash(ROOT / 'train/manifest.json')
    old, old_metadata = load_predictor('checkpoints/best.pt')
    new, new_metadata = load_predictor(CHECKPOINT)
    initial, _ = load_checkpoint('checkpoints/timur_forward/initial.pt')
    original_initial, _ = load_checkpoint('checkpoints/initial.pt')
    same_initial = all(torch.equal(value, original_initial.state_dict()[name])
                       for name,value in initial.state_dict().items())
    assert same_initial
    train_targets = torch.cat([targets for _,targets in train_samples])
    results = {name:[] for name in ['old_model','forward_model','nearest_train_mask','source_phase_oracle']}
    pairs = []
    for sample, targets in test_samples:
        sample_id = sample['id']
        metadata = {'sample_id':sample_id,'split':'test','category':sample['category'],
                    'manifest_sha256':file_hash(ROOT / 'test/manifest.json')}
        for name,predictor,predictor_metadata in [('old_model',old,old_metadata),('forward_model',new,new_metadata)]:
            result = infer_sample(predictor,targets,OUT / name / sample_id,
                                  metadata={**predictor_metadata,**metadata})
            results[name].append(result)
        nearest = int((train_targets-targets).square().mean((1,2,3)).argmin())
        neighbor = train_samples[nearest][0]
        controls = [('nearest_train_mask',ROOT / 'train' / neighbor['phase_path']),
                    ('source_phase_oracle',ROOT / 'test' / sample['phase_path'])]
        for name,path in controls:
            result = export_result(targets,load_phase_png(str(path)),OUT / name / sample_id,
                                   metadata={**metadata,'method':name,'source_phase_sha256':file_hash(path),
                                             'neighbor_parent_id':neighbor['parent_sample_id'] if name=='nearest_train_mask' else None})
            results[name].append(result)
        assert results['source_phase_oracle'][-1]['mean']['ssim'] == 1.0
        pairs.append({'sample_id':sample_id,'category':sample['category'],
                      'ssim_delta':results['forward_model'][-1]['mean']['ssim']-results['old_model'][-1]['mean']['ssim'],
                      'pearson_delta':results['forward_model'][-1]['mean']['pearson']-results['old_model'][-1]['mean']['pearson']})
    categories = sorted({sample['category'] for sample,_ in test_samples})
    summary = {'protocol':current_protocol(),'seed':42,'test_samples':len(test_samples),
               'train_samples':len(train_samples),'initial_weights_identical':same_initial,
               'selected_step':payload['step'],'selected_samples_seen':payload['samples_seen'],
               'checkpoint_sha256':file_hash(CHECKPOINT),'baseline_checkpoint_sha256':file_hash('checkpoints/best.pt'),
               'model_config':payload['model_config'],'parameters':sum(p.numel() for p in new.parameters()),
               'methods':{name:aggregate_results(values) for name,values in results.items()},
               'categories':{category:{name:aggregate_results([value for value,(sample,_) in zip(values,test_samples)
                                                              if sample['category']==category])
                                       for name,values in results.items()} for category in categories},
               'paired_changes':pairs}
    summary['paired_change_summary'] = {}
    for key in ('ssim_delta','pearson_delta'):
        values = np.array([pair[key] for pair in pairs])
        means = np.random.default_rng(42).choice(values,size=(10000,len(values)),replace=True).mean(1)
        summary['paired_change_summary'][key] = {
            'mean':float(values.mean()),'positive_cases':int((values>0).sum()),
            'paired_bootstrap_95ci':np.quantile(means,[.025,.975]).tolist()}
    control_manifest, control_samples = load_manifest('data/timur_control/arbitrary_test/manifest.json')
    arbitrary = {'old_model': [], 'forward_model': []}
    for sample, targets in control_samples:
        for name, predictor, predictor_metadata in [('old_model',old,old_metadata),('forward_model',new,new_metadata)]:
            arbitrary[name].append(infer_sample(
                predictor,targets,OUT / 'arbitrary_control' / name / sample['id'],
                metadata={**predictor_metadata,'sample_id':sample['id'],'split':'arbitrary_test',
                          'category':sample['category'],'seed':control_manifest['seed']}))
    summary['arbitrary_control'] = {
        'samples':len(control_samples),'seed':control_manifest['seed'],
        'manifest_sha256':file_hash('data/timur_control/arbitrary_test/manifest.json'),
        'methods':{name:aggregate_results(values) for name,values in arbitrary.items()}}
    write_json(OUT / 'summary.json',summary)
    for name,value in summary['methods'].items():
        print(name,value['mean'],value['worst_plane_mean'])
    for name,value in summary['arbitrary_control']['methods'].items():
        print('arbitrary_control',name,value['mean'])


if __name__ == '__main__':
    main()
