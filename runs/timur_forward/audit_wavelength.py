"""Audit all stored optical PNGs against explicit 532 nm propagation and NumPy."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_CONFIG, OpticalConfig
from src.data import load_manifest
from src.io import load_four_png, load_phase_png, write_json
from src.physics import output_pitch_m
from src.reconstruction import normalize_recons, reconstruct_four

SPEC = OpticalConfig(wavelength_m=532e-9)
OUT = Path('outputs/timur_forward/wavelength_audit.json')


def numpy_reference(phase, z, config):
    # Unit-modulus global and output chirps cancel when taking the magnitude.
    n = phase.shape[-1]
    axis = (np.arange(n) - n//2) * config.slm_pitch_m
    radius = axis[:,None]**2 + axis[None,:]**2
    field = np.exp(1j * phase.astype(np.float64)) * np.exp(1j*np.pi*radius/(config.wavelength_m*z))
    amplitude = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field))))
    amplitude *= config.slm_pitch_m**2 / (config.wavelength_m*z)
    values = amplitude if config.reconstruction_value == 'amplitude' else amplitude**2
    return values / max(values.max(),config.normalization_eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fill-metadata', action='store_true', help='record verified protocol where absent')
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert DEFAULT_CONFIG.snapshot() == SPEC.snapshot(), 'Configured protocol differs from required 532 nm'
    independent_error, negative_matches = 0.0, 0
    verified, missing_metadata = [], []
    alternative = replace(SPEC,wavelength_m=638e-9)

    def check(phase_path, image_paths):
        nonlocal independent_error, negative_matches
        phase = load_phase_png(str(phase_path))
        saved = load_four_png([str(path) for path in image_paths])
        with torch.no_grad():
            values = normalize_recons(reconstruct_four(phase,config=SPEC),config=SPEC)
        expected = (values*255).round()/255
        assert torch.equal(saved,expected), f'532 nm PNG mismatch: {phase_path}'
        all_alternative_match = True
        for plane,z in enumerate(SPEC.distances_m):
            reference = numpy_reference(phase.numpy()[0,0],z,SPEC)
            error = float(np.abs(reference-values.numpy()[0,plane]).max())
            independent_error = max(independent_error,error)
            assert error < 1e-4, f'Independent reference mismatch: {phase_path}, plane {plane+1}'
            wrong = np.rint(numpy_reference(phase.numpy()[0,0],z,alternative)*255).astype(np.uint8)
            actual = (saved.numpy()[0,plane]*255).round().astype(np.uint8)
            all_alternative_match &= np.array_equal(wrong,actual)
        negative_matches += int(all_alternative_match)
        verified.append(str(phase_path))

    output_cases = sorted(Path('outputs').rglob('phase.png'))
    for phase_path in output_cases:
        check(phase_path,[phase_path.parent/f'recon_{i}.png' for i in range(1,5)])
        metrics_path = phase_path.parent/'metrics.json'
        result = json.loads(metrics_path.read_text())
        metadata = result['metadata']
        if 'protocol' in metadata:
            assert metadata['protocol'] == SPEC.snapshot(), f'Metadata mismatch: {metrics_path}'
        else:
            missing_metadata.append(str(metrics_path))
            if args.fill_metadata:
                metadata.update(protocol=SPEC.snapshot(),dx_out_m=[output_pitch_m(z,128,config=SPEC) for z in SPEC.distances_m])
                write_json(metrics_path,result)

    manifests = [*Path('data').rglob('manifest.json'),Path('examples/manifest.json')]
    forward_cases, wavelength_independent = 0, []
    for path in manifests:
        manifest,samples = load_manifest(path)
        assert manifest['z_m'] == list(SPEC.distances_m), f'Distance order mismatch: {path}'
        if manifest.get('generator') == 'forward-v1':
            assert manifest['protocol'] == SPEC.snapshot()
            for sample,_ in samples:
                check(path.parent/sample['phase_path'],[path.parent/source['path'] for source in sample['sources']])
                forward_cases += 1
        else:
            wavelength_independent.append(str(path))

    checkpoints = {}
    for path in sorted(Path('checkpoints').rglob('*.pt')):
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        assert checkpoint['protocol'] == SPEC.snapshot(), f'Checkpoint mismatch: {path}'
        checkpoints[str(path)] = checkpoint['protocol']['wavelength_m']

    records = []
    def scan(value,path):
        if isinstance(value,dict):
            for key,item in value.items():
                if key == 'wavelength_m':
                    assert item == SPEC.wavelength_m, f'JSON wavelength mismatch: {path}'
                    records.append(str(path))
                scan(item,path)
        elif isinstance(value,list):
            for item in value:
                scan(item,path)
    for root in ('data','runs','outputs','examples'):
        for path in Path(root).rglob('*.json'):
            if path != OUT:
                scan(json.loads(path.read_text()),path)
    report = {'required_wavelength_m':SPEC.wavelength_m,'required_wavelength_nm':532,
              'requirements_sources':['AGENTS.md:16','PIPELINE.md:21'],'protocol':SPEC.snapshot(),
              'output_phase_cases':len(output_cases),'forward_dataset_cases':forward_cases,
              'verified_optical_pngs':len(verified)*4,'independent_numpy_max_abs_error':independent_error,
              'negative_control_nm':638,'negative_control_exact_matches':negative_matches,
              'checkpoints':checkpoints,'json_wavelength_records':len(records),
              'wavelength_independent_manifests':wavelength_independent,
              'metadata_missing_before_audit':missing_metadata,'metadata_filled':args.fill_metadata,
              'errors':[]}
    assert negative_matches == 0, 'Negative wavelength control unexpectedly matched'
    write_json(OUT,report)
    print({key:report[key] for key in ['required_wavelength_nm','output_phase_cases','forward_dataset_cases',
          'verified_optical_pngs','independent_numpy_max_abs_error','negative_control_exact_matches']})
    print('Checkpoints:',len(checkpoints),'metadata absent:',len(missing_metadata),'filled:',args.fill_metadata)


if __name__ == '__main__':
    main()
