#!/usr/bin/env python3
"""Prepare all 1024 historical parameter vectors at the upstream covariance fix."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import prepare_code_update_pilot as pilot

REVISION = '57c1ef2e7651695217155f6be1e4bc0e359ec4a0'
DATASETS = '3db63ead28772fa2a03321008895544347b126f5'
pilot.REQUIRED_PRS = (1499, 1665, 1682, 1760)


def check_design(source):
    import numpy as np
    from scipy.stats import qmc
    design = json.loads((source/'campaign_design.json').read_text())
    with (source/'samples.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1024 or design['n_eval'] != 1024 or design['seed'] != 42:
        raise ValueError('Expected original 1024-model seed-42 design')
    names = design['free_parameter_short_names']
    if len(names) != 20:
        raise ValueError('Expected 20 parameters')
    np.testing.assert_allclose([[float(r[n+'_quantile']) for n in names] for r in rows],
        qmc.Sobol(d=20, scramble=True, seed=42).random_base2(10), atol=1e-15, rtol=0)
    for i, row in enumerate(rows):
        if not row['evaluation_id'].endswith(f'-eval-{i:04d}'):
            raise ValueError('Unexpected evaluation ordering')
        old = source/'evaluations'/row['evaluation_id']
        pilot.validate_parameters(ET.parse(old/'params.xml').getroot(), row, design['free_parameters'])
        if not (old/'galacticus_input_values.json').is_file():
            raise FileNotFoundError(old/'galacticus_input_values.json')
    return design, rows


def original_version(path):
    import h5py
    with h5py.File(path, 'r') as f:
        if f.attrs.get('statusCompletion', -1) != 0:
            raise ValueError(f'Historical output incomplete: {path}')
        version = {k: pilot.text(v) for k, v in f['Version'].attrs.items()}
    if 'gitHashDatasets' not in version:
        raise ValueError(f'Historical datasets revision missing: {path}')
    return version


def stage(args):
    source, out, previous = args.original.resolve(), args.pilot.resolve(), args.previous_pilot.resolve()
    if out.exists():
        raise FileExistsError(out)
    _, rows = check_design(source)
    previous_meta = json.loads((previous/'pilot.json').read_text())
    previous_config = json.loads((previous/'updated/prepared.json').read_text())
    if pilot.sha(source/'campaign_design.json') != previous_meta['original_design_sha256']:
        raise ValueError('Design differs from the pilot')
    if previous_config['datasets_selection']['revision'] != DATASETS:
        raise ValueError('Unexpected pilot datasets')
    fixes = previous_meta['required_fix_commits'].copy()
    fixes['1760'] = REVISION
    if set(fixes) != {'1499','1665','1682','1760'}:
        raise ValueError('Missing required fix provenance')
    out.mkdir(parents=True)
    shutil.copy2(source/'campaign_design.json', out/'original_campaign_design.json')
    shutil.copy2(source/'samples.csv', out/'samples.csv')
    records = []
    for i, row in enumerate(rows):
        name = row['evaluation_id']
        src, dst = source/'evaluations'/name, out/'baseline'/name
        dst.mkdir(parents=True)
        for name_file in ['params.xml', 'galacticus_input_values.json']:
            shutil.copy2(src/name_file, dst/name_file)
        records.append(dict(index=i, evaluation_id=name, parameters_sha256=pilot.sha(dst/'params.xml'),
            original_version=original_version(src/'galacticus.hdf5')))
    pilot.write_json(out/'pilot.json', dict(target_revision=REVISION, required_fix_commits=fixes,
        required_pull_requests=list(pilot.REQUIRED_PRS), upstream_branch='master',
        selection='All 1024 original physical parameter vectors; no resampling',
        original_campaign=str(source), original_design_sha256=pilot.sha(source/'campaign_design.json'),
        samples_sha256=pilot.sha(out/'samples.csv'), evaluations=records))
    pilot.write_json(out/'inherited_environment.json', dict(environment=previous_config['environment'],
        datasets_selection=previous_config['datasets_selection']))
    shutil.copy2(__file__, out/'campaign.py')
    shutil.copy2(pilot.__file__, out/'prepare_code_update_pilot.py')
    print('Staged all 1024 inputs:', out)


def retain_numerical_frequency_prior(root, row, specs):
    # Verify the documented upstream conversion before deliberately overriding it.
    pilot.validate_parameters(root, row, specs, {'diskSFRFrequencyNorm': 1.0/1.36})
    spec = next(s for s in specs if s['short_name'] == 'diskSFRFrequencyNorm')
    pilot.one(root, spec['path']).set('value', str(row['diskSFRFrequencyNorm']))
    pilot.validate_parameters(root, row, specs)


def prepare(args):
    root = args.pilot.resolve()
    inherited = json.loads((root/'inherited_environment.json').read_text())
    os.environ.update(inherited['environment'])
    datasets = Path(os.environ['GALACTICUS_DATA_PATH'])
    revision = subprocess.check_output(['git','-C',str(datasets),'rev-parse','HEAD'], text=True).strip()
    if revision != DATASETS:
        raise ValueError('Datasets checkout moved since pilot')
    subprocess.run(['git','-C',str(datasets),'diff','--exit-code','HEAD','--','static'], check=True)
    unit = datasets/'static/simulations/UNIT.xml'
    if not unit.is_file():
        raise FileNotFoundError(unit)
    if json.loads((root/'pilot.json').read_text())['target_revision'] != REVISION:
        raise ValueError('This migration allowance is specific to the pinned campaign revision')
    args.post_migration = retain_numerical_frequency_prior
    pilot.prepare(args)
    config_path = root/'updated/prepared.json'
    config = json.loads(config_path.read_text())
    config['parameter_conventions'] = dict(
        diskSFRFrequencyNorm=dict(input_scale_from_original=1.0, upstream_migration_scale=1.0/1.36,
            migration_commit='4b5e7bf18d', physics_commit='ea6b55717d',
            note='Deliberately retain the original numerical prior under the new per-molecular-gas (including helium) definition. Upstream rescaling is checked then overridden. samples.csv, emulator inputs and final input.xml use the same original numbers; migrated.xml preserves the upstream conversion for audit.'))
    config['datasets_selection'] = inherited['datasets_selection']
    config['unit_xml_sha256'] = pilot.sha(unit)
    config['launcher_sha256'] = {name: pilot.sha(root/name) for name in ['campaign.py','prepare_code_update_pilot.py']}
    pilot.write_json(config_path, config)
    batch = root/'updated/submit.sbatch'
    text = batch.read_text().replace('galacticus-update64', 'galacticus-update1024')
    text = text.replace('#SBATCH --array=0-63%8', '#SBATCH --array=0-1023%128')
    text = text.replace(str(root/'pilot.py'), str(root/'campaign.py'))
    batch.write_text(text)
    pilot.verify_build(args)
    print('Preparation complete. Submit all 1024 models with a cap of 128:')
    print(f"sbatch --array='0-1023%128' {batch}")


def run(args):
    root = args.pilot.resolve()
    config = json.loads((root/'updated/prepared.json').read_text())
    for name, checksum in config['launcher_sha256'].items():
        if pilot.sha(root/name) != checksum:
            raise ValueError(f'Launcher changed: {name}')
    unit = Path(config['environment']['GALACTICUS_DATA_PATH'])/'static/simulations/UNIT.xml'
    if pilot.sha(unit) != config['unit_xml_sha256']:
        raise ValueError('UNIT snapshot definition changed')
    os.environ['PYTHONPATH'] = str(Path(config['executable']).parent/'python') + os.pathsep + os.environ.get('PYTHONPATH', '')
    pilot.run_model(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    s = commands.add_parser('stage')
    s.add_argument('original', type=Path)
    s.add_argument('pilot', type=Path)
    s.add_argument('--previous-pilot', type=Path, required=True)
    s.set_defaults(func=stage)
    s = commands.add_parser('prepare')
    s.add_argument('pilot', type=Path)
    s.add_argument('--source', type=Path, required=True)
    s.set_defaults(func=prepare)
    for name, fn in [('run',run), ('finish',pilot.finish_model)]:
        s = commands.add_parser(name)
        s.add_argument('pilot',type=Path)
        s.add_argument('index',type=int,choices=range(1024))
        s.set_defaults(func=fn)
    s = commands.add_parser('verify-build')
    s.add_argument('pilot',type=Path)
    s.set_defaults(func=pilot.verify_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
