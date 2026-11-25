#!/usr/bin/env python3
"""Matched Caltech bar-energy coefficient experiment.

Written by OpenAI Codex at Andrew Robertson's request.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import prepare_energy_precision_pilot as p

REVISION = 'f6648df263bf65b582f44c29683b72ab3ad27db6'
BASE_SHA = '43cae987c6139647554ab2a3f11f763759647aa13cf8192725ba9ad1f691bd4a'
# The input names fork-only revision 400084ec. Its two commits since this
# upstream ancestor add the bar prescription and transition, which model()
# replaces explicitly. Migrating from that fork revision gives an empty
# --ancestry-path even when its Git object is available.
MIGRATION_BASE = 'ffecc960d43b861eddc701c496ac5f340e0e3f70'
MASK_URL = 'https://zenodo.org/records/10998446/files/sdss_dr72safe0_res6d.pol.gz'


def cases(definition, rows):
    """Two completed OBS points; vary only the two bar-energy ratios."""
    result = []
    for index in (0, 1):
        for alpha in (.5, 1., 2.):
            for beta in (2., 4.):
                name = f'point{index:04d}_alpha{alpha:g}_beta{beta:g}'
                row = dict(rows[index], evaluation_id=name, barSpheroidFormFactor=.5*alpha)
                result.append(dict(task_index=len(result), level='practical', evaluation_id=name,
                                   design_index=index, alpha=alpha, beta=beta, row=row))
    return result


def verify(root, task, definition):
    # Reuse all established physical/numerical checks, except the interaction
    # coefficient, which is the new varied coordinate in this experiment.
    interaction = p.one(root, p.ENERGY+'/formFactorInteraction')
    if abs(p.number(interaction)-.5*task['beta']) > 1e-12:
        raise ValueError('Incorrect interaction coefficient')
    checked = copy.deepcopy(root)
    p.one(checked, p.ENERGY+'/formFactorInteraction').set('value', '1.0')
    p.verify_parameters(checked, task['row'], definition['free_parameters'], 'practical', 1000)
    if p.seeds(root) != task['random_seeds']:
        raise ValueError('Changed random seeds')


def check_mask(path):
    with path.open('rb') as handle:
        header = handle.read(1024)
    if path.stat().st_size < 1000000 or b'polygon' not in header.lower():
        raise ValueError(f'Not an uncompressed SDSS polygon mask: {path}')


def stage_mask(data, out):
    """Reuse existing mask, otherwise fetch once with bounded retries."""
    dest = out/'provenance'/p.SURVEY_MASK.name
    candidate = Path(os.environ.get('ENERGY_COEFFICIENT_SURVEY_MASK', str(data/'dynamic'/p.SURVEY_MASK)))
    if candidate.is_file():
        check_mask(candidate)
        shutil.copy2(candidate, dest)
        source = str(candidate)
    else:
        compressed = dest.with_suffix('.pol.gz')
        subprocess.run(['curl', '--fail', '--location', '--retry', '4', '--connect-timeout', '30',
                        '--max-time', '900', '--output', str(compressed), MASK_URL], check=True)
        with gzip.open(compressed, 'rb') as src, dest.open('wb') as target:
            shutil.copyfileobj(src, target)
        compressed.unlink()
        check_mask(dest)
        source = MASK_URL
    return dict(source=source, snapshot=str(dest), sha256=p.sha(dest), bytes=dest.stat().st_size)


def prepare(args):
    definition, rows = p.design()
    source = args.source.resolve()
    exe = args.executable.resolve()
    out = args.root.absolute()
    if out.exists():
        raise FileExistsError(f'Use a fresh campaign directory: {out}')
    if p.git(source, 'rev-parse', 'HEAD') != REVISION or p.git(source, 'diff', 'HEAD', '--'):
        raise ValueError('Expected clean pinned Galacticus source')
    subprocess.run(['git', '-C', str(source), 'merge-base', '--is-ancestor', MIGRATION_BASE, REVISION], check=True)
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise ValueError(f'Build is missing: {exe}')
    base_path = p.REPO/'inputs/energy_coefficients/historical_base.xml'
    if p.sha(base_path) != BASE_SHA:
        raise ValueError('Historical base checksum differs')
    environment = {}
    for key in ('GALACTICUS_DATA_PATH', 'SHMR_PATH'):
        value = Path(os.environ[key]).resolve()
        if not value.is_dir():
            raise ValueError(f'Missing {key}: {value}')
        environment[key] = str(value)
    data = Path(environment['GALACTICUS_DATA_PATH'])
    environment['GALACTICUS_TOOLS_PATH'] = os.environ.get('GALACTICUS_TOOLS_PATH', str(data/'dynamic'))
    out.mkdir(parents=True)
    for folder in ('provenance', 'executable', 'logs', 'practical', 'tighter'):
        (out/folder).mkdir()
    for name in ('family_definition.json', 'design_1024.csv'):
        shutil.copy2(p.FAMILY/name, out/'provenance'/name)
    shutil.copy2(base_path, out/'provenance/historical_base.xml')
    for name in ('energy_coefficient_sweep.py', 'prepare_energy_precision_pilot.py',
                 'analyze_energy_precision_pilot.py', 'analyze_campaign_compact_tail.py'):
        shutil.copy2(p.REPO/'scripts'/name, out/name)
    shutil.copy2(__file__, out/'pilot.py')
    env = dict(os.environ, **environment, GALACTICUS_EXEC_PATH=str(source))
    env['PYTHONPATH'] = str(source/'python')+os.pathsep+env.get('PYTHONPATH', '')
    migrated = out/'base_parameters.xml'
    p.dump(out/'provenance/migration_basis.json', dict(
        input_revision=ET.parse(base_path).getroot().find('lastModified').get('revision'),
        migration_base=MIGRATION_BASE, target_revision=REVISION,
        reason='Input revision is on the original bar-model fork; use its verified upstream common ancestor. The fork-only bar-model settings are explicitly replaced.'))
    with (out/'provenance/migration.log').open('w') as log:
        subprocess.run([sys.executable, str(source/'scripts/aux/parametersMigrate.py'),
                        str(base_path), str(migrated), '--validate', 'no',
                        '--lastModifiedRevision', MIGRATION_BASE], cwd=source, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    base = ET.parse(migrated).getroot()
    seeds = p.seeds(ET.parse(base_path).getroot())
    if p.seeds(base) != seeds:
        raise ValueError('Migration changed seeds')
    for element in base.findall('version'):
        if element.get('value') is None:
            base.remove(element)
    for element in base.iter():
        for attr in ('value', 'active'):
            value = element.get(attr)
            if value and (attr == 'active' or value.lstrip().startswith('=')):
                element.set(attr, re.sub(r'\[([^\]]*)\]', lambda m: '['+re.sub(r':+', '/', m[1])+']', value))
    if p.one(base, 'lastModified').get('revision') != REVISION:
        raise ValueError('Unexpected migrated revision')
    p.xml(base, migrated)
    frozen_exe = out/'executable/Galacticus.exe'
    shutil.copy2(exe, frozen_exe)
    mask = stage_mask(data, out)
    tasks = cases(definition, rows)
    for task in tasks:
        folder = out/'practical/evaluations'/task['evaluation_id']
        folder.mkdir(parents=True)
        root = p.model(base, task['row'], definition['free_parameters'], 'practical', folder, 1000)
        p.child(p.one(root, p.ENERGY), 'formFactorInteraction', .5*task['beta'])
        task['random_seeds'] = seeds
        verify(root, task, definition)
        p.xml(root, folder/'params.xml')
        task['input_sha256'] = p.sha(folder/'params.xml')
        p.dump(folder/'galacticus_input_values.json', task['row'])
    p.write_rows(out/'practical/samples.csv', [t['row'] for t in tasks])
    # The existing reducer accepts an empty tighter level; this is a physics
    # experiment, not a numerical-precision comparison.
    (out/'tighter/samples.csv').write_text((out/'practical/samples.csv').read_text().splitlines()[0]+'\n')
    for level in ('practical', 'tighter'):
        p.dump(out/level/'campaign_design.json', dict(experiment='bar_energy_coefficients',
               n_eval=len(tasks) if level == 'practical' else 0, coefficient_grid=dict(alpha=[.5, 1., 2.], beta=[2., 4.]),
               physical_points=[0, 1], seed_policy='preserved historical seeds; matched coefficient comparisons',
               free_parameters=definition['free_parameters']))
    p.dump(out/'tasks.json', tasks)
    try:
        dataset_revision = p.git(data, 'rev-parse', 'HEAD')
    except subprocess.CalledProcessError:
        dataset_revision = None
    scripts = {path.name: p.sha(path) for path in out.glob('*.py')}
    meta = dict(source=str(source), source_revision=REVISION, executable=str(frozen_exe),
                executable_sha256=p.sha(frozen_exe), runner_sha256=p.sha(out/'pilot.py'),
                tasks_sha256=p.sha(out/'tasks.json'), environment=environment, cpus=8,
                scripts_sha256=scripts, datasets_revision=dataset_revision, python=sys.executable,
                isolated_dynamic_cache=True, survey_mask=mask, iteration_limit=1000,
                trees_per_decade=500, experiment='bar_energy_coefficients',
                base_sha256=BASE_SHA, source_executable=str(exe))
    p.dump(out/'manifest.json', meta)
    # Exercise every distinct XML before releasing the simulations. Each dry
    # run has its own directory, and uses the same frozen binary as evolution.
    for task in tasks:
        execute(out, task['task_index'], dry=True)
    (out/'PREPARED').touch()
    print('Prepared and validated 12 cases:', out, flush=True)


def execute(root, index, dry=False):
    root, meta, tasks, env = p.load_run(root)
    for name, checksum in meta['scripts_sha256'].items():
        if p.sha(root/name) != checksum:
            raise ValueError(f'Frozen script changed: {name}')
    if p.git(Path(meta['source']), 'diff', 'HEAD', '--'):
        raise ValueError('Runtime source has tracked edits')
    if not 0 <= index < len(tasks):
        raise ValueError('Invalid task index')
    task = tasks[index]
    original = root/'practical/evaluations'/task['evaluation_id']
    if p.sha(original/'params.xml') != task['input_sha256']:
        raise ValueError('Input XML changed')
    folder = root/'validation'/task['evaluation_id'] if dry else original
    folder.mkdir(parents=True, exist_ok=True)
    if not dry and not (root/'PREPARED').exists():
        raise ValueError('Preparation did not finish')
    if any((folder/name).exists() for name in ('SUCCESS', 'RUNNING', 'galacticus.hdf5')):
        raise FileExistsError('Existing output; use a fresh destination for retries')
    if dry:
        params = ET.parse(original/'params.xml').getroot()
        p.child(params, 'outputFileName', str(folder/'galacticus.hdf5'))
        for element in params.findall('outputScratchFileName'):
            element.set('value', str(folder/'galacticus.hdf5'))
        p.xml(params, folder/'params.xml')
    env = p.case_environment(meta, env, folder)
    env['PYTHONPATH'] = str(Path(meta['source'])/'python')+os.pathsep+env.get('PYTHONPATH', '')
    definition = json.loads((root/'provenance/family_definition.json').read_text())
    result = dict(status='failed', task=task)
    start = time.monotonic()
    (folder/'RUNNING').touch(exist_ok=False)
    try:
        command = [meta['executable'], 'params.xml', '--output-processed-parameters', 'params_used.xml']
        if dry:
            command.append('--dry-run')
        with (folder/'galacticus.out').open('w') as stdout, (folder/'galacticus.err').open('w') as stderr:
            subprocess.run(command, cwd=folder, env=env, stdout=stdout, stderr=stderr, check=True)
        verify(ET.parse(folder/'params_used.xml').getroot(), task, definition)
        if not dry:
            p.completion(folder/'galacticus.hdf5', meta['source_revision'])
        (folder/'SUCCESS').touch()
        result['status'] = 'ok'
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-start, threads=int(env['OMP_NUM_THREADS']), host=os.uname().nodename)
        p.dump(folder/'run_status.json', result)
        (folder/'RUNNING').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--executable', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('root', type=Path)
    run.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    else:
        execute(args.root, args.task)


if __name__ == '__main__':
    main()
