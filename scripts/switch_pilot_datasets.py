#!/usr/bin/env python3
"""Archive pilot attempts and prepare the same inputs with pinned updated datasets."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

DATASETS_REVISION = '3db63ead28772fa2a03321008895544347b126f5'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def switch(pilot, datasets):
    pilot, datasets = pilot.resolve(), datasets.resolve()
    jobs = command('squeue', '--noheader', '--user', getpass.getuser(), '--format=%j')
    if 'galacticus-update64' in jobs.splitlines():
        raise RuntimeError('A pilot job is still queued/running. Wait for it to finish before switching.')
    revision = command('git', '-C', str(datasets), 'rev-parse', 'HEAD')
    if revision != DATASETS_REVISION:
        raise ValueError(f'Expected datasets {DATASETS_REVISION}, found {revision}')
    subprocess.run(['git', '-C', str(datasets), 'diff', '--exit-code', 'HEAD', '--', 'static'], check=True)
    for name in ['ssp_gen.f90.patch', 'sps_vars.f90.patch', 'Makefile.patch']:
        if not (datasets/'static/patches/FSPS'/name).is_file():
            raise FileNotFoundError(name)
    archive = datasets/'dynamic/FSPS_4.0.tar.gz'
    print('Validating the existing FSPS download...', flush=True)
    subprocess.run(['gzip', '-t', str(archive)], check=True)
    updated = pilot/'updated'
    config = json.loads((updated/'prepared.json').read_text())
    if sha(config['executable']) != config['executable_sha256']:
        raise ValueError('Executable changed since preparation; do not silently change the experiment')
    if config.get('datasets_selection', {}).get('revision') == revision:
        raise ValueError('Pilot already switched to this datasets revision; inspect before repeating')
    meta = json.loads((pilot/'pilot.json').read_text())
    names = [r['evaluation_id'] for r in meta['evaluations']]
    if len(names) != 64 or set(names) != set(config['input_sha256']):
        raise ValueError('Expected the complete 64-model pilot')
    for name in names:
        if sha(updated/'evaluations'/name/'input.xml') != config['input_sha256'][name]:
            raise ValueError(f'Prepared input changed: {name}')
    stage = pilot/'updated_newdatasets_staging'
    stage.mkdir()  # Refuse to overwrite a partial preparation.
    (stage/'logs').mkdir()
    shutil.copy2(updated/'submit.sbatch', stage/'submit.sbatch')
    for name in names:
        src, dst = updated/'evaluations'/name, stage/'evaluations'/name
        dst.mkdir(parents=True)
        for file in ['galacticus_input_values.json', 'migration_input.xml', 'migrated.xml', 'migration.log']:
            if (src/file).exists():
                shutil.copy2(src/file, dst/file)
        root = ET.parse(src/'input.xml').getroot()
        # Correct an ignored option added by the initial pilot generator.
        for bar in root.findall(".//nodeOperator[@value='barInstability']"):
            for selector in bar.findall('barInstabilitySpheroidAngularMomentum'):
                if selector.get('value') != 'angularMomentumTransfer':
                    raise ValueError(f'Unexpected bar model in {name}')
                bar.remove(selector)
        ET.indent(root)
        ET.ElementTree(root).write(dst/'input.xml', encoding='utf-8', xml_declaration=True)
        config['input_sha256'][name] = sha(dst/'input.xml')
    config['environment']['GALACTICUS_DATA_PATH'] = str(datasets)
    config['environment']['GALACTICUS_TOOLS_PATH'] = str(datasets/'dynamic')
    config.pop('fsps_repair_provenance', None)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    previous = pilot/'attempts'/('before_updated_datasets_'+stamp)
    config['datasets_selection'] = dict(revision=revision, path=str(datasets),
        selected_at=stamp, fsps_archive_sha256=sha(archive), previous_attempt=str(previous),
        note='Full official datasets checkout; new tool directory and fresh per-evaluation caches.')
    (stage/'prepared.json').write_text(json.dumps(config, indent=2)+'\n')
    previous.parent.mkdir(exist_ok=True)
    if previous.exists():
        raise FileExistsError(previous)
    updated.rename(previous)
    try:
        stage.rename(updated)
    except Exception:
        previous.rename(updated)
        raise
    print('Previous runs and configuration preserved at:', previous)
    print('Pilot datasets:', datasets)
    print('Datasets revision:', revision)
    print('Ready. Submit only evaluation 0000:')
    print(f'sbatch --array=0 "{updated}/submit.sbatch"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('datasets', type=Path)
    args = parser.parse_args()
    switch(args.pilot, args.datasets)
