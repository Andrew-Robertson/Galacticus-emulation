#!/usr/bin/env python3
"""Package and prepare a paired, historical-design Galacticus update pilot."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

REQUIRED_PRS = (1499, 1665, 1682)


def github_json(path):
    request = urllib.request.Request(
        'https://api.github.com/repos/galacticusorg/galacticus/' + path,
        headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'Galacticus-code-update-pilot'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def pin_merged_master(args):
    """Select master once, only after all requested PRs have merged into it."""
    path = args.pilot.resolve()/'pilot.json'
    meta = json.loads(path.read_text())
    if meta.get('required_pull_requests') != list(REQUIRED_PRS):
        raise ValueError('Use the revised pilot bundle requiring PRs #1499, #1665 and #1682')
    if meta.get('target_revision'):
        if set(meta.get('required_fix_commits', {})) != {str(pr) for pr in REQUIRED_PRS}:
            raise ValueError('Existing pin lacks required fix provenance')
        print(meta['target_revision'])
        return
    fixes = {}
    for number in REQUIRED_PRS:
        pr = github_json(f'pulls/{number}')
        if not pr.get('merged') or not pr.get('merge_commit_sha'):
            raise ValueError(f'PR #{number} is not merged yet. No build submitted; rerun setup_caltech.sh after it merges.')
        fixes[str(number)] = pr['merge_commit_sha']
    branch = meta['upstream_branch']
    revision = github_json('commits/' + branch)['sha']
    for number, commit in fixes.items():
        comparison = github_json(f'compare/{commit}...{revision}')
        if comparison['status'] not in ('ahead', 'identical') or comparison['behind_by'] != 0:
            raise ValueError(f'Master does not contain merged PR #{number}; no build submitted')
    meta.update(target_revision=revision, required_fix_commits=fixes,
                revision_selected_at=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix('.json.tmp')
    write_json(temporary, meta)
    temporary.replace(path)
    print(revision)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def one(root, path):
    found = root.findall(path)
    if len(found) != 1:
        raise ValueError(f'Expected exactly one {path}, found {len(found)}')
    return found[0]


def set_child(root, name, value):
    node = root.find(name)
    if node is None:
        node = ET.SubElement(root, name)
    node.set('value', str(value))


def validate_parameters(root, row, specs):
    for spec in specs:
        actual = float(one(root, spec['path']).get('value'))
        if not math.isclose(actual, float(row[spec['short_name']]), rel_tol=2e-13):
            raise ValueError(f"Parameter changed: {spec['short_name']}: {actual}")
    if one(root, 'galacticStructureSolver').get('value') != 'simple':
        raise ValueError('Pilot must retain simple sizes')
    if root.find(".//barInstabilitySpheroidAngularMomentum[@value='bindingEnergy']") is not None:
        raise ValueError('Energy-based bar sizes are outside this pilot')


def stage(args):
    import h5py
    import numpy as np
    source, out = args.campaign.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    definition = json.loads((source/'campaign_design.json').read_text())
    with (source/'samples.csv').open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)[:64]
    if len(rows) != 64 or definition['n_eval'] != 1024 or definition['seed'] != 42:
        raise ValueError('Expected the original 1024-model, seed-42 campaign')
    # Audit the stored quantiles; do not regenerate or round the physical samples.
    from scipy.stats import qmc
    names = definition['free_parameter_short_names']
    quantiles = np.array([[float(r[n+'_quantile']) for n in names] for r in rows])
    expected = qmc.Sobol(d=len(names), scramble=True, seed=42).random_base2(6)
    np.testing.assert_allclose(quantiles, expected, rtol=0, atol=1e-15)
    out.mkdir(parents=True)
    shutil.copy2(source/'campaign_design.json', out/'original_campaign_design.json')
    with (out/'samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    records = []
    for i, row in enumerate(rows):
        name = row['evaluation_id']
        if not name.endswith(f'-eval-{i:04d}'):
            raise ValueError('Unexpected design ordering')
        src = source/'evaluations'/name
        dst = out/'baseline'/name
        dst.mkdir(parents=True)
        validate_parameters(ET.parse(src/'params.xml').getroot(), row, definition['free_parameters'])
        for file in ['params.xml', 'model_changes.xml', 'galacticus_input_values.json']:
            shutil.copy2(src/file, dst/file)
        with h5py.File(src/'galacticus.hdf5') as old, h5py.File(dst/'galacticus.hdf5', 'w') as new:
            if old.attrs.get('statusCompletion', -1) != 0:
                raise ValueError(f'Original evaluation incomplete: {name}')
            version = {k: text(v) for k,v in old['Version'].attrs.items()}
            for k,v in old.attrs.items():
                new.attrs[k] = v
            for key in ['analyses', 'Parameters', 'Version']:
                old.copy(key, new)
        shutil.copytree(src/'emission_line_dust', dst/'emission_line_dust')
        records.append(dict(index=i, evaluation_id=name, original_version=version,
                            parameters_sha256=sha(dst/'params.xml')))
    write_json(out/'pilot.json', dict(target_revision=None, upstream_branch='master',
               required_pull_requests=list(REQUIRED_PRS),
               selection='Exact first 64 rows of original scrambled Sobol sequence',
               original_campaign=str(source), original_design_sha256=sha(source/'campaign_design.json'),
               samples_sha256=sha(out/'samples.csv'), evaluations=records,
               note='Full upstream update including PRs #1499, #1665 and #1682. EPS seeds retained; identical trees are not guaranteed across versions.'))
    shutil.copy2(__file__, out/'pilot.py')
    shutil.copy2(Path(__file__).with_name('compare_code_update_pilot.py'), out/'compare.py')
    print(out)


def prepare(args):
    pilot = args.pilot.resolve()
    meta = json.loads((pilot/'pilot.json').read_text())
    source = args.source.resolve()
    exe = source/'Galacticus.exe'
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != meta['target_revision']:
        raise ValueError(f'Expected pinned source {meta["target_revision"]}, found {head}')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD'], check=True)
    if set(meta.get('required_fix_commits', {})) != {str(pr) for pr in REQUIRED_PRS}:
        raise ValueError('Pin a merged master containing all required fixes before preparation')
    for fix in meta['required_fix_commits'].values():
        subprocess.run(['git', '-C', str(source), 'merge-base', '--is-ancestor', fix, 'HEAD'], check=True)
    if not exe.is_file():
        raise FileNotFoundError(exe)
    for key in ['GALACTICUS_DATA_PATH', 'SHMR_PATH', 'GALACTICUS_EMU_ROOT']:
        if not os.environ.get(key) or not Path(os.environ[key]).is_dir():
            raise ValueError(f'Set {key} to the existing Caltech directory')
    emu = Path(os.environ['GALACTICUS_EMU_ROOT']).resolve()
    dust = emu/'scripts/process_emission_line_dust_evaluation.py'
    if not dust.is_file():
        raise FileNotFoundError(dust)
    run = pilot/'updated'
    run.mkdir()  # Never overwrite an earlier preparation or run.
    (run/'logs').mkdir()
    rows = list(csv.DictReader((pilot/'samples.csv').open()))
    specs = json.loads((pilot/'original_campaign_design.json').read_text())['free_parameters']
    environment = dict(os.environ)
    environment['GALACTICUS_EXEC_PATH'] = str(source)
    environment['PYTHONPATH'] = str(source/'python') + os.pathsep + environment.get('PYTHONPATH', '')
    checksums = {}
    for record, row in zip(meta['evaluations'], rows, strict=True):
        name = record['evaluation_id']
        old = pilot/'baseline'/name
        dest = run/'evaluations'/name
        dest.mkdir(parents=True)
        if sha(old/'params.xml') != record['parameters_sha256']:
            raise ValueError(f'Archived input changed: {name}')
        root = ET.parse(old/'params.xml').getroot()
        # Historical sibling expression predates a syntax migration; evaluate it exactly.
        bar = one(root, ".//nodeOperator[@value='barInstability']/galacticDynamicsBarInstability")
        disk = one(bar, 'fractionAngularMomentumRetainedDisk')
        if disk.get('value').replace(' ', '') == '=1.0-[.::fractionAngularMomentumRetainedSpheroid]':
            disk.set('value', format(1-float(one(bar, 'fractionAngularMomentumRetainedSpheroid').get('value')), '.17g'))
        write_xml(root, dest/'migration_input.xml')
        with (dest/'migration.log').open('w') as log:
            subprocess.run([sys.executable, str(source/'scripts/aux/parametersMigrate.py'),
                            str(dest/'migration_input.xml'), str(dest/'migrated.xml'), '--validate', 'no'],
                           cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        root = ET.parse(dest/'migrated.xml').getroot()
        for node in root.iter():
            for attr in ['value', 'active']:
                value = node.get(attr)
                if value and (attr == 'active' or value.lstrip().startswith('=')):
                    node.set(attr, re.sub(r'\[([^\]]*)\]', lambda m: '['+re.sub(r':+', '/', m[1])+']', value))
        sf = one(root, "starFormationRateSurfaceDensityDisks[@value='blitz2006']")
        set_child(sf, 'pressureCharacteristic', '35000.0')
        for tabulation in sf.findall('useTabulation'):
            sf.remove(tabulation)
        # The pinned master implements angular-momentum transfer directly.
        # The optional selector exists only on the separate energy-model branch.
        for key in ['outputFileName', 'outputScratchFileName']:
            set_child(root, key, str(dest/'galacticus.hdf5'))
        validate_parameters(root, row, specs)
        write_xml(root, dest/'input.xml')
        checksums[name] = sha(dest/'input.xml')
        shutil.copy2(old/'galacticus_input_values.json', dest/'galacticus_input_values.json')
    config = dict(executable=str(exe), executable_sha256=sha(exe), source_revision=head,
                  input_sha256=checksums, emu_root=str(emu), dust_script_sha256=sha(dust),
                  environment={k: os.environ[k] for k in ['GALACTICUS_DATA_PATH', 'SHMR_PATH', 'GALACTICUS_EMU_ROOT']},
                  python=sys.executable)
    for key in ['GALACTICUS_SED_CALC_PATH', 'GALACTICUS_TOOLS_PATH']:
        if os.environ.get(key):
            config['environment'][key] = os.environ[key]
    # Keep tool binaries accessible while giving each evaluation its own fresh cache.
    config['environment'].setdefault('GALACTICUS_TOOLS_PATH', str(Path(os.environ['GALACTICUS_DATA_PATH'])/'dynamic'))
    write_json(run/'prepared.json', config)
    body = f'''#!/bin/bash
#SBATCH --job-name=galacticus-update64
#SBATCH --partition=expansion
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-63%8
#SBATCH --output={run}/logs/slurm-%A_%a.out
#SBATCH --error={run}/logs/slurm-%A_%a.err
set -euo pipefail
ulimit -c 0
export GFORTRAN_ERROR_DUMPCORE=NO
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
{shlex.quote(sys.executable)} {shlex.quote(str(pilot/'pilot.py'))} run {shlex.quote(str(pilot))} "$SLURM_ARRAY_TASK_ID"
'''
    (run/'submit.sbatch').write_text(body)
    print(f'Prepared {len(rows)} models. Submit task 0 first: sbatch --array=0 {run}/submit.sbatch')


def checked_revision(version, pilot, config, meta):
    if version == meta['target_revision']:
        return {'method': 'output_metadata', 'reported_revision': version}
    if version != 'unknown':
        raise ValueError(f'Executable reports unexpected revision: {version}')
    evidence = json.loads((pilot/'verified_build.json').read_text())
    if (evidence['source_revision'] != meta['target_revision'] or
        evidence['executable_sha256'] != config['executable_sha256'] or
        sha(config['executable']) != evidence['executable_sha256']):
        raise ValueError('Verified build does not match this executable and target')
    return dict(evidence, reported_revision=version)


def verify_build(args):
    pilot = args.pilot.resolve()
    config = json.loads((pilot/'updated/prepared.json').read_text())
    meta = json.loads((pilot/'pilot.json').read_text())
    source = Path(config['executable']).parent
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty or revision != meta['target_revision'] or sha(config['executable']) != config['executable_sha256']:
        raise ValueError('Source revision, source cleanliness, or prepared executable checksum mismatch')
    write_json(pilot/'verified_build.json', dict(source_revision=revision,
        executable_sha256=config['executable_sha256'], method='prepared_checksum_and_clean_checkout',
        verified_at=datetime.now(timezone.utc).isoformat(),
        limitation='Checkout and prepared binary checksum verified; missing embedded revision is not reconstructed.'))
    print('Recorded external build provenance; original output metadata is unchanged.')


def finish_model(args):
    import h5py
    pilot = args.pilot.resolve()
    meta = json.loads((pilot/'pilot.json').read_text())
    config = json.loads((pilot/'updated/prepared.json').read_text())
    record = meta['evaluations'][args.index]
    name = record['evaluation_id']
    dest = pilot/'updated/evaluations'/name
    if (dest/'complete.json').exists():
        raise FileExistsError(dest/'complete.json')
    if sha(config['executable']) != config['executable_sha256'] or sha(dest/'input.xml') != config['input_sha256'][name]:
        raise ValueError('Executable or input changed')
    output = dest/'galacticus.hdf5'
    with h5py.File(output) as f:
        if f.attrs.get('statusCompletion', -1) != 0:
            raise ValueError('Evolution incomplete')
        version = text(f['Version'].attrs['gitHash'])
        evidence = checked_revision(version, pilot, config, meta)
        datasets = text(f['Version'].attrs['gitHashDatasets'])
        if float(f['Parameters/starFormationRateSurfaceDensityDisks'].attrs['pressureCharacteristic']) != 35000:
            raise ValueError('Incorrect Blitz pressure')
    if datasets != config.get('datasets_selection', {}).get('revision', datasets):
        raise ValueError('Unexpected datasets revision')
    script = Path(config['emu_root'])/'scripts/process_emission_line_dust_evaluation.py'
    if sha(script) != config['dust_script_sha256']:
        raise ValueError('Dust script changed')
    dust = dest/'emission_line_dust'
    details = json.loads((dust/'metadata.json').read_text())
    if (details['evaluation_id'] != name or details['n_dust_draws'] != 16 or
        details['scatter_mode'] != 'expected' or details['z_pivot'] != 1.0 or
        Path(details['galacticus_file']).resolve() != output.resolve()):
        raise ValueError('Unexpected dust processing metadata')
    files = ['dust_draws.csv', 'emission_line_dust_lf_long.csv',
             'emission_line_dust_lf_emulator_table.csv', 'metadata.json', 'manifest.csv']
    for filename in files:
        if not (dust/filename).stat().st_size:
            raise ValueError(f'Empty dust output: {filename}')
    with (dust/'dust_draws.csv').open() as stream:
        draws = list(csv.DictReader(stream))
    if {int(row['dust_draw_index']) for row in draws} != set(range(16)) or len(draws) != 16:
        raise ValueError('Incomplete dust draws')
    if f'finished emission-line dust post-processing for {name}' not in (dest/'dust.log').read_text():
        raise ValueError('Dust completion message missing')
    write_json(dest/'complete.json', dict(source_revision=meta['target_revision'],
        reported_source_revision=version, revision_provenance=evidence,
        output_sha256=sha(output), dust_sha256={file: sha(dust/file) for file in files},
        datasets_revision=datasets, historical_datasets_revision=record['original_version']['gitHashDatasets']))
    print(f'Completed {name}; simulation output was not modified.')


def run_model(args):
    import h5py
    pilot = args.pilot.resolve()
    meta = json.loads((pilot/'pilot.json').read_text())
    record = meta['evaluations'][args.index]
    config = json.loads((pilot/'updated/prepared.json').read_text())
    name = record['evaluation_id']
    dest = pilot/'updated/evaluations'/name
    exe = Path(config['executable'])
    if sha(exe) != config['executable_sha256'] or sha(dest/'input.xml') != config['input_sha256'][name]:
        raise ValueError('Executable or prepared parameters changed')
    # Exclusive claim prevents duplicate submissions from overwriting results.
    with (dest/'started.json').open('x') as f:
        json.dump({'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'index': args.index}, f)
    environment = dict(os.environ, **config['environment'])
    environment['GALACTICUS_EXEC_PATH'] = str(exe.parent)
    environment['GALACTICUS_DYNAMIC_DATA_PATH'] = str(dest/'cache')
    (dest/'cache').mkdir()
    output = dest/'galacticus.hdf5'
    if output.exists():
        raise FileExistsError(output)
    def execute(cmd, log):
        with (dest/log).open('w') as f:
            subprocess.run(cmd, cwd=dest, env=environment, stdout=f, stderr=subprocess.STDOUT, check=True)
    execute([str(exe), str(dest/'input.xml'), '--dry-run'], 'dry_run.log')
    execute([str(exe), str(dest/'input.xml'), '--output-processed-parameters', str(dest/'params.xml')], 'galacticus.log')
    with h5py.File(output) as f:
        if f.attrs.get('statusCompletion', -1) != 0:
            raise ValueError('Galacticus did not complete successfully')
        version = text(f['Version'].attrs['gitHash'])
        checked_revision(version, pilot, config, meta)
        if float(f['Parameters/starFormationRateSurfaceDensityDisks'].attrs['pressureCharacteristic']) != 35000:
            raise ValueError('Incorrect Blitz pressure in actual run')
        actual_datasets = text(f['Version'].attrs['gitHashDatasets'])
    script = Path(config['emu_root'])/'scripts/process_emission_line_dust_evaluation.py'
    if sha(script) != config['dust_script_sha256']:
        raise ValueError('Dust postprocessor changed since preparation')
    execute([sys.executable, str(script), str(output), '--evaluation-id', name,
             '--output-dir', str(dest/'emission_line_dust'), '--n-dust-draws', '16',
             '--base-seed', '12345', '--evaluation-index', str(args.index),
             '--scatter-mode', 'expected', '--z-pivot', '1.0',
             '--input-json', str(dest/'galacticus_input_values.json')], 'dust.log')
    finish_model(args)
    print(f'Completed {name}; datasets revision {actual_datasets}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('verify-build')
    s.add_argument('pilot', type=Path)
    s.set_defaults(func=verify_build)
    s = sub.add_parser('finish')
    s.add_argument('pilot', type=Path)
    s.add_argument('index', type=int, choices=range(64))
    s.set_defaults(func=finish_model)
    s = sub.add_parser('stage')
    s.add_argument('campaign', type=Path)
    s.add_argument('output', type=Path)
    s.set_defaults(func=stage)
    s = sub.add_parser('pin-merged-master')
    s.add_argument('pilot', type=Path)
    s.set_defaults(func=pin_merged_master)
    s = sub.add_parser('prepare')
    s.add_argument('pilot', type=Path)
    s.add_argument('--source', type=Path, required=True)
    s.set_defaults(func=prepare)
    s = sub.add_parser('run')
    s.add_argument('pilot', type=Path)
    s.add_argument('index', type=int, choices=range(64))
    s.set_defaults(func=run_model)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
