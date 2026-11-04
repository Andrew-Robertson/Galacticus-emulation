#!/usr/bin/env python3
"""Isolated diagnostic retry for the eval-0026 covariance failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

REVISION = '33645e68d767e9f24f51e4380c8c1666f3db19f7'
SOURCE_FILE = 'source/output/analyses/mean_function_1d.F90'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def patched_source(source):
    start = source.index('  subroutine meanFunction1DFinalizeAnalysis(self)')
    end = source.index('  end subroutine meanFunction1DFinalizeAnalysis', start)
    body = source[start:end]
    marker = '    character       (len=43                      )                                :: label'
    assert body.count(marker) == 1
    body = body.replace(marker, marker+'\n    character(len=512) :: guardMessage')
    marker = '             if (unweightedValue(j) > 0.0d0) then\n'
    assert body.count(marker) == 1
    guard = '''                ! Diagnostic guard: avoid squaring weights into underflow.
                ! Means remain unchanged; skipped covariance entries remain zero.
                ! These entries must be reviewed before using model covariances.
                if (min(unweightedValue(i),unweightedValue(j)) <= sqrt(tiny(1.0d0))) then
                   write(guardMessage,'(a,a,a,2i6,a,2es24.16)') &
                        'COVARIANCE_WEIGHT_GUARD analysis=',char(self%label), &
                        ' bins=',i,j,' weights=',unweightedValue(i),unweightedValue(j)
                   call displayMessage(trim(guardMessage))
                   cycle
                end if
'''
    body = body.replace(marker, marker+guard)
    return source[:start]+body+source[end:]


def prepare(pilot):
    config = json.loads((pilot/'updated/prepared.json').read_text())
    original = pilot/'updated/evaluations/sobol_1024_20p_simpleSizes_moreHalos-eval-0026'
    if sha(original/'input.xml') != config['input_sha256'][original.name]:
        raise ValueError('Original input changed')
    retry = pilot/'covariance_retry_0026'
    retry.mkdir()  # Never overwrite an earlier attempt.
    source = retry/'source'
    subprocess.run(['git', '-C', str(pilot/'source'), 'worktree', 'add', '--detach', str(source), REVISION], check=True)
    file = source/SOURCE_FILE
    file.write_text(patched_source(file.read_text()))
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--', SOURCE_FILE], text=True)
    (retry/'covariance_guard.patch').write_text(patch)
    run = retry/'run'
    run.mkdir()
    # Preserve all physics parameters; redirect paths into the separate attempt.
    xml = (original/'input.xml').read_text().replace(str(original), str(run))
    root = ET.fromstring(xml)
    outputs = list(root.iter('outputFileName'))
    if len(outputs) != 1:
        raise ValueError('Expected exactly one outputFileName')
    outputs[0].set('value', str(run/'galacticus.hdf5'))
    ET.ElementTree(root).write(run/'input.xml', encoding='utf-8', xml_declaration=True)
    (retry/'retry.json').write_text(json.dumps(dict(pilot=str(pilot), environment=config['environment'],
        original_input_sha256=sha(original/'input.xml'), retry_input_sha256=sha(run/'input.xml'),
        base_revision=REVISION, patched_source_sha256=sha(file), patch_sha256=sha(retry/'covariance_guard.patch'),
        note='Diagnostic covariance guard; original evolution parameters retained. Guarded covariance entries require review.'), indent=2))
    script = retry/'retry.py'
    script.write_text(Path(__file__).read_text())
    (retry/'submit.sbatch').write_text(f'''#!/bin/bash
#SBATCH --job-name=update64-covariance-0026
#SBATCH --partition=expansion
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --output={retry}/slurm-%j.out
#SBATCH --error={retry}/slurm-%j.err
set -euo pipefail
ulimit -c 0
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
{shlex.quote(sys.executable)} {shlex.quote(str(script))} execute {shlex.quote(str(retry))}
''')
    print('Prepared:', retry)
    print('Build source/Galacticus.exe on an interactive compute node, then submit submit.sbatch.')


def execute(retry):
    import h5py
    config = json.loads((retry/'retry.json').read_text())
    source, run = retry/'source', retry/'run'
    if sha(source/SOURCE_FILE) != config['patched_source_sha256'] or sha(run/'input.xml') != config['retry_input_sha256']:
        raise ValueError('Retry source/input changed')
    with (retry/'started.json').open('x') as stream:
        json.dump(dict(executable_sha256=sha(source/'Galacticus.exe'), job_id=os.environ.get('SLURM_JOB_ID')), stream)
    (run/'cache').mkdir()
    env = dict(os.environ, **config['environment'])
    env.update(GALACTICUS_EXEC_PATH=str(source), GALACTICUS_DYNAMIC_DATA_PATH=str(run/'cache'))
    for name, extra in [('dry_run.log', ['--dry-run']), ('galacticus.log', ['--output-processed-parameters', str(run/'params.xml')])]:
        with (run/name).open('w') as stream:
            subprocess.run([str(source/'Galacticus.exe'), str(run/'input.xml'), *extra], cwd=run, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    with h5py.File(run/'galacticus.hdf5') as f:
        status = int(f.attrs.get('statusCompletion', -1))
    if status != 0:
        raise ValueError('Retry evolution/finalization incomplete')
    lines = [line for line in (run/'galacticus.log').read_text(errors='replace').splitlines() if 'COVARIANCE_WEIGHT_GUARD' in line]
    (retry/'guard_report.txt').write_text('\n'.join(lines)+'\n')
    (retry/'simulation_finished.json').write_text(json.dumps(dict(statusCompletion=status, guard_messages=len(lines))))
    print('Simulation finished. Guard messages:', len(lines))
    print('Review guard_report.txt before dust processing or accepting this retry into the pilot.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'execute'])
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else execute)(args.path.resolve())
