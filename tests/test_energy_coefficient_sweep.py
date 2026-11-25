"""Matched coordinates, processed-XML checks, and dependent submission.

Written by OpenAI Codex at Andrew Robertson's request.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
from test_energy_precision_pilot import fixture_base, staged
import prepare_energy_precision_pilot as p
import energy_coefficient_sweep as sweep


def test_grid_changes_only_energy_ratios(tmp_path):
    definition, rows = p.design()
    cases = sweep.cases(definition, rows)
    assert len(cases) == 12
    baseline = {}
    for task in cases:
        root = p.model(fixture_base(), task['row'], definition['free_parameters'], 'practical', tmp_path, 1000)
        p.child(p.one(root, p.ENERGY), 'formFactorInteraction', .5*task['beta'])
        task['random_seeds'] = p.seeds(root)
        sweep.verify(root, task, definition)
        assert p.number(p.one(root, p.ENERGY+'/formFactorSpheroid')) == .5*task['alpha']
        canonical = copy.deepcopy(root)
        for name in ('formFactorSpheroid', 'formFactorInteraction'):
            p.one(canonical, p.ENERGY+'/'+name).set('value', 'NORMALIZED')
        text = ET.tostring(canonical)
        assert text == baseline.setdefault(task['design_index'], text)
        p.one(root, p.ENERGY+'/formFactorInteraction').set('value', '100')
        with pytest.raises(ValueError, match='interaction'):
            sweep.verify(root, task, definition)


def test_prepare_run_and_guards(staged, tmp_path, monkeypatch):
    # Reuse the mock Galacticus/migration fixture; no Fortran execution claimed.
    repo = tmp_path/'launcher'
    (repo/'inputs/energy_coefficients').mkdir(parents=True)
    shutil.copytree(p.REPO/'scripts', repo/'scripts')
    shutil.copy2(staged.original/'base_parameters.xml', repo/'inputs/energy_coefficients/historical_base.xml')
    monkeypatch.setattr(sweep, 'BASE_SHA', p.sha(staged.original/'base_parameters.xml'))
    monkeypatch.setattr(sweep, 'REVISION', p.git(staged.source, 'rev-parse', 'HEAD'))
    monkeypatch.setattr(sweep, 'MIGRATION_BASE', sweep.REVISION)
    commands = []
    real_run = subprocess.run
    def recording_run(command, *args, **kwargs):
        commands.append(command)
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', recording_run)
    monkeypatch.setattr(p, 'REPO', repo)
    mask = Path(os.environ['GALACTICUS_DATA_PATH'])/'dynamic'/p.SURVEY_MASK
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b'1 polygons\n'+b' '*1000000)
    root = tmp_path/'coefficients'
    args = SimpleNamespace(root=root, source=staged.source, executable=staged.executable)
    sweep.prepare(args)
    migration = next(c for c in commands if any(str(x).endswith('parametersMigrate.py') for x in c))
    assert migration[migration.index('--lastModifiedRevision')+1] == sweep.MIGRATION_BASE
    assert json.loads((root/'provenance/migration_basis.json').read_text())['migration_base'] == sweep.MIGRATION_BASE
    assert (root/'PREPARED').exists()
    tasks = json.loads((root/'tasks.json').read_text())
    assert len(tasks) == 12
    assert all((root/'validation'/t['evaluation_id']/'SUCCESS').exists() for t in tasks)
    assert len(p.read_rows(root/'tighter/samples.csv')) == 0
    sweep.execute(root, 0)
    folder = root/'practical/evaluations'/tasks[0]['evaluation_id']
    assert json.loads((folder/'run_status.json').read_text())['status'] == 'ok'
    with pytest.raises(FileExistsError):
        sweep.execute(root, 0)
    with pytest.raises(FileExistsError):
        sweep.prepare(args)
    (root/'analyze_campaign_compact_tail.py').write_text('changed')
    with pytest.raises(ValueError, match='Frozen script'):
        sweep.execute(root, 1)


def test_submission_dependencies_without_built_executable(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    bindir = tmp_path/'bin'
    bindir.mkdir()
    sbatch = bindir/'sbatch'
    sbatch.write_text('#!/usr/bin/env python3\nimport json,os,sys\np=os.environ["SBATCH_RECORD"]\nwith open(p,"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(100+len(open(p).readlines()))\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
               SBATCH_RECORD=str(tmp_path/'submitted.jsonl'), GALACTICUS_EXEC_PATH=str(tmp_path/'not-built'),
               GALACTICUS_DATA_PATH=str(tmp_path), SHMR_PATH=str(tmp_path),
               ENERGY_COEFFICIENT_OUTPUT=str(tmp_path/'campaign'))
    subprocess.run(['bash', str(repo/'scripts/submit_energy_coefficients_caltech.sh'), '8135663'], env=env, check=True)
    calls = [json.loads(s) for s in (tmp_path/'submitted.jsonl').read_text().splitlines()]
    assert '--dependency=afterok:8135663' in calls[0]
    assert '--dependency=afterok:101' in calls[1]
    assert '--array=0-11%4' in calls[1]
    assert '--dependency=afterok:101,afterany:102' in calls[2]
    assert all('--kill-on-invalid-dep=yes' in c for c in calls)
    # A completed build may have aged out of Slurm's dependency records.
    executable = tmp_path/'existing.exe'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o755)
    env.update(ENERGY_COEFFICIENT_OUTPUT=str(tmp_path/'retry'), ENERGY_COEFFICIENT_EXECUTABLE=str(executable))
    subprocess.run(['bash', str(repo/'scripts/submit_energy_coefficients_caltech.sh'), 'built'], env=env, check=True)
    calls = [json.loads(s) for s in (tmp_path/'submitted.jsonl').read_text().splitlines()]
    assert not any(arg.startswith('--dependency=') for arg in calls[3])
    assert '--dependency=afterok:104' in calls[4]
