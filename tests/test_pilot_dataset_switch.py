import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import pytest

path = Path(__file__).resolve().parents[1]/'scripts/switch_pilot_datasets.py'
spec = importlib.util.spec_from_file_location('switch_pilot_datasets', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_active_job_blocks_switch_before_modifications(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'command', lambda *args: 'galacticus-update64')
    with pytest.raises(RuntimeError, match='still queued/running'):
        module.switch(tmp_path/'pilot', tmp_path/'datasets')
    assert list(tmp_path.iterdir()) == []


def test_switch_preserves_attempt_and_rearms_identical_parameter_points(tmp_path, monkeypatch):
    pilot, data = tmp_path/'pilot', tmp_path/'datasets'
    patches = data/'static/patches/FSPS'
    patches.mkdir(parents=True)
    for name in ['ssp_gen.f90.patch', 'sps_vars.f90.patch', 'Makefile.patch']:
        (patches/name).write_text('patch')
    (data/'dynamic').mkdir()
    with gzip.open(data/'dynamic/FSPS_4.0.tar.gz', 'wb') as f:
        f.write(b'validated download fixture')
    updated = pilot/'updated'
    (updated/'logs').mkdir(parents=True)
    (updated/'logs/job.err').write_text('old failure')
    (updated/'submit.sbatch').write_text('old script')
    exe = pilot/'Galacticus.exe'
    exe.write_text('unchanged executable')
    names = [f'eval-{i:04d}' for i in range(64)]
    checks = {}
    for i, name in enumerate(names):
        folder = updated/'evaluations'/name
        folder.mkdir(parents=True)
        (folder/'input.xml').write_text(f'<parameters><value value="{i}"/>'
            '<nodeOperator value="barInstability"><barInstabilitySpheroidAngularMomentum '
            'value="angularMomentumTransfer"/></nodeOperator></parameters>')
        (folder/'galacticus_input_values.json').write_text(json.dumps({'value': i}))
        checks[name] = module.sha(folder/'input.xml')
    first = updated/'evaluations'/names[0]
    (first/'started.json').write_text('{}')
    (first/'complete.json').write_text('{}')
    (first/'galacticus.hdf5').write_text('preserved output')
    config = dict(executable=str(exe), executable_sha256=module.sha(exe), input_sha256=checks,
                  environment={'GALACTICUS_DATA_PATH':'old', 'GALACTICUS_TOOLS_PATH':'old-tools'},
                  fsps_repair_provenance='prior overlay')
    (updated/'prepared.json').write_text(json.dumps(config))
    original = (updated/'prepared.json').read_bytes()
    (pilot/'pilot.json').write_text(json.dumps({'evaluations':[{'evaluation_id':n} for n in names]}))
    monkeypatch.setattr(module, 'command', lambda *args: '' if args[0]=='squeue' else module.DATASETS_REVISION)
    run = subprocess.run
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs:
        subprocess.CompletedProcess(args, 0) if args[0]=='git' else run(args, **kwargs))
    module.switch(pilot, data)
    new = json.loads((updated/'prepared.json').read_text())
    saved = Path(new['datasets_selection']['previous_attempt'])
    assert (saved/'prepared.json').read_bytes() == original
    assert (saved/'evaluations'/names[0]/'galacticus.hdf5').read_text() == 'preserved output'
    assert not (first/'started.json').exists()
    assert not (first/'complete.json').exists()
    assert new['environment']['GALACTICUS_DATA_PATH'] == str(data)
    assert new['environment']['GALACTICUS_TOOLS_PATH'] == str(data/'dynamic')
    assert 'fsps_repair_provenance' not in new
    for i, name in enumerate(names):
        path = updated/'evaluations'/name/'input.xml'
        root = ET.parse(path).getroot()
        assert root.find('value').get('value') == str(i)
        assert root.find('.//barInstabilitySpheroidAngularMomentum') is None
        assert module.sha(path) == new['input_sha256'][name]
    with pytest.raises(ValueError, match='already switched'):
        module.switch(pilot, data)
