"""Full-campaign launch boundaries and preservation of pilot provenance."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('full_campaign', scripts/'prepare_code_update_campaign.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_prepare_submits_only_failed_pilot_case_and_records_dataset(tmp_path, monkeypatch):
    data = tmp_path/'datasets'
    unit = data/'static/simulations/UNIT.xml'
    unit.parent.mkdir(parents=True)
    unit.write_text('<simulation/>')
    inherited = dict(environment={'GALACTICUS_DATA_PATH':str(data)}, datasets_selection={'revision':c.DATASETS})
    (tmp_path/'inherited_environment.json').write_text(json.dumps(inherited))
    (tmp_path/'pilot.json').write_text(json.dumps({'target_revision':c.REVISION}))
    for name in ['campaign.py','prepare_code_update_pilot.py']:
        (tmp_path/name).write_text('script')
    def prepare(args):
        updated = tmp_path/'updated'
        updated.mkdir()
        (updated/'prepared.json').write_text('{}')
        (updated/'submit.sbatch').write_text(f'#SBATCH --job-name=galacticus-update64\n#SBATCH --array=0-63%8\npython {tmp_path}/pilot.py run')
    monkeypatch.setattr(c.pilot,'prepare',prepare)
    monkeypatch.setattr(c.pilot,'verify_build',lambda a: None)
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**k: c.DATASETS+'\n')
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k: None)
    c.prepare(SimpleNamespace(pilot=tmp_path,source=tmp_path/'source'))
    batch = (tmp_path/'updated/submit.sbatch').read_text()
    assert '#SBATCH --array=26' in batch
    assert str(tmp_path/'campaign.py') in batch
    assert 'update1024' in batch and 'pilot.py run' not in batch
    config = json.loads((tmp_path/'updated/prepared.json').read_text())
    assert config['datasets_selection']['revision'] == c.DATASETS
    assert config['unit_xml_sha256'] == c.pilot.sha(unit)


def test_run_rejects_changed_launcher_before_execution(tmp_path, monkeypatch):
    (tmp_path/'updated').mkdir()
    (tmp_path/'campaign.py').write_text('changed')
    (tmp_path/'updated/prepared.json').write_text(json.dumps({'launcher_sha256':{'campaign.py':'wrong'}}))
    with pytest.raises(ValueError,match='Launcher changed'):
        c.run(SimpleNamespace(pilot=tmp_path,index=1023))


def test_cli_accepts_last_model(tmp_path, monkeypatch):
    seen=[]
    monkeypatch.setattr(c,'run',lambda args: seen.append(args.index))
    monkeypatch.setattr(sys,'argv',['campaign.py','run',str(tmp_path),'1023'])
    c.main()
    assert seen == [1023]


def test_only_documented_blitz_conversion_is_allowed():
    import xml.etree.ElementTree as ET
    old = 3.978380036836035e-9
    root = ET.fromstring('<parameters><galacticStructureSolver value="simple"/><frequency value="2.9252794388500254e-9"/><other value="2"/></parameters>')
    row = {'diskSFRFrequencyNorm':old, 'other':2}
    specs = [{'short_name':'diskSFRFrequencyNorm','path':'frequency'}, {'short_name':'other','path':'other'}]
    scales = {'diskSFRFrequencyNorm':1/1.36}
    c.pilot.validate_parameters(root,row,specs,scales)
    with pytest.raises(ValueError,match='diskSFRFrequencyNorm'):
        c.pilot.validate_parameters(root,row,specs)
    root.find('frequency').set('value',str(old/1.36**2))
    with pytest.raises(ValueError,match='diskSFRFrequencyNorm'):
        c.pilot.validate_parameters(root,row,specs,scales)
    root.find('frequency').set('value',str(old/1.36))
    root.find('other').set('value','3')
    with pytest.raises(ValueError,match='other'):
        c.pilot.validate_parameters(root,row,specs,scales)
