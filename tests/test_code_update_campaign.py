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


def test_prepare_submits_requested_full_array_and_records_dataset(tmp_path, monkeypatch):
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
    assert '#SBATCH --array=0-1023%128' in batch
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


def test_restore_original_prior_only_after_verified_migration():
    import xml.etree.ElementTree as ET
    old = '3.978380036836035e-9'
    root = ET.fromstring('<parameters><galacticStructureSolver value="simple"/><frequency value="2.9252794388500254e-9"/><other value="2"/></parameters>')
    row = {'diskSFRFrequencyNorm':old, 'other':2}
    specs = [{'short_name':'diskSFRFrequencyNorm','path':'frequency'}, {'short_name':'other','path':'other'}]
    c.retain_numerical_frequency_prior(root,row,specs)
    assert root.find('frequency').get('value') == old
    c.pilot.validate_parameters(root,row,specs)
    # A repeated conversion/restoration must fail rather than silently hide an unexpected migration.
    with pytest.raises(ValueError,match='diskSFRFrequencyNorm'):
        c.retain_numerical_frequency_prior(root,row,specs)


def test_original_version_requires_completed_historical_output(tmp_path):
    import h5py
    fpath = tmp_path/'galacticus.hdf5'
    with h5py.File(fpath,'w') as f:
        f.attrs['statusCompletion'] = 0
        f.create_group('Version').attrs['gitHashDatasets'] = 'historical-datasets'
    assert c.original_version(fpath)['gitHashDatasets'] == 'historical-datasets'
    with h5py.File(fpath,'r+') as f:
        f.attrs['statusCompletion'] = 1
    with pytest.raises(ValueError,match='incomplete'):
        c.original_version(fpath)


def test_finish_full_campaign_record_preserves_simulation(tmp_path):
    import h5py
    root=tmp_path
    name='model0026'
    dest=root/'updated/evaluations'/name
    dest.mkdir(parents=True)
    (dest/'input.xml').write_text('<parameters/>')
    exe=root/'Galacticus.exe'
    exe.write_bytes(b'executable')
    dust_script=root/'scripts/process_emission_line_dust_evaluation.py'
    dust_script.parent.mkdir()
    dust_script.write_text('processor')
    result=dest/'galacticus.hdf5'
    with h5py.File(result,'w') as f:
        f.attrs['statusCompletion']=0
        v=f.create_group('Version')
        v.attrs['gitHash']='new-code'
        v.attrs['gitHashDatasets']='new-datasets'
        f.create_group('Parameters/starFormationRateSurfaceDensityDisks').attrs['pressureCharacteristic']=35000.
    before=c.pilot.sha(result)
    config=dict(executable=str(exe),executable_sha256=c.pilot.sha(exe),
        input_sha256={name:c.pilot.sha(dest/'input.xml')},emu_root=str(root),
        dust_script_sha256=c.pilot.sha(dust_script),datasets_selection={'revision':'new-datasets'})
    (root/'updated/prepared.json').write_text(json.dumps(config))
    (root/'pilot.json').write_text(json.dumps(dict(target_revision='new-code',evaluations=[dict(
        evaluation_id=name,original_version={'gitHashDatasets':'old-datasets'})])))
    dust=dest/'emission_line_dust'
    dust.mkdir()
    (dust/'metadata.json').write_text(json.dumps(dict(evaluation_id=name,n_dust_draws=16,
        scatter_mode='expected',z_pivot=1.,galacticus_file=str(result))))
    (dust/'dust_draws.csv').write_text('dust_draw_index\n'+''.join(f'{i}\n' for i in range(16)))
    for f in ['emission_line_dust_lf_long.csv','emission_line_dust_lf_emulator_table.csv','manifest.csv']:
        (dust/f).write_text('data\n')
    (dest/'dust.log').write_text(f'finished emission-line dust post-processing for {name}')
    c.pilot.finish_model(SimpleNamespace(pilot=root,index=0))
    done=json.loads((dest/'complete.json').read_text())
    assert done['historical_datasets_revision']=='old-datasets'
    assert done['datasets_revision']=='new-datasets'
    assert before == c.pilot.sha(result)
