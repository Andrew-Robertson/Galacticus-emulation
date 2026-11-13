"""Regression checks for physical-coordinate preservation and portable execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_energy_precision_pilot as pilot
import analyze_energy_precision_pilot as analysis


def fixture_base():
    definition,_=pilot.design()
    root=ET.Element('parameters')
    paths=[s['path'] for s in definition['free_parameters'] if not s['path'].startswith(pilot.ENERGY)]
    paths += [pilot.ACCRETION, pilot.DYNAMICS,
        "mergerTreeOutputter/mergerTreeOutputter[@value='standard']",
        "nodePropertyExtractor[@value='multi']",'mergerTreeBuildMasses/treesPerDecade',
        'galacticStructureSolver','darkMatterProfile','mergerTreeConstructor',
        'mergerTreeNodeEvolver/odeToleranceRelative','mergerTreeNodeEvolver/odeToleranceAbsolute',
        'randomNumberGenerator/seed','outputTimes/redshifts','outputAnalysis']
    for path in paths:
        node=root
        for step in path.split('/'):
            e=node.find(step)
            if e is None:
                m=re.fullmatch(r"(\w+)(?:\[@value='([^']+)'])?",step);assert m
                e=ET.SubElement(node,m[1],value=m[2] or '1.0')
            node=e
    for tag,value in [('mergerTreeConstructor','build'),('galacticStructureSolver','simple'),
                      ('darkMatterProfile','adiabaticGnedin2004'),('starFormationRateSurfaceDensityDisks','blitz2006'),
                      ('outputTimes/redshifts','2 1 0'),('randomNumberGenerator/seed','42')]:
        pilot.one(root,tag).set('value',value)
    ET.SubElement(root,'lastModified',revision='original')
    ET.SubElement(root,'version').text='0.9.4'
    return root


def canonical(root):
    return ET.tostring(root)


def test_every_physical_point_preserved_and_only_precision_differs(tmp_path):
    definition,rows=pilot.design();base=fixture_base();old=canonical(base)
    ids=pilot.selection(pilot.FAMILY)
    assert len(ids)==len(set(ids))==32 and 18 in ids
    assert json.loads((pilot.FAMILY/'precision_pilot_selection.json').read_text())['records'][0]['design_index']==0
    for row in rows:
        p=pilot.model(base,row,definition['free_parameters'],'practical',tmp_path)
        t=pilot.model(base,row,definition['free_parameters'],'tighter',tmp_path)
        for spec in definition['free_parameters']:
            assert pilot.number(pilot.one(p,spec['path']))==float(row[spec['short_name']])
        pilot.one(t,pilot.ENERGY+'/stepFiniteDifferenceRelative').set('value',str(pilot.SETTINGS['practical']['fd']))
        pilot.one(t,'galacticStructureSolver/solutionTolerance').set('value','1e-07')
        assert canonical(p)==canonical(t)
        assert pilot.one(p,'randomNumberGenerator/seed').get('value')=='42'
        assert pilot.one(p,'outputTimes/redshifts').get('value')=='2 1 0'
    assert canonical(base)==old


def test_wrong_processed_coordinates_and_obsolete_options_rejected(tmp_path):
    definition,rows=pilot.design();row=rows[0]
    root=pilot.model(fixture_base(),row,definition['free_parameters'],'practical',tmp_path)
    pilot.child(pilot.one(root,pilot.ENERGY),'finiteDifferenceStepRelative',.01)
    with pytest.raises(ValueError,match='Obsolete'):pilot.verify_parameters(root,row,definition['free_parameters'],'practical')
    pilot.one(root,pilot.ENERGY).remove(pilot.one(root,pilot.ENERGY+'/finiteDifferenceStepRelative'))
    pilot.one(root,definition['free_parameters'][0]['path']).set('value','0')
    with pytest.raises(ValueError,match='coordinate'):pilot.verify_parameters(root,row,definition['free_parameters'],'practical')


@pytest.fixture
def staged(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    old=tmp_path/'historical';old.mkdir()
    definition,rows=pilot.design()
    pilot.xml(fixture_base(),old/'base_parameters.xml')
    pilot.dump(old/'run_manifest.json',{'base_parameters_sha256':pilot.sha(old/'base_parameters.xml')})
    pilot.write_rows(old/'samples.csv',rows)
    model=source/'source/galactic_dynamics/bar_instability/spheroid_angular_momentum/binding_energy.F90'
    model.parent.mkdir(parents=True);model.write_text('<name>stepFiniteDifferenceRelative</name>')
    migration=source/'scripts/aux/parametersMigrate.py';migration.parent.mkdir(parents=True)
    migration.write_text("import sys,subprocess,xml.etree.ElementTree as E\nt=E.parse(sys.argv[1]);t.getroot().find('lastModified').set('revision',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());t.write(sys.argv[2])\n")
    exe=source/'Galacticus.exe'
    exe.write_text(f'#!{sys.executable}\n'+'''import sys,os,subprocess,xml.etree.ElementTree as E
from pathlib import Path
import h5py
root=E.parse(sys.argv[1]).getroot()
out=sys.argv[sys.argv.index('--output-processed-parameters')+1]
E.ElementTree(root).write(out)
if '--dry-run' not in sys.argv:
 with h5py.File(root.find('outputFileName').get('value'),'w') as f:
  f.attrs['statusCompletion']=0
  f.create_group('Outputs');f.create_group('analyses')
  f.create_group('Version').attrs['gitHash']=subprocess.check_output(['git','-C',os.environ['GALACTICUS_EXEC_PATH'],'rev-parse','HEAD'],text=True).strip()
''')
    exe.chmod(0o755)
    for command in [['git','init','-q'],['git','add','.'],['git','-c','user.name=Test','-c','user.email=test@example.org','commit','-qm','fixture']]:
        subprocess.run(command,cwd=source,check=True)
    revision=pilot.git(source,'rev-parse','HEAD');monkeypatch.setattr(pilot,'MERGE',revision)
    data=tmp_path/'datasets';data.mkdir();shmr=tmp_path/'shmr';shmr.mkdir()
    monkeypatch.setenv('GALACTICUS_DATA_PATH',str(data));monkeypatch.setenv('SHMR_PATH',str(shmr))
    # Non-Git dataset trees are allowed and explicitly record a null revision.
    args=SimpleNamespace(family=pilot.FAMILY,original=old,source=source,executable=exe,
                         output=tmp_path/'runs',partition='obs',cpus=8,concurrency=0,walltime='72:00:00')
    pilot.prepare(args)
    return args


def test_staging_execution_validation_and_no_overwrite(staged):
    root=staged.output
    tasks=json.loads((root/'tasks.json').read_text())
    assert len(tasks)==160 and sum(t['level']=='practical' for t in tasks)==128
    assert '#SBATCH --array=0-159\n' in (root/'submit.slurm').read_text()
    assert '--nodelist' not in (root/'submit.slurm').read_text()
    assert (root/'analyze_energy_precision_pilot.py').is_file()
    for script in list(root.glob('*.slurm'))+list(root.glob('*/evaluations/*/run_eval.sh')):
        subprocess.run(['bash','-n',str(script)],check=True)
    pilot.validate(SimpleNamespace(root=root,task=None))
    assert len(list(root.glob('validation/*/params_used.xml')))==2
    pilot.run(SimpleNamespace(root=root,task=0))
    folder=root/'practical/evaluations/pilot_128-eval-0000'
    assert (folder/'SUCCESS').is_file() and not (folder/'RUNNING').exists()
    with pytest.raises(FileExistsError):pilot.run(SimpleNamespace(root=root,task=0))
    with pytest.raises(FileExistsError):pilot.prepare(staged)
    # A changed executable must be rejected before any simulation is started.
    with (root/'executable/Galacticus.exe').open('a') as f:f.write('\n# changed\n')
    with pytest.raises(ValueError,match='Executable snapshot'):pilot.run(SimpleNamespace(root=root,task=1))


def test_matched_bins_use_tighter_stellar_mass_and_preserve_outliers():
    n=100
    fine=dict(mergerTreeIndex=np.zeros(n,dtype=int),nodeIndex=np.arange(n),spheroidRadius=np.ones(n),
              spheroidMassStellar=np.full(n,3e8),diskMassStellar=np.full(n,7e8),weight=np.ones(n))
    practical={k:v.copy() for k,v in fine.items()}
    practical['spheroidMassStellar'][:]=3e7
    practical['spheroidRadius'][-1]=1e-3
    result=analysis.matched(practical,fine)[1]
    assert result['log_mass_low']==8 and result['matched_count']==100
    assert result['mean']==pytest.approx(-.03)
    assert result['rms']==pytest.approx(.3)
    assert result['abs_p90']==0
    assert result['rms_squared_fraction_from_largest_1percent']==pytest.approx(1)
    practical['nodeIndex'][1]=0
    with pytest.raises(ValueError,match='Duplicate'):analysis.matched(practical,fine)


def test_reduction_reports_failed_runs_without_zero_filling(staged,tmp_path):
    # Simulate one completed catalog and leave the others missing. The toy output
    # has no galaxy snapshots, so diagnostic failures should also remain visible.
    pilot.run(SimpleNamespace(root=staged.output,task=0))
    dest=tmp_path/'reduced'
    with pytest.raises(SystemExit,match='failed'):analysis.analyze(SimpleNamespace(root=staged.output,output=dest))
    report=json.loads((dest/'analysis/summary.json').read_text())
    assert report['successful_runs']==1 and report['expected_runs']==160
    assert len(analysis.rows(dest/'analysis/status.csv'))==160
    assert len(analysis.rows(dest/'analysis/population_differences.csv'))==0
    import h5py
    with h5py.File(dest/'practical/evaluations/pilot_128-eval-0000/galacticus.hdf5') as f:
        assert 'Outputs' not in f and 'analyses' in f


def test_completion_requires_real_lowercase_analysis_group(tmp_path):
    import h5py
    path=tmp_path/'catalog.hdf5'
    with h5py.File(path,'w') as f:
        f.attrs['statusCompletion']=0
        f.create_group('Version').attrs['gitHash']='revision'
        f.create_group('Outputs')
        f.create_group('Analyses')
    with pytest.raises(ValueError,match='Missing catalogs'):
        pilot.completion(path,'revision')
    with h5py.File(path,'a') as f:f.move('Analyses','analyses')
    pilot.completion(path,'revision')
    with h5py.File(path,'a') as f:f.attrs['statusCompletion']=1
    with pytest.raises(ValueError,match='did not complete'):
        pilot.completion(path,'revision')


def test_recovery_preserves_data_and_original_failure(staged):
    import h5py
    import revalidate_energy_precision_pilot as recovery
    root=staged.output
    pilot.run(SimpleNamespace(root=root,task=0))
    task=json.loads((root/'tasks.json').read_text())[0]
    folder=root/task['level']/'evaluations'/task['evaluation_id']
    path=folder/'galacticus.hdf5'
    with h5py.File(path,'a') as f:
        f['Outputs'].create_group('Output1')
        f['analyses'].create_group('exampleAnalysis')
    status_path=folder/'run_status.json'
    status=json.loads(status_path.read_text())
    status.update(status='failed',error=recovery.OLD_ERROR)
    pilot.dump(status_path,status)
    original=status_path.read_bytes(); data_hash=pilot.sha(path)
    frozen_hash=pilot.sha(root/'pilot.py')
    (folder/'SUCCESS').unlink()
    recovery.revalidate(SimpleNamespace(root=root,recover=False))
    assert status_path.read_bytes()==original and not (folder/'SUCCESS').exists()
    recovery.revalidate(SimpleNamespace(root=root,recover=True))
    assert (folder/'SUCCESS').exists()
    assert (folder/'run_status_before_validation_recovery.json').read_bytes()==original
    assert json.loads(status_path.read_text())['status']=='ok'
    assert pilot.sha(path)==data_hash and pilot.sha(root/'pilot.py')==frozen_hash
    recovery.revalidate(SimpleNamespace(root=root,recover=True))
    report=json.loads((root/'validation_recovery/recovery.json').read_text())
    assert report['counts']['previously_recovered']==1


def test_recovery_does_not_accept_incomplete_catalog(staged):
    import h5py
    import revalidate_energy_precision_pilot as recovery
    root=staged.output
    pilot.run(SimpleNamespace(root=root,task=0))
    task=json.loads((root/'tasks.json').read_text())[0]
    folder=root/task['level']/'evaluations'/task['evaluation_id']
    with h5py.File(folder/'galacticus.hdf5','a') as f:f.attrs['statusCompletion']=1
    status=json.loads((folder/'run_status.json').read_text())
    status.update(status='failed',error=recovery.OLD_ERROR)
    pilot.dump(folder/'run_status.json',status)
    (folder/'SUCCESS').unlink()
    with pytest.raises(SystemExit,match='failed validation'):
        recovery.revalidate(SimpleNamespace(root=root,recover=True))
    assert not (folder/'SUCCESS').exists()
    assert json.loads((folder/'run_status.json').read_text())['status']=='failed'
