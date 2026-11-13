#!/usr/bin/env python3
"""Repeat the frozen energy pilot with explicit structural accuracy settings."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
FAMILY = REPO / 'runs/campaigns/equilibriumSizesEnergy'
MERGE = 'c0c8731b0b689cf4827229ee8a66f6600f768e42'
BAR = "nodeOperator/nodeOperator[@value='barInstability']"
DYNAMICS = BAR + '/galacticDynamicsBarInstability'
ENERGY = BAR + '/barInstabilitySpheroidAngularMomentum'
ACCRETION = "nodeOperator/nodeOperator[@value='haloAngularMomentumInterpolate']"
SETTINGS = {'practical': {'fd': 10**-3.5, 'equilibrium': 1e-7},
            'tighter': {'fd': 1e-4, 'equilibrium': 1e-8}}
OLD_NAMES = {'finiteDifferenceStepRelative', 'interactionEnergyFactor',
             'spheroidBaryonFractionTransitionMinimum', 'spheroidBaryonFractionTransitionMaximum',
             'finiteDifferenceStepLimitSpheroid'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def one(root, path):
    found = root.findall(path)
    if len(found) != 1:
        raise ValueError(f'Expected one {path}, found {len(found)}')
    return found[0]


def child(parent, name, value):
    found = parent.findall(name)
    if len(found) > 1:
        raise ValueError(f'Duplicate {name}')
    e = found[0] if found else ET.SubElement(parent, name)
    e.set('value', str(value))
    return e


def xml(root, path):
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def number(e):
    return float(e.get('value').replace('D', 'e').replace('d', 'e'))


def design(family=FAMILY):
    definition = json.loads((family/'family_definition.json').read_text())
    if sha(family/'design_1024.csv') != definition['design_sha256']:
        raise ValueError('Frozen design checksum mismatch')
    rows = read_rows(family/'design_1024.csv')[:128]
    if len(rows) != 128 or [int(r['design_index']) for r in rows] != list(range(128)):
        raise ValueError('Expected the original first 128 ordered design points')
    return definition, rows


def select(args):
    """One-time reproducible selection from historical diagnostics, then freeze it."""
    import numpy as np
    definition, rows = design(args.family)
    source = args.historical.resolve()
    tails_path = source/'analysis/compact_tail_z0/mass_bin_summary.csv'
    slopes_path = source/'scratch/prior_observable_comparison/size_slopes/individual_model_slopes.csv'
    coords = np.array([[float(r[s['short_name']+'_quantile']) for s in definition['free_parameters']] for r in rows])
    chosen, reasons = [], {}

    def add(i, reason):
        if i in chosen:
            reasons[i].append(reason)
            return False
        chosen.append(i); reasons[i] = [reason]
        return True

    def spread():
        distance = ((coords[:, None]-coords[chosen][None])**2).sum(axis=2).min(axis=1)
        distance[chosen] = -1
        return int(np.argmax(distance))

    add(0, 'space-filling seed')
    while len(chosen) < 16:
        add(spread(), 'maximin distance in 21-dimensional prior-quantile coordinates')
    broad = list(chosen)
    add(18, 'historical ODE failure')
    for key in ['barSpheroidFormFactor', 'barFractionAngularMomentumRetainedDisk']:
        order = sorted(range(128), key=lambda i: float(rows[i][key]))
        add(order[0], key+' minimum'); add(order[-1], key+' maximum')
    tails = {}
    for r in read_rows(tails_path):
        if float(r['log_mass_low']) == 8.25:
            i = int(r['evaluation_id'].rsplit('-',1)[1]); v = float(r['below_minus1'])
            if np.isfinite(v): tails[i] = v
    added = 0
    for i in sorted(tails, key=lambda i: (-tails[i], i)):
        if add(i, f'large compact fraction in spheroid mass bin 8.25-8.75: {tails[i]:.8g}'):
            added += 1
        if added == 5: break
    slopes = {}
    for r in read_rows(slopes_path):
        if (r['mode']=='full_range' and r['campaign']=='Equilibrium E (21p)' and r['complete']=='True'
                and r['analysis'] in [f'stellarSizeMassRelationvanDerWel2014Sample{i}' for i in (1,3,5)]):
            i = int(r['evaluation_id'].rsplit('-',1)[1]); v = float(r['beta'])
            if np.isfinite(v): slopes[i] = min(v, slopes.get(i, v))
    added = 0
    for i in sorted(slopes, key=lambda i: (slopes[i], i)):
        if add(i, f'negative star-forming size slope: minimum over three epochs {slopes[i]:.8g}'):
            added += 1
        if added == 3: break
    while len(chosen) < 32:
        add(spread(), 'additional maximin coverage after targeted selection')
    if len(chosen) != 32:
        raise ValueError('Selection must contain 32 distinct points')
    if args.output.exists(): raise FileExistsError(args.output)
    dump(args.output, dict(schema_version=1, master_design_sha256=definition['design_sha256'],
        selection='16 maximin points, then ODE failure, coefficient extremes, five compact-tail cases, three negative-slope cases; fill remaining slots by maximin',
        diagnostics={str(p.relative_to(source)):sha(p) for p in [tails_path,slopes_path]},
        records=[dict(design_index=i, group='coverage' if i in broad else 'targeted', reasons=reasons[i]) for i in chosen]))
    print('Frozen tighter comparison IDs:', ', '.join(map(str, chosen)))


def selection(family):
    definition, _ = design(family)
    config = json.loads((family/'precision_pilot_selection.json').read_text())
    ids = [r['design_index'] for r in config['records']]
    if (config['master_design_sha256'] != definition['design_sha256'] or len(ids)!=32
            or len(set(ids))!=32 or any(type(i)!=int or not 0<=i<128 for i in ids)):
        raise ValueError('Invalid frozen tighter selection')
    return ids


def model(base, row, specs, level, folder):
    root = copy.deepcopy(base)
    if one(root, 'mergerTreeConstructor').get('value') != 'build':
        raise ValueError('Expected EPS mergerTreeConstructor=build')
    if one(root, 'darkMatterProfile').get('value') != 'adiabaticGnedin2004':
        raise ValueError('Expected Gnedin contraction')
    if one(root,'starFormationRateSurfaceDensityDisks').get('value') != 'blitz2006':
        raise ValueError('Expected Blitz2006 disk star formation')
    if not root.findall('.//randomNumberGenerator/seed'):
        raise ValueError('Expected explicit random seeds')
    solver = one(root, 'galacticStructureSolver')
    solver.set('value','equilibrium')
    child(solver,'solutionTolerance',SETTINGS[level]['equilibrium'])
    child(solver,'includeBaryonGravity','true')
    child(solver,'convergenceFailureIsFatal','true')
    child(one(root,'darkMatterProfile'),'toleranceRelative',1e-10)
    bar = one(root,BAR)
    for old in bar.findall('barInstabilitySpheroidAngularMomentum'): bar.remove(old)
    energy = ET.SubElement(bar,'barInstabilitySpheroidAngularMomentum',value='bindingEnergy')
    for name,value in {'stepFiniteDifferenceRelative':SETTINGS[level]['fd'], 'formFactorDisk':.5,
                       'formFactorInteraction':1., 'fractionBaryonicSpheroidTransitionMinimum':.003,
                       'fractionBaryonicSpheroidTransitionMaximum':.010}.items(): child(energy,name,value)
    child(one(root,DYNAMICS),'fractionAngularMomentumRetainedSpheroid',1.)
    child(one(root,ACCRETION),'angularMomentumSpecificGrowthFromAccretionRate','false')
    sf=one(root,'starFormationRateSurfaceDensityDisks')
    child(sf,'pressureCharacteristic',35000.);child(sf,'pressureExponent',.92)
    for spec in specs:
        path=spec['path'];parent,tag=path.rsplit('/',1)
        child(one(root,parent),tag,row[spec['short_name']])
    for key in ('odeToleranceAbsolute','odeToleranceRelative'):
        child(one(root,'mergerTreeNodeEvolver'),key,1e-4)
    child(one(root,'mergerTreeBuildMasses'),'treesPerDecade',500)
    one(root,"mergerTreeOutputter/mergerTreeOutputter[@value='standard']")
    extractors=one(root,"nodePropertyExtractor[@value='multi']")
    for name in ('indicesTree','nodeIndices'):
        if not extractors.findall(f"nodePropertyExtractor[@value='{name}']"):
            ET.SubElement(extractors,'nodePropertyExtractor',value=name)
    child(root,'outputFileName',str(folder/'galacticus.hdf5'))
    for e in root.findall('outputScratchFileName'):e.set('value',str(folder/'galacticus.hdf5'))
    verify_parameters(root,row,specs,level)
    return root


def verify_parameters(root, row, specs, level):
    import math
    for spec in specs:
        if not math.isclose(number(one(root,spec['path'])),float(row[spec['short_name']]),rel_tol=2e-12,abs_tol=0):
            raise ValueError('Changed sampled coordinate: '+spec['short_name'])
    expected={ENERGY+'/stepFiniteDifferenceRelative':SETTINGS[level]['fd'],
              'galacticStructureSolver/solutionTolerance':SETTINGS[level]['equilibrium'],
              'darkMatterProfile/toleranceRelative':1e-10,
              ENERGY+'/formFactorDisk':.5,ENERGY+'/formFactorInteraction':1.,
              ENERGY+'/fractionBaryonicSpheroidTransitionMinimum':.003,
              ENERGY+'/fractionBaryonicSpheroidTransitionMaximum':.010,
              DYNAMICS+'/fractionAngularMomentumRetainedSpheroid':1.,
              'mergerTreeBuildMasses/treesPerDecade':500,
              'starFormationRateSurfaceDensityDisks/pressureCharacteristic':35000.,
              'starFormationRateSurfaceDensityDisks/pressureExponent':.92,
              'mergerTreeNodeEvolver/odeToleranceAbsolute':1e-4,
              'mergerTreeNodeEvolver/odeToleranceRelative':1e-4}
    for path,value in expected.items():
        if not math.isclose(number(one(root,path)),value,rel_tol=2e-12): raise ValueError(f'Incorrect {path}')
    for path,value in [(ENERGY,'bindingEnergy'),('galacticStructureSolver','equilibrium'),
                       ('darkMatterProfile','adiabaticGnedin2004'),(ACCRETION+'/angularMomentumSpecificGrowthFromAccretionRate','false'),
                       ('galacticStructureSolver/includeBaryonGravity','true')]:
        if one(root,path).get('value').lower()!=value.lower():raise ValueError(f'Incorrect {path}')
    if any(e.tag in OLD_NAMES for e in one(root,ENERGY).iter()):raise ValueError('Obsolete energy option')
    if one(root,ENERGY).find('galacticStructureSolver') is not None:raise ValueError('Unexpected nested structure solver')


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True,stderr=subprocess.PIPE).strip()


def prepare(args):
    definition, rows = design(args.family)
    ids = selection(args.family)
    original, source, exe = args.original.resolve(), args.source.resolve(), args.executable.resolve()
    out=args.output.absolute()
    if any(c.isspace() for c in str(out)):
        raise ValueError('Use an output path without whitespace for Slurm directives')
    if out.exists() or out.is_symlink():raise FileExistsError(out)
    if not exe.is_file() or not os.access(exe,os.X_OK):raise ValueError(f'Executable missing or not executable: {exe}')
    revision=git(source,'rev-parse','HEAD')
    subprocess.run(['git','-C',str(source),'merge-base','--is-ancestor',MERGE,'HEAD'],check=True)
    if git(source,'diff','HEAD','--'):
        raise ValueError('Galacticus has tracked modifications; use the merged, unpatched implementation')
    energy_source=source/'source/galactic_dynamics/bar_instability/spheroid_angular_momentum/binding_energy.F90'
    text=energy_source.read_text()
    if '<name>stepFiniteDifferenceRelative</name>' not in text or 'StepLimitSpheroid' in text:
        raise ValueError('Expected the merged uncapped bindingEnergy implementation')
    old_manifest=json.loads((original/'run_manifest.json').read_text())
    if sha(original/'base_parameters.xml')!=old_manifest['base_parameters_sha256']:
        raise ValueError('Historical base XML checksum mismatch')
    old_rows=read_rows(original/'samples.csv')
    if len(old_rows)!=128:raise ValueError('Expected 128 historical sample rows')
    for i,(old,new) in enumerate(zip(old_rows,rows)):
        for spec in definition['free_parameters']:
            name=spec['short_name']
            if float(old[name])!=float(new[name]):raise ValueError(f'Historical point {i} differs: {name}')
    environment={}
    for key in ['GALACTICUS_DATA_PATH','SHMR_PATH']:
        if not os.environ.get(key) or not Path(os.environ[key]).is_dir():raise ValueError(f'Set {key} to an existing directory')
        environment[key]=str(Path(os.environ[key]).resolve())
    for key in ['GALACTICUS_SED_CALC_PATH','GALACTICUS_TOOLS_PATH']:
        if os.environ.get(key):environment[key]=os.environ[key]
    out.mkdir(parents=True)
    (out/'logs').mkdir();(out/'provenance').mkdir();(out/'executable').mkdir()
    for name in ['family_definition.json','design_1024.csv','precision_pilot_selection.json']:
        shutil.copy2(args.family/name,out/'provenance'/name)
    shutil.copy2(original/'base_parameters.xml',out/'provenance/historical_base.xml')
    shutil.copy2(original/'run_manifest.json',out/'provenance/historical_run_manifest.json')
    # Migrate the common base, then apply the saved numerical parameter vectors in
    # the current definitions (including the corrected Blitz frequency convention).
    env=dict(os.environ,**environment,GALACTICUS_EXEC_PATH=str(source))
    env['PYTHONPATH']=str(source/'python')+os.pathsep+env.get('PYTHONPATH','')
    migrated=out/'provenance/migrated_base.xml'
    with (out/'provenance/migration.log').open('w') as log:
        subprocess.run([sys.executable,str(source/'scripts/aux/parametersMigrate.py'),
            str(out/'provenance/historical_base.xml'),str(migrated),'--validate','no'],
            cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    base=ET.parse(migrated).getroot()
    # The legacy free-standing code-version tag is not a parameter; its value is
    # retained in the archived original. Current parsers reject its empty value.
    for e in base.findall('version'):
        if e.get('value') is None:base.remove(e)
    for e in base.iter():
        for attr in ('value','active'):
            v=e.get(attr)
            if v and (attr=='active' or v.lstrip().startswith('=')):
                e.set(attr,re.sub(r'\[([^\]]*)\]',lambda m:'['+re.sub(r':+','/',m[1])+']',v))
    if one(base,'lastModified').get('revision')!=revision:raise ValueError('Migration revision does not match checkout')
    xml(base,out/'base_parameters.xml')
    snapshot=out/'executable/Galacticus.exe';shutil.copy2(exe,snapshot)
    runner=out/'pilot.py';shutil.copy2(__file__,runner)
    shutil.copy2(REPO/'scripts/analyze_campaign_compact_tail.py',out/'analyze_campaign_compact_tail.py')
    shutil.copy2(REPO/'scripts/analyze_energy_precision_pilot.py',out/'analyze_energy_precision_pilot.py')
    tasks=[]
    for level,indices in [('practical',list(range(128))),('tighter',sorted(ids))]:
        dest=out/level;dest.mkdir()
        selected=[]
        for i in indices:
            name=f'pilot_128-eval-{i:04d}'
            folder=dest/'evaluations'/name;folder.mkdir(parents=True)
            row=dict(evaluation_id=name,**rows[i]);selected.append(row)
            xml(model(base,row,definition['free_parameters'],level,folder),folder/'params.xml')
            dump(folder/'galacticus_input_values.json',{k:(v if k=='evaluation_id' else float(v)) for k,v in row.items()})
            task=dict(task_index=len(tasks),level=level,design_index=i,evaluation_id=name,
                      input_sha256=sha(folder/'params.xml'))
            tasks.append(task)
        write_rows(dest/'samples.csv',selected)
        dump(dest/'campaign_design.json',dict(schema_version=2,campaign_name='energy_precision_'+level,
            n_eval=len(selected),seed=definition['seed'],free_parameters=definition['free_parameters'],
            free_parameter_short_names=[s['short_name'] for s in definition['free_parameters']],
            master_design_sha256=definition['design_sha256'],numerics=dict(**SETTINGS[level],ac=1e-10,ode=1e-4),
            selection='first 128' if level=='practical' else 'frozen 32-point comparison; not an independent prior sample'))
    dump(out/'tasks.json',tasks)
    try: datasets_revision=git(Path(environment['GALACTICUS_DATA_PATH']),'rev-parse','HEAD')
    except subprocess.CalledProcessError:datasets_revision=None
    dump(out/'manifest.json',dict(source=str(source),source_revision=revision,original=str(original),
        executable=str(snapshot),executable_sha256=sha(snapshot),python=sys.executable,environment=environment,
        datasets_revision=datasets_revision,runner_sha256=sha(runner),tasks_sha256=sha(out/'tasks.json'),
        family_definition_sha256=sha(args.family/'family_definition.json'),settings=SETTINGS,ac=1e-10,ode=1e-4,
        trees_per_decade=500,cpus=args.cpus,concurrency=args.concurrency,
        source_executable=str(exe),energy_source_sha256=sha(energy_source),
        parameter_convention='Use the frozen numerical values in current upstream definitions, including Blitz frequency; migrations can change unvaried physics.'))
    q=shlex.quote
    common=['#!/bin/bash',f'#SBATCH --partition={args.partition}','#SBATCH --nodes=1','#SBATCH --ntasks=1',
            f'#SBATCH --chdir={out}']
    throttle=f'%{args.concurrency}' if args.concurrency else ''
    (out/'submit.slurm').write_text('\n'.join(common+['#SBATCH --job-name=energy-precision-pilot',
        f'#SBATCH --cpus-per-task={args.cpus}','#SBATCH --mem-per-cpu=10G',f'#SBATCH --time={args.walltime}',
        f'#SBATCH --array=0-159{throttle}',f'#SBATCH --output={out}/logs/slurm-%A_%a.out',
        f'#SBATCH --error={out}/logs/slurm-%A_%a.err','set -euo pipefail',
        f'exec {q(sys.executable)} {q(str(runner))} run {q(str(out))} --task "$SLURM_ARRAY_TASK_ID"','']))
    (out/'analyze.slurm').write_text('\n'.join(common+['#SBATCH --job-name=energy-pilot-analysis',
        '#SBATCH --cpus-per-task=1','#SBATCH --mem=32G','#SBATCH --time=08:00:00',
        f'#SBATCH --output={out}/logs/analysis-%j.out',f'#SBATCH --error={out}/logs/analysis-%j.err',
        'set -euo pipefail',f'exec {q(sys.executable)} {q(str(runner))} analyze {q(str(out))} --output {q(str(out)+"_reduced")}', '']))
    for task in tasks:
        folder=out/task['level']/'evaluations'/task['evaluation_id']
        (folder/'run_eval.sh').write_text('#!/bin/bash\nset -euo pipefail\n'+
            f'exec {q(sys.executable)} {q(str(runner))} run {q(str(out))} --task {task["task_index"]}\n')
    print('Prepared 128 practical + 32 tighter runs:',out)
    print('Next: validate the generated inputs, then sbatch submit.slurm and dependent analyze.slurm')


def load_run(root):
    root=root.resolve();meta=json.loads((root/'manifest.json').read_text())
    if sha(root/'tasks.json')!=meta['tasks_sha256']:raise ValueError('Task manifest changed')
    if sha(root/'pilot.py')!=meta['runner_sha256']:raise ValueError('Frozen runner changed')
    if sha(meta['executable'])!=meta['executable_sha256']:raise ValueError('Executable snapshot changed')
    if git(Path(meta['source']),'rev-parse','HEAD')!=meta['source_revision']:
        raise ValueError('Source checkout moved since preparation; runtime auxiliary files must remain consistent')
    env=dict(os.environ,**meta['environment'],GALACTICUS_EXEC_PATH=meta['source'])
    env['OMP_NUM_THREADS']=str(os.environ.get('SLURM_CPUS_PER_TASK',meta['cpus']))
    env['OMP_PLACES']='cores';env['OMP_PROC_BIND']='spread'
    return root,meta,json.loads((root/'tasks.json').read_text()),env


def completion(path,revision):
    import h5py
    with h5py.File(path,'r') as f:
        if f.attrs.get('statusCompletion',-1)!=0:raise ValueError('Catalog did not complete successfully')
        if 'Outputs' not in f or 'analyses' not in f:raise ValueError('Missing catalogs or calibration analyses')
        value=f['Version'].attrs['gitHash']
        if isinstance(value,bytes):value=value.decode()
        if str(value).strip()!=revision:raise ValueError(f'Executable was built from {value}, expected {revision}; rebuild it')


def run(args):
    root,meta,tasks,env=load_run(args.root)
    if not 0<=args.task<len(tasks):raise ValueError('Task index out of range')
    task=tasks[args.task];folder=root/task['level']/'evaluations'/task['evaluation_id']
    if sha(folder/'params.xml')!=task['input_sha256']:raise ValueError('Input XML changed')
    if any((folder/x).exists() for x in ['SUCCESS','RUNNING','galacticus.hdf5']):
        raise FileExistsError('Existing output/run marker; archive the failed evaluation before retrying')
    definition=json.loads((root/'provenance/family_definition.json').read_text())
    rows=read_rows(root/task['level']/'samples.csv')
    row=next(r for r in rows if r['evaluation_id']==task['evaluation_id'])
    start=time.monotonic();(folder/'RUNNING').touch(exist_ok=False)
    result=dict(status='failed',task=task)
    try:
        with (folder/'galacticus.out').open('w') as stdout,(folder/'galacticus.err').open('w') as stderr:
            subprocess.run([meta['executable'],'params.xml','--output-processed-parameters','params_used.xml'],
                cwd=folder,env=env,stdout=stdout,stderr=stderr,check=True)
        verify_parameters(ET.parse(folder/'params_used.xml').getroot(),row,definition['free_parameters'],task['level'])
        completion(folder/'galacticus.hdf5',meta['source_revision'])
        (folder/'SUCCESS').touch();result['status']='ok'
    except Exception as e:
        result['error']=str(e);raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,threads=int(env['OMP_NUM_THREADS']),host=os.uname().nodename)
        dump(folder/'run_status.json',result);(folder/'RUNNING').unlink(missing_ok=True)


def validate(args):
    root,meta,tasks,env=load_run(args.root)
    indices=args.task if args.task is not None else [0,128]
    definition=json.loads((root/'provenance/family_definition.json').read_text())
    for i in indices:
        if not 0<=i<len(tasks):raise ValueError('Task index out of range')
        task=tasks[i];folder=root/task['level']/'evaluations'/task['evaluation_id']
        if sha(folder/'params.xml')!=task['input_sha256']:raise ValueError('Input XML changed')
        dest=root/'validation'/str(i);dest.mkdir(parents=True,exist_ok=True)
        params=ET.parse(folder/'params.xml').getroot()
        child(params,'outputFileName',str(dest/'validation.hdf5'))
        for e in params.findall('outputScratchFileName'):e.set('value',str(dest/'validation.hdf5'))
        xml(params,dest/'params.xml')
        with (dest/'validation.log').open('w') as log:
            subprocess.run([meta['executable'],str(dest/'params.xml'),'--output-processed-parameters',str(dest/'params_used.xml'),'--dry-run'],
                           cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        row=next(r for r in read_rows(root/task['level']/'samples.csv') if r['evaluation_id']==task['evaluation_id'])
        verify_parameters(ET.parse(dest/'params_used.xml').getroot(),row,definition['free_parameters'],task['level'])
        print('Validated:',task['level'],task['evaluation_id'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    s=sub.add_parser('select');s.add_argument('--family',type=Path,default=FAMILY)
    s.add_argument('--historical',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('prepare');p.add_argument('--family',type=Path,default=FAMILY)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partition',choices=['obs','dmtheory'],default='obs')
    p.add_argument('--cpus',type=int,default=8);p.add_argument('--concurrency',type=int,default=0)
    p.add_argument('--walltime',default='72:00:00')
    r=sub.add_parser('run');r.add_argument('root',type=Path);r.add_argument('--task',type=int,required=True)
    v=sub.add_parser('validate');v.add_argument('root',type=Path);v.add_argument('--task',type=int,action='append')
    a=sub.add_parser('analyze');a.add_argument('root',type=Path);a.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare' and (args.cpus<1 or args.concurrency<0 or not re.fullmatch(r'\d+:\d{2}:\d{2}',args.walltime)):
        parser.error('Invalid CPU count, concurrency or walltime')
    if args.action=='analyze':
        from analyze_energy_precision_pilot import analyze
        analyze(args)
    else:globals()[args.action](args)


if __name__=='__main__':main()
