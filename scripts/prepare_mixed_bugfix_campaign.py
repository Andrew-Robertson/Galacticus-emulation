#!/usr/bin/env python3
"""Freeze completed original + PR1785 fallback evaluations into a preliminary view.

prepare runs on a login node; build (dust processing and table merge) runs in Slurm.
Source simulations and their sidecars are never modified. No comparisons/ runs used.
"""
import argparse
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
import xml.etree.ElementTree as ET

OLD = '57c1ef2e7651695217155f6be1e4bc0e359ec4a0'
NEW = 'b780b39e0856d942196021fc80655fc59821fb03'
NEW_SHA = 'ece5f40e1d3a203a2566bd92b4974a0b58b3eacd67c403338643b87bf09674c3'
FILES = ['dust_draws.csv', 'emission_line_dust_lf_long.csv',
         'emission_line_dust_lf_emulator_table.csv', 'metadata.json', 'manifest.csv']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def xml_values(path):
    root = ET.parse(path).getroot()
    for tag in ('outputFileName', 'outputScratchFileName'):
        for node in root.findall(tag):
            root.remove(node)
    def signature(node):
        return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
                tuple(signature(c) for c in node))
    return signature(root)


def check_sim(folder, revision):
    import h5py
    with h5py.File(folder/'galacticus.hdf5', 'r') as f:
        if f.attrs.get('statusCompletion', -1) != 0:
            raise ValueError('simulation incomplete')
        reported = f['Version'].attrs['gitHash']
        if isinstance(reported, bytes):
            reported = reported.decode()
        if str(reported) not in ('unknown', revision):
            raise ValueError(f'unexpected embedded revision {reported}')
        if float(f['Parameters/starFormationRateSurfaceDensityDisks'].attrs['pressureCharacteristic']) != 35000:
            raise ValueError('unexpected Blitz pressure')
        if not len(f['analyses']):
            raise ValueError('no native analyses')


def check_dust(folder, name):
    for file in FILES:
        if not (folder/file).is_file() or not (folder/file).stat().st_size:
            raise ValueError(f'missing/empty dust file {folder/file}')
    meta = read(folder/'metadata.json')
    if (meta['evaluation_id'] != name or meta['n_dust_draws'] != 16 or
        meta['scatter_mode'] != 'expected' or meta['z_pivot'] != 1.0):
        raise ValueError('dust metadata differs')
    with (folder/'dust_draws.csv').open() as f:
        draws = list(csv.DictReader(f))
    if len(draws) != 16 or {int(r['dust_draw_index']) for r in draws} != set(range(16)):
        raise ValueError('dust draws incomplete')


def prepare(campaign, view):
    config = read(campaign/'updated/prepared.json')
    samples = list(csv.DictReader((campaign/'samples.csv').open()))
    meta = read(campaign/'pilot.json')
    if meta['target_revision'] != OLD:
        raise ValueError('unexpected original revision')
    # Search only failed-model retries, plus the separately successful eval0043 test.
    retry_candidates = {}
    for root in sorted((campaign/'retries').glob('pr1785_32_*')):
        for marker in root.rglob('simulation_finished.json'):
            match = re.search(r'-eval-(\d+)$', marker.parent.name)
            if not match:
                raise ValueError(f'Cannot identify retry evaluation: {marker}')
            retry_candidates.setdefault(int(match[1]), []).append(marker.parent)
    special = campaign/'diagnostics/eval0043_pr1785_20261113T234018Z'
    if (special/'simulation_finished.json').exists():
        retry_candidates.setdefault(43, []).append(special)
    records, status = [], []
    for index, row in enumerate(samples):
        name = row['evaluation_id']
        original = campaign/'updated/evaluations'/name
        selected = None
        try:
            if (original/'complete.json').exists():
                proof = read(original/'complete.json')
                if proof['source_revision'] != OLD:
                    raise ValueError('original completion revision differs')
                if sha(original/'input.xml') != config['input_sha256'][name]:
                    raise ValueError('original prepared input changed')
                check_sim(original, OLD)
                check_dust(original/'emission_line_dust', name)
                selected = dict(source=str(original), source_revision=OLD,
                    executable_sha256=config['executable_sha256'], kind='original',
                    completion_record=str(original/'complete.json'))
            else:
                candidates = retry_candidates.get(index, [])
                if len(candidates) > 1:
                    raise ValueError('multiple successful retries: choose explicitly')
                if candidates:
                    src = candidates[0]
                    proof = read(src/'simulation_finished.json')
                    if (proof.get('statusCompletion') != 0 or
                        proof.get('revision', proof.get('source_revision')) != NEW or
                        proof.get('executable_sha256') != NEW_SHA):
                        raise ValueError('retry provenance incomplete or mismatched')
                    if proof.get('input_sha256') != sha(src/'input.xml'):
                        raise ValueError('retry input checksum mismatch')
                    if xml_values(src/'input.xml') != xml_values(original/'input.xml'):
                        raise ValueError('retry input differs beyond output paths')
                    check_sim(src, NEW)
                    selected = dict(source=str(src), source_revision=NEW,
                        executable_sha256=NEW_SHA, kind='pr1785_retry',
                        completion_record=str(src/'simulation_finished.json'))
            if selected:
                selected.update(index=index, evaluation_id=name,
                    input_values=str(original/'galacticus_input_values.json'),
                    input_sha256=sha(Path(selected['source'])/'input.xml'))
                if not Path(selected['input_values']).is_file():
                    raise ValueError('missing input parameter JSON')
                records.append(selected)
                status.append(dict(index=index, evaluation_id=name, status=selected['kind'], detail=''))
            else:
                status.append(dict(index=index, evaluation_id=name, status='excluded', detail='no completed eligible source'))
        except Exception as exc:
            status.append(dict(index=index, evaluation_id=name, status='needs_inspection', detail=str(exc)))
    view.mkdir(parents=True, exist_ok=False)
    (view/'logs').mkdir()
    shutil.copy2(__file__, view/'assemble.py')
    shutil.copy2(campaign/'samples.csv', view/'all_samples.csv')
    shutil.copy2(campaign/'original_campaign_design.json', view/'original_campaign_design.json')
    write(view/'assembly.json', dict(campaign=str(campaign), configuration=config,
        records=records, status=status, selection='Original successful outputs preferred; completed PR1785 failed-model retries only. Comparison reruns excluded.'))
    with (view/'status.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(status[0]));w.writeheader();w.writerows(status)
    for category in ('original', 'pr1785_retry', 'excluded', 'needs_inspection'):
        print(category, sum(s['status']==category for s in status))
    for s in status:
        if s['status'] in ('excluded', 'needs_inspection'):
            print(s['evaluation_id'], s['status'], s['detail'])
    if any(s['status']=='needs_inspection' for s in status):
        raise RuntimeError(f'Inspect {view}/status.csv before assembling; no job script generated')
    if not records:
        raise RuntimeError('No eligible evaluations')
    batch=f'''#!/bin/bash
#SBATCH --job-name=mixed-campaign-dust
#SBATCH --partition=expansion
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output={view}/logs/build-%j.out
#SBATCH --error={view}/logs/build-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
{shlex.quote(sys.executable)} {shlex.quote(str(view/'assemble.py'))} build {shlex.quote(str(view))}
'''
    (view/'build.sbatch').write_text(batch)
    print('Prepared snapshot:', view)
    print('Submit:', 'sbatch', view/'build.sbatch')


def build(view):
    import h5py
    import numpy as np
    import pandas as pd
    manifest=read(view/'assembly.json');config=manifest['configuration']
    script=Path(config['emu_root'])/'scripts/process_emission_line_dust_evaluation.py'
    if sha(script)!=config['dust_script_sha256']:
        raise ValueError('Dust script changed since campaign preparation')
    if (view/'READY.json').exists():
        raise FileExistsError('Already assembled')
    with (view/'BUILD_STARTED.json').open('x') as f:
        json.dump({'job_id':os.environ.get('SLURM_JOB_ID')},f)
    env=dict(os.environ,**config['environment'])
    source=Path(manifest['campaign'])/'diagnostics/eval0043_pr1785_20261113T234018Z/source'
    env['GALACTICUS_EXEC_PATH']=str(source)
    env['PYTHONPATH']=str(source/'python')+os.pathsep+env.get('PYTHONPATH','')
    warnings=[]
    for record in manifest['records']:
        name=record['evaluation_id'];src=Path(record['source']);dst=view/'evaluations'/name
        check_sim(src,record['source_revision'])
        if sha(src/'input.xml')!=record['input_sha256']:
            raise ValueError(f'Input changed: {name}')
        dst.mkdir(parents=True,exist_ok=False)
        for file in ('galacticus.hdf5','input.xml','params.xml'):
            if not (src/file).is_file():raise FileNotFoundError(src/file)
            (dst/file).symlink_to(src/file)
        (dst/'galacticus_input_values.json').symlink_to(record['input_values'])
        shutil.copy2(record['completion_record'],dst/'source_completion.json')
        if record['kind']=='original':
            (dst/'emission_line_dust').symlink_to(src/'emission_line_dust',target_is_directory=True)
        else:
            cmd=[sys.executable,str(script),str(dst/'galacticus.hdf5'),
                 '--evaluation-id',name,'--output-dir',str(dst/'emission_line_dust'),
                 '--n-dust-draws','16','--base-seed','12345','--evaluation-index',str(record['index']),
                 '--scatter-mode','expected','--z-pivot','1.0',
                 '--input-json',str(dst/'galacticus_input_values.json')]
            print('Dust processing:',name,flush=True)
            with (dst/'dust.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        check_dust(dst/'emission_line_dust',name)
        # Audit without silently modifying or flooring problematic native data.
        with h5py.File(dst/'galacticus.hdf5','r') as f:
            for key,g in f['analyses'].items():
                for attr in ('yDataset','yCovariance'):
                    dataset=g.attrs.get(attr,'')
                    if isinstance(dataset,bytes):dataset=dataset.decode()
                    if dataset and dataset in g:
                        a=g[dataset][...]
                        values=np.diag(a) if attr=='yCovariance' else a
                        nonfinite=int(np.count_nonzero(~np.isfinite(values)))
                        nonpositive=int(np.count_nonzero(values<=0))
                        if nonfinite or nonpositive:
                            warnings.append(dict(evaluation_id=name,analysis=key,attribute=attr,
                                nonfinite=nonfinite,nonpositive=nonpositive))
        write(dst/'preliminary_source.json',record)
    ids=[r['evaluation_id'] for r in manifest['records']]
    samples=pd.read_csv(view/'all_samples.csv');samples=samples[samples.evaluation_id.isin(ids)]
    if len(samples)!=len(ids) or samples.evaluation_id.duplicated().any():
        raise ValueError('Sample identity mismatch')
    samples.to_csv(view/'samples.csv',index=False)
    design=read(view/'original_campaign_design.json')
    design['n_eval']=len(samples)
    design['preliminary_mixed_campaign']=dict(source_campaign=manifest['campaign'],
        original_n_eval=1024,provenance_file='source_manifest.csv',
        note='Subset of original Sobol design; mixed executable versions; no resampling.')
    write(view/'campaign_design.json',design)
    pd.DataFrame(manifest['records']).to_csv(view/'source_manifest.csv',index=False)
    pd.DataFrame(warnings,columns=['evaluation_id','analysis','attribute','nonfinite','nonpositive']).to_csv(view/'native_value_audit.csv',index=False)
    for filename in ('emission_line_dust_lf_emulator_table.csv','emission_line_dust_lf_long.csv','dust_draws.csv'):
        frames=[pd.read_csv(view/'evaluations'/name/'emission_line_dust'/filename) for name in ids]
        for name,frame in zip(ids,frames):
            if 'evaluation_id' not in frame or set(frame.evaluation_id)!= {name}:
                raise ValueError(f'Sidecar evaluation IDs differ: {name}/{filename}')
        pd.concat(frames,ignore_index=True).to_csv(view/filename,index=False)
    excluded=[s for s in manifest['status'] if s['status']=='excluded']
    (view/'excluded_evaluations.txt').write_text(''.join(s['evaluation_id']+'\n' for s in excluded))
    write(view/'READY.json',dict(n_evaluations=len(ids),n_original=sum(r['kind']=='original' for r in manifest['records']),
        n_pr1785=sum(r['kind']=='pr1785_retry' for r in manifest['records']),n_excluded=len(excluded),
        note='Assembly and dust checks passed; inspect native_value_audit.csv with training masks before GP fitting. HDF5 files are linked, not copied.'))
    print('READY:',view,flush=True)
    print(json.dumps(read(view/'READY.json'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('campaign',type=Path);p.add_argument('view',type=Path)
    p=subs.add_parser('build');p.add_argument('view',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.campaign.resolve(),args.view.resolve())
    else:build(args.view.resolve())
