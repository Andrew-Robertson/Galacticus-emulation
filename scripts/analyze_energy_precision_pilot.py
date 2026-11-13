#!/usr/bin/env python3
"""Portable population and matched-galaxy diagnostics for the precision pilot."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

import analyze_campaign_compact_tail as compact


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write_rows(path,records):
    with Path(path).open('w',newline='') as f:
        if records:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def population(data):
    m=data['spheroidMassStellar'];d=data['diskMassStellar'];r=data['spheroidRadius'];w=data['weight']
    good=np.isfinite(m+d+r+w)&(m>=1e6)&(d>=0)&(m+d>=1e8)&(r>0)&(w>0)
    lm=np.log10(m[good]);lr=np.log10(r[good]);ww=w[good]
    records=[]
    for lo in np.arange(6,12,.25):
        s=(lm>=lo)&(lm<lo+.25)
        q=compact.quantile(lr[s],ww[s],[.16,.5,.84]) if s.sum()>=20 else [np.nan]*3
        records.append(dict(log_mass_low=float(lo),log_mass_high=float(lo+.25),count=int(s.sum()),
            effective_count=compact.neff(ww[s]),p16=float(q[0]),median=float(q[1]),p84=float(q[2]),width68=float((q[2]-q[0])/2)))
    return records


def matched(practical,tighter):
    """Bins and abundance weights come from the tighter run; no clipping."""
    def keys(data):
        k=list(zip(data['mergerTreeIndex'].tolist(),data['nodeIndex'].tolist()))
        if len(set(k))!=len(k):raise ValueError('Duplicate (tree,node) IDs; matching is ambiguous')
        return k
    pk,tk=keys(practical),keys(tighter);lookup={k:i for i,k in enumerate(pk)}
    b=np.array([i for i,k in enumerate(tk) if k in lookup],dtype=int)
    a=np.array([lookup[tk[i]] for i in b],dtype=int)
    rm=tighter['spheroidMassStellar'];rd=tighter['diskMassStellar'];rr=tighter['spheroidRadius'];w=tighter['weight']
    rp=practical['spheroidRadius'][a]
    valid=np.isfinite(rr[b]+rp+w[b])&(rr[b]>0)&(rp>0)&(w[b]>0)
    with np.errstate(divide='ignore',invalid='ignore'):delta=np.log10(rp/rr[b])
    records=[]
    for lo in range(7,12):
        eligible=np.isfinite(rm+rd)&(rd>=0)&(rm>=10.**lo)&(rm<10.**(lo+1))&(rm+rd>=1e8)&(w>0)
        s=eligible[b]&valid
        ww=w[b[s]];x=delta[s]
        absq=compact.quantile(np.abs(x),ww,[.5,.9,.99]) if len(x) else [np.nan]*3
        signed=compact.quantile(x,ww,[.5])[0] if len(x) else np.nan
        variance=float(np.sum(ww*x*x))
        n_top=max(1,int(np.ceil(len(x)*.01)))
        largest=np.argsort(np.abs(x))[-n_top:]
        records.append(dict(log_mass_low=lo,log_mass_high=lo+1,reference_count=int(eligible.sum()),matched_count=int(s.sum()),
            matched_weight_coverage=float(ww.sum()/w[eligible].sum()) if eligible.any() else np.nan,
            mean=float(np.average(x,weights=ww)) if len(x) else np.nan,median=float(signed),
            rms=float(np.sqrt(variance/ww.sum())) if len(x) else np.nan,
            abs_p50=float(absq[0]),abs_p90=float(absq[1]),abs_p99=float(absq[2]),
            rms_squared_fraction_from_largest_1percent=float(np.sum(ww[largest]*x[largest]**2)/variance) if variance>0 else np.nan))
    return records


def analyze(args):
    root=args.root.resolve();out=args.output.absolute()
    if out.exists() or out.is_symlink():raise FileExistsError('Use a fresh reduced/analysis destination')
    out.mkdir(parents=True);(out/'analysis').mkdir()
    meta=json.loads((root/'manifest.json').read_text());tasks=json.loads((root/'tasks.json').read_text())
    shutil.copy2(root/'manifest.json',out/'manifest.json');shutil.copy2(root/'tasks.json',out/'tasks.json')
    shutil.copytree(root/'provenance',out/'provenance')
    status=[];good={};profiles={};comparisons=[];matches=[];diagnostic_status=[]
    for level in ('practical','tighter'):
        dest=out/level;dest.mkdir()
        for name in ('samples.csv','campaign_design.json'):shutil.copy2(root/level/name,dest/name)
    for task in tasks:
        level,name=task['level'],task['evaluation_id'];src=root/level/'evaluations'/name;dest=out/level/'evaluations'/name
        dest.mkdir(parents=True)
        for f in ['params.xml','params_used.xml','galacticus_input_values.json','run_status.json','SUCCESS','galacticus.out','galacticus.err']:
            if (src/f).is_file():shutil.copy2(src/f,dest/f)
        item=dict(task_index=task['task_index'],level=level,evaluation_id=name,status='failed',detail='')
        try:
            if not (src/'SUCCESS').is_file():raise ValueError('Missing SUCCESS marker; inspect copied logs/run_status.json')
            with h5py.File(src/'galacticus.hdf5','r') as f:
                if f.attrs.get('statusCompletion',-1)!=0:raise ValueError('Catalog is incomplete')
                revision=f['Version'].attrs['gitHash'];revision=revision.decode() if isinstance(revision,bytes) else str(revision)
                if revision.strip()!=meta['source_revision']:raise ValueError('Unexpected executable revision')
                if 'analyses' not in f or 'Outputs' not in f:raise ValueError('Missing analyses or catalogs')
                with h5py.File(dest/'galacticus.hdf5','w') as reduced:
                    for k,v in f.attrs.items():reduced.attrs[k]=v
                    for k in f:
                        if k!='Outputs':f.copy(k,reduced)
            item['status']='ok';good[(level,name)]=src/'galacticus.hdf5'
        except (OSError,ValueError,KeyError) as e:item['detail']=str(e)
        status.append(item)
        print(f"{task['task_index']+1}/160 {level} {name}: {item['status']}",flush=True)
        write_rows(out/'analysis/status.csv',status)
    # Use the established reader (dendros when installed) and compact-tail definition.
    for z in (0.,1.,2.):
        for level in ('practical','tighter'):
            dest=out/level/'analysis'/f'compact_tail_z{z:g}';dest.mkdir(parents=True)
            tail_rows=[];pop_rows=[]
            for (lev,name),path in good.items():
                if lev!=level:continue
                item=dict(level=level,evaluation_id=name,redshift=z,status='failed',detail='')
                try:
                    data,info=compact.read_catalog(path,redshift=z,reader='auto')
                    result,stats,_,worst,extra=compact.analyze(data,min_count=20,log_mass_min=8.)
                    folder=dest/'evaluations'/name;folder.mkdir(parents=True)
                    np.savez_compressed(folder/'compact_tail.npz',**result)
                    dump(folder/'metadata.json',dict(**info,**extra));write_rows(folder/'extreme_galaxies.csv',worst)
                    tail_rows.extend(dict(evaluation_id=name,**r) for r in stats)
                    pp=population(data);profiles[(level,name,z)]=pp
                    pop_rows.extend(dict(evaluation_id=name,**r) for r in pp)
                    item['status']='ok'
                except (OSError,ValueError,KeyError,RuntimeError) as e:item['detail']=str(e)
                diagnostic_status.append(item)
            write_rows(dest/'mass_bin_summary.csv',tail_rows);write_rows(dest/'population_relations.csv',pop_rows)
        for (level,name),fine_path in good.items():
            if level!='tighter' or ('practical',name) not in good:continue
            p=profiles.get(('practical',name,z));t=profiles.get(('tighter',name,z))
            if p and t:
                for a,b in zip(p,t):
                    comparisons.append(dict(evaluation_id=name,redshift=z,log_mass_low=b['log_mass_low'],
                        practical_count=a['count'],tighter_count=b['count'],
                        delta_median=a['median']-b['median'],delta_width68=a['width68']-b['width68']))
            try:
                practical,_=compact.read_catalog(good[('practical',name)],redshift=z,reader='auto')
                tighter,_=compact.read_catalog(fine_path,redshift=z,reader='auto')
                matches.extend(dict(evaluation_id=name,redshift=z,**r) for r in matched(practical,tighter))
            except (OSError,ValueError,KeyError,RuntimeError) as e:
                diagnostic_status.append(dict(level='matched',evaluation_id=name,redshift=z,status='failed',detail=str(e)))
    write_rows(out/'analysis/diagnostic_status.csv',diagnostic_status)
    write_rows(out/'analysis/population_differences.csv',comparisons)
    write_rows(out/'analysis/matched_size_differences.csv',matches)
    dump(out/'analysis/definitions.json',dict(delta='practical minus tighter, log10(radius/kpc)',
        population='Own stellar spheroid-mass bins, 0.25 dex; total stellar mass >= 1e8 Msun; >=20 galaxies per bin; abundance weighted',
        width68='(84th - 16th percentile)/2 in log radius',
        matched='Unique (mergerTreeIndex,nodeIndex); bins and weights from tighter run; no clipping; positive radii in both',
        compact='Own-model median log radius interpolated in 0.25 dex stellar spheroid-mass bins; no B/T or sigma cut',
        caveat='Tighter subset is deliberately enriched in extremes and is not a prior sample; compare it only against its matching practical points'))
    failed=[s for s in status+diagnostic_status if s['status']!='ok']
    dump(out/'analysis/summary.json',dict(successful_runs=len(good),expected_runs=160,failures=failed))
    print('Copy this entire reduced directory:',out)
    if failed:raise SystemExit('Some simulations or diagnostics failed; see analysis/summary.json and copied logs')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    analyze(p.parse_args())
