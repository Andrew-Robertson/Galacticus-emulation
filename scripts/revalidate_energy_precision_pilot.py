#!/usr/bin/env python3
"""Audit/recover completed pilot outputs rejected by the old Analyses/analyses check.

Never reruns Galacticus or modifies an HDF5 file. Original failed status is backed
up before recovery. The prepared runner and its checksum remain untouched.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET

import h5py
import prepare_energy_precision_pilot as pilot

OLD_ERROR = 'Missing catalogs or calibration analyses'


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        temp = Path(f.name)
    os.replace(temp, path)


def revalidate(args):
    root = args.root.resolve()
    meta = json.loads((root/'manifest.json').read_text())
    # Verify the original snapshot, not the current source checkout or diagnostic build.
    if pilot.sha(meta['executable']) != meta['executable_sha256']:
        raise ValueError('Original executable snapshot changed')
    tasks = json.loads((root/'tasks.json').read_text())
    if pilot.sha(root/'tasks.json') != meta['tasks_sha256']:
        raise ValueError('Task manifest changed')
    definition = json.loads((root/'provenance/family_definition.json').read_text())
    if pilot.sha(root/'provenance/family_definition.json') != meta['family_definition_sha256']:
        raise ValueError('Frozen family definition changed')
    rows = {level:{r['evaluation_id']:r for r in pilot.read_rows(root/level/'samples.csv')}
            for level in ('practical','tighter')}
    records = []
    for task in tasks:
        folder = root/task['level']/'evaluations'/task['evaluation_id']
        status_file = folder/'run_status.json'
        item = dict(task_index=task['task_index'],level=task['level'],evaluation_id=task['evaluation_id'])
        if (folder/'RUNNING').exists() or not status_file.exists():
            item['result']='not_a_completed_candidate'
            records.append(item)
            continue
        status = json.loads(status_file.read_text())
        if status.get('error') != OLD_ERROR:
            item['result']='previously_recovered' if 'validation_recovery' in status else 'not_the_group_name_error'
            records.append(item)
            continue
        try:
            if pilot.sha(folder/'params.xml') != task['input_sha256']:
                raise ValueError('Original input changed')
            pilot.verify_parameters(ET.parse(folder/'params_used.xml').getroot(),
                rows[task['level']][task['evaluation_id']], definition['free_parameters'],task['level'])
            pilot.completion(folder/'galacticus.hdf5', meta['source_revision'])
            with h5py.File(folder/'galacticus.hdf5','r') as f:
                item['root_groups']=list(f)
                item['outputs']=len(f['Outputs'])
                item['analyses']=len(f['analyses'])
                if item['outputs']==0 or item['analyses']==0:
                    raise ValueError('Empty catalog or analysis group')
            item['result']='validated_for_recovery'
            if args.recover:
                original=status_file.read_bytes()
                backup=folder/'run_status_before_validation_recovery.json'
                if backup.exists():
                    if backup.read_bytes()!=original:
                        raise ValueError('Existing backup differs from original status')
                else:
                    with backup.open('xb') as f:f.write(original)
                status['status']='ok'
                status['previous_validation_error']=status.pop('error')
                status['validation_recovery']=dict(
                    reason='Correct HDF5 group name: analyses (lowercase)',
                    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    script_sha256=pilot.sha(Path(__file__)),
                    original_status_file=backup.name)
                atomic_json(status_file,status)
                (folder/'SUCCESS').touch(exist_ok=True)
                item['result']='recovered'
        except (OSError,ValueError,KeyError,ET.ParseError) as e:
            item.update(result='validation_failed',detail=str(e))
        records.append(item)
    dest=root/'validation_recovery'
    dest.mkdir(exist_ok=True)
    counts={label:sum(r['result']==label for r in records) for label in sorted({r['result'] for r in records})}
    atomic_json(dest/('recovery.json' if args.recover else 'audit.json'),dict(counts=counts,records=records))
    print(json.dumps(counts,indent=2))
    print('Report:',dest)
    if any(r['result']=='validation_failed' for r in records):
        raise SystemExit('Some candidates failed validation; inspect the report. They were not recovered.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--recover',action='store_true',help='Back up failed status and mark fully revalidated catalogs successful')
    revalidate(p.parse_args())
