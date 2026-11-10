#!/usr/bin/env python3
"""Restore omitted historical metadata; finalize terminal jobs without rerunning."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import prepare_code_update_campaign as campaign


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--array', required=True, type=int)
    args = p.parse_args()
    root = args.root.resolve()
    manifest = root/'pilot.json'
    before = manifest.read_bytes()
    meta = json.loads(before)
    source = Path(meta['original_campaign'])
    changed = False
    for record in meta['evaluations']:
        src = source/'evaluations'/record['evaluation_id']
        if campaign.pilot.sha(src/'params.xml') != record['parameters_sha256']:
            raise ValueError(f'Historical input mismatch: {src}')
        version = campaign.original_version(src/'galacticus.hdf5')
        if 'original_version' in record:
            if record['original_version'] != version:
                raise ValueError(f'Existing historical metadata differs: {src}')
        else:
            record['original_version'] = version
            changed = True
    if changed:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = root/f'pilot.before_original_versions_{stamp}.json'
        with backup.open('xb') as f:
            f.write(before)
        temporary = root/f'pilot.repair_{stamp}.json'
        temporary.write_text(json.dumps(meta, indent=2)+'\n')
        if manifest.read_bytes() != before:
            raise RuntimeError('Manifest changed during repair')
        temporary.replace(manifest)
        print('Restored historical metadata; backup:', backup, flush=True)
    else:
        print('Historical metadata already correct.', flush=True)
    # Only finalize jobs Slurm says have terminated; never inspect running HDF5 outputs.
    output = subprocess.check_output(['sacct','-n','-X','-P','-j',str(args.array),
        '--format=JobID%64,State'], text=True)
    terminal = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
    selected = set()
    for line in output.splitlines():
        fields = line.split('|')
        match = re.fullmatch(str(args.array)+r'_(\d+)', fields[0].strip())
        if match and fields[1].strip().split()[0].rstrip('+') in terminal:
            selected.add(int(match[1]))
    failures = []
    for index in sorted(selected):
        if not 0 <= index < 64:
            raise ValueError('Recovery expects the first 64-model array')
        dest = root/'updated/evaluations'/meta['evaluations'][index]['evaluation_id']
        if (dest/'complete.json').exists():
            print('Already finalized:', index, flush=True)
            continue
        try:
            campaign.pilot.finish_model(SimpleNamespace(pilot=root,index=index))
        except Exception as error:
            failures.append((index,str(error)))
            print('NEEDS INSPECTION:',index,str(error),flush=True)
    print('Terminal indices checked:', sorted(selected))
    print('Unresolved:', failures)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
