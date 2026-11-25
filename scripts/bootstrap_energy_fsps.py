#!/usr/bin/env python3
"""Build pristine FSPS in a private tools directory, preserving shared tools.

Written by OpenAI Codex at Andrew Robertson's request. Mirrors FSPS 4.0 setup
in Galacticus f6648df source/interface/FSPS.F90, with noninteractive patches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile

VERSION = '4.0'
DATASET_REVISION = '77d9aa7bf4eae9c01be89f0566b8eb6e3b46989e'
PATCHES = ('imf.f90', 'ssp_gen.f90', 'sps_vars.f90', 'autosps.f90', 'Makefile')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, destination):
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                    '--max-time', '1800', '--output', str(destination), url], check=True)


def build(shared, destination, patch_directory=None):
    shared, destination = shared.resolve(), destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Use a fresh private tools directory: {destination}')
    if not shared.is_dir():
        raise ValueError(f'Missing shared tools: {shared}')
    compiler = shutil.which(os.environ.get('FCCOMPILER', 'gfortran'))
    if compiler is None:
        raise ValueError('No Fortran compiler in the batch environment')
    destination.mkdir(parents=True)
    patch_dir = destination/'fsps_patches'
    patch_dir.mkdir()
    for name in ['galacticus_IMF.f90']+[name+'.patch' for name in PATCHES]:
        if patch_directory is None:
            download(f'https://raw.githubusercontent.com/galacticusorg/datasets/{DATASET_REVISION}/static/patches/FSPS/{name}', patch_dir/name)
        else:
            shutil.copy2(patch_directory/name, patch_dir/name)
    # All FSPS source, markers, archives and locks are private. Other tools
    # retain their existing paths through symlinks; this script never builds them.
    for entry in shared.iterdir():
        if not entry.name.lower().startswith('fsps'):
            (destination/entry.name).symlink_to(entry)
    archive = destination/'FSPS_4.0.tar.gz'
    original = shared/archive.name
    if original.is_file():
        shutil.copy2(original, archive)
        archive_source = str(original)
    else:
        archive_source = 'https://github.com/cconroy20/fsps/archive/refs/tags/v4.0.tar.gz'
        download(archive_source, archive)
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle.getmembers():
            if member.name != 'fsps-4.0' and not member.name.startswith('fsps-4.0/'):
                raise ValueError(f'Unexpected archive member: {member.name}')
        handle.extractall(destination, filter='data')
    fsps = destination/'fsps-4.0'
    src = fsps/'src'
    shutil.copy2(patch_dir/'galacticus_IMF.f90', src/'galacticus_IMF.f90')
    provenance = dict(version=VERSION, archive_source=archive_source, archive_sha256=sha(archive),
                      compiler=compiler, patches={},
                      dataset_revision=DATASET_REVISION if patch_directory is None else None,
                      imf_module_sha256=sha(patch_dir/'galacticus_IMF.f90'))
    for name in PATCHES:
        patch = patch_dir/(name+'.patch')
        # Explicit target names avoid the old imf patch's misleading .orig header.
        for dry in (True, False):
            command = ['patch', '--batch', '--forward']+(['--dry-run'] if dry else [])
            with patch.open('rb') as handle:
                subprocess.run(command+[name], cwd=src, stdin=handle, check=True)
        provenance['patches'][name] = sha(patch)
    # Match Galacticus's dynamic-build Makefile changes and compiler selection.
    makefile = src/'Makefile'
    text = makefile.read_text()
    text = re.sub(r'^(F90FLAGS := .*?)\s+-static(?=\s|$)', r'\1', text, flags=re.M)
    text = text.replace('gfortran', compiler)
    makefile.write_text(text)
    env = dict(os.environ, SPS_HOME=str(fsps))
    prefix = '' if platform.machine().lower() in ('aarch64', 'arm64') else '-mcmodel=medium '
    env['F90FLAGS'] = prefix+env.get('GALACTICUS_FCFLAGS', '')
    subprocess.run([compiler, '--version'], check=True)
    subprocess.run(['make', 'clean'], cwd=src, env=env, check=True)
    subprocess.run(['make', '-j1'], cwd=src, env=env, check=True)
    exe = src/'autosps.exe'
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise RuntimeError('FSPS build did not produce an executable autosps.exe')
    # Write the completion marker only after every patch AND the build succeeded.
    (src/'patched.status').write_text('success\n')
    provenance.update(executable_sha256=sha(exe), fortran_flags=env['F90FLAGS'])
    (destination/'fsps_build_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('FSPS BUILD SUCCESSFUL:', exe, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-tools', type=Path, required=True)
    parser.add_argument('--patch-directory', type=Path, help='Optional offline patch source; default: pinned official dataset revision')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.shared_tools, args.output, args.patch_directory)
