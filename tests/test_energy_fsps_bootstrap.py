"""Exercise private installation, patch failure and completion-marker handling."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import bootstrap_energy_fsps as fsps


def fixture(tmp_path, monkeypatch):
    shared = tmp_path/'shared'
    shared.mkdir()
    (shared/'fsps-4.0/src').mkdir(parents=True)
    stale = shared/'fsps-4.0/src/imf.f90'
    stale.write_text('already patched\n')
    (shared/'other-tool').mkdir()
    (shared/'fsps4.0.lock').touch()
    patch_dir = tmp_path/'static/patches/FSPS'
    patch_dir.mkdir(parents=True)
    (patch_dir/'galacticus_IMF.f90').write_text('fixture\n')
    makefile = '# old\nall:\n\tprintf "#!/bin/sh\\nexit 0\\n" > autosps.exe\n\tchmod +x autosps.exe\nclean:\n\trm -f autosps.exe\n'
    with tarfile.open(shared/'FSPS_4.0.tar.gz', 'w:gz') as tar:
        for name in fsps.PATCHES:
            data = (makefile if name == 'Makefile' else 'old\n').encode()
            member = tarfile.TarInfo('fsps-4.0/src/'+name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
            old, new = ('# old', '# new') if name == 'Makefile' else ('old', 'new')
            (patch_dir/(name+'.patch')).write_text(f'--- {name}\n+++ {name}\n@@ -1 +1 @@\n-{old}\n+{new}\n')
    compiler = tmp_path/'compiler'
    compiler.write_text('#!/bin/sh\necho fixture compiler\n')
    compiler.chmod(0o755)
    monkeypatch.setenv('FCCOMPILER', str(compiler))
    return shared, tmp_path/'static', tmp_path/'private'


def test_private_pristine_build_and_no_overwrite(tmp_path, monkeypatch):
    shared, static, private = fixture(tmp_path, monkeypatch)
    fsps.build(shared, private, static/'patches/FSPS')
    assert (shared/'fsps-4.0/src/imf.f90').read_text() == 'already patched\n'
    assert (private/'fsps-4.0/src/imf.f90').read_text() == 'new\n'
    assert (private/'fsps-4.0/src/patched.status').read_text() == 'success\n'
    assert os.access(private/'fsps-4.0/src/autosps.exe', os.X_OK)
    assert not (private/'fsps-4.0').is_symlink()
    assert not (private/'fsps4.0.lock').exists()
    assert (private/'other-tool').resolve() == shared/'other-tool'
    meta = json.loads((private/'fsps_build_provenance.json').read_text())
    assert len(meta['patches']) == 5
    with pytest.raises(FileExistsError):
        fsps.build(shared, private, static/'patches/FSPS')


def test_partial_patch_failure_never_marked_complete(tmp_path, monkeypatch):
    shared, static, private = fixture(tmp_path, monkeypatch)
    patch = static/'patches/FSPS/ssp_gen.f90.patch'
    patch.write_text(patch.read_text().replace('-old', '-not present'))
    with pytest.raises(subprocess.CalledProcessError):
        fsps.build(shared, private, static/'patches/FSPS')
    assert (private/'fsps-4.0/src/imf.f90').read_text() == 'new\n'
    assert not (private/'fsps-4.0/src/patched.status').exists()
    assert not (private/'fsps-4.0/src/autosps.exe').exists()
    assert (shared/'fsps-4.0/src/imf.f90').read_text() == 'already patched\n'
