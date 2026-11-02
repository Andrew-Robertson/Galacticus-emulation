"""Scientific comparison invariants for the paired code-update pilot."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest


def load(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load('compare_code_update_pilot')
prepare = load('prepare_code_update_pilot')


def make_output(path, y, *, x=(1, 2), target=(3, 4), logarithmic=True):
    with h5py.File(path, 'w') as f:
        g = f.create_group('analyses/test')
        g.attrs.update(type='function1D', xDataset='x', yDataset='y',
                       yAxisIsLog=int(logarithmic), yDatasetTarget='target',
                       yCovarianceTarget='cov')
        g['x'], g['y'], g['target'], g['cov'] = x, y, target, np.diag([4., 9.])


def test_linear_and_already_logged_observables_are_distinguished(tmp_path):
    a, b = tmp_path/'a.h5', tmp_path/'b.h5'
    make_output(a, [1., 10.])
    make_output(b, [10., 100.])
    with h5py.File(a) as old, h5py.File(b) as new:
        rows = compare.direct_rows(old, new, 'model')
    np.testing.assert_allclose([r['shift'] for r in rows], [1, 1])
    np.testing.assert_allclose([r['delta_over_observational_sigma'] for r in rows], [4.5, 30])
    make_output(a, [-3., -2.], logarithmic=False)
    make_output(b, [-2.5, -1.5], logarithmic=False)
    with h5py.File(a) as old, h5py.File(b) as new:
        rows = compare.direct_rows(old, new, 'model')
    np.testing.assert_allclose([r['shift'] for r in rows], [.5, .5])


@pytest.mark.parametrize('changed', [{'x': [1, 3]}, {'target': [3, 5]}])
def test_changed_bins_or_targets_are_rejected(tmp_path, changed):
    a, b = tmp_path/'a.h5', tmp_path/'b.h5'
    make_output(a, [1, 2])
    make_output(b, [1, 2], **changed)
    with h5py.File(a) as old, h5py.File(b) as new:
        with pytest.raises(AssertionError):
            compare.direct_rows(old, new, 'model')


def test_zero_and_negative_lf_values_are_not_floored_into_apparent_agreement():
    result = compare.shifts([0, -1, 1, np.nan], [0, 2, 10, 2], True)
    assert np.isnan(result[[0, 1, 3]]).all()
    assert result[2] == 1


def test_migration_cannot_change_sample_or_enable_new_size_physics():
    root = ET.fromstring('<parameters><galacticStructureSolver value="simple"/>'
                         '<test value="0.3"/></parameters>')
    row, specs = {'test': '0.3'}, [{'short_name': 'test', 'path': 'test'}]
    prepare.validate_parameters(root, row, specs)
    root.find('test').set('value', '0.4')
    with pytest.raises(ValueError, match='Parameter changed'):
        prepare.validate_parameters(root, row, specs)
    root.find('test').set('value', '0.3')
    ET.SubElement(root, 'barInstabilitySpheroidAngularMomentum', value='bindingEnergy')
    with pytest.raises(ValueError, match='Energy-based'):
        prepare.validate_parameters(root, row, specs)


def pilot_metadata(tmp_path):
    path = tmp_path/'pilot.json'
    path.write_text(json.dumps(dict(target_revision=None, upstream_branch='master',
                                   required_pull_requests=[1499, 1665, 1682])))
    return path


def test_open_pr_prevents_pinning_without_changing_manifest(tmp_path, monkeypatch):
    path = pilot_metadata(tmp_path)
    before = path.read_bytes()
    def api(endpoint):
        return {'merged': endpoint != 'pulls/1682', 'merge_commit_sha': 'a'*40}
    monkeypatch.setattr(prepare, 'github_json', api)
    with pytest.raises(ValueError, match='#1682 is not merged'):
        prepare.pin_merged_master(SimpleNamespace(pilot=tmp_path))
    assert path.read_bytes() == before


def test_merged_pr_must_actually_be_in_selected_master(tmp_path, monkeypatch):
    path = pilot_metadata(tmp_path)
    before = path.read_bytes()
    def api(endpoint):
        if endpoint.startswith('pulls/'):
            return {'merged': True, 'merge_commit_sha': 'a'*40}
        if endpoint.startswith('commits/'):
            return {'sha': 'b'*40}
        return {'status': 'diverged', 'behind_by': 1}
    monkeypatch.setattr(prepare, 'github_json', api)
    with pytest.raises(ValueError, match='does not contain'):
        prepare.pin_merged_master(SimpleNamespace(pilot=tmp_path))
    assert path.read_bytes() == before


def test_successful_pin_records_fixes_and_is_not_moved_on_retry(tmp_path, monkeypatch):
    path = pilot_metadata(tmp_path)
    def api(endpoint):
        if endpoint.startswith('pulls/'):
            return {'merged': True, 'merge_commit_sha': endpoint.split('/')[-1].zfill(40)}
        if endpoint.startswith('commits/'):
            return {'sha': 'b'*40}
        return {'status': 'ahead', 'behind_by': 0}
    monkeypatch.setattr(prepare, 'github_json', api)
    prepare.pin_merged_master(SimpleNamespace(pilot=tmp_path))
    meta = json.loads(path.read_text())
    assert meta['target_revision'] == 'b'*40
    assert set(meta['required_fix_commits']) == {'1499', '1665', '1682'}
    before = path.read_bytes()
    def no_network(endpoint):
        pytest.fail('An existing successful pin must not follow a moving branch')
    monkeypatch.setattr(prepare, 'github_json', no_network)
    prepare.pin_merged_master(SimpleNamespace(pilot=tmp_path))
    assert path.read_bytes() == before


def test_unknown_revision_requires_verified_binary(tmp_path):
    exe = tmp_path/'Galacticus.exe'
    exe.write_bytes(b'prepared executable')
    config = {'executable': str(exe), 'executable_sha256': prepare.sha(exe)}
    meta = {'target_revision': 'expected'}
    with pytest.raises(FileNotFoundError):
        prepare.checked_revision('unknown', tmp_path, config, meta)
    evidence = dict(source_revision='expected', executable_sha256=prepare.sha(exe),
                    method='prepared_checksum_and_clean_checkout')
    (tmp_path/'verified_build.json').write_text(json.dumps(evidence))
    assert prepare.checked_revision('unknown', tmp_path, config, meta)['reported_revision'] == 'unknown'
    with pytest.raises(ValueError):
        prepare.checked_revision('different-revision', tmp_path, config, meta)
    exe.write_bytes(b'changed executable')
    with pytest.raises(ValueError):
        prepare.checked_revision('unknown', tmp_path, config, meta)


def test_comparison_binds_unknown_revision_to_output(tmp_path):
    directory = tmp_path/'updated/evaluations/model'
    directory.mkdir(parents=True)
    output = directory/'galacticus.hdf5'
    output.write_bytes(b'original result')
    evidence = dict(source_revision='expected', executable_sha256='binary-hash',
                    method='prepared_checksum_and_clean_checkout')
    (tmp_path/'verified_build.json').write_text(json.dumps(evidence))
    (directory/'complete.json').write_text(json.dumps(dict(
        source_revision='expected', revision_provenance=evidence,
        output_sha256=prepare.sha(output))))
    compare.validate_revision('unknown', 'expected', directory)
    output.write_bytes(b'changed result')
    with pytest.raises(ValueError, match='Output changed'):
        compare.validate_revision('unknown', 'expected', directory)


def test_blanc_covariance_correction_preserves_both_scales(tmp_path):
    a, b = tmp_path/'a.h5', tmp_path/'b.h5'
    for path, values, variances in [(a, [8., 8.], [0.000025, 0.]),
                                    (b, [8.065, 8.07], [0.004225, 0.0049])]:
        make_output(path, values, logarithmic=False)
        with h5py.File(path, 'r+') as f:
            f.move('analyses/test', 'analyses/massMetallicityBlanc2019')
            f['analyses/massMetallicityBlanc2019/cov'][...] = np.diag(variances)
    with h5py.File(a) as old, h5py.File(b) as new:
        rows = compare.direct_rows(old, new, 'model')
    np.testing.assert_allclose([r['delta_over_updated_sigma'] for r in rows], [1., 1.])
    assert rows[0]['delta_over_historical_sigma'] == pytest.approx(13.)
    assert np.isnan(rows[1]['delta_over_historical_sigma'])
    assert all(r['target_covariance_changed'] for r in rows)
    with h5py.File(b, 'r+') as f:
        f['analyses/massMetallicityBlanc2019/cov'][0, 1] = 0.001
    with h5py.File(a) as old, h5py.File(b) as new:
        with pytest.raises(AssertionError):
            compare.direct_rows(old, new, 'model')


def test_unreviewed_covariance_change_is_rejected(tmp_path):
    a, b = tmp_path/'a.h5', tmp_path/'b.h5'
    make_output(a, [1., 2.])
    make_output(b, [1., 2.])
    with h5py.File(b, 'r+') as f:
        f['analyses/test/cov'][0, 0] = 5.
    with h5py.File(a) as old, h5py.File(b) as new:
        with pytest.raises(ValueError, match='Unreviewed'):
            compare.direct_rows(old, new, 'model')
