#!/usr/bin/env python3
"""Compare paired direct observables; these diagnostics are not a recalibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def validate_revision(version, target, directory):
    if version == target:
        return
    if version != 'unknown':
        raise ValueError('Wrong executable revision')
    completion = json.loads((directory/'complete.json').read_text())
    evidence = completion.get('revision_provenance', {})
    pilot = directory.parent.parent.parent
    verified = json.loads((pilot/'verified_build.json').read_text())
    if (completion.get('source_revision') != target or
        evidence.get('source_revision') != target or
        evidence.get('method') != 'prepared_checksum_and_clean_checkout' or
        evidence.get('executable_sha256') != verified.get('executable_sha256') or
        verified.get('source_revision') != target):
        raise ValueError('Missing or mismatched external revision provenance')
    with (directory/'galacticus.hdf5').open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    if checksum != completion.get('output_sha256'):
        raise ValueError('Output changed since completion validation')


def shifts(old, new, logarithmic):
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    valid = np.isfinite(old) & np.isfinite(new)
    if logarithmic:
        valid &= (old > 0) & (new > 0)
    delta = np.full(old.shape, np.nan)
    delta[valid] = np.log10(new[valid]/old[valid]) if logarithmic else new[valid]-old[valid]
    return delta


def direct_rows(old, new, name):
    result = []
    if set(old['analyses']) != set(new['analyses']):
        raise ValueError('Analysis inventory changed; inspect before comparing')
    for key, a in old['analyses'].items():
        if text(a.attrs.get('type', '')) != 'function1D':
            continue
        b = new['analyses'][key]
        xkey, ykey = (text(a.attrs[k]) for k in ['xDataset', 'yDataset'])
        for attr in ['xDataset', 'yDataset', 'yAxisIsLog']:
            if text(a.attrs[attr]) != text(b.attrs[attr]):
                raise ValueError(f'Changed observable metadata: {key}/{attr}')
        x, old_y, new_y = a[xkey][...], a[ykey][...], b[ykey][...]
        np.testing.assert_allclose(x, b[xkey][...], rtol=1e-12, atol=0)
        if old_y.shape != new_y.shape or old_y.ndim != 1:
            raise ValueError(f'Changed shape: {key}')
        logarithmic = bool(a.attrs['yAxisIsLog'])
        delta = shifts(old_y, new_y, logarithmic)
        sigma = np.full(old_y.shape, np.nan)
        new_sigma = np.full(old_y.shape, np.nan)
        covariance_changed = False
        target = np.full(old_y.shape, np.nan)
        target_key = text(a.attrs.get('yDatasetTarget', ''))
        cov_key = text(a.attrs.get('yCovarianceTarget', ''))
        if target_key in a:
            target = a[target_key][...]
            np.testing.assert_allclose(target, b[target_key][...], rtol=1e-12, atol=0, equal_nan=True)
        if cov_key in a:
            covariance = a[cov_key][...]
            new_covariance = b[cov_key][...]
            if covariance.shape != new_covariance.shape or covariance.shape != (len(x), len(x)):
                raise ValueError(f'Changed covariance shape: {key}')
            covariance_changed = not np.allclose(covariance, new_covariance, rtol=1e-12, atol=0, equal_nan=True)
            if covariance_changed:
                if key != 'massMetallicityBlanc2019':
                    raise ValueError(f'Unreviewed target covariance change: {key}')
                off_diagonal = ~np.eye(len(x), dtype=bool)
                np.testing.assert_allclose(covariance[off_diagonal], new_covariance[off_diagonal],
                                           rtol=1e-12, atol=0, equal_nan=True,
                                           err_msg=f'Changed off-diagonal covariance: {key}')
                if (not np.all(np.isfinite(np.diag(new_covariance))) or
                    not np.all(np.diag(new_covariance) > 0) or
                    not np.all(np.diag(new_covariance) >= np.diag(covariance))):
                    raise ValueError(f'Unexpected Blanc variance change: {key}')
            with np.errstate(invalid='ignore'):
                sigma = np.sqrt(np.diag(covariance))
                new_sigma = np.sqrt(np.diag(new_covariance))
        # Native-space shift / historical observational sigma, not a likelihood change.
        ratio = np.divide(new_y-old_y, sigma, out=np.full(old_y.shape, np.nan),
                          where=np.isfinite(sigma) & (sigma > 0))
        new_ratio = np.divide(new_y-old_y, new_sigma, out=np.full(old_y.shape, np.nan),
                              where=np.isfinite(new_sigma) & (new_sigma > 0))
        for i in range(len(x)):
            result.append(dict(evaluation_id=name, observable=key, bin_index=i, x=float(x[i]),
                old=float(old_y[i]), new=float(new_y[i]), shift=float(delta[i]),
                shift_units='dex' if logarithmic else 'native_y',
                delta_over_observational_sigma=float(ratio[i]), target=float(target[i]),
                target_sigma=float(sigma[i]),
                historical_target_sigma=float(sigma[i]), updated_target_sigma=float(new_sigma[i]),
                delta_over_historical_sigma=float(ratio[i]), delta_over_updated_sigma=float(new_ratio[i]),
                target_covariance_changed=covariance_changed,
                status='valid' if np.isfinite(delta[i]) else 'nonpositive_or_nonfinite'))
    return result


def dust_rows(old_dir, new_dir, name):
    with (old_dir/'dust_draws.csv').open() as f:
        old_draws = list(csv.DictReader(f))
    with (new_dir/'dust_draws.csv').open() as f:
        new_draws = list(csv.DictReader(f))
    if len(old_draws) != len(new_draws):
        raise ValueError('Dust draw count changed')
    for a,b in zip(old_draws, new_draws, strict=True):
        for k in ['delta_0', 'delta_z', 'delta_M', 'delta_Mz', 'attenuation_scatter']:
            if not np.isclose(float(a[k]), float(b[k]), rtol=1e-12, atol=1e-14):
                raise ValueError(f'Dust draw changed: {k}')
    keys = ['dust_draw_index', 'observable', 'sample_label', 'bin_index']
    def load(folder):
        with (folder/'emission_line_dust_lf_long.csv').open() as f:
            rows = list(csv.DictReader(f))
        result = {tuple(r[k] for k in keys):r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate dust comparison keys')
        return result
    old, new = load(old_dir), load(new_dir)
    if old.keys() != new.keys():
        raise ValueError('Dust LF bins changed')
    results = []
    for k,a in old.items():
        b = new[k]
        for field in ['log10_luminosity_min', 'log10_luminosity_center', 'log10_luminosity_max', 'redshift']:
            if not np.isclose(float(a[field]), float(b[field]), rtol=1e-12, atol=1e-12):
                raise ValueError(f'Dust LF grid changed: {field}')
        before, after = float(a['phi_mpc3_dex']), float(b['phi_mpc3_dex'])
        delta = shifts([before], [after], True)[0]
        results.append(dict(evaluation_id=name, observable=a['observable']+'_'+a['sample_label'],
            dust_draw_index=a['dust_draw_index'], bin_index=a['bin_index'],
            x=a['log10_luminosity_center'], old=before, new=after, shift=float(delta),
            shift_units='dex', status='valid' if np.isfinite(delta) else 'nonpositive_or_nonfinite'))
    return results


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.pilot/'pilot.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    direct, dust, status = [], [], []
    for record in meta['evaluations']:
        name = record['evaluation_id']
        old, new = (args.pilot/k/name for k in ['baseline', 'updated/evaluations'])
        if not (new/'complete.json').exists():
            status.append(dict(evaluation_id=name, status='not_complete', detail='No complete.json'))
            continue
        try:
            with h5py.File(old/'galacticus.hdf5') as a, h5py.File(new/'galacticus.hdf5') as b:
                if b.attrs.get('statusCompletion', -1) != 0:
                    raise ValueError('Nonzero/missing completion status')
                validate_revision(text(b['Version'].attrs['gitHash']), meta['target_revision'], new)
                these_direct = direct_rows(a, b, name)
                data_note = ('same_datasets_revision' if text(a['Version'].attrs['gitHashDatasets']) ==
                             text(b['Version'].attrs['gitHashDatasets']) else 'datasets_revision_changed')
            these_dust = dust_rows(old/'emission_line_dust', new/'emission_line_dust', name)
            direct.extend(these_direct)
            dust.extend(these_dust)
            status.append(dict(evaluation_id=name, status='compared', detail=data_note))
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            status.append(dict(evaluation_id=name, status='comparison_failed', detail=str(exc)))
    write_csv(args.output/'evaluation_status.csv', status)
    write_csv(args.output/'direct_observable_shifts.csv', direct)
    uncertainty_fields = ['evaluation_id', 'observable', 'bin_index', 'x',
                          'historical_target_sigma', 'updated_target_sigma',
                          'delta_over_historical_sigma', 'delta_over_updated_sigma']
    write_csv(args.output/'target_uncertainty_changes.csv',
              [{k: row[k] for k in uncertainty_fields} for row in direct if row['target_covariance_changed']])
    write_csv(args.output/'dust_lf_shifts.csv', dust)
    summary = []
    for family, rows in [('direct',direct), ('dust',dust)]:
        for observable in sorted({r['observable'] for r in rows}):
            selected = [r for r in rows if r['observable'] == observable]
            delta = np.array([r['shift'] for r in selected])
            good = delta[np.isfinite(delta)]
            summary.append(dict(family=family, observable=observable,
                n_evaluations=len({r['evaluation_id'] for r in selected}),
                n_bins_or_draw_bins=len(selected), n_invalid=int(np.sum(~np.isfinite(delta))),
                shift_units=selected[0]['shift_units'],
                median_signed_shift=float(np.median(good)) if len(good) else float('nan'),
                median_absolute_shift=float(np.median(abs(good))) if len(good) else float('nan'),
                p90_absolute_shift=float(np.percentile(abs(good),90)) if len(good) else float('nan'),
                max_absolute_shift=float(np.max(abs(good))) if len(good) else float('nan')))
    write_csv(args.output/'summary.csv', summary)
    completed = sum(r['status'] == 'compared' for r in status)
    (args.output/'README.txt').write_text(
        f'{completed}/64 evaluations compared. Read evaluation_status.csv before interpreting summaries.\n'
        'Shifts are updated minus historical. Positive linear observables use log10(new/old);\n'
        'already-logarithmic relations retain native units. Zero/negative/nonfinite values\n'
        'remain in detailed tables and are counted as invalid, not silently floored.\n'
        'Observational-sigma ratios use native-space diagonal target covariance only;\n'
        'Historical and updated sigma ratios are both reported. Legacy target_sigma and\n'
        'delta_over_observational_sigma columns retain the historical convention. Zero\n'
        'historical variance yields NaN, with no artificial floor. The reviewed Blanc2019\n'
        'diagonal covariance correction (PR #1590) is allowed and reported separately;\n'
        'other covariance changes, changed target means, and changed grids are rejected.\n'
        'they are not the paper likelihood and do not include model or emulator errors.\n'
        'Relation bins with sentinel zeros need their original calibration masks applied.\n'
        'Dust summaries include all 16 matched draws, not independent galaxy evaluations.\n'
        'This screens a broad prior sample. Small shifts do not establish posterior stability;\n'
        'check the old MAP/posterior region before deciding to defer recalibration.\n')
    print(f'{completed}/64 compared; {args.output}')
    if completed != 64:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
