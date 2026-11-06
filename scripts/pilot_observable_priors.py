#!/usr/bin/env python3
"""Export compact pilot predictions and plot empirical prior distributions."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import h5py
import numpy as np


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native(path):
    result = {}
    with h5py.File(path, 'r') as f:
        for key, g in f['analyses'].items():
            if text(g.attrs.get('type', '')) != 'function1D':
                continue
            attrs = {k: text(v) for k, v in g.attrs.items()}
            x, y = (g[text(g.attrs[k])][...].astype(float) for k in ['xDataset', 'yDataset'])
            target_key = text(g.attrs.get('yDatasetTarget', ''))
            cov_key = text(g.attrs.get('yCovarianceTarget', ''))
            target = g[target_key][...].astype(float) if target_key in g else np.full_like(y, np.nan)
            sigma = np.sqrt(np.maximum(0, np.diag(g[cov_key][...]))) if cov_key in g else np.full_like(y, np.nan)
            result[key] = dict(x=x.tolist(), y=y.tolist(), target=target.tolist(), sigma=sigma.tolist(),
                               x_log=bool(g.attrs['xAxisIsLog']), y_log=bool(g.attrs['yAxisIsLog']), attrs=attrs)
    return result


def dust(path):
    groups = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = row['observable']+'_'+row['sample_label']
            draw = int(row['dust_draw_index'])
            groups.setdefault(key, {}).setdefault(draw, []).append(row)
    result = {}
    for key, draws in groups.items():
        if set(draws) != set(range(16)):
            raise ValueError(f'Incomplete dust draws: {path}, {key}')
        curves, grid = [], None
        for draw, rows in sorted(draws.items()):
            rows.sort(key=lambda r: int(r['bin_index']))
            x = [float(r['log10_luminosity_center']) for r in rows]
            if grid is not None:
                np.testing.assert_allclose(grid, x)
            grid = x
            curves.append([float(r['phi_mpc3_dex']) for r in rows])
        result[key] = dict(x=grid, y=curves)
    return result


def process_retry_dust(pilot, config, retry, original):
    script = Path(config['emu_root'])/'scripts/process_emission_line_dust_evaluation.py'
    if sha(script) != config['dust_script_sha256']:
        raise ValueError('Dust postprocessor changed')
    out = retry/'run/emission_line_dust'
    if out.exists():
        raise FileExistsError(f'{out}: inspect existing processing instead of overwriting')
    env = dict(os.environ, **config['environment'])
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (retry/'run/dust.log').open('x') as log:
        subprocess.run([sys.executable, str(script), str(retry/'run/galacticus.hdf5'),
            '--evaluation-id', original.name, '--output-dir', str(out), '--n-dust-draws', '16',
            '--base-seed', '12345', '--evaluation-index', '26', '--scatter-mode', 'expected',
            '--z-pivot', '1.0', '--input-json', str(original/'galacticus_input_values.json')],
            cwd=retry/'run', env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def export(args):
    pilot = args.pilot.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((pilot/'pilot.json').read_text())
    config = json.loads((pilot/'updated/prepared.json').read_text())
    records = []
    for record in meta['evaluations']:
        name = record['evaluation_id']
        original = pilot/'updated/evaluations'/name
        current = original
        provenance = {}
        if record['index'] == 26:
            retry = pilot/'covariance_retry_0026'
            done = json.loads((retry/'simulation_finished.json').read_text())
            if done['statusCompletion'] != 0:
                raise ValueError('Retry did not complete')
            rc = json.loads((retry/'retry.json').read_text())
            started = json.loads((retry/'started.json').read_text())
            if rc['base_revision'] != meta['target_revision']:
                raise ValueError('Wrong retry base revision')
            if sha(retry/'run/input.xml') != rc['retry_input_sha256'] or sha(original/'input.xml') != rc['original_input_sha256']:
                raise ValueError('Retry input provenance mismatch')
            if sha(retry/'source/Galacticus.exe') != started['executable_sha256']:
                raise ValueError('Retry executable changed')
            if sha(retry/'covariance_guard.patch') != rc['patch_sha256']:
                raise ValueError('Retry patch changed')
            if args.process_retry_dust:
                process_retry_dust(pilot, config, retry, original)
            current = retry/'run'
            provenance = dict(kind='covariance_guard_retry', retry=rc, started=started,
                guard_report=(retry/'guard_report.txt').read_text(),
                caveat='Sample5 model covariance is diagnostic only; plots use mean predictions, not model covariance.')
        else:
            provenance = json.loads((original/'complete.json').read_text())
            if provenance['source_revision'] != meta['target_revision']:
                raise ValueError(f'Unexpected revision: {name}')
        with h5py.File(current/'galacticus.hdf5', 'r') as f:
            if f.attrs.get('statusCompletion', -1) != 0:
                raise ValueError(f'Incomplete simulation: {name}')
            datasets = text(f['Version'].attrs['gitHashDatasets'])
            if datasets != config['datasets_selection']['revision']:
                raise ValueError(f'Datasets mismatch: {name}')
        old = pilot/'baseline'/name
        before, after = native(old/'galacticus.hdf5'), native(current/'galacticus.hdf5')
        if before.keys() != after.keys():
            raise ValueError(f'Analysis inventory mismatch: {name}')
        for key in before:
            np.testing.assert_allclose(before[key]['x'], after[key]['x'], rtol=1e-12, atol=0)
            np.testing.assert_allclose(before[key]['target'], after[key]['target'], rtol=1e-12, atol=0, equal_nan=True)
            if (before[key]['x_log'], before[key]['y_log']) != (after[key]['x_log'], after[key]['y_log']):
                raise ValueError(f'Axis mismatch: {name}/{key}')
        old_dust, new_dust = dust(old/'emission_line_dust/emission_line_dust_lf_long.csv'), dust(current/'emission_line_dust/emission_line_dust_lf_long.csv')
        if old_dust.keys() != new_dust.keys():
            raise ValueError('Dust inventory changed')
        for key in old_dust:
            np.testing.assert_allclose(old_dust[key]['x'], new_dust[key]['x'], rtol=1e-12, atol=0)
        records.append(dict(name=name, index=record['index'], provenance=provenance,
                            old=before, new=after, old_dust=old_dust, new_dust=new_dust))
        print('Exported', name, flush=True)
    if len(records) != 64:
        raise ValueError('Expected all 64 paired models')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'wt') as stream:
        json.dump(dict(pilot=meta, datasets_revision=config['datasets_selection']['revision'], records=records), stream)
    print('Portable predictions:', args.output)


def transform(values, log_axis, missing_zero=False):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    if log_axis:
        values[values <= 0] = np.nan
        values = np.log10(values)
    elif missing_zero:
        values[values == 0] = np.nan
    return values


def draw_panel(ax, x, old, new, title, xlabel, ylabel, target=None, lower=None, upper=None):
    ax.set_title(title, fontsize=8)
    coverage = []
    limits = []
    for label, values, color in [('Original', old, '#557a9b'), ('Updated', new, '#c65c2d')]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            quantiles = np.nanpercentile(values, [16, 50, 84], axis=0)
        ax.plot(x, values.T, color=color, lw=.35, alpha=.10, rasterized=True)
        ax.fill_between(x, quantiles[0], quantiles[2], color=color, alpha=.20)
        ax.plot(x, quantiles[1], color=color, lw=1.5, label=label)
        limits.extend(quantiles[[0, 2]].ravel())
        coverage.append(np.sum(np.isfinite(values), axis=0).tolist())
    if target is not None:
        ax.plot(x, target, 'k.', ms=3, label='Target (updated errors)')
        if lower is not None:
            ax.vlines(x, lower, upper, color='black', lw=.7)
            limits.extend(lower); limits.extend(upper)
        limits.extend(target)
    finite = np.asarray(limits)[np.isfinite(limits)]
    if len(finite):
        lo, hi = finite.min(), finite.max()
        pad = max(.1*(hi-lo), .05)
        ax.set_ylim(lo-pad, hi+pad)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.15)
    ax.text(.02, .02, f'Valid/bin: old {min(coverage[0])}–{max(coverage[0])}; new {min(coverage[1])}–{max(coverage[1])}',
            transform=ax.transAxes, fontsize=6)
    return dict(observable=title, old_valid_per_bin=coverage[0], new_valid_per_bin=coverage[1])


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap
    with gzip.open(args.bundle, 'rt') as stream:
        data = json.load(stream)
    records = data['records']
    args.output.mkdir(parents=True, exist_ok=False)
    coverage = []
    for family in ['native', 'dust']:
        keys = list(records[0]['old' if family == 'native' else 'old_dust'])
        with PdfPages(args.output/f'{family}_prior_comparison.pdf') as pdf:
            for page, offset in enumerate(range(0, len(keys), 6), 1):
                fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
                for ax, key in zip(axes.flat, keys[offset:offset+6]):
                    if family == 'native':
                        ref = records[0]['new'][key]
                        missing = 'Relation' in key and 'Scatter' not in key
                        x = transform(ref['x'], ref['x_log'])
                        old, new = [transform([r[side][key]['y'] for r in records], ref['y_log'], missing) for side in ['old', 'new']]
                        target = transform(ref['target'], ref['y_log'])
                        lower = transform(np.array(ref['target'])-ref['sigma'], ref['y_log'])
                        upper = transform(np.array(ref['target'])+ref['sigma'], ref['y_log'])
                        xlabel = ('log10 ' if ref['x_log'] else '')+ref['attrs'].get('xAxisLabel', ref['attrs']['xDataset']).replace('\\\\', '\\')
                        ylabel = ('log10 ' if ref['y_log'] else '')+ref['attrs'].get('yAxisLabel', ref['attrs']['yDataset']).replace('\\\\', '\\')
                        entry = draw_panel(ax, x, old, new, '\n'.join(textwrap.wrap(key, 43)), xlabel, ylabel, target, lower, upper)
                        if key == 'stellarSizeMassRelationvanDerWel2014Sample5':
                            ax.text(.02, .95, '0026: covariance guard; means retained', transform=ax.transAxes, fontsize=6, va='top')
                    else:
                        ref = records[0]['new_dust'][key]
                        x = np.array(ref['x'])
                        old, new = [transform(np.concatenate([r[side][key]['y'] for r in records]), True) for side in ['old_dust', 'new_dust']]
                        entry = draw_panel(ax, x, old, new, key, 'log10 L [erg/s]', 'log10 phi [Mpc^-3 dex^-1]')
                    entry.update(family=family, observable=key)
                    coverage.append(entry)
                for ax in list(axes.flat)[len(keys[offset:offset+6]):]:
                    ax.set_visible(False)
                axes.flat[0].legend(fontsize=7)
                count = len(records)
                subtitle = f'{count} paired physical-parameter draws' if family == 'native' else f'{count} physical models × 16 dust draws (not {count*16} independent models)'
                fig.suptitle('Empirical prior on observables — '+subtitle+'\nMedian and pointwise 16–84% range; faint individual curves. Axes show central bands/targets; extreme curves may be clipped.', fontsize=11)
                pdf.savefig(fig)
                fig.savefig(args.output/f'{family}_page_{page:02d}.png', dpi=140)
                plt.close(fig)
    (args.output/'coverage.json').write_text(json.dumps(coverage, indent=2))
    (args.output/'README.txt').write_text('Empirical prior predictions, not posterior fits or emulator predictions.\nNative plots use updated observational errors. Dust plots show predictions only.\nAll 64 models are equally weighted; dust plots give each model 16 equally weighted draws.\nNonpositive log-axis values and zero sentinel values in mean relations are omitted per bin; coverage.json reports remaining counts. Bands are conditional on valid predictions, not a treatment of missing data.\nAxes show central 16–84% bands and targets; extreme curves may be clipped.\nModel 0026 uses the separately recorded covariance guard; its model covariance is not used in these plots.\n')
    print(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export')
    p.add_argument('pilot', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--process-retry-dust', action='store_true')
    p.set_defaults(func=export)
    p = sub.add_parser('plot')
    p.add_argument('bundle', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=plot)
    args = parser.parse_args()
    args.func(args)
