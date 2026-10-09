#!/usr/bin/env python3
"""Extract small, EPS-weighted compact-tail diagnostics from full Galacticus catalogs.

Standalone: requires numpy, h5py and matplotlib; prefers dendros when installed.
Never modifies catalogs. Run `extract --help` or `plot --help` for usage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np

MASS_EDGES = np.arange(6., 12.001, .25)
MASS_BINS = np.array([[x, x + .5] for x in np.arange(7.75, 10.26, .5)])
PDF_EDGES = np.linspace(-5., 2., 71)
CDF_X = np.linspace(-5., 2., 141)
SIGMA_EDGES = np.arange(1.9, 2.701, .1)
QUANTILES = np.array([.01, .16, .5, .84, .99])


def quantile(x, w, q):
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = np.asarray(x)[valid], np.asarray(w)[valid]
    if not len(x):
        return np.full(np.shape(q), np.nan)
    order = np.argsort(x)
    x, w = x[order], w[order]
    return np.interp(q, (np.cumsum(w) - .5 * w) / w.sum(), x)


def neff(w):
    return float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) and w.sum() > 0 else 0.


def log(x):
    result = np.full(np.shape(x), np.nan)
    np.log10(x, out=result, where=np.isfinite(x) & (x > 0))
    return result


def expand_weights(output, n):
    """Require an exact, non-overlapping partition of node rows into trees."""
    result = np.full(n, np.nan)
    arrays = [np.asarray(output[k]) for k in
              ('mergerTreeStartIndex', 'mergerTreeCount', 'mergerTreeWeight')]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError('Inconsistent tree metadata lengths')
    for start, count, weight in zip(*arrays):
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Invalid tree weight')
        a, b = int(start), int(start + count)
        if start != a or count != int(count) or a < 0 or b < a or b > n:
            raise ValueError('Invalid tree start/count')
        if np.any(np.isfinite(result[a:b])):
            raise ValueError('Overlapping tree row ranges')
        result[a:b] = weight
    sub = np.asarray(output['nodeData/nodeSubsamplingWeight'])
    if sub.shape != (n,) or not np.all(np.isfinite(sub) & (sub >= 0)):
        raise ValueError('Invalid subsampling weights')
    if not np.all(np.isfinite(result) & (result >= 0)):
        raise ValueError('Invalid or incomplete tree weights')
    weights = result * sub
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite combined weights')
    return weights


def unit_factor(dataset, expected_si):
    """Galacticus units metadata: SI scale, descriptive name, unit name, h power."""
    if 'units' not in dataset.attrs:
        raise ValueError(f'Missing units metadata: {dataset.name}')
    unit = dataset.attrs['units']
    if len(unit) != 4 or float(unit[3]) != 0:
        raise ValueError(f'Unsupported units/h scaling for {dataset.name}: {unit}')
    factor = float(unit[0]) / expected_si
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f'Invalid unit scale for {dataset.name}')
    return factor


def read_catalog(path, redshift=0., tolerance=.02, reader='auto'):
    use_dendros = reader == 'dendros' or (reader == 'auto' and importlib.util.find_spec('dendros') is not None)
    with h5py.File(path, 'r') as f:
        if f.attrs.get('statusCompletion') != 0:
            raise RuntimeError(f"Incomplete catalog: statusCompletion={f.attrs.get('statusCompletion')}")
        if 'Outputs' not in f:
            raise ValueError('No /Outputs: full catalog required, not a reduced campaign')
        snapshots = []
        for name, g in f['Outputs'].items():
            kind = g.attrs.get('outputType', b'snapshot')
            if isinstance(kind, bytes):
                kind = kind.decode()
            a = float(g.attrs.get('outputExpansionFactor', np.nan))
            if kind == 'snapshot' and np.isfinite(a) and a > 0 and 'nodeData' in g:
                snapshots.append((name, 1 / a - 1))
        if not snapshots:
            raise ValueError('No snapshot outputs')
        name, actual_z = min(snapshots, key=lambda item: abs(item[1] - redshift))
        if abs(actual_z - redshift) > tolerance:
            raise ValueError(f'Nearest snapshot z={actual_z} is outside requested tolerance')
        output = f[f'Outputs/{name}']
        nodes = output['nodeData']
        required = ['spheroidMassStellar', 'diskMassStellar', 'spheroidRadius']
        optional = ['velocityDispersion', 'blackHoleMass', 'nodeIndex', 'mergerTreeIndex']
        names = required + [key for key in optional if key in nodes]
        for key in required:
            if key not in nodes:
                raise KeyError(f'Missing {key}')
        factors = {key: unit_factor(nodes[key], 3.08567758135e19 if key == 'spheroidRadius'
                                   else 1000. if key == 'velocityDispersion' else 1.98892e30)
                   for key in names if key not in ('nodeIndex', 'mergerTreeIndex')}
        weights = expand_weights(output, len(nodes['spheroidRadius']))
        data = {} if use_dendros else {key: np.asarray(nodes[key]) for key in names}
        version = {key: value.decode() if isinstance(value, bytes) else str(value)
                   for key, value in f['Version'].attrs.items()} if 'Version' in f else {}
    if use_dendros:
        from dendros import open_outputs
        with open_outputs(str(path)) as collection:
            collection.validate_completion()
            data = collection.read(name, {key: f'nodeData/{key}' for key in names}, as_quantity=False)
    for key in names:
        value = np.asarray(data[key])
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.shape != weights.shape:
            raise ValueError(f'{key} is not scalar per galaxy: {value.shape}; ambiguous dispersion columns are not selected automatically')
        data[key] = value * factors[key] if key in factors else value
    data['weight'] = weights
    return data, dict(source=str(path.resolve()), output=name, redshift=actual_z,
                      reader='dendros' if use_dendros else 'h5py', version=version)


def residuals(x, y, w, min_count=20):
    centers = .5 * (MASS_EDGES[1:] + MASS_EDGES[:-1])
    medians = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    effective = np.zeros(len(centers))
    for i, (lo, hi) in enumerate(zip(MASS_EDGES[:-1], MASS_EDGES[1:])):
        sel = (x >= lo) & (x < hi)
        counts[i], effective[i] = sel.sum(), neff(w[sel])
        if counts[i] >= min_count:
            medians[i] = quantile(y[sel], w[sel], .5)
    delta = np.full(len(x), np.nan)
    # Interpolate adjacent supported bins only; never bridge gaps or extrapolate.
    for i in range(len(centers) - 1):
        if np.isfinite(medians[i:i+2]).all():
            sel = (x >= centers[i]) & (x <= centers[i+1])
            delta[sel] = y[sel] - np.interp(x[sel], centers[i:i+2], medians[i:i+2])
    return delta, centers, medians, counts, effective


def distribution(r, w):
    """Normalize to ALL residual weights, including outside the histogram range."""
    total = w.sum()
    if not len(r) or total <= 0:
        return np.full(len(PDF_EDGES)-1, np.nan), np.full(len(CDF_X), np.nan)
    pdf = np.histogram(r, PDF_EDGES, weights=w)[0] / total / np.diff(PDF_EDGES)
    order = np.argsort(r)
    cumulative = np.r_[0., np.cumsum(w[order])] / total
    cdf = cumulative[np.searchsorted(r[order], CDF_X, side='left')]
    return pdf, cdf


def analyze(data, min_count=20, log_mass_min=8.):
    m = data['spheroidMassStellar']
    total = m + data['diskMassStellar']
    radius, weights = data['spheroidRadius'], data['weight']
    valid = (np.isfinite(total) & (total >= 10**log_mass_min) & (data['diskMassStellar'] >= 0) & np.isfinite(m) & (m >= 1e6)
             & (m < 1e12) & np.isfinite(radius) & (radius > 0) & (weights > 0))
    x, y, w = log(m[valid]), log(radius[valid]), weights[valid]
    delta, centers, medians, counts, eff = residuals(x, y, w, min_count)
    supported = np.isfinite(delta)
    pdf, cdf, rows = [], [], []
    for lo, hi in MASS_BINS:
        eligible = (x >= lo) & (x < hi)
        sel = eligible & supported
        r, ww = delta[sel], w[sel]
        pp, cc = distribution(r, ww)
        pdf.append(pp); cdf.append(cc)
        rows.append(dict(log_mass_low=lo, log_mass_high=hi, count=int(sel.sum()),
                         effective_count=neff(ww), eligible_count=int(eligible.sum()),
                         weight_coverage=float(ww.sum()/w[eligible].sum()) if eligible.any() else np.nan,
                         below_minus1=float(ww[r < -1].sum()/ww.sum()) if len(ww) else np.nan,
                         below_minus2=float(ww[r < -2].sum()/ww.sum()) if len(ww) else np.nan,
                         pdf_underflow=float(ww[r < PDF_EDGES[0]].sum()/ww.sum()) if len(ww) else np.nan,
                         pdf_overflow=float(ww[r > PDF_EDGES[-1]].sum()/ww.sum()) if len(ww) else np.nan))
    result = dict(mass_bins=MASS_BINS, pdf_edges=PDF_EDGES, cdf_x=CDF_X,
                  pdf=np.array(pdf), cdf=np.array(cdf), relation_mass=centers,
                  relation_median=medians, relation_count=counts, relation_neff=eff,
                  residual_quantiles=np.array([quantile(delta[(x>=lo)&(x<hi)&supported], w[(x>=lo)&(x<hi)&supported], QUANTILES)
                                               for lo, hi in MASS_BINS]),
                  sigma_edges=SIGMA_EDGES)
    sigma_rows = []
    if 'velocityDispersion' in data:
        s = log(data['velocityDispersion'][valid])
        bh = log(data.get('blackHoleMass', np.full(len(total), np.nan))[valid])
        for lo, hi in zip(SIGMA_EDGES[:-1], SIGMA_EDGES[1:]):
            sel = (s >= lo) & (s < hi)
            known = sel & supported
            compact = known & (delta < -1)
            noncompact = known & (delta >= -1)
            def mean_bh(mask):
                mask = mask & np.isfinite(bh)
                return float(np.average(bh[mask], weights=w[mask])) if mask.any() else np.nan
            sigma_rows.append(dict(log_sigma_low=lo, log_sigma_high=hi, count=int(sel.sum()),
                                   effective_count=neff(w[sel]),
                                   residual_supported_count=int(known.sum()),
                                   noncompact_count=int(noncompact.sum()),
                                   noncompact_effective_count=neff(w[noncompact]),
                                   residual_coverage=float(w[known].sum()/w[sel].sum()) if sel.any() else np.nan,
                                   compact_fraction=float(w[compact].sum()/w[known].sum()) if known.any() else np.nan,
                                   log_bh_mean_all=mean_bh(sel), log_bh_mean_supported=mean_bh(known),
                                   log_bh_mean_noncompact=mean_bh(noncompact),
                                   log_stellar_mass_median=float(quantile(log(total[valid])[sel], w[sel], .5))))
    result['sigma_compact_fraction'] = np.array([r['compact_fraction'] for r in sigma_rows])
    result['sigma_residual_coverage'] = np.array([r['residual_coverage'] for r in sigma_rows])
    for key in ['log_bh_mean_all', 'log_bh_mean_supported', 'log_bh_mean_noncompact']:
        result['sigma_'+key] = np.array([r[key] for r in sigma_rows])
    # A bounded list of actual galaxies for follow-up; extreme by residual, not sigma.
    selected = np.flatnonzero(valid)
    worst = np.flatnonzero(supported)[np.argsort(delta[supported])[:100]]
    outliers = []
    for i in worst:
        j = selected[i]
        outliers.append(dict(row_index=int(j), node_index=int(data['nodeIndex'][j]) if 'nodeIndex' in data else '',
                             tree_index=int(data['mergerTreeIndex'][j]) if 'mergerTreeIndex' in data else '',
                             delta_log_radius=float(delta[i]), log_spheroid_mass=float(x[i]),
                             log_stellar_mass=float(log(total[j])), radius_kpc=float(radius[j]),
                             stellar_bt=float(m[j]/total[j]), weight=float(weights[j]),
                             sigma_km_s=float(data['velocityDispersion'][j]) if 'velocityDispersion' in data else np.nan,
                             black_hole_mass=float(data['blackHoleMass'][j]) if 'blackHoleMass' in data else np.nan))
    info = dict(catalog_rows=len(total), selected_rows=int(valid.sum()), residual_rows=int(supported.sum()),
                zero_or_invalid_radius_rows=int(np.count_nonzero((total >= 10**log_mass_min) & (m > 0) & (~np.isfinite(radius) | (radius <= 0)))),
                optional_fields_missing=[k for k in ('velocityDispersion','blackHoleMass') if k not in data])
    return result, rows, sigma_rows, outliers, info


def write_csv(path, rows):
    with path.open('w') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def extract(args):
    root = args.campaign.resolve()
    if (root/'results/evaluations').is_dir():
        root = (root/'results').resolve()
    with (root/'samples.csv').open() as handle:
        ids = [r['evaluation_id'] for r in csv.DictReader(handle)]
    if len(ids) != len(set(ids)) or any(Path(i).name != i or i in ('.','..') for i in ids):
        raise ValueError('Invalid/duplicate evaluation IDs')
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; refusing to mix extractions')
    args.output.mkdir(parents=True)
    status, all_rows, all_sigma = [], [], []
    for index, evaluation in enumerate(ids):
        path = root/'evaluations'/evaluation/args.input_filename
        row = dict(evaluation_id=evaluation, source=str(path), status='error', detail='')
        try:
            data, meta = read_catalog(path, args.redshift, args.redshift_tolerance, args.reader)
            result, rows, sigma_rows, worst, info = analyze(data, args.min_count, args.log_stellar_mass_min)
            folder = args.output/'evaluations'/evaluation
            folder.mkdir(parents=True)
            np.savez_compressed(folder/'compact_tail.npz', **result)
            (folder/'metadata.json').write_text(json.dumps(dict(**meta, **info), indent=2)+'\n')
            write_csv(folder/'extreme_galaxies.csv', worst)
            all_rows.extend(dict(evaluation_id=evaluation, **r) for r in rows)
            all_sigma.extend(dict(evaluation_id=evaluation, **r) for r in sigma_rows)
            row.update(status='ok', detail=f"{info['residual_rows']} galaxies with supported residuals")
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            row['detail'] = str(error)
        status.append(row)
        write_csv(args.output/'status.csv', status)
        print(f"[{index+1}/{len(ids)}] {evaluation}: {row['status']} {row['detail']}", flush=True)
    write_csv(args.output/'mass_bin_summary.csv', all_rows)
    write_csv(args.output/'sigma_bin_summary.csv', all_sigma)
    manifest = dict(campaign=str(root), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    successful=[r['evaluation_id'] for r in status if r['status']=='ok'],
                    failed=[r for r in status if r['status']!='ok'],
                    definition='log10(spheroidRadius/kpc) minus own-model weighted median at spheroid stellar mass; 0.25 dex mass bins; no B/T or sigma cut',
                    interpolation='Adjacent bins with >=min_count galaxies only; no extrapolation or bridging unsupported bins',
                    weights='mergerTreeWeight times nodeSubsamplingWeight; zero-weight galaxies excluded',
                    sigma_note='z snapshot, selected spheroid sample, not an exact reproduction of the outputAnalysis selection. Compact means residual < -1 dex; fractions condition on supported residuals, coverage saved separately.',
                    uncertainty='Effective counts are weight-concentration diagnostics, not independent-tree counts. No claim of bootstrap or observational errors.',
                    normalization='Each model PDF/CDF normalized separately, before equal-model quantiles. PDF tails outside plotted range retained in normalization; CDF uses strict less-than.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not manifest['successful']:
        raise RuntimeError('No catalogs successfully extracted; inspect status.csv')
    plot(args.output, args.label)
    print(f"Finished: {len(manifest['successful'])}/{len(ids)} readable completed catalogs. Transfer {args.output}")


def plot(directory, label):
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'galacticus-compact-tail-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import warnings
    manifest = json.loads((directory/'manifest.json').read_text())
    arrays = []
    for evaluation in manifest['successful']:
        with np.load(directory/'evaluations'/evaluation/'compact_tail.npz') as f:
            arrays.append({k: f[k] for k in f.files})
    figures = directory/'figures'; figures.mkdir(exist_ok=True)
    def bands(ax, x, ys):
        if not len(ys):
            return
        values = np.asarray(ys)
        for y in values:
            ax.plot(x,y,color='#b12424',alpha=.10,lw=.6)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            q = np.nanpercentile(values,[2.5,16,50,84,97.5],axis=0)
        q[:, np.mean(np.isfinite(values),axis=0)<.5] = np.nan
        ax.fill_between(x,q[0],q[4],color='#f4b7b0',alpha=.5)
        ax.fill_between(x,q[1],q[3],color='#df7164',alpha=.5)
        ax.plot(x,q[2],color='#b12424',lw=2)
    for part in range(2):
        fig, axes = plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for col,i in enumerate(range(part*3,part*3+3)):
            lo,hi = MASS_BINS[i]
            bands(axes[0,col],.5*(PDF_EDGES[:-1]+PDF_EDGES[1:]),[a['pdf'][i] for a in arrays])
            bands(axes[1,col],CDF_X,[a['cdf'][i] for a in arrays])
            axes[0,col].set(title=rf'${lo:.2f}\leq\log_{{10}}(M_{{\star,\rm sph}}/M_\odot)<{hi:.2f}$',ylabel='Weighted PDF [dex$^{-1}$]')
            axes[1,col].set(yscale='log',ylim=(1e-4,1.05),ylabel='Weighted fraction below threshold')
            for ax in axes[:,col]:
                ax.axvline(0,color='0.5',lw=.7); ax.grid(alpha=.2)
                ax.set(xlim=(-5,2),xlabel=r'$\Delta\log_{10}R_{\rm sph}$ [dex]')
        fig.suptitle(f'{label}: compact tails relative to each model\'s own mass-size relation')
        fig.supxlabel(f"z ~ {manifest['arguments']['redshift']:g}; log10(Mstar/Msun) >= {manifest['arguments']['log_stellar_mass_min']:g}; no B/T or sigma cut. "
                      f"{len(arrays)} completed runs.\nFaint: individual models; dark/light: central 68/95%; line: median across models.\n"
                      'Zero CDFs are retained in statistics (not visible on log axes). Sparse median bins are excluded; coverage is saved in tables.',fontsize=9)
        for ext in ('png','pdf'):
            fig.savefig(figures/f'compact_tail_mass_bins_{part+1}.{ext}',dpi=160)
        plt.close(fig)
    available = [a for a in arrays if a['sigma_compact_fraction'].size]
    fig, axes = plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    axes[2].sharey(axes[1])
    centers = .5*(SIGMA_EDGES[:-1]+SIGMA_EDGES[1:])
    # A missing residual is unknown, not a noncompact galaxy.
    fractions = [np.where(a['sigma_residual_coverage']>=.5,a['sigma_compact_fraction'],np.nan) for a in available]
    bands(axes[0],centers,fractions)
    bands(axes[1],centers,[a['sigma_log_bh_mean_supported'] for a in available])
    bands(axes[2],centers,[a['sigma_log_bh_mean_noncompact'] for a in available])
    axes[0].set(title='Compact fraction (residual < -1 dex)',ylabel='Weighted fraction',ylim=(0,1))
    for ax,title in zip(axes[1:],['BH mass: residual-supported sample','BH mass: excluding compact systems']):
        ax.set(title=title,ylabel=r'Mean $\log_{10}(M_{\rm BH}/M_\odot)$')
    for ax in axes:
        ax.set(xlabel=r'$\log_{10}(\sigma/{\rm km\,s^{-1}})$',xlim=(SIGMA_EDGES[0],SIGMA_EDGES[-1])); ax.grid(alpha=.2)
    fig.suptitle(f'{label}: does compactness explain the high-sigma population?')
    fig.supxlabel('Faint: individual models; dark/light: central 68/95%; line: median. Same galaxy selection as residual PDFs.\n'
                  'Not an exact reconstruction of the calibration analysis.\n'
                  'Compact fractions require >=50% residual coverage within each model/bin. Missing residuals are not treated as noncompact.',fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(figures/f'compact_tail_sigma_diagnostic.{ext}',dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    get = sub.add_parser('extract',help='Read full catalogs sequentially and create portable summaries and figures')
    get.add_argument('campaign',type=Path,help='Campaign with samples.csv and evaluations/, or a design directory with results/ symlink')
    get.add_argument('--output',type=Path,required=True,help='Fresh output directory')
    get.add_argument('--input-filename',default='galacticus.hdf5')
    get.add_argument('--reader',choices=['auto','dendros','h5py'],default='auto')
    get.add_argument('--redshift',type=float,default=0.)
    get.add_argument('--redshift-tolerance',type=float,default=.02)
    get.add_argument('--min-count',type=int,default=20)
    get.add_argument('--log-stellar-mass-min',type=float,default=8.)
    get.add_argument('--label',default='Equilibrium E pilot')
    draw = sub.add_parser('plot',help='Regenerate plots from summaries only; no full catalogs needed')
    draw.add_argument('directory',type=Path)
    draw.add_argument('--label',default='Equilibrium E pilot')
    args = parser.parse_args()
    if args.action == 'extract':
        if args.min_count < 1 or args.redshift_tolerance < 0:
            parser.error('Require positive min-count and nonnegative redshift tolerance')
        extract(args)
    else:
        plot(args.directory,args.label)


if __name__ == '__main__':
    main()
