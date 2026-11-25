"""Compare saved final joint fits without rerunning MCMC or changing either run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paper/figure_scripts'))
from generate_final_calibrated_parameters_table import load_summary, PARAMETER_GROUPS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--old-results', type=Path, required=True)
    parser.add_argument('--new-results', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--skip-corner', action='store_true')
    args = parser.parse_args()
    paths = [args.old_results.resolve(), args.new_results.resolve()]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    summaries = [load_summary(path) for path in paths]
    if summaries[0].keys() != summaries[1].keys():
        raise ValueError('Old and new parameter sets differ')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []
    lines = ['# Paper versus updated-code posterior', '',
             'Joint MAP and marginal 16th/50th/84th percentiles; units/scales match Table 5.',
             'These are numerical parameter comparisons. The disk SF frequency definition changed between code versions.',
             'The updated result is preliminary and uses the mixed-executable campaign.', '',
             '| Parameter | Scale applied | Old MAP | Old p16 / p50 / p84 | New MAP | New p16 / p50 / p84 |',
             '|---|---:|---:|---|---:|---|']
    for group, parameters in PARAMETER_GROUPS:
        for name, label, scale in parameters:
            row = {'parameter': name, 'group': group, 'paper_label': label, 'scale': scale}
            for tag, summary in zip(('old', 'new'), summaries):
                values = summary[name]
                row[tag + '_map'] = values['map'] * scale
                row[tag + '_q_map'] = values['q_map']
                for i, percentile in enumerate((16, 50, 84)):
                    row[f'{tag}_p{percentile}'] = values['posterior'][i] * scale
                    row[f'{tag}_q{percentile}'] = values['q_posterior'][i]
            records.append(row)
            interval = lambda tag: ' / '.join(f'{row[f"{tag}_p{p}"]:.4g}' for p in (16,50,84))
            lines.append(f'| {name} | {scale:g} | {row["old_map"]:.4g} | {interval("old")} | {row["new_map"]:.4g} | {interval("new")} |')
    with (out / 'table5_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (out / 'table5_comparison.md').write_text('\n'.join(lines) + '\n')
    proof = {}
    for tag, path in zip(('old', 'new'), paths):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        proof[tag] = {'path': str(path), 'sha256': digest.hexdigest()}
        subprocess.run([sys.executable, str(ROOT / 'paper/figure_scripts/generate_final_calibrated_parameters_table.py'),
                        '--results', str(path), '--output', str(out / f'table5_{tag}.tex')], check=True)
    (out / 'provenance.json').write_text(json.dumps(proof, indent=2) + '\n')
    if not args.skip_corner:
        subprocess.run([sys.executable, str(ROOT / 'scripts/plot_observable_mcmc_overlay_getdist.py'),
                        '--posterior', f'Paper={paths[0]}', '--posterior', f'Updated={paths[1]}',
                        '--parameter-set', 'auto', '--label-style', 'paper',
                        '--label-rotation-style', 'straight', '--filled', '--save-provenance',
                        '--output-path', str(out / 'paper_vs_updated_corner.png'), '--dpi', '180'], check=True)
    print('Comparison:', out, flush=True)


if __name__ == '__main__':
    main()
