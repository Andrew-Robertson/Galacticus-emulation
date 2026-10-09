# Campaign Compact-Tail Diagnostics

`scripts/analyze_campaign_compact_tail.py` extracts small, portable diagnostics
from full Galacticus catalogs. It reads one run and one snapshot at a time,
never modifies the catalogs, and uses dendros when available. The h5py fallback
uses the same raw fields and explicitly checks completion, units and weights.
Dependencies are numpy, h5py and matplotlib, already in the emulator environment.

## Definition

Defaults reproduce the scientific selection of the earlier compact-tail figure:

- Snapshot nearest z=0, within 0.02 in redshift (configurable).
- Total stellar mass >= 1e8 Msun, spheroid stellar mass in [1e6,1e12) Msun.
- Finite positive spheroid scale radius and positive EPS weight. No B/T or sigma cut.
- Weights = mergerTreeWeight times nodeSubsamplingWeight, with complete,
  non-overlapping tree row ranges required.
- Residual = log10(spheroidRadius/kpc) minus the model's own weighted median
  relation at the galaxy's spheroid stellar mass. This is not a residual from
  observations or from a common reference model. Scaling all sizes equally
  therefore does not change compactness.
- Medians use 0.25-dex mass bins and at least 20 galaxies per bin. Linear
  interpolation is permitted only between adjacent supported bins: no
  extrapolation and no bridging missing bins. This is a stricter sparse-bin
  safeguard than the earlier illustration. Counts, effective counts, median
  relations and residual coverage are saved so exclusions can be inspected.
- Six 0.5-dex distribution bins span log10(Mstar,sph)=7.75 to 10.75, including
  the original 8.25-8.75 bin. PDFs use 0.1-dex residual bins; CDFs are evaluated
  exactly at 0.05-dex thresholds, using strict residual < threshold.
- PDF normalization includes objects outside the displayed -5 to +2 dex range;
  underflow/overflow fractions are saved. CDF zeros remain genuine zeros.

Each model is normalized independently. Campaign plots use equal-model medians
and central 68/95% bands, not a pooled galaxy distribution. These bands describe
parameter variation, not sampling errors. Effective counts describe weight
concentration and do not account for galaxies sharing the same tree.

## OBS-HPC Pilot

Activate the existing emulator Python environment. From the repository root:

```bash
RUNS=/home/arobertson/Galacticus/Galacticus-emulation/runs/campaigns/equilibriumSizesEnergy/pilot_128/results
OUT="$RUNS/analysis/compact_tail_z0"
PYTHON=$(command -v python)
SCRIPT="$PWD/scripts/analyze_campaign_compact_tail.py"

sbatch --job-name=compact-tail-pilot --partition=obs \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=03:00:00 \
  --output="$RUNS/logs/compact-tail-%j.out" \
  --error="$RUNS/logs/compact-tail-%j.err" \
  --wrap="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '$PYTHON' '$SCRIPT' extract '$RUNS' --output '$OUT' --label 'Equilibrium E pilot'"
```

If the `results` symlink is absent, use the physical run directory instead:
`/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_eps`.
The output directory must be fresh, to avoid mixing extractions with different
settings. One CPU is intentional: the job reads catalogs sequentially and avoids
128 simultaneous large-file reads. The resource request is an initial allowance,
not a benchmark for the full pilot.

Failed/incomplete catalogs are reported in `status.csv` and `manifest.json`, not
silently included as zero-density models. In particular, the known failed
evaluations 0018 and 0076 must not contribute. A readable partial `/analyses`
group is not evidence that a catalog completed. Each input must have root
`statusCompletion=0` and full `/Outputs` data. The script continues past per-run
errors and plots successful extractions, but raises an error if none succeeded.
Always check the reported totals before interpreting the campaign bands.

Download the whole `analysis/compact_tail_z0` directory. It contains:

- PNG and PDF figures, including the 8.25-8.75 mass-bin PDF/CDF.
- `mass_bin_summary.csv`: tail fractions, counts and residual coverage per model/bin.
- `sigma_bin_summary.csv`: compact fraction and BH masses versus sigma.
- `evaluations/*/compact_tail.npz`: small arrays for local overlays without catalogs.
- Per-model metadata and the 100 most negative residual galaxies, with tree/node
  IDs when available, for targeted follow-up on the cluster.
- Completion status and a manifest with definitions and script hash.

## Interpreting the Sigma Diagnostic

For each sigma bin, compute the weighted fraction with residual < -1 dex among
galaxies with supported residuals. Show the weighted mean log BH mass before and
after removing those galaxies, using the same residual-supported parent sample.
Unknown residuals are never classified as noncompact. A coverage column records
their contribution, and compact-fraction curves require >=50% residual coverage.
The unrestricted selected-sample BH mean is also saved in the CSV and NPZ.

This is a diagnostic, not an exact reconstruction of the McConnell & Ma
outputAnalysis: it uses the selected snapshot and the mass cuts above. A rise in
BH mass after removing compact galaxies would support the proposed explanation
for the downturn, but should be followed by inspection of the identified galaxies.
Missing optional BH/sigma fields are recorded. Multiple dispersion columns are
rejected rather than guessing which aperture to use.

## Historical Campaign on Caltech

Use the same standalone script with the **full**, unreduced campaign directory:

```bash
python scripts/analyze_campaign_compact_tail.py extract /path/to/full/campaign \
  --output /path/to/data/old_simple_compact_tail_z0 --label 'Historical simple sizes'
```

The directory must contain `samples.csv` and `evaluations/<evaluation_id>/galacticus.hdf5`.
Use `--input-filename` if the full catalogs have another filename. The local
`sobol_1024_20p_simpleSizes_moreHalos_reduced` files cannot supply these diagnostics:
their individual-galaxy `/Outputs` were removed. Saved means cannot recover size tails.
No Galacticus executable is needed to run this extraction on existing catalogs.

Replot downloaded summaries locally with:

```bash
python scripts/analyze_campaign_compact_tail.py plot /path/to/compact_tail_z0 \
  --label 'Equilibrium E pilot'
```
