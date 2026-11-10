# Repeat the energy-size pilot at tested numerical accuracy

This batch repeats all **128 physical parameter vectors** from the October
21-parameter pilot and adds **32 tighter comparisons**. It uses the merged,
uncapped `bindingEnergy` prescription, with 500 EPS trees per decade. Historical
inputs and outputs are preserved. It does not perform a new calibration.

| Setting | Practical: 128 runs | Tighter: 32 runs |
|---|---:|---:|
| `stepFiniteDifferenceRelative` | 10^-3.5 = 0.000316227766 | 10^-4 |
| `galacticStructureSolver/solutionTolerance` | 10^-7 | 10^-8 |
| `darkMatterProfile/toleranceRelative` | 10^-10 | 10^-10 |
| ODE absolute and relative tolerances | 10^-4 | 10^-4 |

The fixed physical choices remain spheroid retention 1, disk form factor 0.5,
interaction form factor 1, and a baryonic spheroid-fraction transition of
0.003-0.010. Spheroid form factor and independent disk retention retain the
original sampled values. Blitz pressure is 35000 K cm^-3, its exponent is 0.92,
and `angularMomentumSpecificGrowthFromAccretionRate=false`.

The preparer migrates the old common base with the selected Galacticus checkout,
then applies the saved numerical coordinates in the current parameter definitions.
In particular, the sampled Blitz star-formation frequency is retained numerically;
it is not rescaled to undo the upstream frequency-convention correction. Other
upstream fixes may also change predictions. Comparisons between the new practical
and tighter runs isolate numerical accuracy; comparisons against the October
pilot also include intervening code and data changes.

## The 32 tighter points

`runs/campaigns/equilibriumSizesEnergy/precision_pilot_selection.json` records the
IDs, reasons, master-design checksum and hashes of the historical diagnostics.
Sixteen points were chosen by farthest-point sampling in the 21-dimensional
unit-prior coordinates, starting at index 0. The additional sixteen include the
ODE-failure point 18, the extremes of spheroid form factor and disk retention,
five additional points with large compact fractions, three with negative
star-forming size slopes, and further farthest points to fill the selection.
This subset is deliberately enriched in unusual models. Compare it with its 32
matching practical runs, not with the whole prior as if it were representative.

The frozen selection can be reproduced from the reduced historical campaign
(write to a new filename):

```bash
python scripts/prepare_energy_precision_pilot.py select \
  --historical /path/to/pilot_128_eps_reduced \
  --output /tmp/precision_pilot_selection.json
```

## 1. Build the merged Galacticus in the existing checkout

Use the usual compiler/library environment and an interactive compute allocation.
No fresh clone is required. Use a checkout that is not supporting ongoing jobs:
Galacticus also reads auxiliary scripts from it at runtime. The preparation and
run scripts require a clean tracked checkout containing PR #1585 and preserve an
executable copy in the run directory.

The revision below is upstream master checked on 10 November 2026. It includes
the merged size prescription and the later upstream fixes. This makes the code
choice explicit rather than following a moving branch during the batch.

```bash
# From a login shell, if you need an interactive build allocation:
srun --partition=obs --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --time=04:00:00 --pty bash

# Activate the usual Galacticus build environment here, then:
export GALACTICUS_EXEC_PATH=/home/arobertson/Galacticus/galacticus-binding-energy
cd "$GALACTICUS_EXEC_PATH"
(
  set -euo pipefail
  git diff --exit-code
  git diff --cached --exit-code
  git fetch https://github.com/galacticusorg/galacticus.git master
  git switch --detach 71254dfd7c3e41176bb882d6158137f762901430
  if [[ -e Galacticus.exe ]]; then
    mv Galacticus.exe "Galacticus_before_energy_precision_$(date +%Y%m%d_%H%M%S).exe"
  fi
  make -j8 Galacticus.exe
)
```

If the clean-checkout check reports local edits, stop there and preserve those
edits before switching revisions. This recipe does not reset or discard them.
No `make clean` is required. The executable's compiled revision is checked in each
completed output, in addition to checking the executable checksum before running.

## 2. Pull, prepare and submit on OBS-HPC

Activate the existing Python environment used for the pilot (`galacticemu`, for
example). It needs numpy, h5py, matplotlib and lxml. The preparer sets
`PYTHONPATH` to Galacticus's `python/` directory for parameter migration, so the
`List` module need not be installed separately. If needed: `python -m pip install lxml`.

```bash
cd /home/arobertson/Galacticus/Galacticus-emulation
git pull --ff-only origin main
export GALACTICUS_EXEC_PATH=/home/arobertson/Galacticus/galacticus-binding-energy
export GALACTICUS_DATA_PATH=/home/arobertson/Galacticus/datasets
export SHMR_PATH=/home/arobertson/Galacticus/make-SHMR-datasets
bash scripts/submit_energy_precision_pilot.sh
```

The helper performs migration and two Galacticus dry-run validations, then
submits **160 array tasks on `obs`, 8 CPUs and 80 GB per task, 72 hours**.
There is **no node restriction or array throttle**; Slurm schedules against
available resources. Optional overrides before invoking the helper:

```bash
export ENERGY_PILOT_CONCURRENCY=16    # 0 means no throttle (the default)
export ENERGY_PILOT_CPUS=8
export ENERGY_PILOT_WALLTIME=72:00:00
```

Defaults:

```text
Historical catalogs:
/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_eps

New runs:
/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_precision
  practical/evaluations/pilot_128-eval-0000/...    (128)
  tighter/evaluations/pilot_128-eval-0000/...       (32)
  provenance/                                  migration and frozen design
  executable/Galacticus.exe                     copied executable
  tasks.json                                   array-index -> model mapping
  manifest.json                                code/data/env/settings/checksums
  logs/                                        Slurm logs
```

Set `ENERGY_PILOT_ORIGINAL`, `ENERGY_PILOT_OUTPUT` or `ENERGY_PILOT_EXECUTABLE`
to override those paths. `GALACTICUS_EXEC_PATH` identifies the source checkout,
even when the executable itself is elsewhere. Preparation refuses to overwrite
an existing destination. Each evaluation holds `galacticus.out`, `galacticus.err`,
`params.xml`, `params_used.xml`, `run_status.json`, and the full catalog.

Home-space symlinks under `runs/campaigns/equilibriumSizesEnergy/` point to the
new full and reduced campaign directories, so both are accessible in VSCode.
The reduced link remains dangling until post-processing starts.

## 3. Automatic analysis and what to copy back

An analysis job depends on the array with `afterany`, so it reports failures
rather than remaining blocked by them. It writes a sibling directory:

```text
.../equilibriumSizesEnergy/pilot_128_precision_reduced
```

**Copy that whole reduced directory back.** It includes:

- Small HDF5 files retaining the calibration analyses and provenance, with the
  full `/Outputs` catalogs removed; no H-alpha dust post-processing is run.
- Compact-tail PDFs/CDF arrays, extreme-galaxy lists and own-mass median/16th/84th
  percentile radius relations at z = 0, 1 and 2 for both accuracy levels.
- Paired differences in median size and 68% half-width in 0.25-dex spheroid
  stellar-mass bins, with counts for both runs.
- Matched radius differences in 1-dex spheroid stellar-mass bins, assigned using
  the tighter run: signed mean/median, RMS, absolute 50th/90th/99th percentiles,
  match coverage and the fraction of squared residuals due to the largest 1%.
  There is no clipping. The galaxy sample has total stellar mass >= 10^8 Msun.
- Completion status, runtimes, processed XML and logs, including failed models.

The established catalog reader uses dendros when installed. The two precision
levels share the physical inputs and seed, but tree evolution can diverge, so
population convergence is the main test. Missing runs or unsupported mass bins
are flagged, not assigned zero predictions.

Check `analysis/summary.json` first. It should report 160 successful runs and
no failures. The stricter subset's 32 cases may take longer than the main pilot.
The old median runtime of 3 hours is not a forecast for this batch; the 72-hour
limit allows for the tighter solves and parameter-dependent runtimes.

After any reruns, regenerate into a fresh destination:

```bash
RUNS=/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_precision
python "$RUNS/pilot.py" analyze "$RUNS" --output "${RUNS}_reduced_retry"
```

The original reduced output and all full catalogs are left intact. Before
retrying an individual failed task, archive its previous output and log files;
the runner refuses to overwrite a catalog, `RUNNING` or `SUCCESS` marker.
