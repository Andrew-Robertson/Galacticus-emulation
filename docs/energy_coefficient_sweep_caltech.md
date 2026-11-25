# Caltech matched bar-energy coefficient sweep

Prepared with OpenAI Codex at Andrew Robertson's request, 2026-11-24.

This experiment measures sensitivity to the two independent coefficients in
the continuous disk-instability binding-energy prescription. It uses two
physical parameter vectors (indices 0000 and 0001 of the saved 21-dimensional
pilot design), which completed in the current OBS-HPC iteration-limit pilot.
These are prior points, not calibrated models or a representative prior sample.

For each point, run all six combinations of:

* alpha = c_s/c_d = 0.5, 1, 2;
* beta = f_int/c_d = 2, 4.

With c_d fixed to 0.5 this means c_s = 0.25, 0.5, 1 and f_int = 1, 2.
Merger energyOrbital and disk angular-momentum retention retain each point's
sampled values. All other physics and the historical seeds remain fixed within
each comparison. The two physical points also retain the shared historical seed.
The existing sampled c_s is deliberately replaced by the grid values.

All 12 models use 500 EPS trees per decade, equilibrium sizes, bindingEnergy,
stepFiniteDifferenceRelative=10^-3.5, solutionTolerance=1e-7,
AC toleranceRelative=1e-10, ODE absolute/relative tolerance=1e-4, and
countIterationsMaximum=1000. The transition remains 0.003--0.010 in baryonic
B/T; spheroid retention is 1. This is a coefficient sensitivity experiment,
not a test of numerical convergence at every new coefficient value.

## Source and input provenance

Galacticus is pinned to f6648df263bf65b582f44c29683b72ab3ad27db6 (PR #1854),
matching the current OBS pilot. The executable is frozen after the build, with
its checksum recorded. Do not move or modify the source checkout during these
runs: runtime auxiliary scripts still use it. Each completed HDF5 file is
checked for the expected compiled revision and successful completion.

The bundled inputs/energy_coefficients/historical_base.xml is the original
pilot_128_eps_reduced/base_parameters.xml, SHA256
43cae987c6139647554ab2a3f11f763759647aa13cf8192725ba9ad1f691bd4a.
Preparation migrates it using the pinned Galacticus version, then applies the
saved design values in current parameter definitions, as for the OBS pilot.
It also enforces current Blitz parameters and the accretion angular-momentum
setting. Original and migrated inputs, the design and final XML files are saved.

The legacy family_definition.json documents the earlier pilot defaults; the
new tasks.json and final params.xml record the actual coefficients and tighter
numerical settings used here. The six coefficients are not random prior draws.

## Queue behind the build

Export this launcher from branch codex/energy-pilot-precision into a fresh
directory, keeping the existing emulation checkout and its runs symlink intact.
Required files are scripts/energy_coefficient_sweep.py,
scripts/submit_energy_coefficients_caltech.sh,
scripts/prepare_energy_precision_pilot.py,
scripts/analyze_energy_precision_pilot.py,
scripts/analyze_campaign_compact_tail.py, inputs/energy_coefficients,
runs/campaigns/equilibriumSizesEnergy/family_definition.json, and
runs/campaigns/equilibriumSizesEnergy/design_1024.csv.

Activate the existing Caltech galacticus-workspace Python environment (numpy,
h5py, matplotlib, lxml). Dendros is used for catalog reading if available.
Retain the compiler/library environment used for the build.

```bash
export GALACTICUS_EXEC_PATH=/home/arobert2/Galacticus/galacticus
export GALACTICUS_DATA_PATH=/home/arobert2/Galacticus/datasets
export SHMR_PATH=/home/arobert2/GalacticusEmu/make-SHMR-datasets
bash "$SETUP/scripts/submit_energy_coefficients_caltech.sh" 8135663
```

The setup submits three jobs immediately:

1. Preparation afterok the build: migrate, freeze inputs and executable, stage
   the SDSS mask, and dry-run all 12 generated parameter files. Eight CPUs,
   64 GB, four hours.
2. Simulation array afterok preparation: 12 tasks, four concurrent, eight CPUs,
   64 GB per task, 72 hours per task, expansion/normal.
3. Analysis afterany the array and afterok preparation: one CPU, 32 GB, eight hours.

Slurm cancels jobs whose success dependencies become impossible. Submitting
does not require that the executable already exists. Output destinations must
be new; no completed runs are overwritten. If only submission partially fails,
inspect the recorded IDs before submitting again.

Default output:
/resnick/groups/carnegie_poc/arobert2/galacticusEmulationCampaigns/equilibriumSizesEnergy/coefficient_sweep_20261124

The sibling coefficient_sweep_20261124_submission contains all job IDs and
Slurm logs. The final coefficient_sweep_20261124_reduced directory contains
calibration analyses without full catalogs, z=0,1,2 compact-tail summaries and
spheroid mass-size percentiles, parameters, timing/status records and logs.
It retains the reducer's practical/ and empty tighter/ layout; no tighter runs
are requested. Copy the entire reduced directory back for comparison plots.

Each run has its own dynamic cache. Preparation first looks for the SDSS mask
in GALACTICUS_DATA_PATH/dynamic/surveyGeometry/SDSS. If missing, it downloads
the gzip from the same Zenodo URL as Galacticus, with bounded retries, and
checks the decompressed header. ENERGY_COEFFICIENT_SURVEY_MASK can point to
an existing uncompressed copy. The mask's checksum is checked when staging it
into each private cache. Other runtime dependencies are exercised by validation.

Optional environment overrides: ENERGY_COEFFICIENT_OUTPUT,
ENERGY_COEFFICIENT_EXECUTABLE, ENERGY_COEFFICIENT_CONCURRENCY (default 4).

## Local verification

The regression tests check the paired physical coordinates and seeds, all
12 generated and processed parameter files, immutable input/script guards,
refusal to overwrite outputs, and the three Slurm dependency relationships.
They use a fixture executable; actual Galacticus validation and evolution run
on Caltech. The existing precision-pilot tests also remain applicable.
