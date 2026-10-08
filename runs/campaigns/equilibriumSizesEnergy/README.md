# Equilibrium Sizes with Energy-Based Bar Spheroids

This campaign family tests the updated Galacticus model as a whole, then provides
a reproducible parameter design for a future GP training campaign. It is not a
set of one-change-at-a-time comparisons with the historical paper calibration.
No Galacticus executions or scheduler submissions are performed by generation.

## Layout

```text
equilibriumSizesEnergy/
  README.md
  family_definition.json        frozen priors, fixed physics, provenance
  design_1024.csv                master 21-dimensional Sobol design
  pilot_128/                    generated first 128 points
    README.md
    campaign_design.json
    samples.csv
    evaluations/*/model_changes.xml
    evaluations/*/galacticus_input_values.json
  production_1024/              generate later, after reviewing the pilot
```

The family definition and master table are version-controlled; generated child
campaigns and simulation outputs are ignored. A clone can regenerate the pilot
from the saved table without the historical campaign or the same SciPy version.

## Physics and Parameters

All models use equilibrium structure and `bindingEnergy` bar-spheroid growth.
The fixed settings are:

| Parameter | Value |
| --- | --- |
| Spheroid retention fraction | 1 |
| Disk binding form factor, c_d | 0.5 |
| Interaction coefficient, f_int | 1 |
| Baryonic B/T transition | 0.003 to 0.010 |
| Relative finite-difference step | 0.01 |
| angularMomentumSpecificGrowthFromAccretionRate | false |
| Blitz2006 pressureCharacteristic | 35000 K cm^-3 |
| Blitz2006 pressureExponent | 0.92 |

The Blitz values are explicitly pinned to the current defaults inspected when
this family was created, rather than inherited from an old processed XML.
The pressure value is P/k_B in the units Galacticus accepts. Other unspecified
physics comes from the chosen migrated base XML and executable: those must be
recorded and kept consistent across a run set.

There are **21 free parameters**:

- 19 retain the exact prior definitions and XML paths from
  `sobol_1024_20p_simpleSizes_moreHalos_reduced/campaign_design.json`.
- `barFractionAngularMomentumRetainedDisk` replaces the old spheroid-retention
  coordinate. Its prior is exactly the reflected prior implied by the old
  constraint f_d = 1 - f_s, including its shape, not just its 0.05--0.95 bounds.
  Its inverse CDF is 1 - F_old^-1(1-u). The spheroid factor is now fixed to one;
  there is no remaining expression coupling it to disk retention.
- `barSpheroidFormFactor` varies c_s, the spheroid binding-energy coefficient,
  **log-uniformly from 0.25 to 1.0**. This exploratory range spans a factor of two
  either side of 0.5; it is not an observationally inferred prior. Only coefficient
  ratios affect the ideal energy constraint, so c_d and f_int remain fixed.

The complete parameter paths and priors are in `family_definition.json`.
This is a new 21-dimensional Sobol sequence, not a reuse of the exact old
20-dimensional sample points. Scrambling seed: 42. The stored master table is
authoritative; its checksum and NumPy/SciPy generation versions are recorded.
The first 64, 128, and 1024 points are nested designs.

## How It Was Generated

Run from the Galacticus-emulation repository in an environment with its normal
scientific dependencies (for example `galacticus-workspace`). The first command
below documents the one-time initialization and refuses to overwrite the frozen
family. It is **not needed after cloning**.

```bash
python scripts/generate_equilibrium_energy_campaign.py freeze \
  --source-design runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/campaign_design.json \
  --seed 42

python scripts/generate_equilibrium_energy_campaign.py generate \
  --name pilot_128 --n-eval 128
```

Each evaluation contains the common updated physics and its sampled parameters
in `model_changes.xml`. The campaign has the familiar `samples.csv`,
`campaign_design.json`, and `evaluations/` layout. The reflected and log-uniform
prior descriptors are new: legacy MCMC code must be taught to deserialize these
before this family can be used for inference. This pilot is not an emulator or
a calibration result.

To generate the later production design:

```bash
python scripts/generate_equilibrium_energy_campaign.py generate \
  --name production_1024 --n-eval 1024
```

This creates parameter files only. It does not launch 1024 models. If the priors
or fixed physics change after the pilot, create a separately versioned family;
do not overwrite this master table. Pilot outputs are reusable for training only
if tree sampling, executable, base physics, and output definitions also match.

## Prepare on OBS-HPC

The initial staging command supports **EPS**, using the already-migrated paper
MAP EPS base in Galacticus-structure-modelling. It does not rerun parameter
migration. Use a current, rebuilt binding-energy executable. UNIT preparation
is deliberately not wired in yet; select and validate its tree sample before
the production campaign. The parameter design itself is tree-independent.

The pilot defaults to **500 trees per decade**, versus 5000 in the paper MAP
run. It preserves the base mass range, resolution, output epochs, seed, and
on-the-fly analyses, and enables standard galaxy catalogs for size-tail and B/T
diagnostics. EPS trees are regenerated with the shared seed; this does not claim
to reuse an identical saved tree realization across every model. Halpha dust
post-processing and GP fitting are not included in this robustness pilot.

Activate an environment with Galacticus-emulation's Python dependencies first.
Set up the usual compiler/library environment for your Galacticus executable.
Then, from `/home/arobertson/Galacticus/Galacticus-emulation`:

```bash
export GALACTICUS_EXEC_PATH=/home/arobertson/Galacticus/galacticus-binding-energy
export GALACTICUS_DATA_PATH=/home/arobertson/Galacticus/datasets
export SHMR_PATH=/home/arobertson/Galacticus/make-SHMR-datasets

FAMILY=runs/campaigns/equilibriumSizesEnergy
RUNS=/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_eps

python scripts/generate_equilibrium_energy_campaign.py prepare \
  --name pilot_128 \
  --base-parameters /home/arobertson/Galacticus/Galacticus-structure-modelling/inputs/gp_paper_map_eps_params_current_master.xml \
  --executable "$GALACTICUS_EXEC_PATH/Galacticus.exe" \
  --runs-directory "$RUNS" \
  --trees-per-decade 500 --partition obs --concurrency 3 --cpus 24

# A navigable home-space link; ln fails rather than replacing an existing path.
ln -s "$RUNS" "$FAMILY/pilot_128/results"
```

Preparation snapshots the base XML, design, and executable checksum into the
run directory, generates a throttled array, and refuses to overwrite an existing
run set. Each evaluation writes `galacticus.out`, `galacticus.err`,
`params_used.xml`, and `galacticus.hdf5` in its own directory. The executable is
checked against its checksum before each task. Do not rebuild it in place during
the campaign; use an immutable versioned executable for long training runs.

Before launch, validate a representative generated evaluation on the cluster
with Galacticus's normal validation workflow. The local generator tests XML and
bookkeeping; it cannot prove that the cluster executable accepts every migrated
parameter or that dependencies and datasets are installed.

```bash
cd "$RUNS"
sbatch submit.slurm                 # all 128 points, at most three concurrently
```

Alternatively submit only the first half, then the second half after inspection:

```bash
sbatch --array=0-63%3 submit.slurm
# Later, without resubmitting the first half:
sbatch --array=64-127%3 submit.slurm
```

The array uses the established OBS-HPC allocation of 24 CPUs, 10 GB per CPU,
and 20 hours per model. These are starting resource limits, not runtime
predictions. Slurm startup logs are in `logs/`; Galacticus logs are next to each
catalog. Existing outputs are never automatically overwritten on a retry.

## Pilot Review Before Production

Check completion and runtime across the design; finite masses, radii and angular
momenta; compact and oversized tails relative to each model's median relations;
and mass functions, mass-size relations, M-sigma and B/T at several redshifts.
Repeat selected cases with tighter ODE tolerances and smaller finite-difference
steps. Finite-difference convergence is distinct from tree-sampling convergence.
Goodness of fit is informative, but this sparse pilot is not a recalibration.

No historical campaign outputs or templates are modified by these commands.
