# Repeat the energy pilot after the size-solver fixes

Reviewed 20 November 2026. Galacticus upstream master is pinned to
`6ee60569e4ae6e3a3166b378d594187887099509` for this batch. It includes:

- PR #1798 (`9754ac45ed`): applies damping to the accepted radius proposal,
  retains the undamped residual in the convergence check, and resets history.
- PR #1802 (`4f9fcca1f6`): removes the adiabatic-response discontinuity at the
  virial radius. The response joins the unchanged halo at a matching radius
  whose orbit-averaged radius is the virial radius. The initial baryonic budget
  now uses the baryonic rotation curve at the virial radius, excluding the
  halo's outflowed reservoir. This changes physics as well as solver behaviour.
  The same PR guards the Efstathiou timescale against a zero-mass disk.

The boundary tests cover expansion, contraction, and extended exponential
baryonic disks. The merged damping tests exercise the production update, and
existing disk/spheroid reference tests check physical radii. These checks do not
establish that every broad-prior model will now complete; that is the purpose
of this repeat pilot.

## Run design

Repeat the same 128 saved 21-parameter points at practical precision, plus the
same 32-point tighter subset used by the previous precision pilot (160 tasks).
The original loose-precision pilot completed 126/128 models; a later partial
failure archive contained 63 equilibrium failures, 49 completed simulations
rejected by the old HDF5-group check, three cache-lock failures and two ODE
failures. The HDF5 check has already been fixed. All 63 equilibrium-failure
states were close to the estimated virial radius, but only the detailed scans
established the discontinuity directly.

| Parameter | 128 practical | 32 tighter |
| --- | ---: | ---: |
| `stepFiniteDifferenceRelative` | 10^-3.5 | 10^-4 |
| `galacticStructureSolver/solutionTolerance` | 10^-7 | 10^-8 |
| `darkMatterProfile/toleranceRelative` | 10^-10 | 10^-10 |
| ODE absolute and relative tolerances | 10^-4 | 10^-4 |

Use equilibrium sizes, baryonic self-gravity, Gnedin contraction, and
`bindingEnergy` with transition 0.003–0.010. Keep spheroid retention at one,
variable disk retention and spheroid form factor, and the saved priors. Retain
`angularMomentumSpecificGrowthFromAccretionRate=false`, Blitz pressure 35000
K cm^-3 and exponent 0.92. Use 500 EPS trees per decade and full catalogs plus
calibration analyses. No H-alpha dust post-processing or GP fitting is launched.
The saved numerical Blitz frequency values use the current convention, as in
the previous precision pilot. Unvaried parameters are migrated to the pinned
source revision; this is a repeat with all intervening upstream updates.

Historical seeds are preserved and checked before/after migration and in
processed XML. They are stored in the manifest and task records. This is a
matched repeat, so it does not adopt the independent seeds agreed for the future
fresh 1024-model campaign. Identical seeds do not guarantee identical evolved
catalogs across code versions or thread scheduling.

Each evaluation has a private `dynamic/` directory. External tool binaries
remain under `GALACTICUS_TOOLS_PATH` (by default the existing datasets/dynamic).
Static data remain shared. This avoids different evaluations writing the same
dynamic-cache files, at the cost of building and storing separate tables.
Caches and outputs are on nobackup. The run records the data revision; this
setup does not update datasets. Do not move the source checkout or rebuild
external tools during the batch.

## Build on OBS-HPC

Use the existing checkout and build directory on a compute allocation, with
the usual compiler/library environment. This does not require a new clone or
`make clean`. Use a checkout that is not serving active Galacticus jobs.
A subshell confines error-exit behaviour so a failure does not close the terminal.
Local tracked edits cause a stop; save/stash them before switching revisions.

```bash
export GALACTICUS_EXEC_PATH=/home/arobertson/Galacticus/galacticus-binding-energy
(
  set -euo pipefail
  cd "$GALACTICUS_EXEC_PATH"
  git diff --exit-code
  git diff --cached --exit-code
  git fetch https://github.com/galacticusorg/galacticus.git master
  git switch --detach 6ee60569e4ae6e3a3166b378d594187887099509
  if [[ -f Galacticus.exe ]]; then
    mv Galacticus.exe "Galacticus_before_solver_fixed_$(date +%Y%m%d_%H%M%S).exe"
  fi
  make -j8 Galacticus.exe
)
```

## Obtain the launcher without changing home-space symlinks

The historical `runs` symlink can conflict with Git-tracked files underneath
`runs/campaigns`. Export the required files to a fresh directory on nobackup;
this leaves the HPC checkout, local edits and symlinks unchanged.

```bash
SETUP=/carnegie/nobackup/users/arobertson/Galacticus/launchers/energy_solver_fixed_20261120
(
  set -euo pipefail
  cd /home/arobertson/Galacticus/Galacticus-emulation
  git fetch origin codex/energy-pilot-precision
  mkdir -p "$(dirname "$SETUP")"
  mkdir "$SETUP"
  git rev-parse origin/codex/energy-pilot-precision > "$SETUP/launcher_revision.txt"
  git archive origin/codex/energy-pilot-precision \
    scripts/prepare_energy_precision_pilot.py \
    scripts/submit_energy_solver_fixed_pilot.sh \
    scripts/analyze_energy_precision_pilot.py \
    scripts/analyze_campaign_compact_tail.py \
    runs/campaigns/equilibriumSizesEnergy/family_definition.json \
    runs/campaigns/equilibriumSizesEnergy/design_1024.csv \
    runs/campaigns/equilibriumSizesEnergy/precision_pilot_selection.json \
    docs/energy_solver_fixed_pilot.md | tar -x -C "$SETUP"
)
```

## Prepare and submit

Activate the existing `galacticemu` environment (or another with numpy, h5py,
matplotlib and lxml). `PYTHONPATH` for migration is supplied by the preparer.
Keep the usual runtime library environment. The executable must be freshly built
from the pinned revision. Preparation snapshots it and refuses to reuse an
existing output directory. If preparation/validation fails, inspect it before
rerunning; nothing is submitted until validation succeeds.

```bash
export GALACTICUS_EXEC_PATH=/home/arobertson/Galacticus/galacticus-binding-energy
export GALACTICUS_DATA_PATH=/home/arobertson/Galacticus/datasets
export SHMR_PATH=/home/arobertson/Galacticus/make-SHMR-datasets
export ENERGY_PILOT_CONCURRENCY=24
bash "$SETUP/scripts/submit_energy_solver_fixed_pilot.sh"
```

The array uses `obs`, eight CPUs, 80 GB and 72 hours per evaluation, at most 24
concurrent tasks, and no node restriction. The analysis job uses `afterany` so
failed simulations are reported too. Both accuracy levels are dry-run validated
before submission. The frozen runner, executable checksum, source revision,
parameter checks and lowercase HDF5 `analyses` check protect against the earlier
post-processing error. Successful simulation completion is checked separately
from diagnostic extraction.

Run files are at:

```text
/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_solver_fixed_20261120
```

In VSCode, use the existing `/home/arobertson/Galacticus/nobackup` link, then
`Galacticus-emulation/equilibriumSizesEnergy/`. When analysis finishes, copy the
whole sibling `pilot_128_solver_fixed_20261120_reduced` directory to the Mac.
It retains observables, inputs, logs, statuses and compact-tail summaries,
while omitting full catalogs and dynamic caches.

Assess completion first, then practical/tighter convergence of mass-size
relations, scatter, tails and matched systems, and finally the prior-observable
overlays. The 32 tighter models are deliberately enriched in extremes and are
not an unbiased prior sample. Compare them against their corresponding practical
runs. Do not use failed models as zero predictions.

Prepared with OpenAI Codex assistance, at Andrew Robertson's request.
