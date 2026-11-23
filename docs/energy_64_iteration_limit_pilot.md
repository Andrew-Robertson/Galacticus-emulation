# 64-model energy-size pilot with a configurable iteration limit

This batch uses Galacticus PR #1854 at commit
`f6648df263bf65b582f44c29683b72ab3ad27db6`. It includes the merged damping and
virial-boundary fixes, and exposes `galacticStructureSolver/countIterationsMaximum`.
The executable is copied into the campaign and its checksum is recorded.
No diagnostic source patch is applied.

## Design

Run the first 64 points of the saved 21-parameter Sobol design, then repeat
16 of those points with tighter numerical tolerances (80 jobs total).
Select the tighter subset by greedy maximin distance in the 21 prior-quantile
coordinates, beginning at index 0 and breaking ties by lowest index. The IDs are:

```text
0000 0001 0003 0016 0017 0019 0021 0024
0033 0034 0036 0043 0047 0051 0053 0062
```

Keep the historical seed assignment, physics parameters and 500 EPS trees per
decade. These are matched comparisons. The future final 1024-model campaign will
use independent seeds across evaluations, as agreed; that change is not applied
to this pilot. Existing successful instrumented runs used a different executable,
so they are comparison evidence rather than substitutes for these runs.

| Setting | Practical (64) | Tighter (16) |
|---|---:|---:|
| `stepFiniteDifferenceRelative` | 10^-3.5 | 10^-4 |
| `galacticStructureSolver/solutionTolerance` | 10^-7 | 10^-8 |
| `darkMatterProfile/toleranceRelative` | 10^-10 | 10^-10 |
| ODE absolute and relative tolerances | 10^-4 | 10^-4 |
| `galacticStructureSolver/countIterationsMaximum` | 1000 | 1000 |

Both levels use the same frozen executable. The 1000-iteration ceiling does not
force that many iterations: each solve stops when it converges. Generation and
processed-parameter checks enforce the ceiling as well as the physical inputs.
Models retain their historical `pilot_128-eval-NNNN` identifiers so comparisons
can identify the original parameter vectors.

The setup copies the existing SDSS polygon mask into campaign provenance and
into each private dynamic cache before Galacticus runs. This avoids the repeated
Bernardi/SDSS download failure. It checks for an uncompressed polygon header and
records/verifies the file checksum; dry-run validation exercises the input on
OBS-HPC. Other dynamically generated tables remain private to each evaluation.

## 1. Build on an OBS-HPC compute allocation

Use the existing checkout and compiler/library environment. The parentheses
confine `set -e` to a subshell: a failure will not close the interactive terminal.
The working tree must be clean. The executable suffix preserves other binaries
and uses the existing build directory, so this is an incremental build.

```bash
export GALACTICUS_EXEC_PATH=/home/arobertson/Galacticus/galacticus-binding-energy
(
  set -euo pipefail
  cd "$GALACTICUS_EXEC_PATH"
  git diff --exit-code
  git diff --cached --exit-code
  git fetch https://github.com/Andrew-Robertson/galacticus.git feat/equilibrium-iteration-limit
  git switch --detach f6648df263bf65b582f44c29683b72ab3ad27db6
  make -j8 Galacticus.exe SUFFIX=_iterationLimitPilot
  test -x Galacticus.exe_iterationLimitPilot
)
```

Do not switch or modify this Galacticus checkout during the batch: although the
binary is frozen, runtime auxiliary scripts still use `GALACTICUS_EXEC_PATH`.
The runner checks the source revision before each job.

## 2. Export the launcher to nobackup

This avoids changing the repository's `runs` symlink or unrelated local edits.
Use a fresh SETUP directory. All shell blocks can be pasted into an interactive
terminal; failed commands inside the subshell stop that block only.

```bash
SETUP=/carnegie/nobackup/users/arobertson/Galacticus/launchers/energy_64_iteration_limit_20261123
(
  set -euo pipefail
  cd /home/arobertson/Galacticus/Galacticus-emulation
  git fetch origin codex/energy-pilot-precision
  mkdir -p "$(dirname "$SETUP")"
  mkdir "$SETUP"
  git rev-parse origin/codex/energy-pilot-precision > "$SETUP/launcher_revision.txt"
  git archive origin/codex/energy-pilot-precision \
    scripts/prepare_energy_precision_pilot.py \
    scripts/submit_energy_64_iteration_limit_pilot.sh \
    scripts/analyze_energy_precision_pilot.py \
    scripts/analyze_campaign_compact_tail.py \
    runs/campaigns/equilibriumSizesEnergy/family_definition.json \
    runs/campaigns/equilibriumSizesEnergy/design_1024.csv \
    runs/campaigns/equilibriumSizesEnergy/precision_pilot_selection.json \
    docs/energy_64_iteration_limit_pilot.md | tar -x -C "$SETUP"
)
```

## 3. Prepare, validate and submit

Activate the `galacticemu` environment and retain the usual runtime libraries.

```bash
conda activate galacticemu
export GALACTICUS_EXEC_PATH=/home/arobertson/Galacticus/galacticus-binding-energy
export GALACTICUS_DATA_PATH=/home/arobertson/Galacticus/datasets
export SHMR_PATH=/home/arobertson/Galacticus/make-SHMR-datasets
export ENERGY_PILOT_CONCURRENCY=8
bash "$SETUP/scripts/submit_energy_64_iteration_limit_pilot.sh"
```

Defaults: `obs`, eight CPUs and 80 GB per task, 72 hours, eight concurrent tasks,
no node restriction. The array is `0-79%8`: tasks 0–63 are practical and 64–79 are
the tighter comparisons. Nothing is submitted until dry-run validation of both
levels succeeds. The analysis job uses `afterany`, including failed simulations
in its report. Existing campaign output directories are never overwritten.

The mask defaults to the file already found at:

```text
/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_solver_fixed_20261120/practical/evaluations/pilot_128-eval-0000/dynamic/surveyGeometry/SDSS/sdss_dr72safe0_res6d.pol
```

If it has moved, set `ENERGY_PILOT_SURVEY_MASK` to an existing uncompressed copy
before running the launcher. It fails early if the mask is missing or looks like
an unsuccessful download. `ENERGY_PILOT_ORIGINAL`, `ENERGY_PILOT_OUTPUT`,
`ENERGY_PILOT_EXECUTABLE`, `ENERGY_PILOT_CPUS`, and `ENERGY_PILOT_WALLTIME` can also
be overridden explicitly.

## Outputs and assessment

Default output:

```text
/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_64_iteration_limit_20261123
```

Browse via `/home/arobertson/Galacticus/nobackup/Galacticus-emulation/equilibriumSizesEnergy/`.
After analysis, copy the whole sibling directory
`pilot_64_iteration_limit_20261123_reduced` to the Mac. It includes the parameters,
statuses, logs, calibration observables and compact-tail/population summaries,
but omits full galaxy catalogs and dynamic caches.

First assess completion and failure causes. Then compare the paired models'
median sizes, scatter, compact-tail fractions and matched galaxy sizes at
z = 0, 1 and 2. Finally compare prior predictions with observations. The tighter
subset provides a numerical-accuracy check, not an independent prior sample.
The launcher does not start GP fitting or an additional production campaign.

Prepared with OpenAI Codex assistance at Andrew Robertson's request.
