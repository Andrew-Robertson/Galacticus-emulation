# Caltech pilot: 64 historical models with updated Galacticus

The comparison now permits the reviewed Blanc2019 diagonal target-covariance
correction from Galacticus PR #1590. It reports historical and updated target
sigmas and prediction shifts divided by each; these are diagnostics, not the
calibration likelihood. Zero historical variances give NaN without a floor.
`target_uncertainty_changes.csv` lists the affected bins. Bin locations and
target means must still match, other covariance changes are rejected, and
the original simulation files are unchanged. New calibration likelihoods must
use the corrected covariance, with their original masks and other error terms
reviewed separately; this comparator does not modify calibration settings.

## Launcher distribution and current pilot status (2026-11-02)

Pilot code is distributed on GitHub branch `codex/caltech-code-update-pilot`.
Fetch that branch into the Caltech Galacticus-emulation repository, then use
`git show <fixed-commit>:scripts/prepare_code_update_pilot.py` and
`git show <fixed-commit>:scripts/compare_code_update_pilot.py` to install
`pilot.py` and `compare.py` in the existing pilot directory. Back up both files
first, do not replace them during an active pilot job, and record the exact
commit in `launcher_revision.txt`. No repository checkout switch is required.
The launcher revision is separate from the pinned Galacticus physics revision.

The current pilot uses datasets revision
`3db63ead28772fa2a03321008895544347b126f5`, in a separate checkout, supplemented
with the original installation's `static/simulations/UNIT.xml`. Its environment
is stored in `updated/prepared.json`; do not rerun preparation to install a
launcher update. The older setup instructions below describe initial creation.

Model 0000 completed evolution and all 16 dust draws. Its HDF5 reports code
revision `unknown`, but the prepared executable checksum matches and the source
checkout is clean at the intended revision. `pilot.py verify-build PILOT`
records this external evidence (not proof reconstructed from the binary).
`pilot.py finish PILOT 0` validates existing simulation and dust outputs and
writes the completion record without rerunning either stage or altering HDF5.
The comparator accepts this explicit provenance and verifies the output hash.
Other reported revision mismatches remain errors. Models 0001–0063 have not
yet been submitted; compare model 0000 before submitting the remaining array.

Updated 2026-10-28 to require PRs #1499, #1665, and #1682. This bundle
supersedes `code_update_pilot_64_20261027`, whose pin omitted #1682.

PR #1682 merged while this bundle was being prepared. The distributed archive
is pinned to master commit `33645e68d767e9f24f51e4380c8c1666f3db19f7`, verified
through GitHub to contain all three PR merge commits. Their hashes and the
selection timestamp are recorded in `pilot.json`. Setup retains this pin, even
if master advances; preparation independently verifies ancestry in local Git.

When recreating a new bundle from the staging script, the initial manifest is
unpinned. Its first setup checks GitHub for all three merged PRs, verifies their
ancestry, and selects master once. It refuses to build if any fix is unmerged
or absent from master. This is a one-time check, not a background monitor.

This tests the complete upstream update, including other intervening fixes;
it does not isolate these three changes. The existing PR-only OBS executable
is not used, because the tested PR head lacks #1665.

The bundle uses evaluations 0000–0063: the exact first 64 points of the original
20-dimensional scrambled Sobol sequence. Its stored quantiles are checked
against the seed-42 sequence. Physical parameter values are copied, never
resampled. All 20 varied parameters are checked before and after migration.

The original EPS halo mass range, 5,000 trees per decade, seeds, output epochs,
ODE settings, and simple size model are retained. The pinned master's
bar-instability operator implements angular-momentum transfer directly; it does
not accept the `barInstabilitySpheroidAngularMomentum` selector from the separate
energy-model branch. The initial bundle mistakenly added that selector, which
master ignored with a warning. The generator now omits it.
The Blitz characteristic pressure is explicitly 35,000 K cm^-3 and the retired
tabulation option is removed. Other changes are those of current upstream
and its parameter migrations. The archived input and migrated XML are retained
for auditing. Matching EPS seeds does not guarantee identical merger trees
across different code versions.

## Transfer from the Mac

The prepared archive is at `dist/code_update_pilot_64_20261028.tar.gz` in the
local Galacticus-emulation repository. Use your normal Caltech access method:

```bash
scp /Users/arobertson/Documents/Projects/GalacticusEmu/Galacticus-emulation/dist/code_update_pilot_64_20261028.tar.gz \
  arobert2@login.hpc.caltech.edu:/resnick/groups/carnegie_poc/arobert2/galacticusEmulationCampaigns/
```

## On Caltech: build and prepare

Run from the environment you normally use to **compile and run Galacticus**.
The conda activation below supplies the Python environment; retain any compiler
modules and library variables required by your existing Galacticus build.
Use an interactive compute node for compilation. No cluster-specific compiler
stack is guessed here. Existing source and executables are not updated or replaced.

```bash
source /resnick/groups/carnegie_poc/arobert2/miniconda3/etc/profile.d/conda.sh
conda activate galacticus-workspace
export GALACTICUS_EMU_ROOT=/resnick/home/arobert2/GalacticusEmu/Galacticus-emulation

# These must already point to the existing source checkout and data directories.
printf 'Source: %s\nData: %s\nSHMR: %s\n' "$GALACTICUS_EXEC_PATH" "$GALACTICUS_DATA_PATH" "$SHMR_PATH"

cd /resnick/groups/carnegie_poc/arobert2/galacticusEmulationCampaigns
test ! -e code_update_pilot_64_20261028 && \
  tar -xzf code_update_pilot_64_20261028.tar.gz
export PILOT="$PWD/code_update_pilot_64_20261028"
bash "$PILOT/setup_caltech.sh"
```

The setup script now only prepares (or verifies) the separate pinned checkout.
It does not build or submit any jobs. If you already ran an older copy that
submitted `build_and_prepare.sbatch`, check that specific job and cancel it
before compiling in the same directory yourself.

On your usual interactive compute allocation, activate the compiler/library
and Python environment, then run (with at least eight CPUs allocated):

```bash
export PILOT=/resnick/groups/carnegie_poc/arobert2/galacticusEmulationCampaigns/code_update_pilot_64_20261028
export GALACTICUS_EXEC_PATH="$PILOT/source"
export PYTHONPATH="$PILOT/source/python${PYTHONPATH:+:$PYTHONPATH}"
unset BUILDPATH
cd "$GALACTICUS_EXEC_PATH"
set -o pipefail
make -j 8 Galacticus.exe 2>&1 | tee "$PILOT/build.log"
```

Only after compilation succeeds, prepare the 64 evaluations explicitly:

```bash
python "$PILOT/pilot.py" prepare "$PILOT" --source "$PILOT/source"
```

Keep `GALACTICUS_DATA_PATH`, `SHMR_PATH`, and `GALACTICUS_EMU_ROOT` set to the
existing Caltech directories. Preparation checks the pinned revision, all three
fixes' ancestry, and the 20 sampled parameters. It does not submit simulations.
If `updated/prepared.json` already exists from a successful earlier preparation,
do not run preparation over it; verify its executable hash against the rebuild
before submitting, or use a fresh pilot directory for the new build.

The static datasets directory is the one supplied by the existing environment.
Each evaluation has a fresh dynamic cache; external tools remain accessible
through `GALACTICUS_TOOLS_PATH` (defaulting to the original datasets/dynamic).
Both historical and updated dataset revisions are recorded. A changed revision
is flagged by the comparison, and changed target data/binning prevent a silent
comparison. If updated Galacticus requires missing static data or rebuilt
external tools, address that from the first job's log before launching the array.

## Run one full evaluation before releasing the other 63

Once compilation and preparation succeed:

```bash
sbatch --array=0 "$PILOT/updated/submit.sbatch"
```

This runs the original halo sample at evaluation 0000, not a reduced-tree
smoke model. It dry-runs the migrated input, runs Galacticus, checks completion
and the embedded executable revision, and generates the original 16 dust
draws using the original evaluation index and seed. Inspect:

```bash
cat "$PILOT/updated/evaluations/sobol_1024_20p_simpleSizes_moreHalos-eval-0000/complete.json"
```

Only after this exists and the job succeeded:

```bash
sbatch --array='1-63%8' "$PILOT/updated/submit.sbatch"
```

The runtime requests 16 CPUs, 8 GB per CPU, and seven days per evaluation,
matching the archived launcher. At most eight evaluations run simultaneously.
Adjust the concurrency cap if appropriate for your allocation. No full
1,024-model campaign is submitted. Scripts refuse to overwrite attempted runs;
if a task fails, inspect its logs and archive its evaluation directory before
preparing a retry. Do not delete the claim marker and rerun over existing output.

## Compare

```bash
python "$PILOT/compare.py" "$PILOT" --output "$PILOT/comparison"
```

Outputs include per-evaluation status, per-bin direct-observable shifts,
matched-dust luminosity-function shifts, and aggregate summaries. The comparison
returns a nonzero exit status if any of the 64 evaluations cannot be compared;
partial results are still written. Use a new output folder for each comparison.

Direct outputs are compared on the same bins with the same observational data.
Positive linear functions use log10(new/old); quantities already stored in log
space use subtraction. Native-space shifts divided by historical observational
standard deviations are also recorded. These are diagnostic comparisons, not
the paper's masked likelihood, and do not include emulator uncertainty.
Zero/negative/nonfinite values are retained and flagged; relation-bin sentinel
zeros need the original calibration masks before scientific interpretation.
Dust draws and LF binning must match before any LF comparison is accepted.

Small broad-prior shifts would support postponing the expensive recalibration,
but would not prove posterior stability. Before making that decision, compare
the old MAP and a few posterior-region points, assess shifts against the
observational and emulator uncertainties in the fitted bins, and investigate
failed models or large outliers. A sizeable difference can justify earlier
recalibration without waiting for the spheroid-size changes.

## Recreate the bundle locally

```bash
python scripts/prepare_code_update_pilot.py stage \
  runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced \
  dist/code_update_pilot_64_20261028
cp scripts/setup_code_update_pilot_caltech.sh dist/code_update_pilot_64_20261028/setup_caltech.sh
cp docs/code_update_pilot.md dist/code_update_pilot_64_20261028/README.md
tar -czf dist/code_update_pilot_64_20261028.tar.gz -C dist code_update_pilot_64_20261028
```
