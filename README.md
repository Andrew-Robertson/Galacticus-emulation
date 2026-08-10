<p align="center">
  <img src="assets/galacticemu_logo.png" alt="GalacticEmu logo" width="520">
</p>

# GalacticEmu

Thin orchestration and emulation tooling for running Galacticus model evaluations one parameter vector at a time.

## Design

This repository is intended to stay lightweight:

- `GalacticEmu` owns experiment state, manifests, summaries, emulator training, and adaptive model selection.
- `Galacticus-dust-modelling` remains the home for Galacticus-facing helper code, existing calibration workflows, and post-processing utilities.
- `Galacticus-ParameterFiles` remains a separate repository of reusable Galacticus parameter files.

The first goal is a minimal, portable execution contract:

1. define a parameter vector `theta`
2. point to one or more existing Galacticus run-definition changes files
3. map `theta` to a Galacticus parameter changes file
4. run Galacticus with the base file, the selected changes files, and `theta`
5. reduce outputs to summary statistics for emulator training

## Workflow Modes

This repository is intended to support two closely related workflows.

### `low_cost_likelihood`

This mode is for cheap validation runs that are still affordable to sample directly with MCMC.

Typical characteristics:

- a small number of halos
- a small number of output times
- a likelihood-oriented run-definition changes file, such as a UniverseMachine `z=0` comparison
- used to check that emulator-based inference reproduces the behavior of direct MCMC

### `emulator_training`

This mode is for the more expensive runs that motivate emulation in the first place.

Typical characteristics:

- full halo mass-function style runs
- many output times
- direct comparison to more observationally immediate quantities, such as stellar mass functions
- used to train the emulator on a manageable number of expensive Galacticus evaluations

The `mode` field in a manifest is descriptive metadata. It does not change execution semantics by itself. It is there so manifests, outputs, and future analysis scripts can distinguish between validation-style cheap runs and emulator-training runs without changing the underlying command model.

## Layout

- `src/galacticus_emu/`: package code
- `scripts/`: primary entry points for campaign generation, extraction, training, and MCMC
- `scripts/plotting/`: reusable plotting and posterior-visualization helpers
- `scripts/playground/`: exploratory science diagnostics and one-off comparison plots
- `scripts/legacy/`: older or superseded helpers kept for reference
- `config/paths.example.toml`: machine-specific paths and defaults
- `demo_emulators/`: curated saved emulator artifacts used by live interactive demos
- `data/manifests/`: JSON manifests describing one Galacticus evaluation
- `data/summaries/`: reduced outputs suitable for emulator training
- `runs/`: generated changes files, logs, and run products
- `playing/`: local scratch outputs and generated exploratory artifacts; ignored by git

## Script Guide

The top-level `scripts/` directory is intended to contain the main workflow entry points:

- campaign generation:
  - `generate_lhs_campaign.py`
  - `generate_trinity_lhs_campaign.py`
- summary extraction:
  - `extract_campaign_hdf5_groups.py`
  - `extract_shmr_summary.py`
  - `extract_trinity_summary.py`
  - `extract_trinity_mass_metallicity_summary.py`
- emulator fitting / validation:
  - `fit_trinity_gp.py`
  - `fit_trinity_gp_all_outputs_cv.py`
  - `fit_trinity_mass_metallicity_gp_cv.py`
  - `train_trinity_all_outputs.py`
  - `compare_gp_subset_sizes.py`
  - `compare_trinity_gp_subset_sizes.py`
- emulator-based inference:
  - `run_emulator_mcmc.py`
  - `run_trinity_emulator_mcmc.py`
  - `run_trinity_mcmc_ablation_suite.py`

Plotting helpers now live under `scripts/plotting/`, and exploratory diagnostics such as mass-metallicity, `M-\sigma`, and Faber-Jackson checks live under `scripts/playground/`.
Older or superseded workflow helpers that are not the recommended starting point live under `scripts/legacy/`.
As a rule of thumb, keep a script at top level only when it is a reusable workflow CLI that another user should run directly. Put one-off diagnostics, comparison plots, and exploratory checks in `scripts/playground/` or `paper/figure_scripts/` if they exist solely to build manuscript figures. Generated outputs belong in `playing/` or `runs/`, not in `scripts/`.

## HPC HDF5 Reduction

For expensive Galacticus campaigns, the full `/Outputs` tree in each HDF5 file can be too large to copy back from an HPC. To mirror a campaign while excluding `/Outputs` and retaining lightweight top-level groups such as `/Build`, `/Parameters`, `/Version`, and `/analyses`, run:

```bash
python scripts/extract_campaign_hdf5_groups.py \
  runs/campaigns/disk_feedback_velocity_1d_mass_function-mMax1e14_32 \
  --output-root runs/campaigns/disk_feedback_velocity_1d_mass_function-mMax1e14_32_reduced \
  --copy-sidecars
```

Add `--overwrite` to replace previously reduced files. The reduced campaign directory is the one intended to transfer back for emulator-building and plotting when the raw Galacticus outputs are too large.

## Quick Start

Copy the example config and edit the paths:

```bash
cp config/paths.example.toml config/paths.toml
```

Create a starter manifest:

```bash
python scripts/smoke_test.py create-manifest
```

Print the Galacticus commands that would be run:

```bash
python scripts/smoke_test.py show-commands
```

This scaffold does not yet train a Gaussian process. It sets up the execution and bookkeeping layer that the emulator will build on.

Each manifest includes:

- a base Galacticus parameter file
- one or more existing Galacticus run-definition changes files
- a set of model parameter updates that will be written to `theta.xml`
- optional descriptive metadata such as workflow `mode`

The starter example is:

- [z0_validation_mcmc_bridge.json](/Users/arobertson/Documents/Projects/GalacticusEmu/Galacticus-emulation/data/manifests/z0_validation_mcmc_bridge.json)

It is intended to be a low-cost validation case using the existing UniverseMachine `z=0` likelihood setup.

## Environment Variables

The local workflow is intended to follow the standard Galacticus environment-variable pattern:

- `GALACTICUS_EXEC_PATH`
- `GALACTICUS_DATA_PATH`
- `GALACTICUS_PARAMETER_FILES`
- `GALACTICUS_DUST_ROOT`
- `SHMR_PATH`

The emulator config resolves paths through these variables, so the same repo can be reused across different machines with minimal path edits.

## Dry-Run LHS Campaigns

For emulator preparation, the repository can generate a dry-run Latin hypercube campaign:

```bash
python scripts/generate_lhs_campaign.py --n-eval 32
```

By default this creates a campaign under:

- `runs/campaigns/lhs_z2_mcmc_analog/`

with:

- `campaign.json`: campaign-level definition and parameter priors
- `samples.csv`: one sampled parameter vector per evaluation
- `manifests/`: one manifest JSON per evaluation
- `commands.txt`: one Galacticus command per line
- `commands.sh`: Galacticus commands for all evaluations
- generated `theta.xml` files in each evaluation run directory

The default dry-run campaign uses the 7-parameter MCMC-analog setup and composes:

- `romanEPS_local.xml`
- `handCalibratedChanges.xml`
- `z2_Likelihood.xml`
- `seed1234.xml`
- one generated `theta.xml`

This is intended as a starting point for low-cost validation runs before moving on to more expensive emulator-training campaigns.

To also generate a SLURM array submission script:

```bash
python scripts/generate_lhs_campaign.py --n-eval 128 --write-slurm-array
```

This adds:

- `submit_slurm_array.sh`

inside the campaign directory. The SLURM array uses one task per evaluation and reads commands from `commands.txt`.
The commands and generated `output.xml` files use campaign-relative paths, so the whole campaign directory can be copied to another machine and submitted there.

For OBS-HPC specifically, a useful pattern is:

```bash
python scripts/generate_lhs_campaign.py \
  --n-eval 128 \
  --campaign-name lhs_z2_mcmc_bridge_128 \
  --write-slurm-array \
  --platform-config obs-hpc \
  --slurm-conda-env galacticus-workspace
```

Then copy `runs/campaigns/lhs_z2_mcmc_bridge_128/` to OBS-HPC, `cd` into that campaign directory on the cluster, and submit:

```bash
sbatch submit_slurm_array.sh
```

After some or all evaluations have completed, extract the 3-bin SHMR mean/scatter summaries with:

```bash
python scripts/extract_shmr_summary.py runs/campaigns/lhs_z2_mcmc_analog
```

This writes:

- `summary.csv`
- `summary.jsonl`

inside the campaign directory, with the emulator-ready targets:

- `mass_stellar_log10_0..2`
- `mass_stellar_log10_scatter_0..2`

along with the corresponding target values, errors, and log-likelihood terms already stored by Galacticus.

To make some first-pass diagnostic plots of how the sampled parameters affect the SHMR means, run:

```bash
python scripts/plotting/plot_shmr_dependence.py runs/campaigns/lhs_z2_mcmc_analog
```

This writes:

- `emulator_table.csv`: `samples.csv` and `summary.csv` merged by `evaluation_id`
- `figures/main_effects_mean.png`: one panel per input parameter and mean-SHMR output
- `figures/spearman_heatmap_mean.png`: input-output Spearman correlations

To also make the analogous plots for the SHMR scatter outputs:

```bash
python scripts/plotting/plot_shmr_dependence.py runs/campaigns/lhs_z2_mcmc_analog --include-scatter
```

For historical reference, the original rough fixed-kernel GP script now lives under `scripts/legacy/`:

```bash
python scripts/legacy/fit_rough_gp.py runs/campaigns/lhs_z2_mcmc_analog
```

It fits one exact GP per output in prior-quantile input space and writes:

- `emulator/gp_metrics_mean.csv`: cross-validation metrics such as RMSE, MAE, and `R^2`
- `emulator/gp_cv_predictions_mean.csv`: cross-validated predictions and uncertainties for each evaluation
- `emulator/gp_fit_summary_mean.json`: fitted kernel summaries for the full-data fits
- `figures/gp_cv_parity_mean.png`: predicted-vs-actual parity plots for the three mean outputs

To compare nested pseudo-`128`, `256`, and `512` training sets from a larger completed campaign:

```bash
python scripts/compare_gp_subset_sizes.py runs/campaigns/lhs_z2_mcmc_bridge_512 --sizes 128 256 512
```

This writes:

- `emulator/nested_subset_order.csv`: deterministic space-filling ordering of the evaluations
- `emulator/subset_train_ids_n*.txt`: evaluation IDs used in each nested training subset
- `emulator/gp_subset_comparison_metrics_mean.csv`: metrics for each subset size
- `figures/gp_subset_parity_mean_n*.png`: parity plots for each subset size
- `figures/gp_subset_metric_rmse_mean.png`
- `figures/gp_subset_metric_r2_mean.png`

To run an `emcee` MCMC using the GP emulator in place of direct Galacticus calls:

```bash
python scripts/run_emulator_mcmc.py runs/campaigns/lhs_z2_mcmc_bridge_512 --train-size 256 --include-scatter
```

This fits GP emulators to the selected training subset, then evaluates a Gaussian likelihood against the SHMR target vector using:

- GP predictive mean as the model prediction
- target covariance from the Galacticus analysis group
- GP predictive variance added on the diagonal as emulator uncertainty

It writes:

- `emulator_mcmc/posterior_samples.csv`
- `emulator_mcmc/run_summary.json`
- `figures/emulator_mcmc_trace.png`
- `figures/emulator_mcmc_corner.png`

To visualize one-at-a-time GP predictions around the emulator MAP point:

```bash
python scripts/plotting/plot_map_one_at_a_time.py runs/campaigns/lhs_z2_mcmc_bridge_512
```

This holds all parameters fixed at the MAP point from `emulator_mcmc/posterior_samples.csv`,
varies one parameter at a time across its prior range, and writes:

- `figures/map_one_at_a_time_mean.png`

With `--include-scatter`, it also includes the three SHMR scatter outputs:

```bash
python scripts/plotting/plot_map_one_at_a_time.py runs/campaigns/lhs_z2_mcmc_bridge_512 --include-scatter
```
