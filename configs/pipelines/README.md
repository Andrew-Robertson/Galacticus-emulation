# Campaign pipeline templates

These YAML files are reusable inputs for `scripts/run_campaign_pipeline.py`.
They are intentionally kept outside `runs/`, because campaign run products are
ignored by git.

The main variables to override for a new campaign are usually:

- `ROOT`: reduced campaign directory containing `samples.csv` and `evaluations/`.
- `PIPELINE_OUTPUT_ROOT`: where emulators, MCMCs, and figures should be written.
- PCA and dust-draw settings, either with `--set` or one of the named profiles.

Example:

```bash
python scripts/run_campaign_pipeline.py \
  --config configs/pipelines/standard_observables_plus_halpha.yaml \
  --set ROOT=runs/campaigns/my_reduced_campaign \
  --set PIPELINE_OUTPUT_ROOT=runs/campaigns/my_reduced_campaign/pipeline \
  --stage train_standard_emulators
```

On a Slurm machine, the same config can be submitted as a small dependency
graph. The `standard_halpha_mcmc` workflow trains the two emulator bundles,
runs the independent MCMCs concurrently where possible, and then runs the
corner-plot stages after their inputs are complete.

```bash
python scripts/submit_campaign_pipeline_slurm.py \
  --config configs/pipelines/standard_observables_plus_halpha.yaml \
  --workflow standard_halpha_mcmc \
  --set ROOT=/home/arobert2/GalacticusEmu/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced \
  --set PIPELINE_OUTPUT_ROOT=/home/arobert2/GalacticusEmu/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/pipeline \
  --setup-command 'source /resnick/groups/carnegie_poc/arobert2/miniconda3/etc/profile.d/conda.sh' \
  --setup-command 'conda activate galacticus-workspace' \
  --partition expansion \
  --qos normal \
  --cpus-per-task 16 \
  --mem-per-cpu 8G \
  --time 120:00:00 \
  --submit
```

Omit `--submit` first to write the batch scripts and inspect the dependency
commands without sending anything to Slurm.

Use `--workflow standard_halpha_mcmc_with_bestfits` instead if the all-standard
and all-standard-plus-H-alpha MAP Galacticus runs should also be submitted.
Those runs are written under `bestFitModel_GalacticusRun/` inside the relevant
MCMC directory.
