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
