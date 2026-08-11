# Interactive Emulator App

This is the recommended local path for exploring a trained Galacticus emulator
with browser sliders.

The definitive example campaign is:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced
```

The app uses the trained products in that campaign's pipeline output:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/pipeline/emulators/standard_observables/pca_99/standard_observables_bundle.joblib
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/pipeline/emulators/emission_line_lfs/pca_99/halpha_sobral_1dustdraw_pca99.joblib
```

## Use My Trained Emulators

From the repository root, run:

```bash
python scripts/serve_pipeline_emulator_app.py
```

Then open:

```text
http://127.0.0.1:8010/
```

The launcher accepts a campaign root, a `pipeline/` directory, or a
`pipeline/emulators/` directory:

```bash
python scripts/serve_pipeline_emulator_app.py \
  runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced
```

The landing page should show:

- `Interactive Observable Suite`, backed by the standard-observable PCA-GP
  bundle.
- `Interactive Emission-Line LFs`, backed by the H-alpha Sobral sidecar LF
  PCA-GP bundle, when that bundle is present.

By default this launcher hides the older one-off SMF and H-alpha demo bundles,
so the local app is focused on the campaign that is intended to be the reusable
example. To show those legacy demos as well, pass:

```bash
python scripts/serve_pipeline_emulator_app.py --include-demo-defaults
```

## Run Your Own Campaign

The corresponding "run your own campaign" layer is the pipeline that produced
the files above. The main reusable inputs are:

```text
configs/pipelines/standard_observables_plus_halpha.yaml
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/campaign_design.json
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/samples.csv
```

For a new campaign, use the same pipeline shape and then point the local app at
the new campaign root:

```bash
python scripts/serve_pipeline_emulator_app.py runs/campaigns/YOUR_CAMPAIGN
```

The app discovers:

```text
pipeline/emulators/standard_observables/*/standard_observables_bundle.joblib
pipeline/emulators/emission_line_lfs/*/*.joblib
```

The full Galacticus HDF5 outputs are not needed to run the browser app once the
emulator bundles have been trained. They are only needed if someone wants to
derive a new observable from the original Galacticus runs and then train or
validate a new emulator for that derived observable.

## Direct Uvicorn Entry Point

For deployment-like testing, the FastAPI app can also be run directly:

```bash
PYTHONPATH=src uvicorn galacticus_emu.interactive_fastapi:app \
  --host 127.0.0.1 \
  --port 8010
```

In that mode, set `INTERACTIVE_OBSERVABLES_BUNDLE_PATH` and
`INTERACTIVE_SIDECAR_LF_BUNDLE_PATH` if you want to force particular bundle
files. When the definitive campaign exists locally, the FastAPI defaults prefer
its trained products over the small checked-in demo artifacts; the launcher
above also sets those variables explicitly and hides the older one-off demos.
