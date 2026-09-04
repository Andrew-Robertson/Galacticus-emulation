# Interactive Emulator App

The interactive app uses browser sliders to show how changes to Galacticus
parameters affect predicted observables without running Galacticus or training
an emulator. The hosted version uses compact, mean-only copies of the paper
emulators, while the local launcher automatically uses the full paper
emulators when they are available.

## Hosted Demo

The hosted multi-observable emulator is available at:

<https://galacticus-emulation.onrender.com>

The hosted standard-observable and H-alpha emulators use the same Ntrain=1024
Gaussian-process mean predictions as the paper analysis. Predictive
uncertainties are omitted from the hosted bundles to reduce their disk and
memory requirements.

## Run The Demonstrations Locally

After installing GalacticEmu, run this command from the repository root:

```bash
python scripts/serve_interactive_fastapi_demo.py
```

Then open <http://127.0.0.1:8010/>. When the paper campaign is present in its
default location, the landing page provides the standard-observable suite and
the emission-line luminosity functions, including predictive uncertainty.
Without those files, it falls back to the two compact paper emulators used by
the hosted app.

To use the compact deployment bundles even when the full campaign is present,
run:

```bash
python scripts/serve_interactive_fastapi_demo.py --deployment
```

## Use The Full Paper Emulators

The larger emulator bundles trained for the paper are not stored in git. Once
they are available from the planned data archive, the app can instead be
pointed at the extracted campaign data. The paper campaign is:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced
```

The standard-observable emulator comes from the definitive paper pipeline:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/automatedPipeline_transformedParams_finalPaper_definitive/emulators/standard_observables/pca_99/standard_observables_bundle.joblib
```

The final MCMC used the Sobral-log-error H-alpha emulator at:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/scratch/sobral_log_errors_local/emulators/emission_line_lfs/pca_99/halpha_sobral_1dustdraw_pca99_sobralLogErr.joblib
```

The default interactive launcher selects these paths when they are present.
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

With both paper bundles present, the landing page should show:

- `Interactive Observable Suite`, backed by the standard-observable PCA-GP
  bundle.
- `Interactive Emission-Line LFs`, backed by the H-alpha Sobral sidecar LF
  PCA-GP bundle, when that bundle is present.

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

The emulator predictions do not require the full Galacticus HDF5 outputs once
the bundles have been trained. The local paper launcher uses the reduced files
to reconstruct the displayed training-preview curves with the same missing-data
treatment used during training. Those processed previews are already embedded
in the hosted bundles.

## Direct Uvicorn Entry Point

For deployment-like testing, the FastAPI app can also be run directly:

```bash
PYTHONPATH=src uvicorn galacticus_emu.interactive_fastapi:app \
  --host 127.0.0.1 \
  --port 8010
```

In that mode, set `INTERACTIVE_OBSERVABLES_BUNDLE_PATH` and
`INTERACTIVE_SIDECAR_LF_BUNDLE_PATH` if you want to force particular bundle
files. The launcher above sets these variables explicitly and exposes only the
standard-observable and emission-line interfaces.
