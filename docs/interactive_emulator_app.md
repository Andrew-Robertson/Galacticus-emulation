# Interactive Emulator App

The interactive app uses browser sliders to show how changes to Galacticus
parameters affect predicted observables without running Galacticus or training
an emulator. The hosted version uses a compact pre-trained bundle, while the
local launcher automatically uses the larger paper emulators when they are
available.

## Hosted Demo

The hosted multi-observable emulator is available at:

<https://galacticus-emulation.onrender.com>

The current hosted bundle was trained on 512 runs spanning 19 Galacticus
parameters. It is a compact predecessor of the paper emulator, not the final
1024-run, 20-parameter model used in the analysis.

## Run The Demonstrations Locally

After installing GalacticEmu, run this command from the repository root:

```bash
python scripts/serve_interactive_fastapi_demo.py
```

Then open <http://127.0.0.1:8010/>. When the paper campaign is present in its
default location, the landing page provides the standard-observable suite and
the emission-line luminosity functions. Without those files, it falls back to
the compact checked-in standard-observable emulator.

## Use The Full Paper Emulators

The larger emulator bundles trained for the paper are not stored in git. Once
they are available from the planned data archive, the app can instead be
pointed at the extracted campaign data. The paper campaign is:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced
```

The app uses the trained products in that campaign's pipeline output:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/pipeline/emulators/standard_observables/pca_99/standard_observables_bundle.joblib
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/pipeline/emulators/emission_line_lfs/pca_99/halpha_sobral_1dustdraw_pca99.joblib
```

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
