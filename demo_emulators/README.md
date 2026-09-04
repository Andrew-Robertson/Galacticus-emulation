# Demo Emulators

The `paper/` directory contains compact, mean-only copies of the Ntrain=1024
emulators used for the paper. They include the data needed by the browser UI
but omit the Gaussian-process state used to calculate predictive uncertainty,
making them small enough for the hosted interactive demo.

When the full campaign products are available locally, the app instead loads
the definitive standard-observable emulator from:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/automatedPipeline_transformedParams_finalPaper_definitive/emulators
```

It loads the Sobral-log-error H-alpha emulator used by the final MCMC from:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/scratch/sobral_log_errors_local/emulators
```

Run it with:

```bash
python scripts/serve_interactive_fastapi_demo.py
```

To reproduce the hosted configuration locally, add `--deployment`.
