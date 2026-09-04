# Paper Emulator Deployment Bundles

These are compact, self-contained copies of the Ntrain=1024 emulators used for
the paper and by the hosted interactive demo. They preserve the Gaussian-process
mean predictions exactly but omit the matrices needed to calculate predictive
uncertainty.

The standard-observable bundle was exported from the definitive paper pipeline.
The H-alpha bundle is the Sobral-log-error version used by the final joint MCMC.
Both bundles contain 64 processed training curves per observable and the final
joint best-fit parameters, so the hosted app does not require the campaign files.
The numerical package versions used to serialize them are recorded in
`render-constraints.txt` at the repository root.

They can be regenerated from a checkout containing the full campaign with:

```bash
python scripts/export_mean_only_emulator_bundle.py \
  runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/automatedPipeline_transformedParams_finalPaper_definitive/emulators/standard_observables/pca_99/standard_observables_bundle.joblib \
  demo_emulators/paper/standard_observables_mean_only.joblib \
  --campaign-root runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced \
  --best-fit-summary paper/figure_data/final_calibration/mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr_run_summary.json \
  --training-preview-rows 64

python scripts/export_mean_only_emulator_bundle.py \
  runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/scratch/sobral_log_errors_local/emulators/emission_line_lfs/pca_99/halpha_sobral_1dustdraw_pca99_sobralLogErr.joblib \
  demo_emulators/paper/halpha_sobral_log_error_mean_only.joblib \
  --campaign-root runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced \
  --best-fit-summary paper/figure_data/final_calibration/mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr_run_summary.json \
  --training-preview-rows 64
```
