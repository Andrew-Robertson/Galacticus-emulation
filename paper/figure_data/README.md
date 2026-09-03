# Paper Figure Data

This directory contains compact, derived inputs for reproducing, auditing, or
modifying paper figures without adding multi-GB campaign products to git. These
files are useful when a reader wants to check plotted values, make a quick
variant of a figure, or compare the manuscript curves with a new plotting
choice.

The full training campaign, emulator bundles, MCMC chains, and direct
Galacticus HDF5 outputs should be archived in the Zenodo data release described
in `docs/data_release_plan.md`.

## Contents

### `smf_z0_cv/`

Cached cross-validation products for the SMF z~0 emulator validation figures:

- `smf_z0_pca_threshold_cv_predictions.csv`
- `smf_z0_pca_threshold_cv_metrics.csv`
- `smf_z0_pca_threshold_cv_splits.csv`
- `smf_z0_pca_threshold_cv_summary.json`

These files come from the PCA-threshold/bin-by-bin comparison run for the
1024-point Sobol campaign. The generating validation script is
`scripts/run_smf_z0_pca_threshold_cv.py`; it is a reusable analysis workflow
rather than a manuscript-only renderer.

### `combining_constraints/`

Cached posterior-prediction summaries for the standard-observable
constraint-combination figure:

- `smf_sfrf_sizes_corner_observable_composite.csv`
- `smf_sfrf_sizes_individual_vs_joint_posterior_mean_bands.csv`
- `smf_sfrf_sizes_individual_vs_joint_posterior_mean_bands_with_training_prior.csv`
- `mcmc_smf_sfrf_sizes_best_fit_observables.csv`
- `mcmc_smf_sfrf_sizes_run_summary.json`

The MCMC HDF5 files used to make these summaries are not tracked here and
belong in the Zenodo archive.

### `final_calibration/`

Compact products from the final standard-observable plus H-alpha calibration:

- best-fit standard-observable and sidecar luminosity-function CSVs;
- direct-run standard-observable and H-alpha luminosity-function CSVs for the
  fiducial Galacticus and UNIT validation runs;
- maximum-a-posteriori model-change XML files;
- the generated calibrated-parameter table used by the manuscript;
- the final MCMC run summary JSON.

The final MCMC chains and direct-run HDF5 catalogues are intentionally omitted
from git. Recomputing posterior intervals or rebuilding the direct-Galacticus
validation panels from first principles requires those Zenodo products.
The compact `galacticus_run_standard_observable_actuals.csv` file is extracted
from the final EPS MAP HDF5 output so the plotted validation curves can be
regenerated without archiving that 1.8 GiB HDF5 file in the paper-reproduction
package.

## Local Source Campaign

The checked-in data products were copied from:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced
```

Public users should get the full source products from the associated Zenodo
data release once it is minted.
