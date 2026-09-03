# Paper Reproduction Notes

This directory contains the manuscript source for the GalacticEmu paper, the
static PDF figures needed to compile it, compact derived figure inputs, and the
paper-specific scripts used to build manuscript products.

## Compile The Manuscript

From this directory:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable, use the equivalent PDFLaTeX/BibTeX sequence
required by the local TeX installation.

## Included Figures

The figures used by `main.tex` are checked in under `paper/figures/`:

- `smf_z0_pca99_holdout_curves_highlight_bins.pdf`
- `smf_z0_pca99_parity_highlight_bins_random_quarter.pdf`
- `smf_z0_pca_threshold_rmse_r2_vs_training_size.pdf`
- `smf_sfrf_sizes_corner_observable_composite.pdf`
- `final_map_validation_all_observables_4x3.pdf`
- `final_posterior_corner_galacticus20_with_dust5_inset_inwards.pdf`

These are static manuscript products. The larger training campaign,
cross-validation products, emulator bundles, MCMC chains, and direct
Galacticus validation runs should be archived separately in the Zenodo data
release.

## Figure Data

Small derived inputs are checked in under `paper/figure_data/`:

- `smf_z0_cv/`: cached SMF z~0 cross-validation predictions, metrics, splits,
  and summary metadata.
- `combining_constraints/`: cached posterior-observable summaries for the
  standard-observable constraint-combination figure.
- `final_calibration/`: best-fit prediction tables, MAP XML files, emission-line
  LF tables from direct runs, run metadata, and the generated calibrated
  parameter table.

These files are intended for quick inspection and lightweight figure audits.
They do not replace the Zenodo data release, which should hold the reduced
campaign outputs, emulator bundles, MCMC HDF5 files, and direct Galacticus
HDF5 products.

## Data Products Needed To Rebuild Figures

The current manuscript was built from the reduced campaign:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced
```

The final calibration products used by the paper live under:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/automatedPipeline_transformedParams_finalPaper_definitive/MCMCs_production_from_exploratory_MAP
```

The public data release should include, at minimum:

- the reduced `evaluations/` tree;
- the final emulator bundles;
- cross-validation metrics and prediction tables;
- MCMC result HDF5 files and MAP parameter XML files;
- direct Galacticus MAP validation outputs;
- the figure/table provenance commands.

## Figure Scripts

Reusable workflow scripts remain in the repository-level `scripts/` directory.
Scripts that only reproduce manuscript figures or tables live in
`paper/figure_scripts/`; see `paper/figure_scripts/README.md` for the current
script inventory and data requirements.
