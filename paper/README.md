# Paper Reproduction Notes

This directory contains the manuscript source for the GalacticEmu paper and
the small PDF figures needed to compile it without access to local `runs/`
directories.

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

## Script Placement

Reusable workflow scripts should remain in the repository-level `scripts/`
directory. Scripts that only reproduce manuscript figures or tables should
live in `paper/figure_scripts/` once they have been reviewed for public use.
