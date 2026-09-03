# Paper Figure Scripts

This directory is for reproducing, auditing, and adapting the plots and tables
used in the paper. A new reader should be able to start here, identify the
script behind a manuscript figure, inspect the compact data products in
`paper/figure_data/`, and then rerun the script with the larger Zenodo data
archive when needed.

The repository has two script areas:

- `paper/figure_scripts/` contains paper-facing renderers and wrappers. These
  scripts are tied to manuscript figures, captions, labels, or table formats.
- `scripts/` contains reusable workflow programs for campaign construction,
  emulator training, validation, MCMC, and general plotting. Some paper figures
  are outputs of those broader workflows, so their generating script stays in
  `scripts/` and is referenced from here.

Run commands from the repository root unless a script says otherwise. Use
`python <script> --help` to inspect the command-line options before adapting a
plot.

## Figure Map

| Manuscript item | Checked-in output | Main script or workflow | Checked-in data | Full data needed |
| --- | --- | --- | --- | --- |
| SMF z~0 holdout curves, parity, and training-size panels | `paper/figures/smf_z0_pca99_holdout_curves_highlight_bins.pdf`, `paper/figures/smf_z0_pca99_parity_highlight_bins_random_quarter.pdf`, `paper/figures/smf_z0_pca_threshold_rmse_r2_vs_training_size.pdf` | `scripts/run_smf_z0_pca_threshold_cv.py`; `paper/figure_scripts/render_smf_z0_cached_cv_paper_figures.py` is a lighter paper-style renderer for cached campaign products | `paper/figure_data/smf_z0_cv/` | Reduced 1024-run campaign from Zenodo to recompute the CV products |
| Standard-observable constraint-combination figure | `paper/figures/smf_sfrf_sizes_corner_observable_composite.pdf` | `paper/figure_scripts/plot_fig4_combining_constraints.py`, which calls `plot_standard_corner_observable_composite.py` | `paper/figure_data/combining_constraints/` | Standard-observable emulator bundle, MCMC HDF5 files, and reduced campaign preview data from Zenodo |
| Final MAP validation figure | `paper/figures/final_map_validation_all_observables_4x3.pdf` | `paper/figure_scripts/plot_final_map_validation_paper_figures.py` | `paper/figure_data/final_calibration/` | Final emulator bundles plus direct Galacticus MAP validation HDF5 outputs from Zenodo |
| Final calibrated-parameter table | Embedded in `paper/main.tex`; mirrored as `paper/figure_data/final_calibration/final_calibrated_parameters_table.tex` | `paper/figure_scripts/generate_final_calibrated_parameters_table.py` | `paper/figure_data/final_calibration/` | Final combined MCMC HDF5 results from Zenodo |
| Final posterior-corner figure with dust inset | `paper/figures/final_posterior_corner_galacticus20_with_dust5_inset_inwards.pdf` | Core corner plots are made with `scripts/plot_observable_mcmc_overlay_getdist.py`; the inset overlay still needs to be promoted from provenance notes into a standalone public script | Existing static PDF only | Final MCMC HDF5 results, dust posterior samples, and the cleaned overlay recipe |

## Script Inventory

- `render_smf_z0_cached_cv_paper_figures.py` renders paper-style SMF z~0
  validation figures from cached campaign cross-validation products.
- `plot_fig4_combining_constraints.py` is the command wrapper for the
  standard-observable constraint-combination figure.
- `plot_standard_corner_observable_composite.py` builds the composite
  posterior-corner plus observable-panel figure used by the wrapper above.
- `plot_standard_observable_posterior_band_overlay.py` builds posterior
  observable-band overlays for standard observables.
- `plot_final_map_validation_paper_figures.py` builds the final MAP validation
  panels, including direct Galacticus/N-body comparisons when the archived
  HDF5 products are available.
- `generate_final_calibrated_parameters_table.py` generates the calibrated
  parameter table from the final MCMC HDF5 results.

## Data Boundaries

The checked-in `paper/figure_data/` files are curated for inspection,
provenance, and lightweight alternative plots. They are not the canonical raw
data release. Scripts that evaluate posterior samples, read trained emulator
bundles, or read direct Galacticus outputs still require the larger archived
products described in `docs/data_release_plan.md`.

Before the public release, every manuscript figure should have either a
documented command that runs against the Zenodo archive or a lightweight
replotter that runs from `paper/figure_data/` alone.
