# Paper Reproduction Guide

This directory contains the material associated with
*Emulator-Assisted Calibration of a Semi-Analytic Galaxy Formation Model for
the Roman Galaxy Redshift Survey*. It is the starting point for readers who
want to inspect how a figure was made, reproduce the reported plots and tables,
or adapt the plotting scripts for a different presentation of the results.

The code associated with the arXiv v1 paper is identified by the
[`roman-grs-calibration-arxiv-v1` tagged release](https://github.com/Andrew-Robertson/Galacticus-emulation/tree/roman-grs-calibration-arxiv-v1),
so that the calculation can be inspected from a fixed version even as the main
branch continues to develop.

## Current Data Availability

The repository includes the manuscript source, the figure PDFs used by the
manuscript, scripts for producing or reproducing them, and compact derived data
for inspection and lightweight replotting. The larger source products are not
stored in git. These include the reduced 1024-run training campaign, trained
emulators, MCMC result files, and full Galacticus validation outputs.

Those larger products will be deposited on Zenodo with the final version of the
paper. The planned archive is described in
[`../docs/data_release_plan.md`](../docs/data_release_plan.md). Until that
archive is public, some figures can be inspected and replotted from the
included compact data, but the complete calculation cannot yet be rerun from
public files alone.

The reduced-campaign archive will include the fully resolved Galacticus
parameter file for every training evaluation, together with the corresponding
parameter changes and run commands. These files preserve the complete model
configuration actually used for each run. The Galacticus,
Galacticus-datasets, and emission-line post-processing software versions will
also be recorded with the archive.

The modular Galacticus parameter files prepared for the arXiv v1 paper release
are in
[`Andrew-Robertson/Galacticus-ParameterFiles` at tag `galacticemu-roman-grs-arxiv-v1`](https://github.com/Andrew-Robertson/Galacticus-ParameterFiles/tree/galacticemu-roman-grs-arxiv-v1).

To reproduce the emission-line dust post-processing, use
[`galacticus_sed_calculator` v0.1.0](https://github.com/roman-grs-pit/galacticus_sed_calculator/tree/v0.1.0)
(commit `9b635751b7dfb3506e90ac328246107771d59066`).

## Directory Layout

- `figures/` contains the PDF figures included by `main.tex`.
- `figure_data/` contains compact derived CSV, JSON, XML, and TeX inputs that
  are small enough to keep in git.
- `figure_scripts/` contains the scripts and wrappers used to make the paper
  figures and calibrated-parameter table.

The larger archive will supply the reduced campaign outputs, emulator bundles,
MCMC files, and direct Galacticus runs required by scripts that cannot operate
from the compact data alone.

## Inspect Or Reproduce A Figure

Start with [`figure_scripts/README.md`](figure_scripts/README.md). Its figure
map gives, for each manuscript figure or table:

- the output file included in the manuscript;
- the script or workflow that generated it;
- the compact data available in this repository;
- any larger products that will be required from the Zenodo archive.

The scripts have command-line options so that their plotting choices can be
inspected and adapted.

The three low-redshift stellar-mass-function emulator-validation figures and
the final MAP validation figure can be replotted from a fresh checkout using
only the compact data tracked in this repository:

```bash
python paper/figure_scripts/rebuild_paper_figures.py --from-figure-data
```

The rebuilt PDFs and PNGs are written under `paper/tmp/rebuilt_figures/`. The
posterior-corner figures and posterior-derived parameter table require the MCMC
samples from the planned Zenodo archive.

The manuscript currently uses these figure files:

- `figures/smf_z0_pca99_holdout_curves_highlight_bins.pdf`
- `figures/smf_z0_pca99_parity_highlight_bins_random_quarter.pdf`
- `figures/smf_z0_pca_threshold_rmse_r2_vs_training_size.pdf`
- `figures/smf_sfrf_sizes_corner_observable_composite.pdf`
- `figures/final_map_validation_all_observables_4x3.pdf`
- `figures/final_posterior_corner_galacticus20_with_dust5_inset_inwards.pdf`

## Included Figure Data

The compact inputs under `figure_data/` are grouped by their role in the
paper:

- `smf_z0_cv/` contains cached stellar-mass-function cross-validation
  predictions, metrics, splits, and summary metadata.
- `combining_constraints/` contains cached posterior-observable summaries for
  the standard-observable constraint-combination figure.
- `final_calibration/` contains best-fit prediction tables, MAP XML files,
  emission-line luminosity-function tables from direct runs, run metadata, and
  the generated calibrated-parameter table.

These files make it possible to audit plotted values and make some alternative
plots without downloading the full campaign. They are derived products rather
than replacements for the larger emulator, chain, and Galacticus output files.

## Rebuild From The Future Data Archive

The planned Zenodo release will be arranged so that its archives can be
extracted into a GalacticEmu checkout. After extracting the reduced-campaign
and paper-reproduction archives from the repository root, install the extra
plotting dependencies and run:

```bash
python -m pip install -e ".[paper]"
python paper/figure_scripts/rebuild_paper_figures.py \
  --data-root . \
  --output-dir paper/tmp/rebuilt_figures
```

The output goes to `paper/tmp/rebuilt_figures` so that a trial rebuild does not
replace the figure PDFs used by the manuscript. This command is the intended
public interface, but the archive it refers to has not yet been published.

## Training Campaign

The final emulators were trained on 1024 Galacticus evaluations spanning the
20 model parameters described in the paper. The parameter points were drawn
from a scrambled Sobol sequence generated with SciPy using seed 42. The
campaign design, sampled parameter values, Galacticus run configuration, and
reduced output files will be included in the planned training-data archive.

That archive will extract into the directory layout expected by the figure
scripts, so readers will not need to recreate the original local paths by
hand.

## Preparing The Archival Data Release

This section is for maintainers preparing the future Zenodo deposit, rather
than for readers reproducing the figures. The staging program selects three
packages: the reduced training campaign, the paper-reproduction products, and
one full Galacticus catalogue at the final MAP parameters.

To inspect what would be included without copying data, run:

```bash
python scripts/stage_zenodo_data_release.py --skip-checksums
```

To create the staging tree and compressed archives under the ignored `dist/`
directory, run:

```bash
python scripts/stage_zenodo_data_release.py --execute
python scripts/stage_zenodo_data_release.py --archives-only
```

The package contents, recorded software versions, and remaining release
decisions are documented in
[`../docs/data_release_plan.md`](../docs/data_release_plan.md).
