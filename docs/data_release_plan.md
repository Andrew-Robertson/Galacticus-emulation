# Zenodo Data Release Plan

This document records the proposed archival data products for the GalacticEmu
paper release. The data should be archived outside git, with the repository
containing only scripts, notebooks, lightweight examples, manifests, and
documentation.

## Proposed Archives

The local staging helper encodes the current archive split:

```bash
python scripts/stage_zenodo_data_release.py
```

This command prints a dry-run summary. To create the local staging tree under
`dist/zenodo_data_release/galacticemu-paper-data-v1/`, run:

```bash
python scripts/stage_zenodo_data_release.py --execute
```

Add `--make-archives` when the staged tree is ready to be compressed into one
`.tar.gz` file per package.

The archives unpack repository-relative data paths directly. For example, from
the repository root:

```bash
tar -xzf galacticemu_sobol1024_reduced_campaign_v1.tar.gz
tar -xzf galacticemu_paper_reproduction_products_v1.tar.gz
```

Package README, manifest, and checksum files unpack under `.zenodo/<package>/`
so multiple archives can be extracted into the same checkout without metadata
name clashes.

### Reduced 1024-Run Training Campaign

Local source:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced
```

Core payload:

```text
campaign_design.json
samples.csv
commands.txt
commands.sh
submit_slurm_array.sh
COMMAND_LOG.md
evaluations/
```

The `evaluations/` directory is about 3.1 GB locally and contains the reduced
per-evaluation Galacticus outputs plus sidecar luminosity-function tables. The
archive should exclude generated desktop files such as `.DS_Store`, Python
caches, logs that are not useful provenance, and temporary plotting outputs.

Staging package/archive name:

```text
galacticemu_sobol1024_reduced_campaign_v1.tar.gz
```

### Paper Reproduction Products

Local source:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/automatedPipeline_transformedParams_finalPaper_definitive/MCMCs_production_from_exploratory_MAP
```

The full production tree is about 8.3 GB locally, so the staged public package
selects the products needed by `paper/figure_scripts/` rather than copying every
backend chain and scratch output. The public archive prioritizes:

- trained emulator bundles and `.meta.json` files;
- cross-validation tables and prediction summaries;
- final MCMC result HDF5 files;
- maximum-a-posteriori model-change XML files;
- compact best-fit direct Galacticus validation outputs;
- paper figure/table inputs and provenance commands.

Compact copies of selected figure/table inputs are also tracked in git under
`paper/figure_data/` for quick inspection. The Zenodo archive should remain the
authoritative source for the larger HDF5 chains, trained emulator bundles, and
direct Galacticus products used to regenerate those derived inputs.

Staging package/archive name:

```text
galacticemu_paper_reproduction_products_v1.tar.gz
```

### Full Galacticus Example Run

Local source:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/automatedPipeline_transformedParams_finalPaper_definitive/MCMCs_production_from_exploratory_MAP/standard_observables_plus_emission_line_lfs/all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr/bestFitModel_UNIT1
```

This directory is about 397 MB locally. It includes a full Galacticus HDF5 file
with `/Outputs`, making it suitable for a public notebook that derives a new
observable from a galaxy catalogue and then compares it with the reduced-output
workflow.

Staging package/archive name:

```text
galacticemu_example_full_unit1_run_v1.tar.gz
```

## Rebuilding The Paper Figures From Staged Data

After extracting the reduced-campaign and paper-reproduction archives from the
repository root, install the optional paper plotting dependencies and run:

```bash
python -m pip install -e ".[paper]"
python paper/figure_scripts/rebuild_paper_figures.py \
  --data-root . \
  --output-dir paper/tmp/rebuilt_figures
```

The driver leaves outputs under `paper/tmp/` by default. Use that directory to
compare against the checked-in manuscript PDFs in `paper/figures/`.

## Metadata To Include Beside Each Archive

Each archive should be accompanied by:

- `README.md` describing the payload and intended use;
- `MANIFEST.tsv` with file paths, byte sizes, and archive membership;
- `SHA256SUMS.txt`;
- software commit for this repository;
- Galacticus commit hash: `a8fd4a98520d4821b5b3d0551a13fa92ca100c90`;
- Galacticus datasets hash: `18d9ec9d7111db34980c822eb24f4ae2aa76a943`;
- parameter-file repository tag:
  `https://github.com/Andrew-Robertson/Galacticus-ParameterFiles/tree/paper-arxiv-v1`;
- `galacticus_sed_calculator` commit or release tag for emission-line dust
  post-processing;
- license and citation instructions.

## Open Decisions

- Choose a data license.
- Decide whether the 8.3 GB final pipeline tree should be archived as one file
  or split into smaller products by purpose.
- Decide whether to include posterior-draw direct Galacticus runs beyond the
  single UNIT example.
- Confirm that `Andrew-Robertson/Galacticus-ParameterFiles` is public and that
  tag `paper-arxiv-v1` has been pushed.
- Record the exact `galacticus_sed_calculator` revision used for the training
  campaign.
