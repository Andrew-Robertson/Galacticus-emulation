# Zenodo Data Release Plan

This document records the proposed archival data products for the GalacticEmu
paper release. The data should be archived outside git, with the repository
containing only scripts, notebooks, lightweight examples, manifests, and
documentation.

## Proposed Archives

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

Suggested archive name:

```text
galacticemu_sobol1024_reduced_campaign_v1.tar.gz
```

### Final Paper Pipeline Products

Local source:

```text
runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/automatedPipeline_transformedParams_finalPaper_definitive/MCMCs_production_from_exploratory_MAP
```

This tree is about 8.3 GB locally. The public archive should prioritize:

- trained emulator bundles and `.meta.json` files;
- cross-validation tables and prediction summaries;
- final MCMC result HDF5 files;
- maximum-a-posteriori model-change XML files;
- best-fit direct Galacticus validation outputs;
- paper figure/table inputs and provenance commands.

Compact copies of selected figure/table inputs are also tracked in git under
`paper/figure_data/` for quick inspection. The Zenodo archive should remain the
authoritative source for the larger HDF5 chains, trained emulator bundles, and
direct Galacticus products used to regenerate those derived inputs.

Suggested archive name:

```text
galacticemu_final_paper_pipeline_products_v1.tar.gz
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

Suggested archive name:

```text
galacticemu_example_full_unit1_run_v1.tar.gz
```

## Metadata To Include Beside Each Archive

Each archive should be accompanied by:

- `README.md` describing the payload and intended use;
- `MANIFEST.tsv` with file paths, byte sizes, and archive membership;
- `SHA256SUMS.txt`;
- software commit for this repository;
- Galacticus commit hash: `a8fd4a98520d4821b5b3d0551a13fa92ca100c90`;
- Galacticus datasets hash: `18d9ec9d7111db34980c822eb24f4ae2aa76a943`;
- parameter-file repository commit or release tag;
- dust-modelling repository commit or release tag, if dust post-processing
  code is required to reproduce a product;
- license and citation instructions.

## Open Decisions

- Choose a data license.
- Decide whether the 8.3 GB final pipeline tree should be archived as one file
  or split into smaller products by purpose.
- Decide whether to include posterior-draw direct Galacticus runs beyond the
  single UNIT example.
- Resolve the final public URL/DOI for the parameter files.
- Clean the sibling `Galacticus-ParameterFiles` and `Galacticus-dust-modelling`
  repositories before recording their provenance commits.
