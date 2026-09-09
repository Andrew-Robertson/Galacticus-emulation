# Public Release Checklist

This checklist tracks the work needed to release GalacticEmu alongside the
paper. It separates the software repository, the archival data products, and
the paper-reproduction materials so that each can be cited and maintained
cleanly.

## Release Objects

- GitHub repository release for the software and lightweight examples.
- Zenodo software archive created from the tagged GitHub release.
- Zenodo data archive containing the reduced training campaign, final pipeline
  products, and one full Galacticus run suitable for deriving new observables.

## Repository Cleanup

- Keep final-paper working-tree changes grouped into small release-preparation
  commits.
- Keep `main` as the release branch; it now contains the former
  `sobral-log-error-likelihood` branch history.
- Keep the software license and citation metadata current. Add Zenodo metadata
  when preparing the archival release, and optionally add `codemeta.json`.
- Replace absolute local paths in the manuscript and docs with repository-local
  paths or release DOI references.
- Keep manuscript-only plotting/table scripts in `paper/figure_scripts/`, and
  leave top-level `scripts/` for reusable public CLIs and workflow pieces.
- Keep one public workflow path centered on
  `configs/pipelines/standard_observables_plus_halpha.yaml` and
  `scripts/run_campaign_pipeline.py`.
- Keep internal HPC helper scripts only if they are generalized enough for
  public use; otherwise preserve their commands in provenance notes instead of
  presenting them as reusable tools.
- Remove generated files from the public surface: `.DS_Store`, `__pycache__`,
  LaTeX intermediates, temporary PDF renders, scratch figures, and local logs.

## Repository State For The arXiv Submission

Status checked on 2026-09-03 after the release-readiness pass:

- Current branch: `main`.
- The former `sobral-log-error-likelihood` branch was a direct descendant of
  local `main`, so local `main` was fast-forwarded to the Sobral likelihood and
  final-paper history.
- Notebook development is kept on the separate `notebook-examples` branch and
  is not part of the arXiv release.

Initial cleanup completed:

- Added CI scaffolding and release-planning docs.
- Made the paper source portable by checking in static PDFs and using relative
  figure paths.
- Cleaned local absolute paths from public docs/scripts.
- Moved manuscript-specific figure/table builders to `paper/figure_scripts/`.
- Added compact derived paper inputs under `paper/figure_data/`.
- Added a clone-only command that replots the three SMF emulator-validation
  figures and the final MAP validation figure from compact checked-in data.
- Generalized the kept HPC helper scripts enough that they no longer expose
  local machine paths.
- Ran the local CI-equivalent checks and compiled the manuscript successfully.
- Added a BSD 3-Clause software license and GitHub citation metadata.

Remaining sequence before assigning the arXiv paper tag:

- Push `main`.
- Tag the exact commit submitted to arXiv.

## Continuous Integration

- Install the package in editable mode.
- Run the lightweight Python tests.
- Compile the package and public workflow entry points.
- Dry-render the public pipeline YAML so broken stage names, missing PyYAML, or
  invalid command definitions fail quickly.
- Do not run Galacticus, train production emulators, execute long MCMCs, or
  download multi-GB data products in CI.

## Zenodo Data Package

See `docs/data_release_plan.md` for the current concrete package boundaries,
source paths, and open data-licensing decisions.

- Package the reduced training campaign as one archive containing
  `campaign_design.json`, `samples.csv`, command/provenance files, and
  `evaluations/`.
- Package final derived products separately: trained emulator bundles,
  cross-validation metrics, MCMC summaries/chains, MAP XML files, and
  manuscript figure inputs.
- Package one full Galacticus example run separately so notebooks can derive
  new catalogue-level observables from `/Outputs`.
- Include a data `README.md`, `MANIFEST.tsv`, `SHA256SUMS.txt`, license,
  Galacticus commit hash, Galacticus datasets hash, parameter-file provenance,
  and citation instructions.

## Notebooks

- Develop the worked notebooks on `notebook-examples` during peer review and
  merge them only after they run from a fresh environment.
- Use one low-redshift stellar mass function as the worked emulator example.
- Add stripped-output notebooks under `notebooks/`.
- Provide an explicit data-download/cache helper; do not auto-download the
  multi-GB Zenodo archive on import.
- Prefer `GALACTICUS_EMU_DATA` for pointing notebooks at an existing data copy.

## Paper Reproduction

- Keep `paper/README.md` current with commands for rebuilding figures and
  manuscript products from the archived data.
- Keep final paper figures in `paper/figures/` and use relative paths in
  `paper/main.tex`.
- Keep compact derived paper inputs in `paper/figure_data/`, with the larger
  source HDF5 products in Zenodo.
- Maintain a figure-by-figure map from manuscript output to script, compact
  data product, and required Zenodo payload.
- Add lightweight replotters for checked-in `paper/figure_data/` products where
  practical, so readers can adapt figure styling without downloading every
  large source file.
- Promote the final posterior-corner inset overlay from provenance notes into a
  standalone public script before tagging the release.
- Record exact source data paths and commands in provenance files, while keeping
  machine-specific absolute paths out of the final manuscript.
- The arXiv version links to GitHub and states that the parameter files and
  larger data products will be archived upon acceptance.
- Update the accepted-paper version with the GitHub release tag, software DOI,
  data DOI, Galacticus commit hash, and parameter-file release tag or DOI.
