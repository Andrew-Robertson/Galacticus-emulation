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
- Add release metadata: `LICENSE`, `CITATION.cff`, `.zenodo.json`, and
  optionally `codemeta.json`.
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

## Repository State After Initial Cleanup

Status checked on 2026-09-03 after the first cleanup pass:

- Current branch: `main`.
- The former `sobral-log-error-likelihood` branch was a direct descendant of
  local `main`, so local `main` was fast-forwarded to the Sobral likelihood and
  final-paper history.
- Local `main` is ahead of `origin/main`; pushing it will publish the existing
  paper/app history as well as the Sobral log-error likelihood work.

Initial cleanup completed:

- Added CI scaffolding and release-planning docs.
- Made the paper source portable by checking in static PDFs and using relative
  figure paths.
- Cleaned local absolute paths from public docs/scripts.
- Moved manuscript-specific figure/table builders to `paper/figure_scripts/`.
- Added compact derived paper inputs under `paper/figure_data/`.
- Generalized the kept HPC helper scripts enough that they no longer expose
  local machine paths.

Remaining sequence before assigning the paper commit:

- Run the local CI-equivalent checks.
- Push `main`.
- Tag the paper release only after release metadata and the data DOI placeholders
  or final DOIs are in place.

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

- Add stripped-output notebooks under `notebooks/`.
- Provide an explicit data-download/cache helper; do not auto-download the
  multi-GB Zenodo archive on import.
- Prefer `GALACTICUS_EMU_DATA` for pointing notebooks at an existing data copy.
- Name the fast MCMC tutorial `05_run_emulator_mcmc_quickstart.ipynb` rather
  than using "smoke" in the public title.

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
- Update the code/data availability text with the GitHub release tag, software
  DOI, data DOI, Galacticus commit hash, and parameter-file release tag or DOI.
- Resolve the remaining manuscript TODO for the parameter-file repository,
  version, or DOI before tagging the paper release.
