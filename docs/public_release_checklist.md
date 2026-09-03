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

- Commit or intentionally discard all final-paper working-tree changes before
  switching branches.
- Merge `sobral-log-error-likelihood` into `main` after the working tree is
  clean. The current local branch graph makes this a fast-forward merge from
  `main` to the Sobral branch tip.
- Add release metadata: `LICENSE`, `CITATION.cff`, `.zenodo.json`, and
  optionally `codemeta.json`.
- Replace absolute local paths in the manuscript and docs with repository-local
  paths or release DOI references.
- Move manuscript-only plotting/table scripts into `paper/figure_scripts/`, or
  leave them in `scripts/` only if they are intended as reusable public CLIs.
- Keep one public workflow path centered on
  `configs/pipelines/standard_observables_plus_halpha.yaml` and
  `scripts/run_campaign_pipeline.py`.
- Keep internal HPC helper scripts only if they are generalized enough for
  public use; otherwise preserve their commands in provenance notes instead of
  presenting them as reusable tools.
- Remove generated files from the public surface: `.DS_Store`, `__pycache__`,
  LaTeX intermediates, temporary PDF renders, scratch figures, and local logs.

## Current Working Tree Triage

Status checked on 2026-09-03:

- Current branch: `sobral-log-error-likelihood`.
- `sobral-log-error-likelihood` contains three commits not yet on local `main`:
  the Sobral log-error likelihood commit and two later paper commits.
- Local `main` has no commits missing from `sobral-log-error-likelihood`, so the
  final merge back to `main` can be a fast-forward after the working tree is
  clean.
- Local `main` is ahead of `origin/main`, so pushing the release branch to
  GitHub will publish the existing paper/app history as well as the Sobral
  likelihood work.

Proposed cleanup buckets:

- Release scaffolding to keep and commit first: `.github/workflows/ci.yml`,
  `pyproject.toml`, `.gitignore`, and this checklist.
- Final paper files to commit only after a paper-owner review: `paper/main.tex`,
  `paper/bibliography.bib`, and `paper/main.pdf`.
- Reusable code changes to review and keep if they are part of the public
  emulator path: `src/galacticus_emu/interactive_observables.py`,
  `src/galacticus_emu/interactive_sidecar_lf.py`,
  `src/galacticus_emu/observable_plot_metadata.py`, and the public plotting/CV
  scripts.
- Paper-only scripts should move to `paper/figure_scripts/` or be documented as
  provenance commands rather than public entry points.
- Local/HPC helper scripts with hard-coded paths should stay out of the public
  interface unless they are generalized.

Merge sequence once cleanup is complete:

- Commit the chosen working-tree changes on `sobral-log-error-likelihood`.
- Switch to `main` and run `git merge --ff-only sobral-log-error-likelihood`.
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

- Add `paper/README.md` with commands for rebuilding figures and manuscript
  products from the archived data.
- Move final paper figures into `paper/figures/` and use relative paths in
  `paper/main.tex`.
- Record exact source data paths and commands in provenance files, while keeping
  machine-specific absolute paths out of the final manuscript.
- Update the code/data availability text with the GitHub release tag, software
  DOI, data DOI, Galacticus commit hash, and parameter-file release tag or DOI.
- Resolve the remaining manuscript TODO for the parameter-file repository,
  version, or DOI before tagging the paper release.
