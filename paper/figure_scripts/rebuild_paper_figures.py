#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path("runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced")
FINAL_MCMC_ROOT = (
    CAMPAIGN
    / "automatedPipeline_transformedParams_finalPaper_definitive"
    / "MCMCs_production_from_exploratory_MAP"
)
FINAL_JOINT_RUN = (
    FINAL_MCMC_ROOT
    / "standard_observables_plus_emission_line_lfs"
    / "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"
)
FINAL_JOINT_PREFIX = "mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"

STEPS = (
    "smf_cv",
    "combining_constraints",
    "final_map_validation",
    "final_posterior_corner",
    "final_parameter_table",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the paper figures/tables from repo scripts plus the Zenodo-style data tree. "
            "By default, outputs go to paper/tmp/rebuilt_figures."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT,
        help="Root containing the repo-relative runs/ tree after extracting the data release.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper/tmp/rebuilt_figures",
        help="Directory for rebuilt figures and tables.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=STEPS,
        help="Run only this step. Repeat for multiple steps. Defaults to all steps.",
    )
    parser.add_argument(
        "--no-cached-smf-cv",
        action="store_true",
        help=(
            "Refit the SMF z~0 cross-validation examples rather than using the compact "
            "cached CSVs in paper/figure_data/smf_z0_cv."
        ),
    )
    parser.add_argument(
        "--skip-final-posterior-corner",
        action="store_true",
        help=(
            "Skip the GetDist/pypdf final posterior corner. Useful when the paper extras "
            "have not been installed yet."
        ),
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _copy_cached_smf_cv(output_cache_dir: Path) -> None:
    source_dir = REPO_ROOT / "paper/figure_data/smf_z0_cv"
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("smf_z0_pca_threshold_cv_*")):
        if source.is_file():
            shutil.copy2(source, output_cache_dir / source.name)


def rebuild_smf_cv(data_root: Path, output_dir: Path, *, use_cache: bool) -> None:
    cache_dir = output_dir / "smf_z0_cv"
    if use_cache:
        _copy_cached_smf_cv(cache_dir)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_smf_z0_pca_threshold_cv.py"),
        str(data_root / CAMPAIGN),
        "--hdf5-filename",
        "galacticus.hdf5",
        "--subset-size",
        "64",
        "--subset-size",
        "128",
        "--subset-size",
        "256",
        "--subset-size",
        "512",
        "--subset-size",
        "1024",
        "--n-folds",
        "5",
        "--validation-mode",
        "kfold_subset",
        "--pca-variance-threshold",
        "0.90",
        "--pca-variance-threshold",
        "0.95",
        "--pca-variance-threshold",
        "0.99",
        "--pca-variance-threshold",
        "0.999",
        "--highlight-x",
        "8.75",
        "--highlight-x",
        "9.75",
        "--highlight-x",
        "10.75",
        "--example-selection",
        "mean_smf_quantiles",
        "--n-example-curves",
        "5",
        "--n-restarts-optimizer",
        "0",
        "--output-dir",
        str(cache_dir),
        "--paper-figures-only",
    ]
    if not use_cache:
        command.append("--force")
    _run(command)


def rebuild_combining_constraints(data_root: Path, output_dir: Path) -> None:
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "paper/figure_scripts/plot_fig4_combining_constraints.py"),
            "--campaign-root",
            str(data_root / CAMPAIGN),
            "--output-dir",
            str(output_dir),
        ]
    )


def rebuild_final_map_validation(data_root: Path, output_dir: Path) -> None:
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "paper/figure_scripts/plot_final_map_validation_paper_figures.py"),
            "--run-dir",
            str(data_root / FINAL_JOINT_RUN),
            "--output-dir",
            str(output_dir),
        ]
    )


def rebuild_final_posterior_corner(data_root: Path, output_dir: Path) -> None:
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "paper/figure_scripts/plot_final_posterior_corner_with_dust_inset.py"),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ]
    )


def rebuild_final_parameter_table(data_root: Path, output_dir: Path) -> None:
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "paper/figure_scripts/generate_final_calibrated_parameters_table.py"),
            "--results",
            str(data_root / FINAL_JOINT_RUN / f"{FINAL_JOINT_PREFIX}_mcmc_results.hdf5"),
            "--output",
            str(output_dir / "final_calibrated_parameters_table.tex"),
        ]
    )


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = list(args.only or STEPS)
    if args.skip_final_posterior_corner and "final_posterior_corner" in selected:
        selected.remove("final_posterior_corner")

    for step in selected:
        print(f"\n== {step} ==", flush=True)
        if step == "smf_cv":
            rebuild_smf_cv(data_root, output_dir, use_cache=not args.no_cached_smf_cv)
        elif step == "combining_constraints":
            rebuild_combining_constraints(data_root, output_dir)
        elif step == "final_map_validation":
            rebuild_final_map_validation(data_root, output_dir)
        elif step == "final_posterior_corner":
            rebuild_final_posterior_corner(data_root, output_dir)
        elif step == "final_parameter_table":
            rebuild_final_parameter_table(data_root, output_dir)
        else:
            raise ValueError(f"Unknown step: {step}")

    print(f"\noutputs: {output_dir}")


if __name__ == "__main__":
    main()
