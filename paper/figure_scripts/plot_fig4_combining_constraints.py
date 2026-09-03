from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CAMPAIGN_ROOT = Path("runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced")
DEFAULT_MCMC_SUBDIR = (
    "automatedPipeline_transformedParams_finalPaper_definitive"
    "/MCMCs_production_from_exploratory_MAP"
    "/standard_observables"
)
DEFAULT_BUNDLE_SUBPATH = (
    "automatedPipeline_transformedParams_finalPaper"
    "/emulators/standard_observables/pca_99/standard_observables_bundle.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate the paper figure comparing individual and combined standard-observable constraints."
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=REPO_ROOT / DEFAULT_CAMPAIGN_ROOT,
        help="Root of the sobol_1024_20p_simpleSizes_moreHalos_reduced campaign.",
    )
    parser.add_argument(
        "--mcmc-root",
        type=Path,
        default=None,
        help=(
            "Directory containing the standard-observable production MCMC subdirectories. "
            "Defaults to CAMPAIGN_ROOT/automatedPipeline_transformedParams_finalPaper_definitive/"
            "MCMCs_production_from_exploratory_MAP/standard_observables."
        ),
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=None,
        help=(
            "Standard-observable emulator bundle used for observable predictions. "
            "Defaults to the bundle used for the submitted-paper figure."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to MCMC_ROOT/smf_sfrf_sizes/figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    mcmc_root = (
        args.mcmc_root.expanduser().resolve()
        if args.mcmc_root is not None
        else campaign_root / DEFAULT_MCMC_SUBDIR
    )
    bundle_path = (
        args.bundle_path.expanduser().resolve()
        if args.bundle_path is not None
        else campaign_root / DEFAULT_BUNDLE_SUBPATH
    )
    figure_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else mcmc_root / "smf_sfrf_sizes" / "figures"
    )

    output_pdf = figure_dir / "smf_sfrf_sizes_corner_observable_composite.pdf"
    output_png = figure_dir / "smf_sfrf_sizes_corner_observable_composite.png"
    corner_png = figure_dir / "smf_sfrf_sizes_corner_observable_composite_corner.png"

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "plot_standard_corner_observable_composite.py"),
        "--generate-corner",
        "--corner-output-path",
        str(corner_png),
        "--bundle-path",
        str(bundle_path),
        "--posterior",
        f"SMF z0+z3={mcmc_root / 'smf_z0_z3' / 'mcmc_smf_z0_z3_mcmc_results.hdf5'}",
        "--posterior",
        f"SFRF={mcmc_root / 'sfr_function_robotham2011' / 'mcmc_sfr_function_robotham2011_mcmc_results.hdf5'}",
        "--posterior",
        f"Sizes={mcmc_root / 'size_mass_vdw2014_sf_q' / 'mcmc_size_mass_vdw2014_sf_q_mcmc_results.hdf5'}",
        "--posterior",
        f"Combined={mcmc_root / 'smf_sfrf_sizes' / 'mcmc_smf_sfrf_sizes_mcmc_results.hdf5'}",
        "--corner-parameter",
        "diskVelocityCharacteristic",
        "--corner-parameter",
        "diskExponent",
        "--corner-parameter",
        "spheroidVelocityCharacteristic",
        "--corner-parameter",
        "coolingMultiplier",
        "--corner-parameter",
        "spheroidSFREfficiency",
        "--corner-parameter",
        "diskSFRFrequencyNorm",
        "--corner-label-style",
        "paper",
        "--corner-axes-labelsize",
        "17",
        "--corner-tick-labelsize",
        "14",
        "--corner-tick-length-scale",
        "1.5",
        "--corner-x-labelpad",
        "6",
        "--corner-y-labelpad",
        "14",
        "--observable",
        "smf_z0",
        "--observable",
        "smf_z3",
        "--observable",
        "size_mass_vdw2014_star_forming_z0",
        "--observable",
        "size_mass_vdw2014_quiescent_z0",
        "--observable",
        "sfr_function_robotham2011",
        "--observable-ymin",
        "smf_z0=-5.5",
        "--observable-ymin",
        "smf_z3=-5.5",
        "--training-campaign-root",
        str(campaign_root),
        "--training-preview-rows",
        "all",
        "--training-alpha",
        "0.08",
        "--training-linewidth",
        "0.28",
        "--panel-font-scale",
        "1.3",
        "--panel-title-font-scale",
        "1.3",
        "--panel-axis-label-scale",
        "1.15",
        "--target-legend-fontsize",
        "6.7",
        "--n-draws",
        "3000",
        "--credible-interval",
        "0.68",
        "--output-path",
        str(output_pdf),
        "--extra-output-path",
        str(output_png),
        "--save-provenance",
        "--dpi",
        "220",
    ]
    print(shlex.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
