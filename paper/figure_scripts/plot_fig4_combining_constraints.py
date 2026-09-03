from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]

CAMPAIGN_ROOT = Path("runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced")
MCMC_ROOT = (
    CAMPAIGN_ROOT
    / "automatedPipeline_transformedParams_finalPaper_definitive"
    / "MCMCs_production_from_exploratory_MAP"
    / "standard_observables"
)
FIGURE_DIR = MCMC_ROOT / "smf_sfrf_sizes" / "figures"


def main() -> None:
    output_pdf = FIGURE_DIR / "smf_sfrf_sizes_corner_observable_composite.pdf"
    output_png = FIGURE_DIR / "smf_sfrf_sizes_corner_observable_composite.png"
    corner_png = FIGURE_DIR / "smf_sfrf_sizes_corner_observable_composite_corner.png"

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "plot_standard_corner_observable_composite.py"),
        "--generate-corner",
        "--corner-output-path",
        str(corner_png),
        "--bundle-path",
        str(
            CAMPAIGN_ROOT
            / "automatedPipeline_transformedParams_finalPaper"
            / "emulators"
            / "standard_observables"
            / "pca_99"
            / "standard_observables_bundle.joblib"
        ),
        "--posterior",
        f"SMF z0+z3={MCMC_ROOT / 'smf_z0_z3' / 'mcmc_smf_z0_z3_mcmc_results.hdf5'}",
        "--posterior",
        f"SFRF={MCMC_ROOT / 'sfr_function_robotham2011' / 'mcmc_sfr_function_robotham2011_mcmc_results.hdf5'}",
        "--posterior",
        f"Sizes={MCMC_ROOT / 'size_mass_vdw2014_sf_q' / 'mcmc_size_mass_vdw2014_sf_q_mcmc_results.hdf5'}",
        "--posterior",
        f"Combined={MCMC_ROOT / 'smf_sfrf_sizes' / 'mcmc_smf_sfrf_sizes_mcmc_results.hdf5'}",
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
        str(CAMPAIGN_ROOT),
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
