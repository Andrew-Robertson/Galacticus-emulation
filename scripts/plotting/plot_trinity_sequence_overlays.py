from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from galacticus_emu import trinity_parameter_specs
from getdist import MCSamples, plots as gd_plots


LOG_PARAMETER_ALIASES = {
    "blackHoleSeedMass",
    "diskSFRFrequencyNorm",
    "spheroidSFREfficiency",
    "bondiEnhancementSpheroid",
    "bondiEnhancementHotHalo",
    "bondiTemperatureSpheroid",
    "BHefficiencyWind",
    "thinDiskMaximum",
}

DISPLAY_LABELS = {
    "diskVelocityCharacteristic": "diskVelocityCharacteristic",
    "diskExponent": "diskExponent",
    "spheroidVelocityCharacteristic": "spheroidVelocityCharacteristic",
    "spheroidExponent": "spheroidExponent",
    "BHefficiencyWind": "BHefficiencyWind",
    "BHefficiencyRadioMode": "BHefficiencyRadioMode",
    "henriquesGamma": "henriquesGamma",
    "henriquesDelta1": "henriquesDelta1",
    "henriquesDelta2": "henriquesDelta2",
    "coreRadiusOverVirialRadius": "hotHaloCoreRadius",
    "diskSFRFrequencyNorm": "normalisationBlitzSFR",
    "barFractionAngularMomentumRetainedSpheroid": "barInstabilityFracAngMomRetSpheroid",
    "coolingMultiplier": "coolingRateMultiplier",
    "spheroidSFREfficiency": "spheroidStarFormationEfficiency",
    "spheroidSFRExponentVelocity": "spheroidStarFormationExponent",
    "thinDiskMaximum": "thinDiskMaxAccretion",
    "bondiEnhancementSpheroid": "bondiHoyleSpheroidEnhancement",
    "bondiEnhancementHotHalo": "bondiHoyleHotHaloEnhancement",
    "bondiTemperatureSpheroid": "bondiHoyleSpheroidTemperature",
    "blackHoleSeedMass": "blackHoleSeedMass",
    "massRatioMajorMerger": "majorMergerMassRatio",
    "energyOrbital": "mergerRemnantOrbitalEnergyFactor",
    "barStabilityThresholdGaseous": "barInstabilityThresholdGaseous",
    "barStabilityThresholdStellar": "barInstabilityThresholdStellar",
}

PLOT_GROUPS = {
    "mstar_sequence": [
        ("z0_mstar_mean", "z=0 Mstar", "#1b9e77", 1.8, "emulator_mcmc_ablations_no1e11Mbh/z0_mstar_mean/posterior_samples.csv"),
        ("z2_mstar_mean", "z=2 Mstar", "#d95f02", 1.8, "emulator_mcmc_ablations_no1e11Mbh/z2_mstar_mean/posterior_samples.csv"),
        ("mstar_z0_z2", "Combined Mstar", "#000000", 2.3, "emulator_mcmc_mstar_z0_z2/posterior_samples.csv"),
    ],
    "mbh_sequence": [
        ("z0_mbh_mean", "z=0 Mbh", "#7570b3", 1.8, "emulator_mcmc_ablations_no1e11Mbh/z0_mbh_mean/posterior_samples.csv"),
        ("z2_mbh_mean", "z=2 Mbh", "#e7298a", 1.8, "emulator_mcmc_ablations_no1e11Mbh/z2_mbh_mean/posterior_samples.csv"),
        ("mbh_z0_z2_no1e11", "Combined Mbh", "#000000", 2.3, "emulator_mcmc_mbh_z0_z2_no1e11/posterior_samples.csv"),
    ],
    "sector_total_sequence": [
        ("mstar_z0_z2", "Combined Mstar", "#1b9e77", 1.8, "emulator_mcmc_mstar_z0_z2/posterior_samples.csv"),
        ("mbh_z0_z2_no1e11", "Combined Mbh", "#7570b3", 1.8, "emulator_mcmc_mbh_z0_z2_no1e11/posterior_samples.csv"),
        ("four_mean_families_no1e11_mbh", "Total", "#000000", 2.3, "emulator_mcmc_four_mean_families_no1e11Mbh/posterior_samples.csv"),
    ],
    "mstar_mzr_total_sequence": [
        ("mstar_z0_z2", "Combined Mstar", "#1b9e77", 1.8, "emulator_mcmc_mstar_z0_z2/posterior_samples.csv"),
        ("mzr_z0_upper", "z=0 MZR", "#d95f02", 1.8, "emulator_mcmc_mzr_z0_upper/posterior_samples.csv"),
        (
            "mstar_z0_z2_plus_mzr_z0_upper",
            "Total Mstar + MZR",
            "#000000",
            2.3,
            "emulator_mcmc_mstar_z0_z2_plus_mzr_z0_upper/posterior_samples.csv",
        ),
    ],
    "mstar_mzr_total_sequence_w5": [
        ("mstar_z0_z2", "Combined Mstar", "#1b9e77", 1.8, "emulator_mcmc_mstar_z0_z2/posterior_samples.csv"),
        ("mzr_z0_upper", "z=0 MZR", "#d95f02", 1.8, "emulator_mcmc_mzr_z0_upper/posterior_samples.csv"),
        (
            "mstar_z0_z2_plus_mzr_z0_upper_w5",
            "Total Mstar + MZR (w=5)",
            "#000000",
            2.3,
            "emulator_mcmc_mstar_z0_z2_plus_mzr_z0_upper_w5/posterior_samples.csv",
        ),
    ],
    "mstar_mbh_mzr_total_sequence_w5": [
        ("mstar_z0_z2", "Combined Mstar", "#1b9e77", 1.8, "emulator_mcmc_mstar_z0_z2/posterior_samples.csv"),
        ("mbh_z0_z2_no1e11", "Combined Mbh", "#7570b3", 1.8, "emulator_mcmc_mbh_z0_z2_no1e11/posterior_samples.csv"),
        ("mzr_z0_upper", "z=0 MZR", "#d95f02", 1.8, "emulator_mcmc_mzr_z0_upper/posterior_samples.csv"),
        (
            "four_mean_families_no1e11_mbh_plus_mzr_z0_upper_w5",
            "Total Mstar + Mbh + MZR (w=5)",
            "#000000",
            2.4,
            "emulator_mcmc_four_mean_families_no1e11_mbh_plus_mzr_z0_upper_w5/posterior_samples.csv",
        ),
    ],
    "mstar_mbh_mzr_total_sequence_gas": [
        ("mstar_z0_z2", "Combined Mstar", "#1b9e77", 1.8, "emulator_mcmc_mstar_z0_z2/posterior_samples.csv"),
        ("mbh_z0_z2_no1e11", "Combined Mbh", "#7570b3", 1.8, "emulator_mcmc_mbh_z0_z2_no1e11/posterior_samples.csv"),
        ("mzr_z0_upper_gas", "z=0 MZR (gas)", "#d95f02", 1.8, "emulator_mcmc_mzr_z0_upper_gas/posterior_samples.csv"),
        (
            "four_mean_families_no1e11_mbh_plus_mzr_z0_upper_gas",
            "Total Mstar + Mbh + MZR (gas)",
            "#000000",
            2.4,
            "emulator_mcmc_four_mean_families_no1e11_mbh_plus_mzr_z0_upper_gas/posterior_samples.csv",
        ),
    ],
}

DEFAULT_PARAMETERS = [
    "coolingMultiplier",
    "diskVelocityCharacteristic",
    "spheroidSFREfficiency",
    "bondiTemperatureSpheroid",
    "diskExponent",
    "blackHoleSeedMass",
    "bondiEnhancementSpheroid",
    "thinDiskMaximum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make sequential getdist overlay plots for Trinity MCMC runs.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--figures-dir-name",
        default="figures_z0_z2_Trinity_no1e11Mbh",
        help="Name of the figures directory inside the campaign root.",
    )
    parser.add_argument(
        "--filled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use filled getdist contours.",
    )
    parser.add_argument(
        "--parameter-set",
        choices=["top", "full"],
        default="top",
        help="Whether to plot the original top-parameter subset or the full Trinity parameter set.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30000,
        help="Maximum number of posterior samples per chain to use for plotting.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(PLOT_GROUPS.keys()),
        default=None,
        help="Optional subset of plot groups to render.",
    )
    return parser.parse_args()


def _parameter_ranges(parameter_columns: list[str]) -> dict[str, list[float]]:
    spec_map = {spec.short_name: spec for spec in trinity_parameter_specs()}
    ranges: dict[str, list[float]] = {}
    for parameter in parameter_columns:
        spec = spec_map[parameter]
        lower = float(spec.prior.lower)
        upper = float(spec.prior.upper)
        if parameter in LOG_PARAMETER_ALIASES:
            ranges[parameter] = [float(np.log10(lower)), float(np.log10(upper))]
        else:
            ranges[parameter] = [lower, upper]
    return ranges


def _transform_samples(df: pd.DataFrame, parameter_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    transformed_columns = []
    labels = []
    for parameter in parameter_columns:
        values = df[parameter].to_numpy(dtype=float)
        if parameter in LOG_PARAMETER_ALIASES:
            values = np.log10(values)
            labels.append(f"log10({DISPLAY_LABELS.get(parameter, parameter)})")
        else:
            labels.append(DISPLAY_LABELS.get(parameter, parameter))
        transformed_columns.append(values)
    return np.column_stack(transformed_columns), labels


def _plot_group(
    campaign_root: Path,
    figures_root: Path,
    group_name: str,
    filled: bool,
    parameter_set: str,
    max_samples: int,
) -> Path:
    if parameter_set == "full":
        parameter_columns = [spec.short_name for spec in trinity_parameter_specs()]
    else:
        parameter_columns = list(DEFAULT_PARAMETERS)
    ranges = _parameter_ranges(parameter_columns)

    roots = []
    contour_colors = []
    line_args = []
    legend_labels = []
    labels = None

    for _, label, color, linewidth, relative_path in PLOT_GROUPS[group_name]:
        df = pd.read_csv(campaign_root / relative_path)
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        transformed, labels = _transform_samples(df, parameter_columns)
        roots.append(
            MCSamples(
                samples=transformed,
                names=parameter_columns,
                labels=labels,
                label=label,
                ranges=ranges,
                settings={"smooth_scale_1D": 0.7, "smooth_scale_2D": 0.7},
            )
        )
        contour_colors.append(color)
        line_args.append({"lw": linewidth, "color": color})
        legend_labels.append(label)

    suffix = "full" if parameter_set == "full" else "top"
    output_path = figures_root / f"{group_name}_getdist_{suffix}.png"
    plotter = gd_plots.get_subplot_plotter(width_inch=11)
    plotter.settings.figure_legend_frame = False
    plotter.settings.alpha_factor_contour_lines = 1.0
    plotter.settings.linewidth = 1.5
    plotter.settings.linewidth_contour = 1.6
    plotter.settings.num_plot_contours = 2
    plotter.settings.legend_fontsize = 13
    plotter.settings.axes_labelsize = 11
    plotter.settings.lab_fontsize = 11
    if filled:
        plotter.settings.alpha_filled_add = 0.4
        plotter.settings.solid_colors = contour_colors
    plotter.triangle_plot(
        roots,
        params=parameter_columns,
        filled=filled,
        contour_colors=contour_colors,
        legend_labels=legend_labels,
        line_args=line_args,
    )
    plotter.export(str(output_path))
    return output_path


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    group_names = args.groups if args.groups is not None else list(PLOT_GROUPS.keys())
    output_paths = []
    for group_name in group_names:
        output_paths.append(
            _plot_group(
                campaign_root,
                figures_root,
                group_name,
                args.filled,
                args.parameter_set,
                args.max_samples,
            )
        )

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
