from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from galacticus_emu import trinity_parameter_specs

try:
    import corner
except ModuleNotFoundError:  # pragma: no cover
    corner = None

try:
    from getdist import MCSamples, plots as gd_plots
except ModuleNotFoundError:  # pragma: no cover
    MCSamples = None
    gd_plots = None


CASE_DEFINITIONS = {
    "z0_mstar_mean": {"color": "#1b9e77", "label": "z=0 Mstar"},
    "z2_mstar_mean": {"color": "#d95f02", "label": "z=2 Mstar"},
    "z0_mbh_mean": {"color": "#7570b3", "label": "z=0 Mbh"},
    "z2_mbh_mean": {"color": "#e7298a", "label": "z=2 Mbh"},
    "four_mean_families_combined": {"color": "#000000", "label": "Combined", "linewidth": 2.2},
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Trinity ablation MCMC corner plots using log10(parameter) for selected logarithmic parameters."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--output-name",
        default="emulator_mcmc_ablation_corner_overlay_top_params.png",
        help="Filename for the overlaid corner plot.",
    )
    parser.add_argument(
        "--include-combined",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlay the combined four-mean-families posterior in black.",
    )
    parser.add_argument(
        "--backend",
        choices=["corner", "getdist"],
        default="corner",
        help="Plotting backend to use.",
    )
    parser.add_argument(
        "--ablation-dir-name",
        default="emulator_mcmc_ablations",
        help="Name of the ablation output directory inside the campaign root.",
    )
    parser.add_argument(
        "--figures-dir-name",
        default="figures",
        help="Name of the figures directory inside the campaign root.",
    )
    parser.add_argument(
        "--selection-file",
        default=None,
        help="Optional path to an overlay_parameter_selection.json file to use instead of the one in the ablation directory.",
    )
    parser.add_argument(
        "--combined-dir-name",
        default="emulator_mcmc_four_mean_families",
        help="Name of the combined MCMC directory inside the campaign root when --include-combined is used.",
    )
    parser.add_argument(
        "--filled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use filled contours for the getdist backend.",
    )
    return parser.parse_args()


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


def _make_corner_overlay(
    transformed_samples_by_case: dict[str, np.ndarray],
    labels: list[str],
    case_names: list[str],
    output_path: Path,
) -> None:
    stacked = np.vstack(list(transformed_samples_by_case.values()))
    ranges = []
    for dim in range(stacked.shape[1]):
        lower = float(np.min(stacked[:, dim]))
        upper = float(np.max(stacked[:, dim]))
        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        ranges.append((lower - margin, upper + margin))

    fig = None
    for case_name in case_names:
        case = CASE_DEFINITIONS[case_name]
        fig = corner.corner(
            transformed_samples_by_case[case_name],
            fig=fig,
            labels=labels,
            range=ranges,
            color=case["color"],
            plot_datapoints=False,
            fill_contours=False,
            no_fill_contours=True,
            plot_density=False,
            levels=(0.393, 0.865),
            hist_kwargs={"density": True, "histtype": "step", "linewidth": case.get("linewidth", 1.8)},
            contour_kwargs={"linewidths": case.get("linewidth", 1.4)},
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=CASE_DEFINITIONS[name]["color"],
            lw=CASE_DEFINITIONS[name].get("linewidth", 2.0),
            label=CASE_DEFINITIONS[name]["label"],
        )
        for name in case_names
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _make_getdist_overlay(
    transformed_samples_by_case: dict[str, np.ndarray],
    labels: list[str],
    parameter_columns: list[str],
    case_names: list[str],
    output_path: Path,
    filled: bool,
) -> None:
    ranges = _parameter_ranges(parameter_columns)
    roots = []
    contour_colors = []
    line_args = []
    for case_name in case_names:
        case = CASE_DEFINITIONS[case_name]
        roots.append(
            MCSamples(
                samples=transformed_samples_by_case[case_name],
                names=parameter_columns,
                labels=labels,
                label=case["label"],
                ranges=ranges,
                settings={"smooth_scale_1D": 0.7, "smooth_scale_2D": 0.7},
            )
        )
        contour_colors.append(case["color"])
        line_args.append({"lw": case.get("linewidth", 1.8), "color": case["color"]})

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
        legend_labels=[CASE_DEFINITIONS[name]["label"] for name in case_names],
        line_args=line_args,
    )
    plotter.export(str(output_path))


def main() -> None:
    args = parse_args()
    if args.backend == "corner" and corner is None:
        raise ModuleNotFoundError("The 'corner' package is required for plot_trinity_ablation_overlay.py with --backend corner.")
    if args.backend == "getdist" and (MCSamples is None or gd_plots is None):
        raise ModuleNotFoundError("The 'getdist' package is required for plot_trinity_ablation_overlay.py with --backend getdist.")

    campaign_root = Path(args.campaign_root).resolve()
    ablation_root = campaign_root / args.ablation_dir_name
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    selection_path = (
        Path(args.selection_file).resolve()
        if args.selection_file is not None
        else ablation_root / "overlay_parameter_selection.json"
    )
    selection = json.loads(selection_path.read_text())
    parameter_columns = selection["parameter_columns"]

    transformed_samples_by_case: dict[str, np.ndarray] = {}
    case_names = ["z0_mstar_mean", "z2_mstar_mean", "z0_mbh_mean", "z2_mbh_mean"]
    if args.include_combined:
        case_names.append("four_mean_families_combined")

    labels: list[str] = []
    for case_name in case_names:
        if case_name == "four_mean_families_combined":
            posterior_path = campaign_root / args.combined_dir_name / "posterior_samples.csv"
        else:
            posterior_path = ablation_root / case_name / "posterior_samples.csv"
        df = pd.read_csv(posterior_path)
        transformed, labels = _transform_samples(df, parameter_columns)
        transformed_samples_by_case[case_name] = transformed

    output_path = figures_root / args.output_name
    if args.backend == "corner":
        _make_corner_overlay(transformed_samples_by_case, labels, case_names, output_path)
    else:
        _make_getdist_overlay(
            transformed_samples_by_case,
            labels,
            parameter_columns,
            case_names,
            output_path,
            filled=args.filled,
        )
    print(output_path)


if __name__ == "__main__":
    main()
