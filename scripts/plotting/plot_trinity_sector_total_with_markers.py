from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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

SECTOR_TOTAL_GROUP = [
    ("Combined Mstar", "#1b9e77", 1.8, "emulator_mcmc_mstar_z0_z2/posterior_samples.csv"),
    ("Combined Mbh", "#7570b3", 1.8, "emulator_mcmc_mbh_z0_z2_no1e11/posterior_samples.csv"),
    ("Total", "#000000", 2.3, "emulator_mcmc_four_mean_families_no1e11Mbh/posterior_samples.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full sector-vs-total Trinity getdist overlay with parameter markers.")
    parser.add_argument("campaign_root", help="Path to the completed Trinity campaign directory.")
    parser.add_argument(
        "--figures-dir-name",
        default="figures_z0_z2_Trinity_no1e11Mbh",
        help="Name of the figures directory inside the campaign root.",
    )
    parser.add_argument(
        "--emulator-marker-xml",
        default=None,
        help="Path to the emulator MAP Galacticus changes file.",
    )
    parser.add_argument(
        "--pso-marker-xml",
        default=None,
        help="Path to the PSO Galacticus changes file.",
    )
    parser.add_argument(
        "--output-name",
        default="sector_total_sequence_getdist_full_with_markers.png",
        help="Filename for the marked overlay figure.",
    )
    parser.add_argument("--max-samples", type=int, default=5000)
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


def _load_marker_values(xml_path: Path, parameter_columns: list[str]) -> dict[str, float]:
    spec_map = {spec.path: spec.short_name for spec in trinity_parameter_specs()}
    root = ET.parse(xml_path).getroot()
    values: dict[str, float] = {}
    for change in root.findall("change"):
        path = change.attrib.get("path")
        value = change.attrib.get("value")
        if path in spec_map and value is not None:
            short_name = spec_map[path]
            numeric = float(value)
            if short_name in LOG_PARAMETER_ALIASES:
                numeric = float(np.log10(numeric))
            values[short_name] = numeric
    return {name: values[name] for name in parameter_columns if name in values}


def _style_dense_triangle_labels(plotter, n_params: int) -> None:
    bottom = n_params - 1
    for col in range(n_params):
        ax = plotter.subplots[bottom, col]
        if ax is None:
            continue
        label = ax.xaxis.label
        label.set_rotation(38)
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")
        label.set_rotation_mode("anchor")
        ax.xaxis.labelpad = 16
    for row in range(n_params):
        ax = plotter.subplots[row, 0]
        if ax is None:
            continue
        label = ax.yaxis.label
        label.set_rotation(38)
        label.set_horizontalalignment("right")
        label.set_verticalalignment("bottom")
        label.set_rotation_mode("anchor")
        ax.yaxis.labelpad = 10


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    parameter_columns = [spec.short_name for spec in trinity_parameter_specs()]
    ranges = _parameter_ranges(parameter_columns)

    roots = []
    contour_colors = []
    line_args = []
    legend_labels = []
    labels = None

    for label, color, linewidth, relative_path in SECTOR_TOTAL_GROUP:
        df = pd.read_csv(campaign_root / relative_path)
        if len(df) > args.max_samples:
            df = df.sample(n=args.max_samples, random_state=42)
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

    plotter = gd_plots.get_subplot_plotter(width_inch=20)
    plotter.settings.figure_legend_frame = False
    plotter.settings.alpha_factor_contour_lines = 1.0
    plotter.settings.linewidth = 1.4
    plotter.settings.linewidth_contour = 1.5
    plotter.settings.num_plot_contours = 2
    plotter.settings.legend_fontsize = 16
    plotter.settings.axes_labelsize = 10
    plotter.settings.lab_fontsize = 10
    plotter.settings.alpha_filled_add = 0.35
    plotter.settings.solid_colors = contour_colors
    plotter.triangle_plot(
        roots,
        params=parameter_columns,
        filled=True,
        contour_colors=contour_colors,
        legend_labels=legend_labels,
        line_args=line_args,
    )
    _style_dense_triangle_labels(plotter, len(parameter_columns))

    if args.emulator_marker_xml is not None:
        emu_markers = _load_marker_values(Path(args.emulator_marker_xml).resolve(), parameter_columns)
        plotter.add_param_markers(emu_markers, color="#d62728", ls="--", lw=1.2)
    if args.pso_marker_xml is not None:
        pso_markers = _load_marker_values(Path(args.pso_marker_xml).resolve(), parameter_columns)
        plotter.add_param_markers(pso_markers, color="#1f77b4", ls="-.", lw=1.2)

    marker_handles = []
    if args.emulator_marker_xml is not None:
        marker_handles.append(Line2D([0], [0], color="#d62728", lw=1.2, ls="--", label="GP MCMC"))
    if args.pso_marker_xml is not None:
        marker_handles.append(Line2D([0], [0], color="#1f77b4", lw=1.2, ls="-.", label="PSO"))
    if marker_handles:
        marker_legend = plotter.fig.legend(
            handles=marker_handles,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            frameon=False,
            fontsize=14,
            title="Markers",
            title_fontsize=14,
        )
        plotter.extra_artists.append(marker_legend)

    output_path = figures_root / args.output_name
    plotter.fig.savefig(
        output_path,
        dpi=220,
        bbox_extra_artists=plotter.extra_artists,
        bbox_inches="tight",
    )
    print(output_path)


if __name__ == "__main__":
    main()
