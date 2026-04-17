from __future__ import annotations

import argparse
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
from getdist import MCSamples, plots as gd_plots


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
    parser = argparse.ArgumentParser(description="Diagnose blackHoleSeedMass posterior correlations across Trinity MCMC runs.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument("--top-n", type=int, default=6, help="Number of secondary parameters to include in the focused overlay.")
    return parser.parse_args()


def _posterior_paths(campaign_root: Path) -> dict[str, Path]:
    return {
        "z0_mstar_mean": campaign_root / "emulator_mcmc_ablations" / "z0_mstar_mean" / "posterior_samples.csv",
        "z2_mstar_mean": campaign_root / "emulator_mcmc_ablations" / "z2_mstar_mean" / "posterior_samples.csv",
        "z0_mbh_mean": campaign_root / "emulator_mcmc_ablations" / "z0_mbh_mean" / "posterior_samples.csv",
        "z2_mbh_mean": campaign_root / "emulator_mcmc_ablations" / "z2_mbh_mean" / "posterior_samples.csv",
        "four_mean_families_combined": campaign_root / "emulator_mcmc_four_mean_families" / "posterior_samples.csv",
    }


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


def _transform_dataframe(df: pd.DataFrame, parameter_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    transformed = []
    labels = []
    for parameter in parameter_columns:
        values = df[parameter].to_numpy(dtype=float)
        if parameter in LOG_PARAMETER_ALIASES:
            values = np.log10(values)
            labels.append(f"log10({DISPLAY_LABELS.get(parameter, parameter)})")
        else:
            labels.append(DISPLAY_LABELS.get(parameter, parameter))
        transformed.append(values)
    return np.column_stack(transformed), labels


def _compute_seed_correlations(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for case_name, path in paths.items():
        df = pd.read_csv(path)
        corr = df.drop(columns=["log_probability"]).corr(method="spearman")["blackHoleSeedMass"].drop("blackHoleSeedMass")
        for parameter, value in corr.items():
            rows.append({"case": case_name, "parameter": parameter, "spearman_rho": float(value)})
    return pd.DataFrame(rows)


def _plot_heatmap(pivot: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(pivot.index))), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([CASE_DEFINITIONS[col]["label"] for col in pivot.columns], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([DISPLAY_LABELS.get(name, name) for name in pivot.index])
    ax.set_title("Spearman correlation with blackHoleSeedMass")
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label("Spearman rho")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_getdist_overlay(paths: dict[str, Path], parameter_columns: list[str], output_path: Path) -> None:
    ranges = _parameter_ranges(parameter_columns)
    roots = []
    contour_colors = []
    line_args = []
    labels = None
    for case_name, path in paths.items():
        df = pd.read_csv(path)
        transformed, labels = _transform_dataframe(df, parameter_columns)
        case = CASE_DEFINITIONS[case_name]
        roots.append(
            MCSamples(
                samples=transformed,
                names=parameter_columns,
                labels=labels,
                label=case["label"],
                ranges=ranges,
                settings={"smooth_scale_1D": 0.7, "smooth_scale_2D": 0.7},
            )
        )
        contour_colors.append(case["color"])
        line_args.append({"lw": case.get("linewidth", 1.8), "color": case["color"]})

    plotter = gd_plots.get_subplot_plotter(width_inch=12)
    plotter.settings.figure_legend_frame = False
    plotter.settings.alpha_factor_contour_lines = 1.0
    plotter.settings.linewidth = 1.5
    plotter.settings.linewidth_contour = 1.6
    plotter.settings.num_plot_contours = 2
    plotter.settings.legend_fontsize = 13
    plotter.settings.axes_labelsize = 11
    plotter.settings.lab_fontsize = 11
    plotter.triangle_plot(
        roots,
        params=parameter_columns,
        filled=False,
        contour_colors=contour_colors,
        legend_labels=[CASE_DEFINITIONS[name]["label"] for name in paths],
        line_args=line_args,
    )
    plotter.export(str(output_path))


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / "figures"
    emulator_root = campaign_root / "emulator_mcmc_ablations"
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    paths = _posterior_paths(campaign_root)
    correlation_df = _compute_seed_correlations(paths)
    correlation_csv = emulator_root / "seed_mass_posterior_correlations.csv"
    correlation_df.to_csv(correlation_csv, index=False)

    pivot = correlation_df.pivot(index="parameter", columns="case", values="spearman_rho")
    top_parameters = (
        pivot.abs().max(axis=1).sort_values(ascending=False).head(args.top_n).index.tolist()
    )
    ordered_parameters = ["blackHoleSeedMass", *top_parameters]

    heatmap_order = pivot.abs().max(axis=1).sort_values(ascending=False).head(12).index.tolist()
    heatmap_pivot = pivot.loc[heatmap_order, list(paths.keys())]
    heatmap_path = figures_root / "emulator_mcmc_seed_mass_correlation_heatmap.png"
    _plot_heatmap(heatmap_pivot, heatmap_path)

    overlay_path = figures_root / "emulator_mcmc_seed_mass_compensator_overlay_getdist.png"
    _plot_getdist_overlay(paths, ordered_parameters, overlay_path)

    summary_path = emulator_root / "seed_mass_top_compensators.json"
    summary_path.write_text(
        pd.Series(top_parameters, name="parameter").to_json(orient="values", indent=2) + "\n"
    )

    print(correlation_csv)
    print(summary_path)
    print(heatmap_path)
    print(overlay_path)


if __name__ == "__main__":
    main()
