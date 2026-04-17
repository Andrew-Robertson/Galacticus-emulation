from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu import fit_scaled_gp, predict_scaled_gp, transform_to_prior_quantiles, trinity_parameter_specs


INPUT_LABELS = {
    "diskVelocityCharacteristic": "disk v char.",
    "diskExponent": "disk exp.",
    "spheroidVelocityCharacteristic": "spheroid v char.",
    "spheroidExponent": "spheroid exp.",
    "coolingMultiplier": "cooling mult.",
    "BHefficiencyRadioMode": "BH radio eff.",
    "BHefficiencyWind": "BH wind eff.",
    "thinDiskMaximum": "thin-disk max",
    "bondiEnhancementSpheroid": "Bondi enh. sph.",
    "bondiEnhancementHotHalo": "Bondi enh. hot",
    "bondiTemperatureSpheroid": "Bondi temp sph.",
    "blackHoleSeedMass": "BH seed mass",
    "spheroidSFREfficiency": "spheroid SFR eff.",
    "spheroidSFRExponentVelocity": "spheroid SFR v exp.",
    "diskSFRFrequencyNorm": "disk SFR norm",
    "coreRadiusOverVirialRadius": "core/virial",
    "henriquesGamma": "Henriques gamma",
    "henriquesDelta1": "Henriques delta1",
    "henriquesDelta2": "Henriques delta2",
    "massRatioMajorMerger": "major merger ratio",
    "energyOrbital": "orbital energy",
    "barFractionAngularMomentumRetainedSpheroid": "bar j retained",
    "barStabilityThresholdGaseous": "bar thresh gas",
    "barStabilityThresholdStellar": "bar thresh star",
}

OUTPUT_LABELS = {
    **{
        f"z0_mass_metallicity_abundance_oxygen_12logoh_{index}": rf"$\langle 12+\log_{{10}}(\mathrm{{O/H}})\rangle$ z=0 bin {index + 1}"
        for index in range(3)
    },
    **{
        f"z0_mass_metallicity_mass_stellar_log10_{index}": rf"$\langle \log_{{10}} M_\star \rangle$ z=0 bin {index + 1}"
        for index in range(3)
    },
    **{
        f"z0_mass_metallicity_mass_stellar_log10_residual_{index}": rf"$\Delta \langle \log_{{10}} M_\star \rangle$ z=0 bin {index + 1}"
        for index in range(3)
    },
    **{
        f"z0_mass_metallicity_abundance_oxygen_12logoh_residual_{index}": rf"$\Delta \langle 12+\log_{{10}}(\mathrm{{O/H}})\rangle$ z=0 bin {index + 1}"
        for index in range(3)
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot main-effects diagnostics for approximate z=0 mass-metallicity summaries.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument("--output-suffix", default="", help="Suffix to append before output filenames.")
    parser.add_argument("--optimize-hyperparameters", action="store_true", help="Optimize GP hyperparameters for the partial-dependence curves.")
    parser.add_argument("--output-column", help="Plot a single explicit output column.")
    parser.add_argument(
        "--output-family",
        choices=["oxygen", "oxygen_residual", "stellar", "stellar_residual", "both"],
        default="oxygen",
        help="Which approximate outputs to visualize.",
    )
    return parser.parse_args()


def _running_binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    edges = np.linspace(0, len(x_sorted), n_bins + 1, dtype=int)
    x_centers: list[float] = []
    y_centers: list[float] = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end <= start:
            continue
        x_centers.append(float(np.median(x_sorted[start:end])))
        y_centers.append(float(np.median(y_sorted[start:end])))
    return np.asarray(x_centers), np.asarray(y_centers)


def _fit_gp_models(df: pd.DataFrame, inputs: list[str], outputs: list[str], optimize_hyperparameters: bool):
    parameter_specs = trinity_parameter_specs()
    x_raw = df[inputs].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)
    gp_models = {}
    for output_name in outputs:
        y = df[output_name].to_numpy(dtype=float)
        gp_models[output_name] = fit_scaled_gp(
            x=x,
            y=y,
            n_restarts_optimizer=1,
            optimize_hyperparameters=optimize_hyperparameters,
        )
    return gp_models, x_raw


def _partial_dependence_curve(gp_model, x_reference_raw: np.ndarray, input_index: int, grid_values: np.ndarray) -> np.ndarray:
    model, y_mean, y_std = gp_model
    parameter_specs = trinity_parameter_specs()
    pd_means: list[float] = []
    for grid_value in grid_values:
        x_modified_raw = x_reference_raw.copy()
        x_modified_raw[:, input_index] = grid_value
        x_modified = transform_to_prior_quantiles(parameter_specs, x_modified_raw)
        pred_mean, _ = predict_scaled_gp(model, y_mean, y_std, x_modified)
        pd_means.append(float(np.mean(pred_mean)))
    return np.asarray(pd_means)


def _plot_main_effects(df: pd.DataFrame, inputs: list[str], outputs: list[str], output_path: Path, optimize_hyperparameters: bool) -> None:
    gp_models, x_reference_raw = _fit_gp_models(df, inputs, outputs, optimize_hyperparameters)
    if len(outputs) == 1:
        ncols = 6
        nrows = 4
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(3.0 * ncols, 2.8 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        flat_axes = list(axes.flat)
        output_name = outputs[0]
        for col_index, input_name in enumerate(inputs):
            axis = flat_axes[col_index]
            x = df[input_name].to_numpy(dtype=float)
            y = df[output_name].to_numpy(dtype=float)
            axis.scatter(x, y, s=14, alpha=0.55, color="#2a6f97", edgecolors="none")
            trend_x, trend_y = _running_binned_trend(x, y)
            axis.plot(trend_x, trend_y, color="#d1495b", linewidth=1.2)
            grid_values = np.linspace(float(np.min(x)), float(np.max(x)), 60)
            pd_curve = _partial_dependence_curve(
                gp_model=gp_models[output_name],
                x_reference_raw=x_reference_raw,
                input_index=col_index,
                grid_values=grid_values,
            )
            axis.plot(grid_values, pd_curve, color="#222222", linewidth=1.5)
            axis.set_xlabel(INPUT_LABELS.get(input_name, input_name), fontsize=8)
            axis.set_ylabel(OUTPUT_LABELS.get(output_name, output_name), fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.2, linewidth=0.4)
        for axis in flat_axes[len(inputs):]:
            axis.axis("off")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        nrows=len(outputs),
        ncols=len(inputs),
        figsize=(2.6 * len(inputs), 2.5 * len(outputs)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, output_name in enumerate(outputs):
        for col_index, input_name in enumerate(inputs):
            axis = axes[row_index, col_index]
            x = df[input_name].to_numpy(dtype=float)
            y = df[output_name].to_numpy(dtype=float)
            axis.scatter(x, y, s=14, alpha=0.55, color="#2a6f97", edgecolors="none")
            trend_x, trend_y = _running_binned_trend(x, y)
            axis.plot(trend_x, trend_y, color="#d1495b", linewidth=1.2)
            grid_values = np.linspace(float(np.min(x)), float(np.max(x)), 60)
            pd_curve = _partial_dependence_curve(
                gp_model=gp_models[output_name],
                x_reference_raw=x_reference_raw,
                input_index=col_index,
                grid_values=grid_values,
            )
            axis.plot(grid_values, pd_curve, color="#222222", linewidth=1.5)
            if row_index == len(outputs) - 1:
                axis.set_xlabel(INPUT_LABELS.get(input_name, input_name), fontsize=8)
            if col_index == 0:
                axis.set_ylabel(OUTPUT_LABELS.get(output_name, output_name), fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.2, linewidth=0.4)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(df: pd.DataFrame, inputs: list[str], outputs: list[str], output_path: Path) -> None:
    corr = df[inputs + outputs].corr(method="spearman").loc[outputs, inputs]
    fig, ax = plt.subplots(figsize=(1.0 * len(inputs) + 2, 0.6 * len(outputs) + 2), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(inputs)), [INPUT_LABELS.get(name, name) for name in inputs], rotation=50, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(outputs)), [OUTPUT_LABELS.get(name, name) for name in outputs], fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Spearman correlation")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    table_path = campaign_root / "mass_metallicity_emulator_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Expected {table_path}; run extract_trinity_mass_metallicity_summary.py first.")
    df = pd.read_csv(table_path)
    emulator_table_path = campaign_root / "emulator_table.csv"
    if not emulator_table_path.exists():
        raise FileNotFoundError(f"Expected {emulator_table_path} for target stellar-mass values.")
    emulator_df = pd.read_csv(emulator_table_path)
    target_columns = ["evaluation_id", *[f"z0_mass_stellar_log10_target_{index}" for index in range(3)]]
    df = df.merge(emulator_df[target_columns], on="evaluation_id", how="inner", validate="one_to_one")
    for index in range(3):
        df[f"z0_mass_metallicity_mass_stellar_log10_residual_{index}"] = (
            df[f"z0_mass_metallicity_mass_stellar_log10_{index}"] - df[f"z0_mass_stellar_log10_target_{index}"]
        )

    inputs = [spec.short_name for spec in trinity_parameter_specs()]
    outputs: list[str] = []
    if args.output_column is not None:
        outputs = [args.output_column]
    else:
        if args.output_family in {"oxygen", "both"}:
            outputs.extend([f"z0_mass_metallicity_abundance_oxygen_12logoh_{index}" for index in range(3)])
        if args.output_family == "oxygen_residual":
            outputs.extend([f"z0_mass_metallicity_abundance_oxygen_12logoh_residual_{index}" for index in (1, 2)])
        if args.output_family in {"stellar", "both"}:
            outputs.extend([f"z0_mass_metallicity_mass_stellar_log10_{index}" for index in range(3)])
        if args.output_family == "stellar_residual":
            outputs.extend([f"z0_mass_metallicity_mass_stellar_log10_residual_{index}" for index in (1, 2)])

    suffix = args.output_suffix
    main_effects_path = figures_root / f"main_effects_mass_metallicity_{args.output_family}{suffix}.png"
    heatmap_path = figures_root / f"spearman_heatmap_mass_metallicity_{args.output_family}{suffix}.png"
    _plot_main_effects(
        df=df,
        inputs=inputs,
        outputs=outputs,
        output_path=main_effects_path,
        optimize_hyperparameters=args.optimize_hyperparameters,
    )
    _plot_heatmap(df=df, inputs=inputs, outputs=outputs, output_path=heatmap_path)

    print(table_path)
    print(main_effects_path)
    print(heatmap_path)


if __name__ == "__main__":
    main()
