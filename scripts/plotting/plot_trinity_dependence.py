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

from galacticus_emu import (
    TRINITY_MEAN_OUTPUT_COLUMNS,
    TRINITY_OUTPUT_LABELS,
    TRINITY_RUN_CONFIGS,
    fit_scaled_gp,
    load_or_build_emulator_table,
    predict_scaled_gp,
    transform_to_prior_quantiles,
    trinity_parameter_specs,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Trinity emulator main-effect style diagnostics.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument("--output-suffix", default="", help="Suffix to append before .png output filenames.")
    parser.add_argument("--optimize-hyperparameters", action="store_true", help="Optimize GP hyperparameters instead of using the fixed starting kernel.")
    parser.add_argument("--run-name", choices=sorted(TRINITY_RUN_CONFIGS.keys()))
    parser.add_argument("--family", choices=["stellar", "black_hole"])
    parser.add_argument("--output-column", help="Plot only a single explicit output column.")
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Only write the Spearman heatmap and skip the main-effects figures.",
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


def _fit_gp_models(df, inputs: list[str], outputs: list[str], optimize_hyperparameters: bool):
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


def _partial_dependence_curve(gp_model, parameter_specs, x_reference_raw: np.ndarray, input_index: int, grid_values: np.ndarray) -> np.ndarray:
    model, y_mean, y_std = gp_model
    pd_means: list[float] = []
    for grid_value in grid_values:
        x_modified_raw = x_reference_raw.copy()
        x_modified_raw[:, input_index] = grid_value
        x_modified = transform_to_prior_quantiles(parameter_specs, x_modified_raw)
        pred_mean, _ = predict_scaled_gp(model, y_mean, y_std, x_modified)
        pd_means.append(float(np.mean(pred_mean)))
    return np.asarray(pd_means)


def _select_outputs(family: str, run_name: str) -> list[str]:
    prefix = f"{run_name}_mass_{family}_log10_"
    return [column for column in TRINITY_MEAN_OUTPUT_COLUMNS if column.startswith(prefix)]


def _resolve_outputs(run_name: str | None, family: str | None, output_column: str | None) -> list[str]:
    if output_column is not None:
        return [output_column]
    if run_name is None or family is None:
        outputs: list[str] = []
        run_names = [run_name] if run_name else list(TRINITY_RUN_CONFIGS.keys())
        families = [family] if family else ["stellar", "black_hole"]
        for family_name in families:
            for run_name_value in run_names:
                outputs.extend(_select_outputs(family_name, run_name_value))
        return outputs
    return _select_outputs(family, run_name)


def _plot_main_effects(df, inputs: list[str], outputs: list[str], output_path: Path, optimize_hyperparameters: bool) -> None:
    parameter_specs = trinity_parameter_specs()
    gp_models, x_reference_raw = _fit_gp_models(df, inputs, outputs, optimize_hyperparameters)
    if len(outputs) == 1:
        ncols = 4
        nrows = int(np.ceil(len(inputs) / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4.0 * ncols, 3.0 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        flat_axes = list(axes.flat)
        output_name = outputs[0]
        for col_index, input_name in enumerate(inputs):
            axis = flat_axes[col_index]
            x = df[input_name].to_numpy(dtype=float)
            y = df[output_name].to_numpy(dtype=float)
            axis.scatter(x, y, s=14, alpha=0.6, color="#2a6f97", edgecolors="none")
            trend_x, trend_y = _running_binned_trend(x, y)
            axis.plot(trend_x, trend_y, color="#d1495b", linewidth=1.3)
            grid_values = np.linspace(float(np.min(x)), float(np.max(x)), 60)
            pd_curve = _partial_dependence_curve(
                gp_model=gp_models[output_name],
                parameter_specs=parameter_specs,
                x_reference_raw=x_reference_raw,
                input_index=col_index,
                grid_values=grid_values,
            )
            axis.plot(grid_values, pd_curve, color="#222222", linewidth=1.6)
            axis.set_xlabel(INPUT_LABELS.get(input_name, input_name), fontsize=8)
            axis.set_ylabel(TRINITY_OUTPUT_LABELS.get(output_name, output_name), fontsize=8)
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
        figsize=(2.6 * len(inputs), 2.4 * len(outputs)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, output_name in enumerate(outputs):
        for col_index, input_name in enumerate(inputs):
            axis = axes[row_index, col_index]
            x = df[input_name].to_numpy(dtype=float)
            y = df[output_name].to_numpy(dtype=float)
            axis.scatter(x, y, s=14, alpha=0.6, color="#2a6f97", edgecolors="none")
            trend_x, trend_y = _running_binned_trend(x, y)
            axis.plot(trend_x, trend_y, color="#d1495b", linewidth=1.3)
            grid_values = np.linspace(float(np.min(x)), float(np.max(x)), 60)
            pd_curve = _partial_dependence_curve(
                gp_model=gp_models[output_name],
                parameter_specs=parameter_specs,
                x_reference_raw=x_reference_raw,
                input_index=col_index,
                grid_values=grid_values,
            )
            axis.plot(grid_values, pd_curve, color="#222222", linewidth=1.6)
            if row_index == len(outputs) - 1:
                axis.set_xlabel(INPUT_LABELS.get(input_name, input_name), fontsize=8)
            if col_index == 0:
                axis.set_ylabel(TRINITY_OUTPUT_LABELS.get(output_name, output_name), fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.2, linewidth=0.4)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_correlation_heatmap(df, inputs: list[str], outputs: list[str], output_path: Path) -> None:
    corr = df[inputs + outputs].corr(method="spearman").loc[outputs, inputs]
    fig, ax = plt.subplots(figsize=(1.0 * len(inputs) + 2, 0.45 * len(outputs) + 2), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(inputs)), [INPUT_LABELS.get(name, name) for name in inputs], rotation=50, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(outputs)), [TRINITY_OUTPUT_LABELS.get(name, name) for name in outputs], fontsize=8)
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

    df = load_or_build_emulator_table(campaign_root)
    inputs = [spec.short_name for spec in trinity_parameter_specs()]

    families = [args.family] if args.family else ["stellar", "black_hole"]
    run_names = [args.run_name] if args.run_name else list(TRINITY_RUN_CONFIGS.keys())

    if args.output_column is not None:
        outputs = [args.output_column]
        if not args.heatmap_only:
            out_name = f"main_effects_mean_{args.output_column}{args.output_suffix}.png"
            _plot_main_effects(
                df=df,
                inputs=inputs,
                outputs=outputs,
                output_path=figures_root / out_name,
                optimize_hyperparameters=args.optimize_hyperparameters,
            )
            print(figures_root / out_name)
        heatmap_name = f"spearman_heatmap_mean_{args.output_column}{args.output_suffix}.png"
        _plot_correlation_heatmap(
            df=df,
            inputs=inputs,
            outputs=outputs,
            output_path=figures_root / heatmap_name,
        )
        return

    if not args.heatmap_only:
        for family in families:
            for run_name in run_names:
                outputs = _select_outputs(family, run_name)
                out_name = f"main_effects_mean_{family}_{run_name}{args.output_suffix}.png"
                _plot_main_effects(
                    df=df,
                    inputs=inputs,
                    outputs=outputs,
                    output_path=figures_root / out_name,
                    optimize_hyperparameters=args.optimize_hyperparameters,
                )

    selected_outputs = _resolve_outputs(args.run_name, args.family, args.output_column)
    _plot_correlation_heatmap(
        df=df,
        inputs=inputs,
        outputs=selected_outputs,
        output_path=figures_root / f"spearman_heatmap_mean{args.output_suffix}.png",
    )

    print(campaign_root / "emulator_table.csv")
    for family in families:
        for run_name in run_names:
            print(figures_root / f"main_effects_mean_{family}_{run_name}{args.output_suffix}.png")
    print(figures_root / f"spearman_heatmap_mean{args.output_suffix}.png")


if __name__ == "__main__":
    main()
