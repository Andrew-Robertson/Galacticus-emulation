from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu import (
    default_parameter_specs,
    fit_scaled_gp,
    predict_scaled_gp,
    transform_to_prior_quantiles,
)


INPUT_COLUMNS = [
    "diskVelocityCharacteristic",
    "diskExponent",
    "henriquesGamma",
    "henriquesDelta1",
    "henriquesDelta2",
    "BHefficiencyRadioMode",
    "spheroidRatioAngularMomentumScaleRadius",
]

MEAN_OUTPUT_COLUMNS = [
    "mass_stellar_log10_0",
    "mass_stellar_log10_1",
    "mass_stellar_log10_2",
]

SCATTER_OUTPUT_COLUMNS = [
    "mass_stellar_log10_scatter_0",
    "mass_stellar_log10_scatter_1",
    "mass_stellar_log10_scatter_2",
]

INPUT_LABELS = {
    "diskVelocityCharacteristic": "disk velocityCharacteristic",
    "diskExponent": "disk exponent",
    "henriquesGamma": "Henriques gamma",
    "henriquesDelta1": "Henriques delta1",
    "henriquesDelta2": "Henriques delta2",
    "BHefficiencyRadioMode": "BH radio-mode efficiency",
    "spheroidRatioAngularMomentumScaleRadius": "spheroid j/r scale ratio",
}

OUTPUT_LABELS = {
    "mass_stellar_log10_0": r"$\langle \log_{10} M_\star \rangle$ low $M_{\rm halo}$",
    "mass_stellar_log10_1": r"$\langle \log_{10} M_\star \rangle$ mid $M_{\rm halo}$",
    "mass_stellar_log10_2": r"$\langle \log_{10} M_\star \rangle$ high $M_{\rm halo}$",
    "mass_stellar_log10_scatter_0": r"$\sigma(\log_{10} M_\star)$ low $M_{\rm halo}$",
    "mass_stellar_log10_scatter_1": r"$\sigma(\log_{10} M_\star)$ mid $M_{\rm halo}$",
    "mass_stellar_log10_scatter_2": r"$\sigma(\log_{10} M_\star)$ high $M_{\rm halo}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SHMR data-vector dependence on LHS input parameters."
    )
    parser.add_argument("campaign_root", help="Path to a completed campaign directory.")
    parser.add_argument(
        "--include-scatter",
        action="store_true",
        help="Also plot the SHMR scatter outputs.",
    )
    return parser.parse_args()


def _running_binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
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


def _merge_campaign_tables(campaign_root: Path) -> pd.DataFrame:
    samples = pd.read_csv(campaign_root / "samples.csv")
    summary = pd.read_csv(campaign_root / "summary.csv")
    merged = samples.merge(summary, on="evaluation_id", how="inner", validate="one_to_one")
    merged.to_csv(campaign_root / "emulator_table.csv", index=False)
    return merged


def _load_map_theta(campaign_root: Path, inputs: list[str]) -> np.ndarray | None:
    posterior_path = campaign_root / "emulator_mcmc" / "posterior_samples.csv"
    if not posterior_path.exists():
        return None
    posterior = pd.read_csv(posterior_path)
    row = posterior.loc[posterior["log_probability"].idxmax()]
    return row[inputs].to_numpy(dtype=float)


def _fit_gp_models(df: pd.DataFrame, inputs: list[str], outputs: list[str]) -> tuple[dict[str, tuple[object, float, float]], np.ndarray]:
    parameter_specs = default_parameter_specs()
    x_raw = df[inputs].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)
    gp_models: dict[str, tuple[object, float, float]] = {}
    for output_name in outputs:
        y = df[output_name].to_numpy(dtype=float)
        gp_models[output_name] = fit_scaled_gp(x=x, y=y, n_restarts_optimizer=1)
    return gp_models, x_raw


def _partial_dependence_curve(
    gp_model: tuple[object, float, float],
    parameter_specs,
    x_reference_raw: np.ndarray,
    input_index: int,
    grid_values: np.ndarray,
) -> np.ndarray:
    model, y_mean, y_std = gp_model
    pd_means: list[float] = []
    for grid_value in grid_values:
        x_modified_raw = x_reference_raw.copy()
        x_modified_raw[:, input_index] = grid_value
        x_modified = transform_to_prior_quantiles(parameter_specs, x_modified_raw)
        pred_mean, _ = predict_scaled_gp(model, y_mean, y_std, x_modified)
        pd_means.append(float(np.mean(pred_mean)))
    return np.asarray(pd_means)


def _map_one_at_a_time_curve(
    gp_model: tuple[object, float, float],
    parameter_specs,
    baseline_theta: np.ndarray,
    input_index: int,
    grid_values: np.ndarray,
) -> np.ndarray:
    model, y_mean, y_std = gp_model
    pred_means: list[float] = []
    for grid_value in grid_values:
        theta = baseline_theta.copy()
        theta[input_index] = grid_value
        x = transform_to_prior_quantiles(parameter_specs, theta[None, :])
        pred_mean, _ = predict_scaled_gp(model, y_mean, y_std, x)
        pred_means.append(float(pred_mean[0]))
    return np.asarray(pred_means)


def _plot_main_effects(
    campaign_root: Path,
    df: pd.DataFrame,
    inputs: list[str],
    outputs: list[str],
    output_path: Path,
) -> None:
    parameter_specs = default_parameter_specs()
    gp_models, x_reference_raw = _fit_gp_models(df, inputs, outputs)
    map_theta = _load_map_theta(campaign_root, inputs)
    fig, axes = plt.subplots(
        nrows=len(outputs),
        ncols=len(inputs),
        figsize=(3.0 * len(inputs), 2.6 * len(outputs)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, output_name in enumerate(outputs):
        for col_index, input_name in enumerate(inputs):
            axis = axes[row_index, col_index]
            x = df[input_name].to_numpy()
            y = df[output_name].to_numpy()
            axis.scatter(x, y, s=22, alpha=0.75, color="#2a6f97", edgecolors="none")
            trend_x, trend_y = _running_binned_trend(x, y)
            axis.plot(trend_x, trend_y, color="#d1495b", linewidth=1.8)
            grid_values = np.linspace(float(np.min(x)), float(np.max(x)), 80)
            pd_curve = _partial_dependence_curve(
                gp_model=gp_models[output_name],
                parameter_specs=parameter_specs,
                x_reference_raw=x_reference_raw,
                input_index=col_index,
                grid_values=grid_values,
            )
            axis.plot(grid_values, pd_curve, color="#222222", linewidth=2.2)
            if map_theta is not None:
                map_curve = _map_one_at_a_time_curve(
                    gp_model=gp_models[output_name],
                    parameter_specs=parameter_specs,
                    baseline_theta=map_theta,
                    input_index=col_index,
                    grid_values=grid_values,
                )
                axis.plot(grid_values, map_curve, color="#2a9d8f", linewidth=2.0, linestyle="--")
                axis.axvline(map_theta[col_index], color="#2a9d8f", linewidth=1.0, linestyle=":")
            if row_index == len(outputs) - 1:
                axis.set_xlabel(INPUT_LABELS.get(input_name, input_name))
            if col_index == 0:
                axis.set_ylabel(OUTPUT_LABELS.get(output_name, output_name))
            axis.grid(alpha=0.2, linewidth=0.5)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame, inputs: list[str], outputs: list[str], output_path: Path) -> None:
    corr = df[inputs + outputs].corr(method="spearman").loc[outputs, inputs]
    fig, ax = plt.subplots(
        figsize=(1.25 * len(inputs) + 1.5, 0.8 * len(outputs) + 1.5),
        constrained_layout=True,
    )
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(inputs)), [INPUT_LABELS.get(name, name) for name in inputs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(outputs)), [OUTPUT_LABELS.get(name, name) for name in outputs])
    for row_index in range(len(outputs)):
        for col_index in range(len(inputs)):
            value = corr.iat[row_index, col_index]
            ax.text(col_index, row_index, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("Spearman correlation")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    df = _merge_campaign_tables(campaign_root)
    _plot_main_effects(
        campaign_root=campaign_root,
        df=df,
        inputs=INPUT_COLUMNS,
        outputs=MEAN_OUTPUT_COLUMNS,
        output_path=figures_root / "main_effects_mean.png",
    )
    _plot_correlation_heatmap(
        df=df,
        inputs=INPUT_COLUMNS,
        outputs=MEAN_OUTPUT_COLUMNS,
        output_path=figures_root / "spearman_heatmap_mean.png",
    )

    print(campaign_root / "emulator_table.csv")
    print(figures_root / "main_effects_mean.png")
    print(figures_root / "spearman_heatmap_mean.png")

    if args.include_scatter:
        _plot_main_effects(
            campaign_root=campaign_root,
            df=df,
            inputs=INPUT_COLUMNS,
            outputs=SCATTER_OUTPUT_COLUMNS,
            output_path=figures_root / "main_effects_scatter.png",
        )
        _plot_correlation_heatmap(
            df=df,
            inputs=INPUT_COLUMNS,
            outputs=SCATTER_OUTPUT_COLUMNS,
            output_path=figures_root / "spearman_heatmap_scatter.png",
        )
        print(figures_root / "main_effects_scatter.png")
        print(figures_root / "spearman_heatmap_scatter.png")


if __name__ == "__main__":
    main()
