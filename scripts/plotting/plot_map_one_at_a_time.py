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

from galacticus_emu import default_parameter_specs, fit_scaled_gp, predict_scaled_gp, transform_to_prior_quantiles


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one-at-a-time GP predictions around the emulator MAP point."
    )
    parser.add_argument("campaign_root", help="Path to a completed campaign directory.")
    parser.add_argument(
        "--include-scatter",
        action="store_true",
        help="Also include the SHMR scatter outputs.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Number of parameter values to evaluate per one-at-a-time curve.",
    )
    return parser.parse_args()


def _load_emulator_table(campaign_root: Path) -> pd.DataFrame:
    emulator_table_path = campaign_root / "emulator_table.csv"
    if emulator_table_path.exists():
        return pd.read_csv(emulator_table_path)
    samples = pd.read_csv(campaign_root / "samples.csv")
    summary = pd.read_csv(campaign_root / "summary.csv")
    merged = samples.merge(summary, on="evaluation_id", how="inner", validate="one_to_one")
    merged.to_csv(emulator_table_path, index=False)
    return merged


def _load_map_theta(campaign_root: Path, input_columns: list[str]) -> np.ndarray:
    posterior_path = campaign_root / "emulator_mcmc" / "posterior_samples.csv"
    posterior = pd.read_csv(posterior_path)
    row = posterior.loc[posterior["log_probability"].idxmax()]
    return row[input_columns].to_numpy(dtype=float)


def _fit_gp_models(df: pd.DataFrame, input_columns: list[str], output_columns: list[str]):
    parameter_specs = default_parameter_specs()
    x_raw = df[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)
    gp_models = {}
    for output_name in output_columns:
        y = df[output_name].to_numpy(dtype=float)
        gp_models[output_name] = fit_scaled_gp(x=x, y=y, n_restarts_optimizer=1)
    return gp_models


def _one_at_a_time_curve(
    gp_model: tuple[object, float, float],
    parameter_specs,
    baseline_theta: np.ndarray,
    input_index: int,
    grid_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model, y_mean, y_std = gp_model
    pred_means = []
    pred_stds = []
    for grid_value in grid_values:
        theta = baseline_theta.copy()
        theta[input_index] = grid_value
        x = transform_to_prior_quantiles(parameter_specs, theta[None, :])
        pred_mean, pred_std = predict_scaled_gp(model, y_mean, y_std, x)
        pred_means.append(float(pred_mean[0]))
        pred_stds.append(float(pred_std[0]))
    return np.asarray(pred_means), np.asarray(pred_stds)


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    parameter_specs = default_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    output_columns = [*MEAN_OUTPUT_COLUMNS, *SCATTER_OUTPUT_COLUMNS] if args.include_scatter else [*MEAN_OUTPUT_COLUMNS]

    df = _load_emulator_table(campaign_root)
    gp_models = _fit_gp_models(df, input_columns, output_columns)
    baseline_theta = _load_map_theta(campaign_root, input_columns)

    fig, axes = plt.subplots(
        nrows=len(output_columns),
        ncols=len(input_columns),
        figsize=(3.2 * len(input_columns), 2.8 * len(output_columns)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, output_name in enumerate(output_columns):
        for col_index, spec in enumerate(parameter_specs):
            axis = axes[row_index, col_index]
            grid_values = np.linspace(float(spec.prior.lower), float(spec.prior.upper), args.grid_size)
            pred_mean, pred_std = _one_at_a_time_curve(
                gp_model=gp_models[output_name],
                parameter_specs=parameter_specs,
                baseline_theta=baseline_theta,
                input_index=col_index,
                grid_values=grid_values,
            )
            axis.plot(grid_values, pred_mean, color="#222222", linewidth=2.0)
            axis.fill_between(
                grid_values,
                pred_mean - pred_std,
                pred_mean + pred_std,
                color="#a6bddb",
                alpha=0.35,
            )
            axis.axvline(baseline_theta[col_index], color="#d1495b", linestyle="--", linewidth=1.4)
            if row_index == len(output_columns) - 1:
                axis.set_xlabel(INPUT_LABELS.get(spec.short_name, spec.short_name))
            if col_index == 0:
                axis.set_ylabel(OUTPUT_LABELS.get(output_name, output_name))
            axis.grid(alpha=0.2, linewidth=0.5)

    output_name = "map_one_at_a_time_mean.png" if not args.include_scatter else "map_one_at_a_time_mean_scatter.png"
    output_path = figures_root / output_name
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
