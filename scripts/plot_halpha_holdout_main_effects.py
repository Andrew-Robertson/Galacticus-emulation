from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from galacticus_emu.lhs import (
    NormalPrior,
    ParameterSpec,
    TruncatedLogNormalPrior,
    TruncatedNormalPrior,
    transform_to_prior_quantiles,
)


INPUT_COLUMNS = [
    "diskVelocityCharacteristic",
    "delta_0",
    "delta_z",
    "delta_M",
    "delta_Mz",
    "attenuation_scatter",
]

INPUT_PARAMETER_SPECS = [
    ParameterSpec(
        path="nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic",
        short_name="diskVelocityCharacteristic",
        prior=TruncatedLogNormalPrior(lower=25.0, upper=300.0, x0=150.0, sigma=0.5),
    ),
    ParameterSpec(path="delta_0", short_name="delta_0", prior=NormalPrior(mean=0.0, sigma=0.5)),
    ParameterSpec(path="delta_z", short_name="delta_z", prior=NormalPrior(mean=0.0, sigma=0.5)),
    ParameterSpec(path="delta_M", short_name="delta_M", prior=NormalPrior(mean=0.0, sigma=0.5)),
    ParameterSpec(path="delta_Mz", short_name="delta_Mz", prior=NormalPrior(mean=0.0, sigma=0.5)),
    ParameterSpec(
        path="attenuation_scatter",
        short_name="attenuation_scatter",
        prior=TruncatedNormalPrior(mean=0.25, sigma=0.1, lower=0.0),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--sobral-label", default="Z4", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--bin-index", type=int, required=True)
    parser.add_argument("--holdout-evaluation-id", required=True)
    parser.add_argument("--train-draws-per-eval", type=int, default=16)
    parser.add_argument("--n-restarts-optimizer", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-log10-lf", type=float, default=None)
    parser.add_argument("--max-abs-delta", type=float, default=None)
    parser.add_argument("--max-attenuation-scatter", type=float, default=None)
    parser.add_argument("--figures-dir-name", default="figures_halpha_dust")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--raw-window", type=int, default=61)
    return parser.parse_args()


def _kernel(n_dim: int):
    return ConstantKernel(1.0, (1.0e-3, 1.0e3)) * Matern(
        length_scale=np.full(n_dim, 0.5, dtype=float),
        length_scale_bounds=(1.0e-2, 1.0e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1.0e-4, noise_level_bounds=(1.0e-8, 1.0))


def _fit_scaled_gp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[GaussianProcessRegressor, float, float]:
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std == 0.0:
        y_std = 1.0
    model = GaussianProcessRegressor(
        kernel=_kernel(x.shape[1]),
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    model.fit(x, (y - y_mean) / y_std)
    return model, y_mean, y_std


def _predict_scaled_gp(
    model: GaussianProcessRegressor,
    y_mean: float,
    y_std: float,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_scaled, pred_std_scaled = model.predict(x, return_std=True)
    return y_mean + y_std * pred_scaled, y_std * pred_std_scaled


def _output_columns(table: pd.DataFrame, sobral_label: str) -> list[str]:
    def _bin_sort_key(column: str) -> int:
        match = re.search(r"_bin(\d+)$", column)
        if match is None:
            raise ValueError(f"Could not parse bin index from column '{column}'")
        return int(match.group(1))

    return sorted(
        [
            column
            for column in table.columns
            if column.startswith(f"halpha_sobral_{sobral_label.lower()}_bin")
            and not column.endswith("_target")
            and not column.endswith("_target_std")
        ],
        key=_bin_sort_key,
    )


def _apply_filters(table: pd.DataFrame, max_abs_delta: float | None, max_attenuation_scatter: float | None) -> pd.DataFrame:
    mask = np.ones(len(table), dtype=bool)
    if max_abs_delta is not None:
        for column in ["delta_0", "delta_z", "delta_M", "delta_Mz"]:
            mask &= np.abs(table[column].to_numpy(dtype=float)) <= max_abs_delta
    if max_attenuation_scatter is not None:
        mask &= table["attenuation_scatter"].to_numpy(dtype=float) <= max_attenuation_scatter
    return table.loc[mask].reset_index(drop=True)


def _log10_with_floor(values: np.ndarray, min_log10_lf: float | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values[values > 0.0]
    floor = 0.5 * float(np.min(positive))
    safe = np.where(values > 0.0, values, floor)
    log_values = np.log10(safe)
    if min_log10_lf is not None:
        log_values = np.maximum(log_values, min_log10_lf)
    return log_values


def _rolling_mean_sorted(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if window < 3:
        return x_sorted, y_sorted
    window = min(window, len(x_sorted))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return x_sorted, y_sorted
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_sorted, kernel, mode="valid")
    half = window // 2
    x_mid = x_sorted[half : len(x_sorted) - half]
    return x_mid, y_smooth


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(campaign_root / args.table_filename)
    table = _apply_filters(table, args.max_abs_delta, args.max_attenuation_scatter)
    output_columns = _output_columns(table, args.sobral_label)
    output_column = output_columns[args.bin_index]

    long_table = pd.read_csv(campaign_root / args.long_filename)
    center = (
        long_table.loc[
            (long_table["sobral_label"] == args.sobral_label) & (long_table["bin_index"] == args.bin_index),
            "log10_luminosity_center",
        ]
        .drop_duplicates()
        .iloc[0]
    )

    train = (
        table.loc[table["evaluation_id"] != args.holdout_evaluation_id]
        .sort_values(["evaluation_id", "dust_draw_index"])
        .groupby("evaluation_id", group_keys=False)
        .head(args.train_draws_per_eval)
        .reset_index(drop=True)
    )
    x_train_raw = train[INPUT_COLUMNS].to_numpy(dtype=float)
    x_train = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_train_raw)
    y_train = _log10_with_floor(train[output_column].to_numpy(dtype=float), args.min_log10_lf)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model, y_mean, y_std = _fit_scaled_gp(
            x_train,
            y_train,
            n_restarts_optimizer=args.n_restarts_optimizer,
            random_state=args.random_state,
        )

    x_ref = np.median(x_train_raw, axis=0)
    sweep_count = 200
    fig, axes = plt.subplots(1, len(INPUT_COLUMNS), figsize=(3.2 * len(INPUT_COLUMNS), 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for col_index, input_name in enumerate(INPUT_COLUMNS):
        axis = axes[col_index]
        values = np.linspace(np.min(x_train_raw[:, col_index]), np.max(x_train_raw[:, col_index]), sweep_count)
        grid_raw = np.tile(x_ref, (sweep_count, 1))
        grid_raw[:, col_index] = values
        grid = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, grid_raw)
        pred, pred_std = _predict_scaled_gp(model, y_mean, y_std, grid)
        axis.scatter(x_train_raw[:, col_index], y_train, s=10, alpha=0.15, color="#345995")
        x_mid, y_smooth = _rolling_mean_sorted(x_train_raw[:, col_index], y_train, args.raw_window)
        axis.plot(x_mid, y_smooth, color="#d1495b", lw=1.8)
        axis.plot(values, pred, color="black", lw=1.8)
        axis.fill_between(values, pred - pred_std, pred + pred_std, color="0.3", alpha=0.15)
        axis.set_xlabel(input_name)
        if col_index == 0:
            axis.set_ylabel(rf"$\log_{{10}}\,\mathrm{{d}}n/\mathrm{{d}}\ln L$ at $\log_{{10}}L_{{\mathrm{{H}}\alpha}}={center:.2f}$")
        axis.grid(alpha=0.2)

    suffix = f"z{args.sobral_label[-1].lower()}_bin{args.bin_index}"
    if args.output_tag:
        suffix += f"_{args.output_tag}"
    path = figures_root / f"main_effects_halpha_gp_{suffix}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(path)
    print(model.kernel_)


if __name__ == "__main__":
    main()
