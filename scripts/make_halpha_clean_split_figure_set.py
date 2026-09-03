from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

DEFAULT_COMBOS = [
    ("Z1", 7),
    ("Z1", 16),
    ("Z4", 1),
    ("Z4", 12),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean Halpha figure set: random LF examples, 2x2 parity, and 4x6 main-effects "
            "panels for selected Sobral redshift/luminosity bins."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--figures-dir-name", default="figures_halpha_dust_clean_split_20260428")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_dust_clean_split_20260428")
    parser.add_argument("--n-train-evals", type=int, default=16)
    parser.add_argument("--train-draws-per-eval", type=int, default=8)
    parser.add_argument("--n-restarts-optimizer", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--n-random-curves", type=int, default=100)
    parser.add_argument("--raw-window", type=int, default=61)
    parser.add_argument("--main-effects-mode", default="slice", choices=["slice", "empirical"])
    parser.add_argument("--pd-background-size", type=int, default=48)
    parser.add_argument("--pd-sweep-count", type=int, default=120)
    return parser.parse_args()


def _kernel(n_dim: int):
    return ConstantKernel(4.0, (1.0e-3, 1.0e3)) * Matern(
        length_scale=np.asarray([1.0, 0.5, 1.5, 1.0, 3.0, 10.0], dtype=float)[:n_dim],
        length_scale_bounds=(1.0e-2, 1.0e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=3.0e-2, noise_level_bounds=(1.0e-8, 1.0))


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


def _bin_sort_key(column: str) -> int:
    match = re.search(r"_bin(\d+)$", column)
    if match is None:
        raise ValueError(f"Could not parse bin index from column '{column}'")
    return int(match.group(1))


def _output_columns(table: pd.DataFrame, sobral_label: str) -> list[str]:
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


def _apply_filters(
    table: pd.DataFrame,
    max_abs_delta: float | None,
    max_attenuation_scatter: float | None,
) -> pd.DataFrame:
    mask = np.ones(len(table), dtype=bool)
    if max_abs_delta is not None:
        for column in ["delta_0", "delta_z", "delta_M", "delta_Mz"]:
            mask &= np.abs(table[column].to_numpy(dtype=float)) <= max_abs_delta
    if max_attenuation_scatter is not None:
        mask &= table["attenuation_scatter"].to_numpy(dtype=float) <= max_attenuation_scatter
    return table.loc[mask].reset_index(drop=True)


def _log10_with_floor(values: np.ndarray, min_log10_lf: float | None) -> tuple[np.ndarray, float, int, int]:
    values = np.asarray(values, dtype=float)
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("No positive LF values found for log10 transform.")
    floor = 0.5 * float(np.min(positive))
    n_floored = int(np.count_nonzero(values <= 0.0))
    safe = np.where(values > 0.0, values, floor)
    log_values = np.log10(safe)
    n_clipped = 0
    if min_log10_lf is not None:
        n_clipped = int(np.count_nonzero(log_values < min_log10_lf))
        log_values = np.maximum(log_values, min_log10_lf)
    return log_values, floor, n_floored, n_clipped


def _space_filling_subset_indices(n_items: int, subset_size: int) -> np.ndarray:
    if subset_size > n_items:
        raise ValueError("subset_size must be <= n_items")
    return np.asarray(sorted({int(round(value)) for value in np.linspace(0, n_items - 1, subset_size)}), dtype=int)


def _choose_train_test_eval_ids(table: pd.DataFrame, n_train_evals: int) -> tuple[list[str], list[str]]:
    summary = (
        table.groupby("evaluation_id", as_index=False)["diskVelocityCharacteristic"]
        .first()
        .sort_values("diskVelocityCharacteristic")
        .reset_index(drop=True)
    )
    indices = _space_filling_subset_indices(len(summary), n_train_evals)
    train_eval_ids = summary.iloc[indices]["evaluation_id"].tolist()
    test_eval_ids = summary.loc[~summary["evaluation_id"].isin(train_eval_ids), "evaluation_id"].tolist()
    return train_eval_ids, test_eval_ids


def _rolling_mean_sorted(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
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


def _empirical_partial_dependence(
    model: GaussianProcessRegressor,
    y_mean: float,
    y_std: float,
    x_train_raw: np.ndarray,
    column_index: int,
    sweep_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros_like(sweep_values, dtype=float)
    stds = np.zeros_like(sweep_values, dtype=float)
    for idx, value in enumerate(sweep_values):
        grid_raw = np.array(x_train_raw, copy=True)
        grid_raw[:, column_index] = value
        grid = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, grid_raw)
        pred, pred_std = _predict_scaled_gp(model, y_mean, y_std, grid)
        mean = float(np.mean(pred))
        second_moment = float(np.mean(pred**2 + pred_std**2))
        means[idx] = mean
        stds[idx] = np.sqrt(max(second_moment - mean**2, 0.0))
    return means, stds


def _subsample_background_rows(x_train_raw: np.ndarray, max_rows: int) -> np.ndarray:
    if len(x_train_raw) <= max_rows:
        return x_train_raw
    indices = _space_filling_subset_indices(len(x_train_raw), max_rows)
    return x_train_raw[indices]


def _reference_slice_dependence(
    model: GaussianProcessRegressor,
    y_mean: float,
    y_std: float,
    x_train_raw: np.ndarray,
    column_index: int,
    sweep_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_ref = np.median(x_train_raw, axis=0)
    grid_raw = np.tile(x_ref, (len(sweep_values), 1))
    grid_raw[:, column_index] = sweep_values
    grid = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, grid_raw)
    return _predict_scaled_gp(model, y_mean, y_std, grid)


def _combo_metadata(long_table: pd.DataFrame, sobral_label: str, bin_index: int) -> tuple[float, float]:
    row = (
        long_table.loc[
            (long_table["sobral_label"] == sobral_label) & (long_table["bin_index"] == bin_index),
            ["redshift", "log10_luminosity_center"],
        ]
        .drop_duplicates()
        .iloc[0]
    )
    return float(row["redshift"]), float(row["log10_luminosity_center"])


def _plot_random_lfs(
    train: pd.DataFrame,
    long_table: pd.DataFrame,
    sobral_label: str,
    figures_root: Path,
    n_curves: int,
    seed: int,
) -> None:
    output_columns = _output_columns(train, sobral_label)
    sub_long = long_table.loc[long_table["sobral_label"] == sobral_label]
    centers = (
        sub_long[["bin_index", "log10_luminosity_center"]]
        .drop_duplicates()
        .sort_values("bin_index")
    )
    x = centers["log10_luminosity_center"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    sample_size = min(n_curves, len(train))
    subset = train.iloc[np.sort(rng.choice(len(train), size=sample_size, replace=False))].copy()
    y, _, _, _ = _log10_with_floor(subset[output_columns].to_numpy(dtype=float), min_log10_lf=-7.0)
    target, _, _, _ = _log10_with_floor(
        subset[[f"{col}_target" for col in output_columns]].iloc[[0]].to_numpy(dtype=float).ravel(),
        min_log10_lf=-7.0,
    )
    target_std_linear = subset[[f"{col}_target_std" for col in output_columns]].iloc[[0]].to_numpy(dtype=float).ravel()
    target_linear = subset[[f"{col}_target" for col in output_columns]].iloc[[0]].to_numpy(dtype=float).ravel()
    target_std_log10 = target_std_linear / (np.maximum(target_linear, 1.0e-30) * np.log(10.0))

    color_values = subset["delta_0"].to_numpy(dtype=float)
    norm = plt.Normalize(vmin=float(np.min(color_values)), vmax=float(np.max(color_values)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
    for curve, color_value in zip(y, color_values, strict=True):
        ax.plot(x, curve, color=cmap(norm(color_value)), alpha=0.32, lw=1.2)
    ax.plot(x, target, color="black", lw=2.2, label="Sobral target")
    ax.fill_between(x, target - target_std_log10, target + target_std_log10, color="black", alpha=0.12)
    ax.set_xlabel(r"$\log_{10} L_{\mathrm{H}\alpha}$")
    ax.set_ylabel(r"$\log_{10}\,\mathrm{d}n/\mathrm{d}\ln L$")
    ax.set_title(f"{sample_size} random z={sub_long['redshift'].iloc[0]:.2f} training LFs")
    ax.set_ylim(-7.0, -1.0)
    ax.grid(alpha=0.2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$\delta_0$")
    path = figures_root / f"random_halpha_lfs_{sobral_label.lower()}_training_colorby_delta_0.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(campaign_root / args.table_filename)
    long_table = pd.read_csv(campaign_root / args.long_filename)
    table = _apply_filters(table, args.max_abs_delta, args.max_attenuation_scatter)
    train_eval_ids, test_eval_ids = _choose_train_test_eval_ids(table, args.n_train_evals)

    train = (
        table.loc[table["evaluation_id"].isin(train_eval_ids)]
        .sort_values(["evaluation_id", "dust_draw_index"])
        .groupby("evaluation_id", group_keys=False)
        .head(args.train_draws_per_eval)
        .reset_index(drop=True)
    )
    test = table.loc[table["evaluation_id"].isin(test_eval_ids)].reset_index(drop=True)

    x_train_raw = train[INPUT_COLUMNS].to_numpy(dtype=float)
    x_test_raw = test[INPUT_COLUMNS].to_numpy(dtype=float)
    x_train = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_train_raw)
    x_test = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_test_raw)
    x_pd_raw = _subsample_background_rows(x_train_raw, args.pd_background_size)

    for sobral_label, seed_offset in [("Z1", 0), ("Z4", 1)]:
        _plot_random_lfs(
            train,
            long_table,
            sobral_label=sobral_label,
            figures_root=figures_root,
            n_curves=args.n_random_curves,
            seed=args.random_state + seed_offset,
        )
        print(f"wrote random LF plot for {sobral_label}", flush=True)

    combo_results: list[dict[str, object]] = []
    kernel_summary: dict[str, str] = {}

    parity_fig, parity_axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    main_fig, main_axes = plt.subplots(4, len(INPUT_COLUMNS), figsize=(3.2 * len(INPUT_COLUMNS), 3.0 * 4), constrained_layout=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for row_index, (sobral_label, bin_index) in enumerate(DEFAULT_COMBOS):
            print(f"fitting combo {row_index + 1}/{len(DEFAULT_COMBOS)}: {sobral_label} bin {bin_index}", flush=True)
            output_column = _output_columns(table, sobral_label)[bin_index]
            y_train, log10_floor, n_floored_train, n_clipped_train = _log10_with_floor(
                train[output_column].to_numpy(dtype=float),
                args.min_log10_lf,
            )
            y_test, _, n_floored_test, n_clipped_test = _log10_with_floor(
                test[output_column].to_numpy(dtype=float),
                args.min_log10_lf,
            )
            model, y_mean, y_std = _fit_scaled_gp(
                x_train,
                y_train,
                n_restarts_optimizer=args.n_restarts_optimizer,
                random_state=args.random_state,
            )
            y_pred, y_pred_std = _predict_scaled_gp(model, y_mean, y_std, x_test)
            redshift, log10_center = _combo_metadata(long_table, sobral_label, bin_index)

            metrics = {
                "sobral_label": sobral_label,
                "redshift": redshift,
                "bin_index": bin_index,
                "log10_luminosity_center": log10_center,
                "rmse_dex": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae_dex": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
                "coverage_1sigma": float(np.mean(np.abs(y_pred - y_test) <= y_pred_std)),
                "coverage_2sigma": float(np.mean(np.abs(y_pred - y_test) <= 2.0 * y_pred_std)),
                "n_floored_values_train": n_floored_train,
                "n_floored_values_test": n_floored_test,
                "n_clipped_values_train": n_clipped_train,
                "n_clipped_values_test": n_clipped_test,
                "log10_floor": log10_floor,
            }
            combo_results.append(metrics)
            kernel_summary[f"{sobral_label}_bin{bin_index}_log10L{log10_center:.2f}"] = str(model.kernel_)

            parity_axis = parity_axes.flat[row_index]
            lower = float(min(np.min(y_test), np.min(y_pred)))
            upper = float(max(np.max(y_test), np.max(y_pred)))
            margin = 0.08 * (upper - lower if upper > lower else 1.0)
            parity_axis.errorbar(y_test, y_pred, yerr=y_pred_std, fmt="o", ms=3.0, alpha=0.65)
            parity_axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
            parity_axis.set_xlim(lower - margin, upper + margin)
            parity_axis.set_ylim(lower - margin, upper + margin)
            parity_axis.set_title(rf"$z={redshift:.2f}$, $\log_{{10}}L_{{\mathrm{{H}}\alpha}}={log10_center:.2f}$")
            parity_axis.set_xlabel("True log10 LF")
            parity_axis.set_ylabel("Predicted log10 LF")
            parity_axis.grid(alpha=0.2)

            for col_index, input_name in enumerate(INPUT_COLUMNS):
                axis = main_axes[row_index, col_index]
                sweep_values = np.linspace(
                    np.min(x_train_raw[:, col_index]),
                    np.max(x_train_raw[:, col_index]),
                    args.pd_sweep_count,
                )
                if args.main_effects_mode == "empirical":
                    pd_mean, pd_std = _empirical_partial_dependence(
                        model,
                        y_mean,
                        y_std,
                        x_pd_raw,
                        col_index,
                        sweep_values,
                    )
                else:
                    pd_mean, pd_std = _reference_slice_dependence(
                        model,
                        y_mean,
                        y_std,
                        x_train_raw,
                        col_index,
                        sweep_values,
                    )
                x_mid, y_smooth = _rolling_mean_sorted(x_train_raw[:, col_index], y_train, args.raw_window)
                axis.scatter(x_train_raw[:, col_index], y_train, s=9, alpha=0.14, color="#345995")
                axis.plot(x_mid, y_smooth, color="#d1495b", lw=1.7)
                axis.plot(sweep_values, pd_mean, color="black", lw=1.8)
                axis.fill_between(sweep_values, pd_mean - pd_std, pd_mean + pd_std, color="0.3", alpha=0.12)
                axis.grid(alpha=0.2)
                if row_index == 0:
                    axis.set_title(input_name)
                if col_index == 0:
                    axis.set_ylabel(
                        rf"$z={redshift:.2f}$" + "\n" + rf"$\log_{{10}}L_{{\mathrm{{H}}\alpha}}={log10_center:.2f}$" + "\n" + r"$\log_{10}\,\mathrm{d}n/\mathrm{d}\ln L$"
                    )
                if row_index == len(DEFAULT_COMBOS) - 1:
                    axis.set_xlabel(input_name)
            print(f"finished combo {row_index + 1}/{len(DEFAULT_COMBOS)}: {sobral_label} bin {bin_index}", flush=True)

    parity_path = figures_root / "gp_split_parity_halpha_z1_z4_2x2.png"
    parity_fig.savefig(parity_path, dpi=220)
    plt.close(parity_fig)

    main_path = figures_root / "main_effects_halpha_split_z1_z4_4x6.png"
    main_fig.savefig(main_path, dpi=220)
    plt.close(main_fig)

    metrics_path = emulator_root / "gp_split_metrics_halpha_z1_z4_4bins.csv"
    pd.DataFrame(combo_results).to_csv(metrics_path, index=False)

    config = {
        "campaign_root": str(campaign_root),
        "figures_dir_name": args.figures_dir_name,
        "emulator_dir_name": args.emulator_dir_name,
        "n_train_evals": args.n_train_evals,
        "train_eval_ids": train_eval_ids,
        "test_eval_ids": test_eval_ids,
        "train_draws_per_eval": args.train_draws_per_eval,
        "n_training_rows": int(len(train)),
        "n_testing_rows": int(len(test)),
        "min_log10_lf": args.min_log10_lf,
        "max_abs_delta": args.max_abs_delta,
        "max_attenuation_scatter": args.max_attenuation_scatter,
        "n_restarts_optimizer": args.n_restarts_optimizer,
        "pd_background_size": args.pd_background_size,
        "pd_sweep_count": args.pd_sweep_count,
        "main_effects_mode": args.main_effects_mode,
        "input_transform": "prior_quantiles",
        "combos": [
            {"sobral_label": sobral_label, "bin_index": bin_index}
            for sobral_label, bin_index in DEFAULT_COMBOS
        ],
        "metrics": combo_results,
        "kernels": kernel_summary,
    }
    config_path = emulator_root / "gp_split_summary_halpha_z1_z4_4bins.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    print(parity_path)
    print(main_path)
    print(metrics_path)
    print(config_path)


if __name__ == "__main__":
    main()
