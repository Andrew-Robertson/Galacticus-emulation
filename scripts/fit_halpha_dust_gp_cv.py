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
from sklearn.model_selection import LeaveOneGroupOut

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
    parser = argparse.ArgumentParser(
        description=(
            "Fit grouped-CV Gaussian-process emulators for dust-attenuated Halpha luminosity-function bins. "
            "Whole Galacticus evaluations are held out together."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--sobral-label", default="Z1", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument(
        "--bin-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of LF bin indices to emulate.",
    )
    parser.add_argument("--n-restarts-optimizer", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--figures-dir-name", default="figures_halpha_dust")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_dust")
    parser.add_argument(
        "--max-draws-per-eval",
        type=int,
        default=None,
        help="If set, keep only the first N dust draws within each evaluation for a lighter exact-GP fit.",
    )
    parser.add_argument(
        "--subset-draws-per-eval",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of dust-draw subset sizes per evaluation for grouped-CV scaling tests.",
    )
    parser.add_argument(
        "--main-effect-bins",
        type=int,
        nargs="*",
        default=None,
        help="Specific Sobral bin indices to use for main-effects plots. Defaults to two representative bins.",
    )
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


def _load_table(campaign_root: Path, table_filename: str, sobral_label: str) -> tuple[pd.DataFrame, list[str]]:
    def _bin_sort_key(column: str) -> int:
        match = re.search(r"_bin(\d+)$", column)
        if match is None:
            raise ValueError(f"Could not parse bin index from column '{column}'")
        return int(match.group(1))

    table = pd.read_csv(campaign_root / table_filename)
    output_columns = sorted(
        (
            column
            for column in table.columns
            if column.startswith(f"halpha_sobral_{sobral_label.lower()}_bin")
            and not column.endswith("_target")
            and not column.endswith("_target_std")
        ),
        key=_bin_sort_key,
    )
    if not output_columns:
        raise ValueError(f"No output columns found for sobral label {sobral_label}")
    required_columns = ["evaluation_id", *INPUT_COLUMNS, *output_columns]
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return table, output_columns


def _select_output_bins(output_columns: list[str], bin_indices: list[int] | None) -> list[str]:
    if bin_indices is None:
        return output_columns
    selected = []
    for bin_index in bin_indices:
        if bin_index < 0 or bin_index >= len(output_columns):
            raise IndexError(f"Requested bin index {bin_index} outside [0, {len(output_columns)-1}]")
        selected.append(output_columns[bin_index])
    return selected


def _suffix_from_draw_limit(max_draws_per_eval: int | None) -> str:
    return "" if max_draws_per_eval is None else f"_trainrows{max_draws_per_eval}"


def _limit_draws_per_eval(table: pd.DataFrame, max_draws_per_eval: int | None) -> pd.DataFrame:
    if max_draws_per_eval is None:
        return table.copy()
    limited = (
        table.sort_values(["evaluation_id", "dust_draw_index"])
        .groupby("evaluation_id", group_keys=False)
        .head(max_draws_per_eval)
        .reset_index(drop=True)
    )
    return limited


def _log10_with_floor(values: np.ndarray) -> tuple[np.ndarray, float, int]:
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("No positive LF values found for log10 transform.")
    floor = 0.5 * float(np.min(positive))
    n_floored = int(np.count_nonzero(values <= 0.0))
    safe = np.where(values > 0.0, values, floor)
    return np.log10(safe), floor, n_floored


def _grouped_cv_predict(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros_like(y)
    pred_std = np.zeros_like(y)
    splitter = LeaveOneGroupOut()
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        test_group = groups[test_index[0]]
        print(f"grouped CV fold {fold_index}: hold out {test_group} ({len(test_index)} rows)")
        for bin_index in range(y.shape[1]):
            model, y_mean, y_std = _fit_scaled_gp(
                x[train_index],
                y[train_index, bin_index],
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=random_state,
            )
            pred[test_index, bin_index], pred_std[test_index, bin_index] = _predict_scaled_gp(
                model,
                y_mean,
                y_std,
                x[test_index],
            )
    return pred, pred_std


def _grouped_cv_predict_train_subset(
    table: pd.DataFrame,
    output_columns: list[str],
    train_draws_per_eval: int,
    *,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_linear = table[output_columns].to_numpy(dtype=float)
    y, _, _ = _log10_with_floor(y_linear)
    x_raw = table[INPUT_COLUMNS].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)
    groups = table["evaluation_id"].to_numpy()
    dust_draw_index = table["dust_draw_index"].to_numpy()

    pred = np.zeros_like(y)
    pred_std = np.zeros_like(y)
    splitter = LeaveOneGroupOut()
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        test_group = groups[test_index[0]]
        print(
            f"grouped CV subset fold {fold_index}: hold out {test_group} ({len(test_index)} rows), "
            f"train draws/eval={train_draws_per_eval}"
        )
        train_groups = groups[train_index]
        train_draws = dust_draw_index[train_index]
        keep_train = np.zeros_like(train_index, dtype=bool)
        for group_name in np.unique(train_groups):
            group_mask = train_groups == group_name
            group_positions = np.flatnonzero(group_mask)
            group_draws = train_draws[group_positions]
            order = np.argsort(group_draws)
            keep_positions = group_positions[order[:train_draws_per_eval]]
            keep_train[keep_positions] = True
        train_subset = train_index[keep_train]
        for bin_index in range(y.shape[1]):
            model, y_mean, y_std = _fit_scaled_gp(
                x[train_subset],
                y[train_subset, bin_index],
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=random_state,
            )
            pred[test_index, bin_index], pred_std[test_index, bin_index] = _predict_scaled_gp(
                model,
                y_mean,
                y_std,
                x[test_index],
            )
    return y, pred, pred_std


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, output_columns: list[str]) -> pd.DataFrame:
    rows = []
    for bin_index, output_column in enumerate(output_columns):
        rows.append(
            {
                "output": output_column,
                "bin_index": bin_index,
                "rmse_dex": float(np.sqrt(mean_squared_error(y_true[:, bin_index], y_pred[:, bin_index]))),
                "mae_dex": float(mean_absolute_error(y_true[:, bin_index], y_pred[:, bin_index])),
                "r2": float(r2_score(y_true[:, bin_index], y_pred[:, bin_index])),
                "coverage_1sigma": float(np.mean(np.abs(y_pred[:, bin_index] - y_true[:, bin_index]) <= y_std[:, bin_index])),
                "coverage_2sigma": float(np.mean(np.abs(y_pred[:, bin_index] - y_true[:, bin_index]) <= 2.0 * y_std[:, bin_index])),
            }
        )
    rows.append(
        {
            "output": "all",
            "bin_index": np.nan,
            "rmse_dex": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
            "mae_dex": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
            "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
            "coverage_1sigma": float(np.mean(np.abs(y_pred - y_true) <= y_std)),
            "coverage_2sigma": float(np.mean(np.abs(y_pred - y_true) <= 2.0 * y_std)),
        }
    )
    return pd.DataFrame(rows)


def _plot_parity(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, output_columns: list[str], path: Path) -> None:
    n_bins = y_true.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_flat = np.ravel(axes)
    for axis, bin_index in zip(axes_flat, range(n_bins), strict=False):
        observed = y_true[:, bin_index]
        predicted = y_pred[:, bin_index]
        lower = float(min(observed.min(), predicted.min()))
        upper = float(max(observed.max(), predicted.max()))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(observed, predicted, yerr=y_std[:, bin_index], fmt="o", ms=2.8, alpha=0.65)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(output_columns[bin_index].replace("halpha_sobral_", "").replace("_", " "))
        axis.set_xlabel("True log10 LF")
        axis.set_ylabel("GP CV log10 LF")
        axis.grid(alpha=0.2)
    for axis in axes_flat[n_bins:]:
        axis.axis("off")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _choose_main_effect_bins(output_columns: list[str], explicit_bins: list[int] | None) -> list[int]:
    if explicit_bins:
        return explicit_bins
    n_bins = len(output_columns)
    if n_bins <= 2:
        return list(range(n_bins))
    return [n_bins // 3, (2 * n_bins) // 3]


def _plot_main_effects(
    x: np.ndarray,
    y: np.ndarray,
    input_columns: list[str],
    output_columns: list[str],
    selected_bins: list[int],
    *,
    n_restarts_optimizer: int,
    random_state: int,
    path: Path,
) -> None:
    x_ref = np.median(x, axis=0)
    sweep_count = 200
    fig, axes = plt.subplots(
        len(selected_bins),
        len(input_columns),
        figsize=(3.2 * len(input_columns), 2.8 * len(selected_bins)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, bin_index in enumerate(selected_bins):
        model, y_mean, y_std = _fit_scaled_gp(
            x,
            y[:, bin_index],
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        for col_index, input_name in enumerate(input_columns):
            axis = axes[row_index, col_index]
            values = np.linspace(np.min(x[:, col_index]), np.max(x[:, col_index]), sweep_count)
            grid = np.tile(x_ref, (sweep_count, 1))
            grid[:, col_index] = values
            pred, pred_std = _predict_scaled_gp(model, y_mean, y_std, grid)
            axis.plot(values, pred, color="#345995", lw=1.8)
            axis.fill_between(values, pred - pred_std, pred + pred_std, color="#345995", alpha=0.2)
            axis.set_xlabel(input_name)
            if col_index == 0:
                axis.set_ylabel(output_columns[bin_index].replace("halpha_sobral_", "").replace("_", " "))
            axis.grid(alpha=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_subset_scaling(metrics_by_subset: dict[int, pd.DataFrame], path: Path) -> None:
    subset_sizes = sorted(metrics_by_subset)
    rmse = [float(metrics_by_subset[size].loc[metrics_by_subset[size]["output"] == "all", "rmse_dex"].iloc[0]) for size in subset_sizes]
    r2 = [float(metrics_by_subset[size].loc[metrics_by_subset[size]["output"] == "all", "r2"].iloc[0]) for size in subset_sizes]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].plot(subset_sizes, rmse, marker="o", lw=1.8)
    axes[0].set_xlabel("Dust draws per evaluation")
    axes[0].set_ylabel("Grouped-CV RMSE [dex]")
    axes[0].grid(alpha=0.2)
    axes[1].plot(subset_sizes, r2, marker="o", lw=1.8)
    axes[1].set_xlabel("Dust draws per evaluation")
    axes[1].set_ylabel("Grouped-CV R2")
    axes[1].grid(alpha=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    full_table, output_columns = _load_table(campaign_root, args.table_filename, args.sobral_label)
    output_columns = _select_output_bins(output_columns, args.bin_indices)
    table = _limit_draws_per_eval(full_table, args.max_draws_per_eval)
    run_suffix = _suffix_from_draw_limit(args.max_draws_per_eval)
    x_raw = table[INPUT_COLUMNS].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)
    y_linear = table[output_columns].to_numpy(dtype=float)
    y, log10_floor, n_floored = _log10_with_floor(y_linear)
    groups = table["evaluation_id"].to_numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        y_pred, y_pred_std = _grouped_cv_predict(
            x,
            y,
            groups,
            n_restarts_optimizer=args.n_restarts_optimizer,
            random_state=args.random_state,
        )

    metrics = _metrics(y, y_pred, y_pred_std, output_columns)
    metrics_path = emulator_root / f"gp_cv_metrics_halpha_{args.sobral_label.lower()}{run_suffix}.csv"
    metrics.to_csv(metrics_path, index=False)

    predictions = table[["evaluation_id", "dust_draw_index", *INPUT_COLUMNS]].copy()
    for bin_index, output_column in enumerate(output_columns):
        predictions[f"{output_column}_true_log10"] = y[:, bin_index]
        predictions[f"{output_column}_pred_log10"] = y_pred[:, bin_index]
        predictions[f"{output_column}_pred_std_log10"] = y_pred_std[:, bin_index]
    predictions_path = emulator_root / f"gp_cv_predictions_halpha_{args.sobral_label.lower()}{run_suffix}.csv"
    predictions.to_csv(predictions_path, index=False)

    summary = {
        "campaign_root": str(campaign_root),
        "table_filename": args.table_filename,
        "sobral_label": args.sobral_label,
        "input_columns": INPUT_COLUMNS,
        "input_transform": "prior_quantiles",
        "output_columns": output_columns,
        "n_rows": int(len(table)),
        "n_evaluations": int(table["evaluation_id"].nunique()),
        "n_dust_draws_mean": float(table.groupby("evaluation_id").size().mean()),
        "log10_floor": log10_floor,
        "n_floored_values": n_floored,
        "metrics": metrics.to_dict(orient="records"),
    }
    summary_path = emulator_root / f"gp_cv_summary_halpha_{args.sobral_label.lower()}{run_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    parity_path = figures_root / f"gp_cv_parity_halpha_{args.sobral_label.lower()}{run_suffix}.png"
    _plot_parity(y, y_pred, y_pred_std, output_columns, parity_path)

    selected_bins = _choose_main_effect_bins(output_columns, args.main_effect_bins)
    main_effects_path = figures_root / f"main_effects_halpha_{args.sobral_label.lower()}{run_suffix}.png"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        _plot_main_effects(
            x,
            y,
            INPUT_COLUMNS,
            output_columns,
            selected_bins,
            n_restarts_optimizer=args.n_restarts_optimizer,
            random_state=args.random_state,
            path=main_effects_path,
        )

    if args.subset_draws_per_eval:
        metrics_by_subset: dict[int, pd.DataFrame] = {}
        for subset_size in args.subset_draws_per_eval:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                subset_y, subset_pred, subset_std = _grouped_cv_predict_train_subset(
                    full_table,
                    output_columns,
                    subset_size,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    random_state=args.random_state,
                )
            subset_metrics = _metrics(subset_y, subset_pred, subset_std, output_columns)
            metrics_by_subset[subset_size] = subset_metrics
            subset_metrics.to_csv(
                emulator_root / f"gp_cv_metrics_halpha_{args.sobral_label.lower()}_fullheldout_train{subset_size}.csv",
                index=False,
            )
            subset_predictions = full_table[["evaluation_id", "dust_draw_index", *INPUT_COLUMNS]].copy()
            for bin_index, output_column in enumerate(output_columns):
                subset_predictions[f"{output_column}_true_log10"] = subset_y[:, bin_index]
                subset_predictions[f"{output_column}_pred_log10"] = subset_pred[:, bin_index]
                subset_predictions[f"{output_column}_pred_std_log10"] = subset_std[:, bin_index]
            subset_predictions.to_csv(
                emulator_root / f"gp_cv_predictions_halpha_{args.sobral_label.lower()}_fullheldout_train{subset_size}.csv",
                index=False,
            )
        subset_scaling_path = figures_root / f"gp_subset_scaling_halpha_{args.sobral_label.lower()}.png"
        _plot_subset_scaling(metrics_by_subset, subset_scaling_path)
        print(subset_scaling_path)

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(parity_path)
    print(main_effects_path)


if __name__ == "__main__":
    main()
