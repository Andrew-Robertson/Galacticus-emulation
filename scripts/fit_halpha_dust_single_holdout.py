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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a single-holdout Halpha dust GP emulator. "
            "One whole Galacticus evaluation is held out, while a limited number of dust draws "
            "from the remaining evaluations are used for training."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--sobral-label", default="Z1", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--bin-indices", type=int, nargs="*", default=None)
    parser.add_argument("--train-draws-per-eval", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--holdout-evaluation-id", default=None)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--figures-dir-name", default="figures_halpha_dust")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_dust")
    parser.add_argument(
        "--min-log10-lf",
        type=float,
        default=None,
        help="Optional lower clip applied after log10-transforming LF values.",
    )
    parser.add_argument(
        "--max-abs-delta",
        type=float,
        default=None,
        help="If set, keep only rows with |delta_0|, |delta_z|, |delta_M|, |delta_Mz| <= this value.",
    )
    parser.add_argument(
        "--max-attenuation-scatter",
        type=float,
        default=None,
        help="If set, keep only rows with attenuation_scatter <= this value.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix tag added to output filenames, e.g. 'clipm7_trunc2sig'.",
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
    return table, output_columns


def _select_output_bins(output_columns: list[str], bin_indices: list[int] | None) -> list[str]:
    if bin_indices is None:
        return output_columns
    return [output_columns[index] for index in bin_indices]


def _log10_with_floor(values: np.ndarray, min_log10_lf: float | None = None) -> tuple[np.ndarray, float, int, int]:
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


def _default_holdout_evaluation_id(table: pd.DataFrame) -> str:
    summary = (
        table.groupby("evaluation_id", as_index=False)["diskVelocityCharacteristic"]
        .first()
        .sort_values("diskVelocityCharacteristic")
        .reset_index(drop=True)
    )
    return str(summary.iloc[len(summary) // 2]["evaluation_id"])


def _select_training_rows(
    table: pd.DataFrame,
    holdout_evaluation_id: str,
    train_draws_per_eval: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = table.loc[table["evaluation_id"] == holdout_evaluation_id].copy()
    if holdout.empty:
        raise ValueError(f"Holdout evaluation_id '{holdout_evaluation_id}' not present in table.")
    train = (
        table.loc[table["evaluation_id"] != holdout_evaluation_id]
        .sort_values(["evaluation_id", "dust_draw_index"])
        .groupby("evaluation_id", group_keys=False)
        .head(train_draws_per_eval)
        .reset_index(drop=True)
    )
    return train, holdout


def _apply_reasonable_region_filters(
    table: pd.DataFrame,
    *,
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
    ncols = min(4, n_bins)
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, bin_index in zip(axes_flat, range(n_bins), strict=False):
        observed = y_true[:, bin_index]
        predicted = y_pred[:, bin_index]
        lower = float(min(observed.min(), predicted.min()))
        upper = float(max(observed.max(), predicted.max()))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(observed, predicted, yerr=y_std[:, bin_index], fmt="o", ms=3.2, alpha=0.75)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(output_columns[bin_index].replace("halpha_sobral_", "").replace("_", " "))
        axis.set_xlabel("True log10 LF")
        axis.set_ylabel("Predicted log10 LF")
        axis.grid(alpha=0.2)
    for axis in axes_flat[n_bins:]:
        axis.axis("off")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_heldout_lf_curves(
    holdout: pd.DataFrame,
    output_columns: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    path: Path,
) -> None:
    n_draws = min(6, len(holdout))
    selected = holdout.sort_values("dust_draw_index").head(n_draws).index.to_numpy()
    bin_numbers = [int(re.search(r"_bin(\d+)$", col).group(1)) for col in output_columns]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for axis, row_index in zip(axes_flat, selected, strict=False):
        loc = holdout.index.get_loc(row_index)
        axis.plot(bin_numbers, y_true[loc], "o-", lw=1.6, ms=3.5, label="True")
        axis.plot(bin_numbers, y_pred[loc], "o-", lw=1.6, ms=3.5, label="GP")
        axis.fill_between(bin_numbers, y_pred[loc] - y_std[loc], y_pred[loc] + y_std[loc], alpha=0.2)
        dust_row = holdout.loc[row_index]
        axis.set_title(
            f"draw {int(dust_row['dust_draw_index'])}: "
            f"d0={dust_row['delta_0']:.2f}, dM={dust_row['delta_M']:.2f}"
        )
        axis.set_xlabel("Sobral bin index")
        axis.set_ylabel("log10 LF")
        axis.grid(alpha=0.2)
    for axis in axes_flat[n_draws:]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    table, output_columns = _load_table(campaign_root, args.table_filename, args.sobral_label)
    table = _apply_reasonable_region_filters(
        table,
        max_abs_delta=args.max_abs_delta,
        max_attenuation_scatter=args.max_attenuation_scatter,
    )
    output_columns = _select_output_bins(output_columns, args.bin_indices)
    holdout_evaluation_id = args.holdout_evaluation_id or _default_holdout_evaluation_id(table)
    print(f"Using holdout evaluation: {holdout_evaluation_id}")

    for train_draws_per_eval in args.train_draws_per_eval:
        train, holdout = _select_training_rows(table, holdout_evaluation_id, train_draws_per_eval)
        x_train_raw = train[INPUT_COLUMNS].to_numpy(dtype=float)
        x_test_raw = holdout[INPUT_COLUMNS].to_numpy(dtype=float)
        x_train = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_train_raw)
        x_test = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_test_raw)
        y_train, log10_floor, n_floored_train, n_clipped_train = _log10_with_floor(
            train[output_columns].to_numpy(dtype=float),
            min_log10_lf=args.min_log10_lf,
        )
        y_test, _, n_floored_test, n_clipped_test = _log10_with_floor(
            holdout[output_columns].to_numpy(dtype=float),
            min_log10_lf=args.min_log10_lf,
        )

        y_pred = np.zeros_like(y_test)
        y_pred_std = np.zeros_like(y_test)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for bin_index in range(y_train.shape[1]):
                model, y_mean, y_std = _fit_scaled_gp(
                    x_train,
                    y_train[:, bin_index],
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    random_state=args.random_state,
                )
                y_pred[:, bin_index], y_pred_std[:, bin_index] = _predict_scaled_gp(
                    model,
                    y_mean,
                    y_std,
                    x_test,
                )

        suffix = f"_singleholdout_{args.sobral_label.lower()}_train{train_draws_per_eval}"
        if args.output_tag:
            suffix += f"_{args.output_tag}"
        metrics = _metrics(y_test, y_pred, y_pred_std, output_columns)
        metrics_path = emulator_root / f"gp_holdout_metrics{suffix}.csv"
        metrics.to_csv(metrics_path, index=False)

        predictions = holdout[["evaluation_id", "dust_draw_index", *INPUT_COLUMNS]].copy()
        for bin_index, output_column in enumerate(output_columns):
            predictions[f"{output_column}_true_log10"] = y_test[:, bin_index]
            predictions[f"{output_column}_pred_log10"] = y_pred[:, bin_index]
            predictions[f"{output_column}_pred_std_log10"] = y_pred_std[:, bin_index]
        predictions_path = emulator_root / f"gp_holdout_predictions{suffix}.csv"
        predictions.to_csv(predictions_path, index=False)

        summary = {
            "campaign_root": str(campaign_root),
            "table_filename": args.table_filename,
            "sobral_label": args.sobral_label,
            "input_columns": INPUT_COLUMNS,
            "input_transform": "prior_quantiles",
            "output_columns": output_columns,
            "holdout_evaluation_id": holdout_evaluation_id,
            "holdout_diskVelocityCharacteristic": float(holdout["diskVelocityCharacteristic"].iloc[0]),
            "n_training_rows": int(len(train)),
            "n_testing_rows": int(len(holdout)),
            "train_draws_per_eval": int(train_draws_per_eval),
            "min_log10_lf": args.min_log10_lf,
            "max_abs_delta": args.max_abs_delta,
            "max_attenuation_scatter": args.max_attenuation_scatter,
            "log10_floor": log10_floor,
            "n_floored_values_train": n_floored_train,
            "n_floored_values_test": n_floored_test,
            "n_clipped_values_train": n_clipped_train,
            "n_clipped_values_test": n_clipped_test,
            "metrics": metrics.to_dict(orient="records"),
        }
        summary_path = emulator_root / f"gp_holdout_summary{suffix}.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        parity_path = figures_root / f"gp_holdout_parity{suffix}.png"
        _plot_parity(y_test, y_pred, y_pred_std, output_columns, parity_path)

        curves_path = figures_root / f"gp_holdout_curves{suffix}.png"
        _plot_heldout_lf_curves(holdout, output_columns, y_test, y_pred, y_pred_std, curves_path)

        print(metrics_path)
        print(predictions_path)
        print(summary_path)
        print(parity_path)
        print(curves_path)


if __name__ == "__main__":
    main()
