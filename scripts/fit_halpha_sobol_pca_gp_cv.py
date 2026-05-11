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
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp
from galacticus_emu.lhs import (
    NormalPrior,
    ParameterSpec,
    TruncatedNormalPrior,
    transform_to_prior_quantiles,
)


DUST_INPUT_COLUMNS = [
    "delta_0",
    "delta_z",
    "delta_M",
    "delta_Mz",
    "attenuation_scatter",
]

DUST_INPUT_PARAMETER_SPECS = [
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
            "Fit a PCA-decomposed grouped-CV GP emulator for one Sobral Halpha LF in the 19D Sobol campaign, "
            "using 19 slow-parameter quantiles plus 5 dust parameters."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--sobral-label", default="Z4", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--train-draws-per-eval", type=int, default=1)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="unscaled")
    parser.add_argument("--use-training-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shot-noise-alpha",
        choices=["conservative", "smoothed_expectation"],
        default="conservative",
        help="Which Halpha shot-noise estimate to use as GP training alpha.",
    )
    parser.add_argument("--min-training-log10-sigma", type=float, default=1.0e-4)
    parser.add_argument("--max-training-log10-sigma", type=float, default=2.0)
    parser.add_argument("--fit-white-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--figures-dir-name", default="figures_halpha_sobol_pca_cv")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_sobol_pca_cv")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


class MeanCenterer:
    def fit(self, y: np.ndarray) -> "MeanCenterer":
        self.mean_ = np.mean(y, axis=0)
        self.scale_ = np.ones(y.shape[1], dtype=float)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y - self.mean_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y + self.mean_


def _make_preprocessor(pca_scaling: str):
    if pca_scaling == "standardized":
        return StandardScaler()
    if pca_scaling == "unscaled":
        return MeanCenterer()
    raise ValueError(f"Unknown pca_scaling={pca_scaling!r}")


def _apply_filters(table: pd.DataFrame, max_abs_delta: float | None, max_attenuation_scatter: float | None) -> pd.DataFrame:
    mask = np.ones(len(table), dtype=bool)
    if max_abs_delta is not None:
        for column in ["delta_0", "delta_z", "delta_M", "delta_Mz"]:
            mask &= np.abs(table[column].to_numpy(dtype=float)) <= max_abs_delta
    if max_attenuation_scatter is not None:
        mask &= table["attenuation_scatter"].to_numpy(dtype=float) <= max_attenuation_scatter
    return table.loc[mask].reset_index(drop=True)


def _limit_draws_per_eval(table: pd.DataFrame, train_draws_per_eval: int | None) -> pd.DataFrame:
    if train_draws_per_eval is None:
        return table.reset_index(drop=True)
    return (
        table.sort_values(["evaluation_id", "dust_draw_index"])
        .groupby("evaluation_id", group_keys=False)
        .head(train_draws_per_eval)
        .reset_index(drop=True)
    )


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


def _log10_with_floor(values: np.ndarray, min_log10_lf: float) -> tuple[np.ndarray, float, int, int]:
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("No positive LF values found for log10 transform.")
    floor = 0.5 * float(np.min(positive))
    n_floored = int(np.count_nonzero(values <= 0.0))
    safe = np.where(values > 0.0, values, floor)
    log_values = np.log10(safe)
    n_clipped = int(np.count_nonzero(log_values < min_log10_lf))
    log_values = np.maximum(log_values, min_log10_lf)
    return log_values, floor, n_floored, n_clipped


def _training_alpha_log10(
    table: pd.DataFrame,
    output_columns: list[str],
    y_linear: np.ndarray,
    floor_linear: float,
    *,
    shot_noise_alpha: str,
    min_training_log10_sigma: float,
    max_training_log10_sigma: float,
) -> np.ndarray:
    sigma_columns = [f"{column}_shot_noise_std_{shot_noise_alpha}" for column in output_columns]
    missing = [column for column in sigma_columns if column not in table.columns]
    if missing:
        raise ValueError(
            "Missing Halpha shot-noise columns needed for --use-training-alpha. "
            "Regenerate or re-merge the Halpha dust LF tables with the updated post-processing scripts. "
            f"First missing column: {missing[0]}"
        )

    sigma_linear = table[sigma_columns].to_numpy(dtype=float)
    denominator = np.maximum(y_linear, floor_linear) * np.log(10.0)
    sigma_log10 = sigma_linear / denominator
    sigma_log10 = np.where(np.isfinite(sigma_log10), sigma_log10, max_training_log10_sigma)
    sigma_log10 = np.clip(sigma_log10, min_training_log10_sigma, max_training_log10_sigma)
    return sigma_log10**2


def _pca_component_alpha(
    alpha_log10: np.ndarray | None,
    scaler,
    pca: PCA,
    n_components_actual: int,
) -> np.ndarray | None:
    if alpha_log10 is None:
        return None
    alpha_scaled = alpha_log10 / (scaler.scale_[None, :] ** 2)
    return alpha_scaled @ (pca.components_[:n_components_actual] ** 2).T


def _slow_quantile_columns(table: pd.DataFrame) -> list[str]:
    columns = [column for column in table.columns if column.endswith("_quantile")]
    if not columns:
        raise ValueError("No slow-parameter quantile columns found in emulator table.")
    return columns


def _input_matrix(table: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    slow_quantile_columns = _slow_quantile_columns(table)
    x_slow = table[slow_quantile_columns].to_numpy(dtype=float)
    x_dust_raw = table[DUST_INPUT_COLUMNS].to_numpy(dtype=float)
    x_dust = transform_to_prior_quantiles(DUST_INPUT_PARAMETER_SPECS, x_dust_raw)
    x = np.hstack([x_slow, x_dust])
    input_labels = slow_quantile_columns + [f"{column}_quantile" for column in DUST_INPUT_COLUMNS]
    return x, input_labels


def _prediction_input_table(table: pd.DataFrame) -> pd.DataFrame:
    slow_quantile_columns = _slow_quantile_columns(table)
    prediction_inputs = table[["evaluation_id", "dust_draw_index", *slow_quantile_columns]].copy()
    dust_quantiles = transform_to_prior_quantiles(
        DUST_INPUT_PARAMETER_SPECS,
        table[DUST_INPUT_COLUMNS].to_numpy(dtype=float),
    )
    for index, column in enumerate(DUST_INPUT_COLUMNS):
        prediction_inputs[f"{column}_quantile"] = dust_quantiles[:, index]
    return prediction_inputs


def _fit_pca_gp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha_train: np.ndarray | None,
    sobral_label: str,
    fold_index: int,
    n_components: int,
    pca_scaling: str,
    n_restarts_optimizer: int,
    fit_white_noise: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scaler = _make_preprocessor(pca_scaling)
    y_train_scaled = scaler.fit_transform(y_train)
    n_components_actual = min(n_components, y_train_scaled.shape[0], y_train_scaled.shape[1])
    pca = PCA(n_components=n_components_actual)
    coefficients = pca.fit_transform(y_train_scaled)
    coefficient_alpha = _pca_component_alpha(alpha_train, scaler, pca, n_components_actual)

    coefficient_predictions = np.zeros((x_test.shape[0], n_components_actual))
    coefficient_stds = np.zeros_like(coefficient_predictions)
    kernel_summaries = []
    for component_index in range(n_components_actual):
        print(
            f"{sobral_label} fold {fold_index}: fitting PCA component "
            f"{component_index + 1}/{n_components_actual}",
            flush=True,
        )
        model, y_mean, y_std = fit_scaled_gp(
            x_train,
            coefficients[:, component_index],
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=True,
            alpha=coefficient_alpha[:, component_index] if coefficient_alpha is not None else None,
            fit_white_noise=fit_white_noise,
        )
        coefficient_predictions[:, component_index], coefficient_stds[:, component_index] = predict_scaled_gp(
            model,
            y_mean,
            y_std,
            x_test,
        )
        kernel_summaries.append(str(model.kernel_))

    y_pred_scaled = pca.inverse_transform(coefficient_predictions)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    component_variance = coefficient_stds**2
    y_var_scaled = component_variance @ (pca.components_**2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * scaler.scale_[None, :]
    metadata = {
        "n_components_actual": int(n_components_actual),
        "pca_scaling": pca_scaling,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "uses_training_alpha": coefficient_alpha is not None,
        "kernel_summaries": kernel_summaries,
    }
    return y_pred, y_std, metadata


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, luminosity_centers: np.ndarray) -> pd.DataFrame:
    rows = []
    for bin_index in range(y_true.shape[1]):
        rows.append(
            {
                "bin_index": bin_index,
                "log10_luminosity_center": luminosity_centers[bin_index],
                "rmse_dex": float(np.sqrt(mean_squared_error(y_true[:, bin_index], y_pred[:, bin_index]))),
                "mae_dex": float(mean_absolute_error(y_true[:, bin_index], y_pred[:, bin_index])),
                "r2": float(r2_score(y_true[:, bin_index], y_pred[:, bin_index])),
                "coverage_1sigma": float(np.mean(np.abs(y_pred[:, bin_index] - y_true[:, bin_index]) <= y_std[:, bin_index])),
                "coverage_2sigma": float(np.mean(np.abs(y_pred[:, bin_index] - y_true[:, bin_index]) <= 2.0 * y_std[:, bin_index])),
            }
        )
    rows.append(
        {
            "bin_index": "all",
            "log10_luminosity_center": np.nan,
            "rmse_dex": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
            "mae_dex": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
            "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
            "coverage_1sigma": float(np.mean(np.abs(y_pred - y_true) <= y_std)),
            "coverage_2sigma": float(np.mean(np.abs(y_pred - y_true) <= 2.0 * y_std)),
        }
    )
    return pd.DataFrame(rows)


def _plot_pca_modes(y: np.ndarray, luminosity_centers: np.ndarray, n_components: int, pca_scaling: str, path: Path) -> dict[str, object]:
    scaler = _make_preprocessor(pca_scaling)
    y_scaled = scaler.fit_transform(y)
    pca = PCA(n_components=min(n_components, y.shape[0], y.shape[1])).fit(y_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    axes[0].plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), "o-")
    axes[0].set_xlabel("PCA component")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.22)
    for component_index, component in enumerate(pca.components_[: min(5, pca.n_components_)], start=1):
        axes[1].plot(luminosity_centers, component, label=f"PC{component_index}")
    axes[1].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    axes[1].set_ylabel("scaled PCA loading" if pca_scaling == "standardized" else r"PCA loading in $\Delta\log_{10}\Phi$")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_cumulative": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "pca_scaling": pca_scaling,
    }


def _plot_parity(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, luminosity_centers: np.ndarray, path: Path) -> None:
    n_bins = y_true.shape[1]
    ncols = min(4, n_bins)
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
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
        axis.set_title(rf"$\log_{{10}}L={luminosity_centers[bin_index]:.2f}$")
        axis.set_xlabel("True log10 LF")
        axis.set_ylabel("Predicted log10 LF")
        axis.grid(alpha=0.2)
    for axis in axes_flat[n_bins:]:
        axis.axis("off")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix or f"halpha_{args.sobral_label.lower()}_draws{args.train_draws_per_eval}"

    table = pd.read_csv(campaign_root / args.table_filename)
    long_table = pd.read_csv(campaign_root / args.long_filename)
    table = _apply_filters(table, args.max_abs_delta, args.max_attenuation_scatter)
    table = _limit_draws_per_eval(table, args.train_draws_per_eval)

    output_columns = _output_columns(table, args.sobral_label)
    centers = np.array(
        [_combo_metadata(long_table, args.sobral_label, bin_index)[1] for bin_index in range(len(output_columns))],
        dtype=float,
    )

    y_linear = table[output_columns].to_numpy(dtype=float)
    y, floor, n_floored, n_clipped = _log10_with_floor(y_linear, args.min_log10_lf)
    alpha_log10 = (
        _training_alpha_log10(
            table,
            output_columns,
            y_linear,
            floor,
            shot_noise_alpha=args.shot_noise_alpha,
            min_training_log10_sigma=args.min_training_log10_sigma,
            max_training_log10_sigma=args.max_training_log10_sigma,
        )
        if args.use_training_alpha
        else None
    )
    x, input_labels = _input_matrix(table)
    groups = table["evaluation_id"].to_numpy()

    pred = np.zeros_like(y)
    pred_std = np.zeros_like(y)
    fold_metadata = []
    splitter = GroupKFold(n_splits=args.n_splits)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        heldout = np.unique(groups[test_index])
        print(
            f"{args.sobral_label}: fold {fold_index}/{args.n_splits}, hold out {len(heldout)} evals / {len(test_index)} rows",
            flush=True,
        )
        fold_pred, fold_std, metadata = _fit_pca_gp_predict(
            x[train_index],
            y[train_index],
            x[test_index],
            alpha_train=alpha_log10[train_index] if alpha_log10 is not None else None,
            sobral_label=args.sobral_label,
            fold_index=fold_index,
            n_components=args.pca_components,
            pca_scaling=args.pca_scaling,
            n_restarts_optimizer=args.n_restarts_optimizer,
            fit_white_noise=args.fit_white_noise,
        )
        pred[test_index] = fold_pred
        pred_std[test_index] = fold_std
        metadata["fold_index"] = fold_index
        metadata["n_train"] = int(len(train_index))
        metadata["n_test"] = int(len(test_index))
        metadata["heldout_evaluations"] = heldout.tolist()
        fold_metadata.append(metadata)
        print(f"{args.sobral_label}: finished fold {fold_index}/{args.n_splits}", flush=True)

    metrics = _metrics(y, pred, pred_std, centers)
    metrics_path = emulator_root / f"gp_cv_metrics_{output_prefix}.csv"
    metrics.to_csv(metrics_path, index=False)

    predictions = _prediction_input_table(table)
    for bin_index, center in enumerate(centers):
        predictions[f"bin{bin_index:02d}_log10L{center:.2f}_true"] = y[:, bin_index]
        predictions[f"bin{bin_index:02d}_log10L{center:.2f}_pred"] = pred[:, bin_index]
        predictions[f"bin{bin_index:02d}_log10L{center:.2f}_std"] = pred_std[:, bin_index]
    predictions_path = emulator_root / f"gp_cv_predictions_{output_prefix}.csv"
    predictions.to_csv(predictions_path, index=False)

    pca_summary = _plot_pca_modes(y, centers, args.pca_components, args.pca_scaling, figures_root / f"pca_modes_{output_prefix}.png")
    _plot_parity(y, pred, pred_std, centers, figures_root / f"gp_cv_parity_{output_prefix}.png")

    summary = {
        "campaign_root": str(campaign_root),
        "sobral_label": args.sobral_label,
        "n_splits": args.n_splits,
        "train_draws_per_eval": args.train_draws_per_eval,
        "n_restarts_optimizer": args.n_restarts_optimizer,
        "pca_components_requested": args.pca_components,
        "pca_scaling": args.pca_scaling,
        "use_training_alpha": args.use_training_alpha,
        "shot_noise_alpha": args.shot_noise_alpha if args.use_training_alpha else None,
        "min_training_log10_sigma": args.min_training_log10_sigma if args.use_training_alpha else None,
        "max_training_log10_sigma": args.max_training_log10_sigma if args.use_training_alpha else None,
        "fit_white_noise": args.fit_white_noise,
        "min_log10_lf": args.min_log10_lf,
        "max_abs_delta": args.max_abs_delta,
        "max_attenuation_scatter": args.max_attenuation_scatter,
        "log10_floor_linear": floor,
        "n_floored_values": n_floored,
        "n_clipped_values": n_clipped,
        "input_transform": "slow prior quantiles + dust prior quantiles",
        "input_columns": input_labels,
        "n_input_dimensions": int(x.shape[1]),
        "n_unique_evaluations": int(pd.Series(groups).nunique()),
        "n_training_rows_total": int(len(table)),
        "pca_summary_full_data": pca_summary,
        "fold_metadata": fold_metadata,
        "luminosity_centers": centers.tolist(),
    }
    summary_path = emulator_root / f"gp_cv_metadata_{output_prefix}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(figures_root / f"pca_modes_{output_prefix}.png")
    print(figures_root / f"gp_cv_parity_{output_prefix}.png")


if __name__ == "__main__":
    main()
