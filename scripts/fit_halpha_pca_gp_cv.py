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

from fit_halpha_allz_gp_cv import (
    INPUT_COLUMNS,
    INPUT_PARAMETER_SPECS,
    _apply_filters,
    _combo_metadata,
    _fit_scaled_gp,
    _limit_draws_per_eval,
    _log10_with_floor,
    _output_columns,
    _predict_scaled_gp,
)
from galacticus_emu.lhs import transform_to_prior_quantiles


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a PCA-decomposed grouped-CV GP emulator for one Sobral Halpha LF across all "
            "luminosity bins at a fixed redshift."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--sobral-label", default="Z4", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--train-draws-per-eval", type=int, default=8)
    parser.add_argument("--n-restarts-optimizer", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="unscaled")
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--figures-dir-name", default="figures_halpha_pca")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_pca")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def _make_preprocessor(pca_scaling: str):
    if pca_scaling == "standardized":
        return StandardScaler()
    if pca_scaling == "unscaled":
        return MeanCenterer()
    raise ValueError(f"Unknown pca_scaling={pca_scaling!r}")


def _fit_pca_gp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    sobral_label: str,
    fold_index: int,
    n_components: int,
    pca_scaling: str,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scaler = _make_preprocessor(pca_scaling)
    y_train_scaled = scaler.fit_transform(y_train)
    n_components_actual = min(n_components, y_train_scaled.shape[0], y_train_scaled.shape[1])
    pca = PCA(n_components=n_components_actual)
    coefficients = pca.fit_transform(y_train_scaled)

    coefficient_predictions = np.zeros((x_test.shape[0], n_components_actual))
    coefficient_stds = np.zeros_like(coefficient_predictions)
    kernel_summaries = []
    for component_index in range(n_components_actual):
        print(
            f"{sobral_label} fold {fold_index}: fitting PCA component "
            f"{component_index + 1}/{n_components_actual}",
            flush=True,
        )
        model, y_mean, y_std = _fit_scaled_gp(
            x_train,
            coefficients[:, component_index],
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        coefficient_predictions[:, component_index], coefficient_stds[:, component_index] = _predict_scaled_gp(
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
        "n_components_actual": n_components_actual,
        "pca_scaling": pca_scaling,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
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

    output_prefix = args.output_prefix or f"halpha_{args.sobral_label.lower()}"

    table = pd.read_csv(campaign_root / args.table_filename)
    long_table = pd.read_csv(campaign_root / args.long_filename)
    table = _apply_filters(table, args.max_abs_delta, args.max_attenuation_scatter)
    table = _limit_draws_per_eval(table, args.train_draws_per_eval)

    output_columns = _output_columns(table, args.sobral_label)
    centers = np.array([
        _combo_metadata(long_table, args.sobral_label, bin_index)[1]
        for bin_index in range(len(output_columns))
    ])

    y_linear = table[output_columns].to_numpy(dtype=float)
    y, floor, n_floored, n_clipped = _log10_with_floor(y_linear, args.min_log10_lf)
    x = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, table[INPUT_COLUMNS].to_numpy(dtype=float))
    groups = table["evaluation_id"].to_numpy()

    pred = np.zeros_like(y)
    pred_std = np.zeros_like(y)
    fold_metadata = []
    for fold_index, (train_index, test_index) in enumerate(GroupKFold(n_splits=args.n_splits).split(x, y, groups), start=1):
        heldout = np.unique(groups[test_index])
        print(
            f"{args.sobral_label}: fold {fold_index}/{args.n_splits}, hold out {len(heldout)} evals / {len(test_index)} rows",
            flush=True,
        )
        fold_pred, fold_std, metadata = _fit_pca_gp_predict(
            x[train_index],
            y[train_index],
            x[test_index],
            sobral_label=args.sobral_label,
            fold_index=fold_index,
            n_components=args.pca_components,
            pca_scaling=args.pca_scaling,
            n_restarts_optimizer=args.n_restarts_optimizer,
            random_state=args.random_state,
        )
        pred[test_index] = fold_pred
        pred_std[test_index] = fold_std
        metadata["fold_index"] = fold_index
        metadata["n_train"] = int(len(train_index))
        metadata["n_test"] = int(len(test_index))
        metadata["heldout_evaluations"] = heldout.tolist()
        fold_metadata.append(metadata)
        print(
            f"{args.sobral_label}: finished fold {fold_index}/{args.n_splits}",
            flush=True,
        )

    metrics = _metrics(y, pred, pred_std, centers)
    metrics_path = emulator_root / f"gp_cv_metrics_{output_prefix}.csv"
    metrics.to_csv(metrics_path, index=False)

    predictions = table[["evaluation_id", "dust_draw_index", *INPUT_COLUMNS]].copy()
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
        "min_log10_lf": args.min_log10_lf,
        "max_abs_delta": args.max_abs_delta,
        "max_attenuation_scatter": args.max_attenuation_scatter,
        "log10_floor_linear": floor,
        "n_floored_values": n_floored,
        "n_clipped_values": n_clipped,
        "input_transform": "prior_quantiles",
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
