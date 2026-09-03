from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from fit_1d_smf_gp_cv import (
    DEFAULT_ANALYSIS,
    _fit_scaled_gp,
    _load_campaign,
    _load_campaign_log10_noise,
    _plot_curves,
    _plot_parameter_sweep_overlay,
    _plot_parity,
    _plot_residual_heatmap,
    _plot_subset_metric,
    _plot_subset_residual_heatmaps,
    _predict_scaled_gp,
    _space_filling_subset_indices,
    _splitter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a PCA-decomposed 1D GP emulator to a Galacticus stellar mass "
            "function analysis and make cross-validation diagnostics."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--input-column", default="diskVelocityCharacteristic")
    parser.add_argument("--quantile-column", default="prior_quantile")
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--cv-folds", type=int, default=32, help="Use >= N for leave-one-out CV.")
    parser.add_argument("--subset-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument(
        "--pca-scaling",
        choices=["standardized", "unscaled"],
        default="standardized",
        help=(
            "'standardized' standardizes each SMF bin before PCA; 'unscaled' "
            "mean-centers log10(Phi) but leaves bin amplitudes in dex."
        ),
    )
    parser.add_argument("--n-restarts-optimizer", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-prefix", default="smf_zfourge_z0")
    parser.add_argument("--figures-dir-name", default="figures_pca_smf_zfourge_z0")
    parser.add_argument("--emulator-dir-name", default="emulator_pca_smf_zfourge_z0")
    return parser.parse_args()


class MeanCenterer:
    def fit(self, y: np.ndarray) -> "MeanCenterer":
        self.mean_ = np.mean(y, axis=0)
        self.scale_ = np.ones(y.shape[1])
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


def _fit_pca_gp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
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

    # First-order propagation through the linear PCA reconstruction. This ignores
    # coefficient covariances, but gives a useful relative uncertainty diagnostic.
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


def _pca_cv_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int,
    n_components: int,
    pca_scaling: str,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    pred = np.zeros_like(y)
    pred_std = np.zeros_like(y)
    fold_metadata = []
    splitter = _splitter(x.shape[0], cv_folds, random_state)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x)):
        fold_pred, fold_std, metadata = _fit_pca_gp_predict(
            x[train_index],
            y[train_index],
            x[test_index],
            n_components=n_components,
            pca_scaling=pca_scaling,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        pred[test_index] = fold_pred
        pred_std[test_index] = fold_std
        metadata["fold_index"] = fold_index
        metadata["n_train"] = int(len(train_index))
        metadata["n_test"] = int(len(test_index))
        fold_metadata.append(metadata)
    return pred, pred_std, fold_metadata


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, mass_bins: np.ndarray) -> pd.DataFrame:
    rows = []
    for bin_index in range(y_true.shape[1]):
        rows.append(
            {
                "bin": bin_index,
                "mass_stellar": mass_bins[bin_index],
                "mass_stellar_log10": np.log10(mass_bins[bin_index]),
                "rmse_dex": float(np.sqrt(mean_squared_error(y_true[:, bin_index], y_pred[:, bin_index]))),
                "mae_dex": float(mean_absolute_error(y_true[:, bin_index], y_pred[:, bin_index])),
                "r2": float(r2_score(y_true[:, bin_index], y_pred[:, bin_index])),
                "coverage_1sigma": float(np.mean(np.abs(y_pred[:, bin_index] - y_true[:, bin_index]) <= y_std[:, bin_index])),
                "coverage_2sigma": float(np.mean(np.abs(y_pred[:, bin_index] - y_true[:, bin_index]) <= 2.0 * y_std[:, bin_index])),
            }
        )
    rows.append(
        {
            "bin": "all",
            "mass_stellar": np.nan,
            "mass_stellar_log10": np.nan,
            "rmse_dex": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
            "mae_dex": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
            "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
            "coverage_1sigma": float(np.mean(np.abs(y_pred - y_true) <= y_std)),
            "coverage_2sigma": float(np.mean(np.abs(y_pred - y_true) <= 2.0 * y_std)),
        }
    )
    return pd.DataFrame(rows)


def _plot_pca_modes(y: np.ndarray, mass_bins: np.ndarray, n_components: int, pca_scaling: str, path: Path) -> dict[str, object]:
    scaler = _make_preprocessor(pca_scaling)
    y_scaled = scaler.fit_transform(y)
    pca = PCA(n_components=min(n_components, y.shape[0], y.shape[1])).fit(y_scaled)
    x_axis = np.log10(mass_bins)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    axes[0].plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), "o-")
    axes[0].set_xlabel("PCA component")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.22)
    for component_index, component in enumerate(pca.components_[: min(5, pca.n_components_)], start=1):
        if pca_scaling == "standardized":
            y_component = component
        else:
            y_component = component
        axes[1].plot(x_axis, y_component, label=f"PC{component_index}")
    axes[1].set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axes[1].set_ylabel("scaled PCA loading" if pca_scaling == "standardized" else r"PCA loading in $\Delta\log_{10}\Phi$")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle(f"PCA decomposition of training SMF variation ({pca_scaling})")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_cumulative": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "pca_scaling": pca_scaling,
    }


def _write_predictions_table(samples: pd.DataFrame, mass_bins: np.ndarray, y: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, path: Path) -> None:
    predictions = samples.copy()
    for bin_index, mass in enumerate(mass_bins):
        label = f"bin{bin_index:02d}_logM{np.log10(mass):.2f}"
        predictions[f"{label}_log10_phi"] = y[:, bin_index]
        predictions[f"{label}_pca_gp_cv_log10_phi"] = y_pred[:, bin_index]
        predictions[f"{label}_pca_gp_cv_std"] = y_std[:, bin_index]
        predictions[f"{label}_pca_gp_cv_residual"] = y_pred[:, bin_index] - y[:, bin_index]
    predictions.to_csv(path, index=False)


def _run_subset_experiment(
    samples: pd.DataFrame,
    mass_bins: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    subset_sizes: list[int],
    n_components: int,
    pca_scaling: str,
    n_restarts_optimizer: int,
    random_state: int,
    output_prefix: str,
    emulator_root: Path,
    figures_root: Path,
    input_column: str,
) -> None:
    metrics_frames = []
    prediction_rows = []
    residuals_by_size = {}
    all_indices = np.arange(x.shape[0])
    for subset_size in subset_sizes:
        train_index = _space_filling_subset_indices(x.shape[0], subset_size)
        test_index = np.setdiff1d(all_indices, train_index)
        pred, pred_std, metadata = _fit_pca_gp_predict(
            x[train_index],
            y[train_index],
            x[test_index],
            n_components=n_components,
            pca_scaling=pca_scaling,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        metrics = _metrics(y[test_index], pred, pred_std, mass_bins)
        metrics.insert(0, "train_size", subset_size)
        metrics.insert(1, "n_test", len(test_index))
        metrics.insert(2, "pca_components_actual", metadata["n_components_actual"])
        metrics.insert(3, "pca_scaling", pca_scaling)
        metrics.insert(4, "pca_explained_variance", metadata["explained_variance_ratio_sum"])
        metrics_frames.append(metrics)
        residuals_by_size[subset_size] = (test_index, pred - y[test_index])
        for local_row, evaluation_index in enumerate(test_index):
            for bin_index, mass in enumerate(mass_bins):
                prediction_rows.append(
                    {
                        "train_size": subset_size,
                        "evaluation_id": samples.iloc[evaluation_index]["evaluation_id"],
                        "evaluation_index": int(evaluation_index),
                        "prior_quantile": float(samples.iloc[evaluation_index]["prior_quantile"]),
                        input_column: float(samples.iloc[evaluation_index][input_column]),
                        "bin": bin_index,
                        "mass_stellar": float(mass),
                        "mass_stellar_log10": float(np.log10(mass)),
                        "log10_phi": float(y[evaluation_index, bin_index]),
                        "pca_gp_log10_phi": float(pred[local_row, bin_index]),
                        "pca_gp_std": float(pred_std[local_row, bin_index]),
                        "residual": float(pred[local_row, bin_index] - y[evaluation_index, bin_index]),
                    }
                )

    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    metrics_path = emulator_root / f"gp_subset_metrics_{output_prefix}.csv"
    metrics_all.to_csv(metrics_path, index=False)
    predictions_path = emulator_root / f"gp_subset_predictions_{output_prefix}.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    _plot_subset_metric(metrics_all, "rmse_dex", "RMSE [dex]", figures_root / f"gp_subset_metric_rmse_{output_prefix}.png")
    _plot_subset_metric(metrics_all, "r2", r"$R^2$", figures_root / f"gp_subset_metric_r2_{output_prefix}.png")
    _plot_subset_residual_heatmaps(
        samples,
        mass_bins,
        residuals_by_size,
        input_column,
        figures_root / f"gp_subset_residual_heatmaps_{output_prefix}.png",
    )
    print(metrics_path)
    print(predictions_path)
    print(figures_root / f"gp_subset_metric_rmse_{output_prefix}.png")
    print(figures_root / f"gp_subset_metric_r2_{output_prefix}.png")
    print(figures_root / f"gp_subset_residual_heatmaps_{output_prefix}.png")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    samples, mass_bins, target, y = _load_campaign(campaign_root, args.analysis, args.hdf5_filename)
    y_data_std = _load_campaign_log10_noise(campaign_root, samples, args.analysis, args.hdf5_filename)
    x = samples[[args.quantile_column]].to_numpy(dtype=float)
    y_pred, y_std, fold_metadata = _pca_cv_predict(
        x,
        y,
        cv_folds=args.cv_folds,
        n_components=args.pca_components,
        pca_scaling=args.pca_scaling,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.random_state,
    )

    metrics = _metrics(y, y_pred, y_std, mass_bins)
    metrics_path = emulator_root / f"gp_cv_metrics_{args.output_prefix}.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions_path = emulator_root / f"gp_cv_predictions_{args.output_prefix}.csv"
    _write_predictions_table(samples, mass_bins, y, y_pred, y_std, predictions_path)
    pca_summary = _plot_pca_modes(y, mass_bins, args.pca_components, args.pca_scaling, figures_root / f"pca_modes_{args.output_prefix}.png")
    metadata_path = emulator_root / f"gp_cv_metadata_{args.output_prefix}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "analysis": args.analysis,
                "hdf5_filename": args.hdf5_filename,
                "input_column": args.input_column,
                "quantile_column": args.quantile_column,
                "target": "log10(massFunction)",
                "pca_components_requested": args.pca_components,
                "pca_scaling": args.pca_scaling,
                "pca_summary_full_data": pca_summary,
                "fold_metadata": fold_metadata,
                "mass_stellar": mass_bins.tolist(),
                "mass_function_target": target.tolist(),
                "cv_folds": args.cv_folds,
                "n_restarts_optimizer": args.n_restarts_optimizer,
            },
            indent=2,
        )
        + "\n"
    )

    _plot_parity(y, y_pred, y_std, mass_bins, figures_root / f"gp_cv_parity_{args.output_prefix}_bins.png")
    _plot_curves(
        samples,
        mass_bins,
        np.log10(target),
        y,
        y_pred,
        y_std,
        args.input_column,
        figures_root / f"gp_cv_curves_{args.output_prefix}.png",
    )
    _plot_parameter_sweep_overlay(
        samples,
        mass_bins,
        np.log10(target),
        y,
        y_pred,
        y_std,
        y_data_std,
        args.input_column,
        figures_root / f"gp_cv_parameter_sweep_{args.output_prefix}.png",
    )
    _plot_residual_heatmap(
        samples,
        mass_bins,
        y_pred - y,
        args.input_column,
        figures_root / f"gp_cv_residual_heatmap_{args.output_prefix}.png",
    )
    if args.subset_sizes:
        _run_subset_experiment(
            samples,
            mass_bins,
            x,
            y,
            subset_sizes=args.subset_sizes,
            n_components=args.pca_components,
            pca_scaling=args.pca_scaling,
            n_restarts_optimizer=args.n_restarts_optimizer,
            random_state=args.random_state,
            output_prefix=args.output_prefix,
            emulator_root=emulator_root,
            figures_root=figures_root,
            input_column=args.input_column,
        )

    print(metrics_path)
    print(predictions_path)
    print(metadata_path)
    print(figures_root / f"pca_modes_{args.output_prefix}.png")
    print(figures_root / f"gp_cv_parity_{args.output_prefix}_bins.png")
    print(figures_root / f"gp_cv_curves_{args.output_prefix}.png")
    print(figures_root / f"gp_cv_parameter_sweep_{args.output_prefix}.png")
    print(figures_root / f"gp_cv_residual_heatmap_{args.output_prefix}.png")


if __name__ == "__main__":
    main()
