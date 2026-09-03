from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut

from fit_1d_smf_gp_cv import (
    _fit_scaled_gp,
    _predict_scaled_gp,
    _space_filling_subset_indices,
)


PRESETS = {
    "mbh_sigma": {
        "analysis": "blackHoleVelocityDispersionRelation",
        "x-dataset": "velocityDispersion",
        "y-dataset": "massBlackHole",
        "target-dataset": "massBlackHoleTarget",
        "covariance-dataset": "massBlackHoleCovariance",
        "target-label": r"$\log_{10}(M_\mathrm{BH}/M_\odot)$",
        "x-label": r"$\sigma\ [\mathrm{km\,s^{-1}}]$",
        "x-transform": "identity",
        "output-prefix": "mbh_sigma",
        "figures-dir-name": "figures_mbh_sigma",
        "emulator-dir-name": "emulator_mbh_sigma",
        "target-name": "McConnell & Ma target",
    },
    "mzr_blanc2019": {
        "analysis": "massMetallicityBlanc2019",
        "x-dataset": "massStellar",
        "y-dataset": "metallicityMean",
        "target-dataset": "metallicityMeanTarget",
        "covariance-dataset": "metallicityMeanCovariance",
        "target-label": r"$12+\log_{10}(\mathrm{O/H})$",
        "x-label": r"$\log_{10}(M_\star/M_\odot)$",
        "x-transform": "log10",
        "output-prefix": "mzr_blanc2019",
        "figures-dir-name": "figures_mzr_blanc2019",
        "emulator-dir-name": "emulator_mzr_blanc2019",
        "target-name": "Blanc+19 target",
    },
    "hi_mass_function_alfalfa": {
        "analysis": "massFunctionHIMartin2010ALFALFA",
        "output-prefix": "hi_mass_function_alfalfa",
        "figures-dir-name": "figures_hi_mass_function_alfalfa",
        "emulator-dir-name": "emulator_hi_mass_function_alfalfa",
    },
    "smf_zfourge_z4": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz4",
        "output-prefix": "smf_zfourge_z4",
        "figures-dir-name": "figures_smf_zfourge_z4",
        "emulator-dir-name": "emulator_smf_zfourge_z4",
    },
    "sfr_function_robotham2011": {
        "analysis": "starFormationRateFunctionRobotham2011",
        "output-prefix": "sfr_function_robotham2011",
        "figures-dir-name": "figures_sfr_function_robotham2011",
        "emulator-dir-name": "emulator_sfr_function_robotham2011",
    },
    "size_mass_vdw2014_star_forming_z0": {
        "analysis": "stellarSizeMassRelationvanDerWel2014Sample1",
        "output-prefix": "size_mass_vdw2014_star_forming_z0",
        "figures-dir-name": "figures_size_mass_vdw2014_star_forming_z0",
        "emulator-dir-name": "emulator_size_mass_vdw2014_star_forming_z0",
    },
    "size_mass_vdw2014_quiescent_z0": {
        "analysis": "stellarSizeMassRelationvanDerWel2014Sample7",
        "output-prefix": "size_mass_vdw2014_quiescent_z0",
        "figures-dir-name": "figures_size_mass_vdw2014_quiescent_z0",
        "emulator-dir-name": "emulator_size_mass_vdw2014_quiescent_z0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit bin-by-bin 1D GPs to a Galacticus analysis relation and make CV diagnostics."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--analysis", default=None)
    parser.add_argument("--x-dataset", default=None)
    parser.add_argument("--y-dataset", default=None)
    parser.add_argument("--target-dataset", default=None)
    parser.add_argument("--covariance-dataset", default=None)
    parser.add_argument("--x-transform", choices=["identity", "log10"], default=None)
    parser.add_argument("--y-transform", choices=["identity", "log10"], default=None)
    parser.add_argument("--x-label", default=None)
    parser.add_argument("--target-label", default=None)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--input-column", default="diskVelocityCharacteristic")
    parser.add_argument("--quantile-column", default="prior_quantile")
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--cv-folds", type=int, default=32, help="Use >= N for leave-one-out CV.")
    parser.add_argument("--subset-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--n-restarts-optimizer", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--figures-dir-name", default=None)
    parser.add_argument("--emulator-dir-name", default=None)
    return parser.parse_args()


def _setting(args: argparse.Namespace, key: str, default=None):
    value = getattr(args, key.replace("-", "_"))
    if value is not None:
        return value
    if args.preset is not None and key in PRESETS[args.preset]:
        return PRESETS[args.preset][key]
    return default


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_analysis_attributes(campaign_root: Path, analysis: str, hdf5_filename: str) -> dict:
    samples = pd.read_csv(campaign_root / "samples.csv")
    first_evaluation = samples.iloc[0]["evaluation_id"]
    path = campaign_root / "evaluations" / first_evaluation / hdf5_filename
    with h5py.File(path, "r") as handle:
        group = handle[f"/analyses/{analysis}"]
        return {key: _decode_attr(value) for key, value in group.attrs.items()}


def _splitter(n_samples: int, cv_folds: int, random_state: int):
    if cv_folds >= n_samples:
        return LeaveOneOut()
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def _axis_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log10":
        if np.any(values <= 0.0):
            raise ValueError("Cannot take log10 of non-positive x values")
        return np.log10(values)
    return values


def _transform_y(
    values: np.ndarray,
    noise: np.ndarray | None,
    target: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float | None, int]:
    if transform == "identity":
        return values, noise, target, None, 0
    if transform != "log10":
        raise ValueError(f"Unknown y transform: {transform}")
    positive = np.concatenate([values[values > 0.0], target[target > 0.0]])
    if positive.size == 0:
        raise ValueError("Cannot take log10 because no positive y or target values were found")
    floor = 0.5 * float(np.nanmin(positive))
    n_floored = int(np.sum(values <= 0.0) + np.sum(target <= 0.0))
    safe_values = np.where(values > 0.0, values, floor)
    safe_target = np.where(target > 0.0, target, floor)
    transformed_values = np.log10(safe_values)
    transformed_target = np.log10(safe_target)
    if noise is None:
        transformed_noise = None
    else:
        transformed_noise = noise / (safe_values * np.log(10.0))
    return transformed_values, transformed_noise, transformed_target, floor, n_floored


def _load_campaign(
    campaign_root: Path,
    *,
    analysis: str,
    x_dataset: str,
    y_dataset: str,
    target_dataset: str,
    covariance_dataset: str | None,
    hdf5_filename: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    samples = pd.read_csv(campaign_root / "samples.csv")
    x_bins: np.ndarray | None = None
    target: np.ndarray | None = None
    y_rows = []
    noise_rows = []
    for sample in samples.itertuples(index=False):
        evaluation_id = sample.evaluation_id
        path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            current_x = np.asarray(group[x_dataset][...], dtype=float)
            current_target = np.asarray(group[target_dataset][...], dtype=float)
            current_y = np.asarray(group[y_dataset][...], dtype=float)
            if covariance_dataset and covariance_dataset in group:
                covariance = np.asarray(group[covariance_dataset][...], dtype=float)
                current_noise = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            else:
                current_noise = np.full_like(current_y, np.nan, dtype=float)
        if x_bins is None:
            x_bins = current_x
            target = current_target
        else:
            if not np.allclose(current_x, x_bins, equal_nan=True):
                raise ValueError(f"x bins differ for {evaluation_id}")
            if not np.allclose(current_target, target, equal_nan=True):
                raise ValueError(f"target differs for {evaluation_id}")
        y_rows.append(current_y)
        noise_rows.append(current_noise)

    assert x_bins is not None
    assert target is not None
    y = np.vstack(y_rows)
    noise = np.vstack(noise_rows)
    if np.all(~np.isfinite(noise)):
        noise = None
    return samples, x_bins, target, y, noise


def _cv_predict_all_bins(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros_like(y)
    pred_std = np.zeros_like(y)
    splitter = _splitter(x.shape[0], cv_folds, random_state)
    for train_index, test_index in splitter.split(x):
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


def _subset_predict_all_bins(
    x: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros((len(test_index), y.shape[1]))
    pred_std = np.zeros_like(pred)
    for bin_index in range(y.shape[1]):
        model, y_mean, y_std = _fit_scaled_gp(
            x[train_index],
            y[train_index, bin_index],
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        pred[:, bin_index], pred_std[:, bin_index] = _predict_scaled_gp(
            model,
            y_mean,
            y_std,
            x[test_index],
        )
    return pred, pred_std


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, x_bins: np.ndarray) -> pd.DataFrame:
    rows = []
    for bin_index in range(y_true.shape[1]):
        rows.append(
            {
                "bin": bin_index,
                "x_bin": x_bins[bin_index],
                "rmse": float(np.sqrt(mean_squared_error(y_true[:, bin_index], y_pred[:, bin_index]))),
                "mae": float(mean_absolute_error(y_true[:, bin_index], y_pred[:, bin_index])),
                "r2": float(r2_score(y_true[:, bin_index], y_pred[:, bin_index])),
                "coverage_1sigma": float(np.mean(np.abs(y_pred[:, bin_index] - y_true[:, bin_index]) <= y_std[:, bin_index])),
                "coverage_2sigma": float(np.mean(np.abs(y_pred[:, bin_index] - y_true[:, bin_index]) <= 2.0 * y_std[:, bin_index])),
            }
        )
    rows.append(
        {
            "bin": "all",
            "x_bin": np.nan,
            "rmse": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
            "mae": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
            "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
            "coverage_1sigma": float(np.mean(np.abs(y_pred - y_true) <= y_std)),
            "coverage_2sigma": float(np.mean(np.abs(y_pred - y_true) <= 2.0 * y_std)),
        }
    )
    return pd.DataFrame(rows)


def _plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    x_plot: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    n_bins = y_true.shape[1]
    ncols = min(5, n_bins)
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, bin_index in zip(axes_flat, range(n_bins), strict=False):
        observed = y_true[:, bin_index]
        predicted = y_pred[:, bin_index]
        lower = float(min(observed.min(), predicted.min()))
        upper = float(max(observed.max(), predicted.max()))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(observed, predicted, yerr=y_std[:, bin_index], fmt="o", ms=3.5, alpha=0.75)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(f"{x_label}={x_plot[bin_index]:.2f}", fontsize=9)
        axis.set_xlabel("Galacticus")
        axis.set_ylabel("GP CV")
        axis.grid(alpha=0.2)
    for axis in axes_flat[n_bins:]:
        axis.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_parameter_sweep_overlay(
    samples: pd.DataFrame,
    x_plot: np.ndarray,
    target: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_data_std: np.ndarray | None,
    *,
    input_column: str,
    x_label: str,
    y_label: str,
    title: str,
    target_name: str,
    path: Path,
    n_curves: int = 5,
) -> None:
    indices = np.asarray(sorted({int(round(value)) for value in np.linspace(0, len(samples) - 1, min(n_curves, len(samples)))}), dtype=int)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(indices)))

    fig, axis = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    axis.plot(x_plot, target, color="0.35", lw=1.3, ls="--", label=target_name)
    for color, row_index in zip(colors, indices, strict=True):
        value = float(samples.iloc[row_index][input_column])
        label = f"{value:.1f} km/s"
        axis.plot(x_plot, y_pred[row_index], color=color, lw=2.0, label=f"GP CV, {label}")
        axis.fill_between(
            x_plot,
            y_pred[row_index] - y_std[row_index],
            y_pred[row_index] + y_std[row_index],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        if y_data_std is None:
            axis.scatter(x_plot, y_true[row_index], color=color, s=34, edgecolor="white", linewidth=0.45, zorder=3)
        else:
            axis.errorbar(
                x_plot,
                y_true[row_index],
                yerr=y_data_std[row_index],
                fmt="o",
                ms=4.4,
                color=color,
                ecolor=color,
                elinewidth=0.85,
                capsize=1.7,
                alpha=0.9,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
            )
    y_for_limits = np.concatenate(
        [
            target[np.isfinite(target)],
            y_true[indices][np.isfinite(y_true[indices])],
            (y_pred[indices] - y_std[indices])[np.isfinite(y_pred[indices] - y_std[indices])],
            (y_pred[indices] + y_std[indices])[np.isfinite(y_pred[indices] + y_std[indices])],
        ]
    )
    if y_for_limits.size:
        lower = float(np.nanmin(y_for_limits))
        upper = float(np.nanmax(y_for_limits))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.set_ylim(lower - margin, upper + margin)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.22)
    axis.legend(title="disk velocity", frameon=False, fontsize=8, title_fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_residual_heatmap(
    samples: pd.DataFrame,
    x_plot: np.ndarray,
    residual: np.ndarray,
    *,
    input_column: str,
    x_label: str,
    title: str,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    y_axis = samples[input_column].to_numpy(dtype=float)
    vmax = np.nanpercentile(np.abs(residual), 95)
    if vmax == 0.0:
        vmax = 1.0
    mesh = axis.pcolormesh(x_plot, y_axis, residual, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axis.set_xlabel(x_label)
    axis.set_ylabel("diskVelocityCharacteristic")
    axis.set_title(title)
    fig.colorbar(mesh, ax=axis, label="residual")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subset_metric(metrics: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    per_bin = metrics[metrics["bin"] != "all"].copy()
    overall = metrics[metrics["bin"] == "all"].copy()
    grouped = per_bin.groupby("train_size")[metric]
    sizes = np.asarray(sorted(per_bin["train_size"].unique()), dtype=int)
    medians = grouped.median().reindex(sizes).to_numpy(dtype=float)
    lows = grouped.min().reindex(sizes).to_numpy(dtype=float)
    highs = grouped.max().reindex(sizes).to_numpy(dtype=float)
    overall_values = overall.set_index("train_size").reindex(sizes)[metric].to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    axis.fill_between(sizes, lows, highs, color="#b8d8e8", alpha=0.55, label="per-bin min-max")
    axis.plot(sizes, medians, "o-", color="#2a6f97", lw=2.0, label="per-bin median")
    axis.plot(sizes, overall_values, "s--", color="#d1495b", lw=1.7, label="all bins combined")
    axis.set_xlabel("training evaluations")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_subset_experiment(
    samples: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    x_bins: np.ndarray,
    *,
    subset_sizes: list[int],
    n_restarts_optimizer: int,
    random_state: int,
    output_prefix: str,
    emulator_root: Path,
    figures_root: Path,
    title_prefix: str,
) -> None:
    metrics_frames = []
    prediction_rows = []
    all_indices = np.arange(x.shape[0])
    for subset_size in subset_sizes:
        train_index = _space_filling_subset_indices(x.shape[0], subset_size)
        test_index = np.setdiff1d(all_indices, train_index)
        pred, pred_std = _subset_predict_all_bins(
            x,
            y,
            train_index,
            test_index,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        metrics = _metrics(y[test_index], pred, pred_std, x_bins)
        metrics.insert(0, "train_size", subset_size)
        metrics.insert(1, "n_test", len(test_index))
        metrics_frames.append(metrics)
        for local_row, evaluation_index in enumerate(test_index):
            for bin_index, x_bin in enumerate(x_bins):
                prediction_rows.append(
                    {
                        "train_size": subset_size,
                        "evaluation_id": samples.iloc[evaluation_index]["evaluation_id"],
                        "evaluation_index": int(evaluation_index),
                        "prior_quantile": float(samples.iloc[evaluation_index]["prior_quantile"]),
                        "bin": bin_index,
                        "x_bin": float(x_bin),
                        "y": float(y[evaluation_index, bin_index]),
                        "gp_y": float(pred[local_row, bin_index]),
                        "gp_std": float(pred_std[local_row, bin_index]),
                        "residual": float(pred[local_row, bin_index] - y[evaluation_index, bin_index]),
                    }
                )

    metrics_all = pd.concat(metrics_frames, ignore_index=True)
    metrics_path = emulator_root / f"gp_subset_metrics_{output_prefix}.csv"
    predictions_path = emulator_root / f"gp_subset_predictions_{output_prefix}.csv"
    metrics_all.to_csv(metrics_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    _plot_subset_metric(metrics_all, "rmse", "RMSE", f"{title_prefix} held-out GP performance: RMSE", figures_root / f"gp_subset_metric_rmse_{output_prefix}.png")
    _plot_subset_metric(metrics_all, "r2", r"$R^2$", f"{title_prefix} held-out GP performance: R2", figures_root / f"gp_subset_metric_r2_{output_prefix}.png")
    print(metrics_path)
    print(predictions_path)
    print(figures_root / f"gp_subset_metric_rmse_{output_prefix}.png")
    print(figures_root / f"gp_subset_metric_r2_{output_prefix}.png")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    campaign_root = args.campaign_root.resolve()
    analysis = _setting(args, "analysis")
    if analysis is None:
        raise ValueError("Provide --preset or explicit --analysis")

    attributes = _read_analysis_attributes(campaign_root, analysis, args.hdf5_filename)
    x_dataset = _setting(args, "x-dataset", attributes.get("xDataset"))
    y_dataset = _setting(args, "y-dataset", attributes.get("yDataset"))
    target_dataset = _setting(args, "target-dataset", attributes.get("yDatasetTarget"))
    covariance_dataset = _setting(args, "covariance-dataset", attributes.get("yCovariance"))
    x_transform = _setting(args, "x-transform", "log10" if int(attributes.get("xAxisIsLog", 0)) else "identity")
    y_transform = _setting(args, "y-transform", "log10" if int(attributes.get("yAxisIsLog", 0)) else "identity")
    x_label = _setting(args, "x-label", attributes.get("xAxisLabel", x_dataset))
    y_label_from_attrs = attributes.get("yAxisLabel", y_dataset)
    if y_transform == "log10" and not str(y_label_from_attrs).startswith(r"$\log"):
        y_label_from_attrs = rf"$\log_{{10}}$ {y_label_from_attrs}"
    y_label = _setting(args, "target-label", y_label_from_attrs)
    target_name = _setting(args, "target-name", attributes.get("targetLabel", "target"))
    output_prefix = _setting(args, "output-prefix", analysis)
    figures_dir_name = _setting(args, "figures-dir-name", f"figures_{output_prefix}")
    emulator_dir_name = _setting(args, "emulator-dir-name", f"emulator_{output_prefix}")
    if any(value is None for value in [x_dataset, y_dataset, target_dataset]):
        raise ValueError("Could not determine x/y/target datasets from arguments or attributes")

    figures_root = campaign_root / figures_dir_name
    emulator_root = campaign_root / emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    samples, x_bins, target, y, y_data_std = _load_campaign(
        campaign_root,
        analysis=analysis,
        x_dataset=x_dataset,
        y_dataset=y_dataset,
        target_dataset=target_dataset,
        covariance_dataset=covariance_dataset,
        hdf5_filename=args.hdf5_filename,
    )
    y, y_data_std, target, log_floor, n_floored = _transform_y(y, y_data_std, target, y_transform)
    x_plot = _axis_values(x_bins, x_transform)
    x = samples[[args.quantile_column]].to_numpy(dtype=float)
    y_pred, y_std = _cv_predict_all_bins(
        x,
        y,
        cv_folds=args.cv_folds,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.random_state,
    )

    metrics = _metrics(y, y_pred, y_std, x_bins)
    metrics_path = emulator_root / f"gp_cv_metrics_{output_prefix}.csv"
    metrics.to_csv(metrics_path, index=False)

    predictions = samples.copy()
    for bin_index, x_bin in enumerate(x_bins):
        label = f"bin{bin_index:02d}"
        predictions[f"{label}_x"] = x_bin
        predictions[f"{label}_y"] = y[:, bin_index]
        predictions[f"{label}_gp_cv_y"] = y_pred[:, bin_index]
        predictions[f"{label}_gp_cv_std"] = y_std[:, bin_index]
        predictions[f"{label}_gp_cv_residual"] = y_pred[:, bin_index] - y[:, bin_index]
    predictions_path = emulator_root / f"gp_cv_predictions_{output_prefix}.csv"
    predictions.to_csv(predictions_path, index=False)

    metadata_path = emulator_root / f"gp_cv_metadata_{output_prefix}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "analysis": analysis,
                "hdf5_filename": args.hdf5_filename,
                "input_column": args.input_column,
                "quantile_column": args.quantile_column,
                "x_dataset": x_dataset,
                "y_dataset": y_dataset,
                "target_dataset": target_dataset,
                "covariance_dataset": covariance_dataset,
                "x_transform": x_transform,
                "y_transform": y_transform,
                "log_floor_linear_y": log_floor,
                "n_floored_y_or_target_values": n_floored,
                "analysis_attributes": attributes,
                "x_bins": x_bins.tolist(),
                "target": target.tolist(),
                "cv_folds": args.cv_folds,
                "n_restarts_optimizer": args.n_restarts_optimizer,
                "y_min": float(np.nanmin(y)),
                "y_max": float(np.nanmax(y)),
                "n_zero_y": int(np.sum(y == 0.0)),
            },
            indent=2,
        )
        + "\n"
    )

    title_prefix = output_prefix.replace("_", " ")
    _plot_parity(
        y,
        y_pred,
        y_std,
        x_plot,
        x_label=x_label,
        y_label=y_label,
        title=f"{title_prefix}: bin-by-bin CV parity",
        path=figures_root / f"gp_cv_parity_{output_prefix}_bins.png",
    )
    _plot_parameter_sweep_overlay(
        samples,
        x_plot,
        target,
        y,
        y_pred,
        y_std,
        y_data_std,
        input_column=args.input_column,
        x_label=x_label,
        y_label=y_label,
        title=f"{title_prefix}: parameter dependence and held-out GP predictions",
        target_name=target_name,
        path=figures_root / f"gp_cv_parameter_sweep_{output_prefix}.png",
    )
    _plot_residual_heatmap(
        samples,
        x_plot,
        y_pred - y,
        input_column=args.input_column,
        x_label=x_label,
        title=f"{title_prefix}: GP CV residuals",
        path=figures_root / f"gp_cv_residual_heatmap_{output_prefix}.png",
    )

    if args.subset_sizes:
        _run_subset_experiment(
            samples,
            x,
            y,
            x_bins,
            subset_sizes=args.subset_sizes,
            n_restarts_optimizer=args.n_restarts_optimizer,
            random_state=args.random_state,
            output_prefix=output_prefix,
            emulator_root=emulator_root,
            figures_root=figures_root,
            title_prefix=title_prefix,
        )

    print(metrics_path)
    print(predictions_path)
    print(metadata_path)
    print(figures_root / f"gp_cv_parity_{output_prefix}_bins.png")
    print(figures_root / f"gp_cv_parameter_sweep_{output_prefix}.png")
    print(figures_root / f"gp_cv_residual_heatmap_{output_prefix}.png")


if __name__ == "__main__":
    main()
