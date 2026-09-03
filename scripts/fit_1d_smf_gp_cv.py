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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.exceptions import ConvergenceWarning


DEFAULT_ANALYSIS = "massFunctionStellarTomczak2014ZFOURGEz0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit bin-by-bin 1D GPs to a Galacticus stellar mass function analysis "
            "and make cross-validation diagnostics."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--input-column", default="diskVelocityCharacteristic")
    parser.add_argument("--quantile-column", default="prior_quantile")
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--cv-folds", type=int, default=32, help="Use >= N for leave-one-out CV.")
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "If supplied, also train on deterministic space-filling subsets of these sizes "
            "and predict the remaining evaluations."
        ),
    )
    parser.add_argument("--n-restarts-optimizer", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-prefix", default="smf_zfourge_z0")
    return parser.parse_args()


def _kernel() -> ConstantKernel:
    return ConstantKernel(1.0, (1.0e-3, 1.0e3)) * Matern(
        length_scale=np.array([0.25]),
        length_scale_bounds=(1.0e-2, 10.0),
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
        kernel=_kernel(),
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


def _splitter(n_samples: int, cv_folds: int, random_state: int):
    if cv_folds >= n_samples:
        return LeaveOneOut()
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def _load_campaign(campaign_root: Path, analysis: str, hdf5_filename: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    samples = pd.read_csv(campaign_root / "samples.csv")
    mass_bins: np.ndarray | None = None
    target: np.ndarray | None = None
    rows = []
    for sample in samples.itertuples(index=False):
        evaluation_id = sample.evaluation_id
        path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            current_mass_bins = np.asarray(group["massStellar"][...], dtype=float)
            current_target = np.asarray(group["massFunctionTarget"][...], dtype=float)
            mass_function = np.asarray(group["massFunction"][...], dtype=float)
        if mass_bins is None:
            mass_bins = current_mass_bins
            target = current_target
        else:
            if not np.allclose(current_mass_bins, mass_bins):
                raise ValueError(f"Mass bins differ for {evaluation_id}")
            if not np.allclose(current_target, target):
                raise ValueError(f"Target SMF differs for {evaluation_id}")
        rows.append(mass_function)

    assert mass_bins is not None
    assert target is not None
    y_linear = np.vstack(rows)
    if np.any(y_linear <= 0.0):
        raise ValueError("SMF contains non-positive values; choose a floor before taking log10.")
    return samples, mass_bins, target, np.log10(y_linear)


def _load_campaign_log10_noise(campaign_root: Path, samples: pd.DataFrame, analysis: str, hdf5_filename: str, noise_floor_dex: float = 1.0e-4) -> np.ndarray:
    rows = []
    for evaluation_id in samples["evaluation_id"]:
        path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            phi = np.asarray(group["massFunction"][...], dtype=float)
            covariance = np.asarray(group["massFunctionCovariance"][...], dtype=float)
        variance = np.maximum(np.diag(covariance), 0.0)
        sigma = np.sqrt(variance) / (phi * np.log(10.0))
        rows.append(np.maximum(sigma, noise_floor_dex))
    return np.vstack(rows)


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


def _space_filling_subset_indices(n_samples: int, subset_size: int) -> np.ndarray:
    if subset_size >= n_samples:
        raise ValueError("subset_size must be smaller than the number of samples when predicting held-out points")
    if subset_size < 2:
        raise ValueError("subset_size must be at least 2")
    return np.asarray(sorted({int(round(value)) for value in np.linspace(0, n_samples - 1, subset_size)}), dtype=int)


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


def _plot_parity(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, mass_bins: np.ndarray, path: Path) -> None:
    n_bins = y_true.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), constrained_layout=True)
    for axis, bin_index in zip(np.ravel(axes), range(n_bins), strict=False):
        observed = y_true[:, bin_index]
        predicted = y_pred[:, bin_index]
        lower = float(min(observed.min(), predicted.min()))
        upper = float(max(observed.max(), predicted.max()))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(observed, predicted, yerr=y_std[:, bin_index], fmt="o", ms=3.5, alpha=0.75)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(rf"$\log_{{10}} M_\star={np.log10(mass_bins[bin_index]):.2f}$")
        axis.set_xlabel("Galacticus")
        axis.set_ylabel("GP CV")
        axis.grid(alpha=0.2)
    for axis in np.ravel(axes)[n_bins:]:
        axis.axis("off")
    fig.suptitle(r"Bin-by-bin CV parity for $\log_{10}\Phi(M_\star)$", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_curves(
    samples: pd.DataFrame,
    mass_bins: np.ndarray,
    target_log10: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    input_column: str,
    path: Path,
) -> None:
    n_samples = len(samples)
    ncols = 4
    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.9 * nrows), sharex=True, sharey=True, constrained_layout=True)
    x_axis = np.log10(mass_bins)
    for axis, row_index in zip(np.ravel(axes), range(n_samples), strict=False):
        value = samples.iloc[row_index][input_column]
        axis.plot(x_axis, y_true[row_index], color="black", lw=1.5, label="Galacticus")
        axis.plot(x_axis, y_pred[row_index], color="#2a6f97", lw=1.3, label="GP CV")
        axis.fill_between(
            x_axis,
            y_pred[row_index] - y_std[row_index],
            y_pred[row_index] + y_std[row_index],
            color="#2a6f97",
            alpha=0.18,
            linewidth=0,
        )
        axis.plot(x_axis, target_log10, color="#d1495b", lw=1.0, alpha=0.75, label="Target")
        axis.set_title(f"eval {row_index:02d}, v={value:.1f}", fontsize=8)
        axis.grid(alpha=0.18)
    for axis in np.ravel(axes)[n_samples:]:
        axis.axis("off")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\log_{10}\Phi$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Leave-out CV predictions for the full stellar mass function", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_parameter_sweep_overlay(
    samples: pd.DataFrame,
    mass_bins: np.ndarray,
    target_log10: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_data_std: np.ndarray | None,
    input_column: str,
    path: Path,
    n_curves: int = 5,
) -> None:
    if n_curves > len(samples):
        n_curves = len(samples)
    indices = np.asarray(sorted({int(round(value)) for value in np.linspace(0, len(samples) - 1, n_curves)}), dtype=int)
    x_axis = np.log10(mass_bins)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(indices)))

    fig, axis = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    axis.plot(x_axis, target_log10, color="0.35", lw=1.3, ls="--", label="Tomczak+14 target")
    for color, row_index in zip(colors, indices, strict=True):
        value = float(samples.iloc[row_index][input_column])
        label = f"{value:.1f} km/s"
        axis.plot(x_axis, y_pred[row_index], color=color, lw=2.0, label=f"GP CV, {label}")
        axis.fill_between(
            x_axis,
            y_pred[row_index] - y_std[row_index],
            y_pred[row_index] + y_std[row_index],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        if y_data_std is None:
            axis.scatter(
                x_axis,
                y_true[row_index],
                color=color,
                s=34,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        else:
            axis.errorbar(
                x_axis,
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
    axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axis.set_ylabel(r"$\log_{10}\Phi$")
    axis.set_title("SMF parameter dependence and held-out GP predictions")
    axis.grid(alpha=0.22)
    axis.legend(title="disk velocity", frameon=False, fontsize=8, title_fontsize=9, ncols=1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_residual_heatmap(samples: pd.DataFrame, mass_bins: np.ndarray, residual: np.ndarray, input_column: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    x_axis = np.log10(mass_bins)
    y_axis = samples[input_column].to_numpy(dtype=float)
    vmax = np.nanpercentile(np.abs(residual), 95)
    mesh = axis.pcolormesh(x_axis, y_axis, residual, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axis.set_ylabel("diskVelocityCharacteristic")
    axis.set_title(r"GP CV residuals: predicted $-$ Galacticus [dex]")
    fig.colorbar(mesh, ax=axis, label="residual [dex]")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subset_metric(metrics: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
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
    axis.set_title(f"SMF held-out GP performance: {ylabel}")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subset_residual_heatmaps(
    samples: pd.DataFrame,
    mass_bins: np.ndarray,
    residuals_by_size: dict[int, tuple[np.ndarray, np.ndarray]],
    input_column: str,
    path: Path,
) -> None:
    sizes = sorted(residuals_by_size)
    ncols = len(sizes)
    fig, axes = plt.subplots(1, ncols, figsize=(4.1 * ncols, 4.2), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = np.nanpercentile(np.abs(np.vstack([residual for _, residual in residuals_by_size.values()])), 95)
    x_axis = np.log10(mass_bins)
    for axis, size in zip(axes, sizes, strict=True):
        test_index, residual = residuals_by_size[size]
        y_axis = samples.iloc[test_index][input_column].to_numpy(dtype=float)
        order = np.argsort(y_axis)
        mesh = axis.pcolormesh(x_axis, y_axis[order], residual[order], shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axis.set_title(f"Ntrain={size}")
        axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        axis.grid(alpha=0.12)
    axes[0].set_ylabel("diskVelocityCharacteristic")
    fig.colorbar(mesh, ax=axes, label="held-out residual [dex]", shrink=0.86)
    fig.suptitle("Subset-trained GP residuals: predicted - Galacticus", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_subset_experiment(
    samples: pd.DataFrame,
    mass_bins: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    subset_sizes: list[int],
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
        pred, pred_std = _subset_predict_all_bins(
            x,
            y,
            train_index,
            test_index,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        metrics = _metrics(y[test_index], pred, pred_std, mass_bins)
        metrics.insert(0, "train_size", subset_size)
        metrics.insert(1, "n_test", len(test_index))
        metrics_frames.append(metrics)
        residuals_by_size[subset_size] = (test_index, pred - y[test_index])
        for local_row, evaluation_index in enumerate(test_index):
            for bin_index, mass in enumerate(mass_bins):
                prediction_rows.append(
                    {
                        "train_size": subset_size,
                        "evaluation_id": samples.iloc[evaluation_index]["evaluation_id"],
                        "evaluation_index": int(evaluation_index),
                        "is_training": False,
                        "prior_quantile": float(samples.iloc[evaluation_index]["prior_quantile"]),
                        input_column: float(samples.iloc[evaluation_index][input_column]),
                        "bin": bin_index,
                        "mass_stellar": float(mass),
                        "mass_stellar_log10": float(np.log10(mass)),
                        "log10_phi": float(y[evaluation_index, bin_index]),
                        "gp_log10_phi": float(pred[local_row, bin_index]),
                        "gp_std": float(pred_std[local_row, bin_index]),
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
    figures_root = campaign_root / "figures"
    emulator_root = campaign_root / "emulator"
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    samples, mass_bins, target, y = _load_campaign(campaign_root, args.analysis, args.hdf5_filename)
    y_data_std = _load_campaign_log10_noise(campaign_root, samples, args.analysis, args.hdf5_filename)
    x = samples[[args.quantile_column]].to_numpy(dtype=float)
    y_pred, y_std = _cv_predict_all_bins(
        x,
        y,
        cv_folds=args.cv_folds,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.random_state,
    )

    metrics = _metrics(y, y_pred, y_std, mass_bins)
    metrics_path = emulator_root / f"gp_cv_metrics_{args.output_prefix}.csv"
    metrics.to_csv(metrics_path, index=False)

    predictions = samples.copy()
    for bin_index, mass in enumerate(mass_bins):
        label = f"bin{bin_index:02d}_logM{np.log10(mass):.2f}"
        predictions[f"{label}_log10_phi"] = y[:, bin_index]
        predictions[f"{label}_gp_cv_log10_phi"] = y_pred[:, bin_index]
        predictions[f"{label}_gp_cv_std"] = y_std[:, bin_index]
        predictions[f"{label}_gp_cv_residual"] = y_pred[:, bin_index] - y[:, bin_index]
    predictions_path = emulator_root / f"gp_cv_predictions_{args.output_prefix}.csv"
    predictions.to_csv(predictions_path, index=False)

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
    print(figures_root / f"gp_cv_parity_{args.output_prefix}_bins.png")
    print(figures_root / f"gp_cv_curves_{args.output_prefix}.png")
    print(figures_root / f"gp_cv_parameter_sweep_{args.output_prefix}.png")
    print(figures_root / f"gp_cv_residual_heatmap_{args.output_prefix}.png")


if __name__ == "__main__":
    main()
