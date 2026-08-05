from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp
from galacticus_emu.interactive_observables import (
    OBSERVABLE_CONFIGS,
    _input_columns_for_samples,
    _load_observable_campaign,
    _make_preprocessor,
    _parameter_specs_for_columns,
    _prepare_training_targets,
    _supported_bin_mask,
    training_bad_mask_override_from_transform,
)


ALL_STANDARD_OBSERVABLES = [
    "bh_halo_mass_trinity_z1",
    "bh_velocity_dispersion",
    "smf_liwhite2009_sdss",
    "smf_z0",
    "smf_z3",
    "mzr_blanc2019",
    "sfr_function_robotham2011",
    "size_mass_vdw2014_star_forming_z0",
    "size_mass_vdw2014_quiescent_z0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a nested-training-size PCA-GP CV diagnostic for standard interactive observables. "
            "For each subset size, the first N Sobol evaluations are split into K folds."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--hdf5-filename", default="galacticus.hdf5")
    parser.add_argument("--observable", action="append", default=None)
    parser.add_argument("--subset-size", type=int, action="append", default=None)
    parser.add_argument(
        "--validation-mode",
        choices=["kfold", "train_rest", "train_rest_plus_full_kfold"],
        default="kfold",
        help=(
            "kfold: K-fold CV within the first N Sobol evaluations. "
            "train_rest: train on the first N evaluations and test on the remaining campaign tail. "
            "train_rest_plus_full_kfold: use train_rest below the full campaign size, "
            "and K-fold CV when the subset size is the full campaign."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=12345)
    parser.add_argument("--min-log10-y", type=float, default=-7.0)
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="standardized")
    parser.add_argument("--pca-variance-threshold", type=float, default=0.99)
    parser.add_argument(
        "--compare-bin-by-bin",
        action="store_true",
        help=(
            "Also run independent bin-by-bin GPs on the same splits and overplot them "
            "as dashed curves against the PCA-GP results."
        ),
    )
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--optimize-hyperparameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-white-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-plotted-bins", type=int, default=5)
    parser.add_argument(
        "--show-training-noise-floor",
        action="store_true",
        help=(
            "On RMSE panels, draw same-colour right-edge ticks at the RMS training alpha sigma "
            "for each plotted bin. Values are also written to the metrics CSV."
        ),
    )
    parser.add_argument(
        "--training-noise-floor-affects-ylim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Let training-noise-floor ticks expand the RMSE y-axis limits. Default is false, "
            "so very uncertain masked/low-count bins do not ruin the RMSE plot scale."
        ),
    )
    parser.add_argument(
        "--show-all-bins-line",
        action="store_true",
        help="Overplot an aggregate all-bins metric curve. Disabled by default to keep per-bin trends clear.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _choose_pca_components(y_scaled: np.ndarray, threshold: float) -> int:
    max_components = min(y_scaled.shape)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("--pca-variance-threshold must be in (0, 1]")
    probe = PCA(n_components=max_components)
    probe.fit(y_scaled)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, threshold) + 1)


def _predict_pca_gp(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha_train: np.ndarray | None,
    pca_scaling: str,
    pca_variance_threshold: float,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    fit_white_noise: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    scaler = _make_preprocessor(pca_scaling)
    y_scaled = scaler.fit_transform(y_train)
    n_components = _choose_pca_components(y_scaled, pca_variance_threshold)
    pca = PCA(n_components=n_components)
    coefficients = pca.fit_transform(y_scaled)

    coefficient_alpha = None
    if alpha_train is not None:
        alpha_scaled = np.asarray(alpha_train, dtype=float) / (scaler.scale_[None, :] ** 2)
        coefficient_alpha = alpha_scaled @ (pca.components_.T ** 2)
        coefficient_alpha = np.maximum(coefficient_alpha, 1.0e-12)

    coefficient_predictions = np.zeros((x_test.shape[0], n_components), dtype=float)
    coefficient_stds = np.zeros_like(coefficient_predictions)
    kernels = []
    for component_index in range(n_components):
        model, y_mean, y_std = fit_scaled_gp(
            x_train,
            coefficients[:, component_index],
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=optimize_hyperparameters,
            alpha=coefficient_alpha[:, component_index] if coefficient_alpha is not None else None,
            fit_white_noise=fit_white_noise,
        )
        pred, pred_std = predict_scaled_gp(model, y_mean, y_std, x_test)
        coefficient_predictions[:, component_index] = pred
        coefficient_stds[:, component_index] = pred_std
        kernels.append(str(model.kernel_))

    y_pred_scaled = coefficient_predictions @ pca.components_ + pca.mean_[None, :]
    y_pred = y_pred_scaled * scaler.scale_[None, :] + scaler.mean_[None, :]
    y_var_scaled = (coefficient_stds**2) @ (pca.components_**2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * scaler.scale_[None, :]
    return y_pred, y_std, {
        "pca_components": int(n_components),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "kernels": kernels,
    }


def _predict_bin_by_bin_gp(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha_train: np.ndarray | None,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    fit_white_noise: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    y_pred = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=float)
    y_std = np.zeros_like(y_pred)
    kernels = []
    for bin_index in range(y_train.shape[1]):
        model, y_mean, y_scale = fit_scaled_gp(
            x_train,
            y_train[:, bin_index],
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=optimize_hyperparameters,
            alpha=alpha_train[:, bin_index] if alpha_train is not None else None,
            fit_white_noise=fit_white_noise,
        )
        pred, pred_std = predict_scaled_gp(model, y_mean, y_scale, x_test)
        y_pred[:, bin_index] = pred
        y_std[:, bin_index] = pred_std
        kernels.append(str(model.kernel_))

    return y_pred, y_std, {
        "pca_components": np.nan,
        "pca_explained_variance": np.nan,
        "kernels": kernels,
    }


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.allclose(y_true, y_true[0]):
        return np.nan
    return float(r2_score(y_true, y_pred))


def _valid_observed_mask(
    values: np.ndarray,
    *,
    bad_training_condition: str,
    bad_mask_override: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.isfinite(values)
    if bad_mask_override is not None:
        return mask & ~np.asarray(bad_mask_override, dtype=bool)
    if bad_training_condition == "nonpositive":
        mask &= values > 0.0
    elif bad_training_condition == "zero_only":
        mask &= values != 0.0
    return mask


def _masked_metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    *,
    bad_training_condition: str,
    bad_mask_override: np.ndarray | None = None,
) -> dict:
    valid = _valid_observed_mask(
        y_true,
        bad_training_condition=bad_training_condition,
        bad_mask_override=bad_mask_override,
    )
    valid &= np.isfinite(y_pred)
    if not np.any(valid):
        return {
            "n_valid": 0,
            "n_valid_uncertainty": 0,
            "rmse": np.nan,
            "bias": np.nan,
            "r2": np.nan,
            "normalized_rmse": np.nan,
            "normalized_mae": np.nan,
            "normalized_bias": np.nan,
            "normalized_std": np.nan,
            "coverage_1sigma": np.nan,
            "coverage_2sigma": np.nan,
        }
    true_valid = y_true[valid]
    pred_valid = y_pred[valid]
    residual = pred_valid - true_valid

    valid_uncertainty = valid & np.isfinite(y_std) & (y_std > 0.0)
    if np.any(valid_uncertainty):
        residual_uncertainty = y_pred[valid_uncertainty] - y_true[valid_uncertainty]
        std_valid = y_std[valid_uncertainty]
        normalized_residual = residual_uncertainty / std_valid
        normalized_rmse = float(np.sqrt(np.mean(normalized_residual**2)))
        normalized_mae = float(np.mean(np.abs(normalized_residual)))
        normalized_bias = float(np.mean(normalized_residual))
        normalized_std = float(np.std(normalized_residual))
        coverage_1sigma = float(np.mean(np.abs(residual_uncertainty) <= std_valid))
        coverage_2sigma = float(np.mean(np.abs(residual_uncertainty) <= 2.0 * std_valid))
    else:
        normalized_rmse = np.nan
        normalized_mae = np.nan
        normalized_bias = np.nan
        normalized_std = np.nan
        coverage_1sigma = np.nan
        coverage_2sigma = np.nan

    return {
        "n_valid": int(valid.sum()),
        "n_valid_uncertainty": int(valid_uncertainty.sum()),
        "rmse": float(np.sqrt(mean_squared_error(true_valid, pred_valid))),
        "bias": float(np.mean(residual)),
        "r2": _safe_r2(true_valid, pred_valid),
        "normalized_rmse": normalized_rmse,
        "normalized_mae": normalized_mae,
        "normalized_bias": normalized_bias,
        "normalized_std": normalized_std,
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma,
    }


def _metric_rows(
    *,
    observable_key: str,
    emulator_type: str,
    subset_size: int,
    validation_mode: str,
    n_train_values: list[int],
    x_plot: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    bad_training_condition: str,
    bad_mask: np.ndarray | None,
    training_sigma_rms_by_bin: np.ndarray | None = None,
    training_sigma_rms_all: float | None = None,
) -> list[dict]:
    rows = []
    for bin_index, x_value in enumerate(x_plot):
        metric_values = _masked_metric_values(
            y_true[:, bin_index],
            y_pred[:, bin_index],
            y_std[:, bin_index],
            bad_training_condition=bad_training_condition,
            bad_mask_override=bad_mask[:, bin_index] if bad_mask is not None else None,
        )
        rows.append(
            {
                "observable_key": observable_key,
                "emulator_type": emulator_type,
                "subset_size": int(subset_size),
                "validation_mode": validation_mode,
                "n_train_mean": float(np.mean(n_train_values)),
                "n_test_total": int(y_true.shape[0]),
                "bin": int(bin_index),
                "x_plot": float(x_value),
                "training_sigma_rms": (
                    np.nan
                    if training_sigma_rms_by_bin is None
                    else float(training_sigma_rms_by_bin[bin_index])
                ),
                **metric_values,
            }
        )
    metric_values = _masked_metric_values(
        y_true.ravel(),
        y_pred.ravel(),
        y_std.ravel(),
        bad_training_condition=bad_training_condition,
        bad_mask_override=bad_mask.ravel() if bad_mask is not None else None,
    )
    rows.append(
        {
            "observable_key": observable_key,
            "emulator_type": emulator_type,
            "subset_size": int(subset_size),
            "validation_mode": validation_mode,
            "n_train_mean": float(np.mean(n_train_values)),
            "n_test_total": int(y_true.size),
            "bin": "all",
            "x_plot": np.nan,
            "training_sigma_rms": np.nan if training_sigma_rms_all is None else float(training_sigma_rms_all),
            **metric_values,
        }
    )
    return rows


def _training_sigma_rms(
    alpha_sigma: np.ndarray | None,
    train_indices_by_split: list[np.ndarray],
) -> tuple[np.ndarray | None, float | None]:
    if alpha_sigma is None or not train_indices_by_split:
        return None, None
    training_sigma = np.vstack([np.asarray(alpha_sigma[index], dtype=float) for index in train_indices_by_split])
    finite = np.isfinite(training_sigma) & (training_sigma >= 0.0)
    rms_by_bin = np.full(training_sigma.shape[1], np.nan, dtype=float)
    for bin_index in range(training_sigma.shape[1]):
        values = training_sigma[finite[:, bin_index], bin_index]
        if values.size:
            rms_by_bin[bin_index] = float(np.sqrt(np.mean(values**2)))
    all_values = training_sigma[finite]
    rms_all = float(np.sqrt(np.mean(all_values**2))) if all_values.size else np.nan
    return rms_by_bin, rms_all


def _selected_bins(n_bins: int, max_plotted_bins: int) -> np.ndarray:
    if n_bins <= max_plotted_bins:
        return np.arange(n_bins, dtype=int)
    return np.asarray(sorted({int(round(v)) for v in np.linspace(0, n_bins - 1, max_plotted_bins)}), dtype=int)


def _method_style(emulator_type: str) -> str:
    return "--" if emulator_type == "bin_by_bin" else "-"


def _method_label(emulator_type: str) -> str:
    if emulator_type == "bin_by_bin":
        return "bin-by-bin GP"
    if emulator_type == "pca":
        return "PCA-GP"
    return emulator_type


def _method_order(values: pd.Series) -> list[str]:
    preferred = ["pca", "bin_by_bin"]
    present = [str(value) for value in values.dropna().unique()]
    ordered = [value for value in preferred if value in present]
    ordered.extend(sorted(value for value in present if value not in ordered))
    return ordered


def _add_plot_train_size(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    validation_modes = set(str(value) for value in metrics.get("validation_mode", pd.Series(dtype=str)).dropna())
    if validation_modes <= {"kfold", "train_rest"}:
        metrics["_plot_train_size"] = metrics["subset_size"].astype(float)
        metrics.attrs["_plot_train_size_label"] = "Training sample size"
    elif "n_train_mean" in metrics:
        metrics["_plot_train_size"] = metrics["n_train_mean"].astype(float)
        metrics.attrs["_plot_train_size_label"] = "mean training evaluations"
    else:
        metrics["_plot_train_size"] = metrics["subset_size"].astype(float)
        metrics.attrs["_plot_train_size_label"] = "Training sample size"
    return metrics


def _format_train_size_tick(value: float) -> str:
    if np.isfinite(value) and abs(value - round(value)) < 1.0e-6:
        return str(int(round(value)))
    return f"{value:.0f}"


def _plot_observable_metrics(
    metrics: pd.DataFrame,
    *,
    observable_key: str,
    x_axis_label: str,
    output_path: Path,
    max_plotted_bins: int,
    show_training_noise_floor: bool,
    training_noise_floor_affects_ylim: bool,
    show_all_bins_line: bool,
) -> None:
    observable_metrics = _add_plot_train_size(metrics.loc[metrics["observable_key"] == observable_key])
    plot_train_size_label = str(observable_metrics.attrs.get("_plot_train_size_label", "Training sample size"))
    if "emulator_type" not in observable_metrics:
        observable_metrics["emulator_type"] = "pca"
    methods = _method_order(observable_metrics["emulator_type"])
    compare_methods = len(methods) > 1
    bin_metrics = observable_metrics.loc[observable_metrics["bin"] != "all"].copy()
    bin_metrics["bin"] = bin_metrics["bin"].astype(int)
    selected = _selected_bins(int(bin_metrics["bin"].max()) + 1, max_plotted_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(selected)))
    noise_tick_values = []
    for color, bin_index in zip(colors, selected, strict=True):
        bin_rows = bin_metrics.loc[bin_metrics["bin"] == bin_index]
        label = f"{x_axis_label}={bin_rows['x_plot'].iloc[0]:.3g}"
        for method_index, method in enumerate(methods):
            rows = bin_rows.loc[bin_rows["emulator_type"] == method].sort_values("_plot_train_size")
            rmse_rows = rows.loc[np.isfinite(rows["rmse"].to_numpy(dtype=float))]
            r2_rows = rows.loc[np.isfinite(rows["r2"].to_numpy(dtype=float))]
            line_label = label if method_index == 0 else "_nolegend_"
            if not rmse_rows.empty:
                axes[0].plot(
                    rmse_rows["_plot_train_size"],
                    rmse_rows["rmse"],
                    marker="o",
                    color=color,
                    ls=_method_style(method),
                    label=line_label,
                )
            if not r2_rows.empty:
                axes[1].plot(
                    r2_rows["_plot_train_size"],
                    r2_rows["r2"],
                    marker="o",
                    color=color,
                    ls=_method_style(method),
                    label=line_label,
                )
        if show_training_noise_floor and "training_sigma_rms" in bin_rows:
            noise_rows = bin_rows.loc[bin_rows["emulator_type"] == methods[0]].sort_values("_plot_train_size")
            finite_noise = noise_rows.loc[np.isfinite(noise_rows["training_sigma_rms"].to_numpy(dtype=float))]
            if not finite_noise.empty:
                noise_value = float(finite_noise.sort_values("_plot_train_size")["training_sigma_rms"].iloc[-1])
                noise_tick_values.append(noise_value)
                transform = blended_transform_factory(axes[0].transAxes, axes[0].transData)
                axes[0].plot(
                    [1.005, 1.045],
                    [noise_value, noise_value],
                    color=color,
                    lw=2.6,
                    transform=transform,
                    clip_on=False,
                    solid_capstyle="butt",
                )

    if show_all_bins_line and not compare_methods:
        all_rows = observable_metrics.loc[observable_metrics["bin"] == "all"].sort_values("_plot_train_size")
        rmse_all = all_rows.loc[np.isfinite(all_rows["rmse"].to_numpy(dtype=float))]
        r2_all = all_rows.loc[np.isfinite(all_rows["r2"].to_numpy(dtype=float))]
        if not rmse_all.empty:
            axes[0].plot(
                rmse_all["_plot_train_size"],
                rmse_all["rmse"],
                color="black",
                lw=2.2,
                ls="--",
                marker="o",
                label="all bins",
            )
        if not r2_all.empty:
            axes[1].plot(
                r2_all["_plot_train_size"],
                r2_all["r2"],
                color="black",
                lw=2.2,
                ls="--",
                marker="o",
                label="all bins",
            )

    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel(r"$R^2$")
    for axis in axes:
        axis.set_xlabel(plot_train_size_label)
        axis.set_xscale("log", base=2)
        positive_train_sizes = sorted(
            value for value in observable_metrics["_plot_train_size"].unique() if np.isfinite(value) and value > 0
        )
        axis.set_xticks(positive_train_sizes)
        axis.set_xticklabels([_format_train_size_tick(value) for value in positive_train_sizes])
        if positive_train_sizes:
            if len(positive_train_sizes) == 1:
                axis.set_xlim(positive_train_sizes[0] / 1.2, positive_train_sizes[0] * 1.2)
            else:
                axis.set_xlim(positive_train_sizes[0] / 1.15, positive_train_sizes[-1] * 1.15)
        axis.grid(alpha=0.25)
    if show_training_noise_floor and noise_tick_values and training_noise_floor_affects_ylim:
        finite_y = observable_metrics["rmse"].to_numpy(dtype=float)
        finite_y = finite_y[np.isfinite(finite_y)]
        finite_noise = np.asarray(noise_tick_values, dtype=float)
        finite_noise = finite_noise[np.isfinite(finite_noise)]
        combined = np.concatenate([finite_y, finite_noise]) if finite_y.size else finite_noise
        if combined.size:
            ymin = min(0.0, float(np.min(combined)))
            ymax = float(np.max(combined))
            if ymax <= ymin:
                ymax = ymin + 1.0
            axes[0].set_ylim(ymin, ymax + 0.08 * (ymax - ymin))
        axes[0].text(
            1.05,
            0.02,
            "RMS training sigma",
            transform=axes[0].transAxes,
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
            color="0.35",
            clip_on=False,
        )
    else:
        finite_y = observable_metrics["rmse"].to_numpy(dtype=float)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size:
            ymax = float(np.max(finite_y))
            axes[0].set_ylim(0.0, ymax * 1.15 if ymax > 0.0 else 1.0)
    axes[1].axhline(0.0, color="0.5", lw=1.0, ls=":")
    finite_r2 = observable_metrics["r2"].to_numpy(dtype=float)
    finite_r2 = finite_r2[np.isfinite(finite_r2)]
    axes[1].set_ylim(bottom=min(-0.2, np.nanmin(finite_r2) - 0.05) if finite_r2.size else -0.2, top=1.03)
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            legend = axis.legend(handles, labels, frameon=False, fontsize=8, loc="best")
            if compare_methods:
                axis.add_artist(legend)
                method_handles = [
                    Line2D([0], [0], color="0.2", lw=2.2, ls=_method_style(method), label=_method_label(method))
                    for method in methods
                ]
                axis.legend(handles=method_handles, frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(observable_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_observable_calibration(
    metrics: pd.DataFrame,
    *,
    observable_key: str,
    x_axis_label: str,
    output_path: Path,
    max_plotted_bins: int,
    show_all_bins_line: bool,
) -> None:
    observable_metrics = _add_plot_train_size(metrics.loc[metrics["observable_key"] == observable_key])
    plot_train_size_label = str(observable_metrics.attrs.get("_plot_train_size_label", "Training sample size"))
    if "emulator_type" not in observable_metrics:
        observable_metrics["emulator_type"] = "pca"
    methods = _method_order(observable_metrics["emulator_type"])
    compare_methods = len(methods) > 1
    bin_metrics = observable_metrics.loc[observable_metrics["bin"] != "all"].copy()
    bin_metrics["bin"] = bin_metrics["bin"].astype(int)
    selected = _selected_bins(int(bin_metrics["bin"].max()) + 1, max_plotted_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(selected)))
    for color, bin_index in zip(colors, selected, strict=True):
        bin_rows = bin_metrics.loc[bin_metrics["bin"] == bin_index]
        label = f"{x_axis_label}={bin_rows['x_plot'].iloc[0]:.3g}"
        for method_index, method in enumerate(methods):
            rows = bin_rows.loc[bin_rows["emulator_type"] == method].sort_values("_plot_train_size")
            normalized_rows = rows.loc[np.isfinite(rows["normalized_rmse"].to_numpy(dtype=float))]
            coverage_rows = rows.loc[np.isfinite(rows["coverage_1sigma"].to_numpy(dtype=float))]
            line_label = label if method_index == 0 else "_nolegend_"
            if not normalized_rows.empty:
                axes[0].plot(
                    normalized_rows["_plot_train_size"],
                    normalized_rows["normalized_rmse"],
                    marker="o",
                    color=color,
                    ls=_method_style(method),
                    label=line_label,
                )
            if not coverage_rows.empty:
                axes[1].plot(
                    coverage_rows["_plot_train_size"],
                    coverage_rows["coverage_1sigma"],
                    marker="o",
                    color=color,
                    ls=_method_style(method),
                    label=line_label,
                )

    if show_all_bins_line and not compare_methods:
        all_rows = observable_metrics.loc[observable_metrics["bin"] == "all"].sort_values("_plot_train_size")
        normalized_all = all_rows.loc[np.isfinite(all_rows["normalized_rmse"].to_numpy(dtype=float))]
        coverage_all = all_rows.loc[np.isfinite(all_rows["coverage_1sigma"].to_numpy(dtype=float))]
        if not normalized_all.empty:
            axes[0].plot(
                normalized_all["_plot_train_size"],
                normalized_all["normalized_rmse"],
                color="black",
                lw=2.2,
                ls="--",
                marker="o",
                label="all bins",
            )
        if not coverage_all.empty:
            axes[1].plot(
                coverage_all["_plot_train_size"],
                coverage_all["coverage_1sigma"],
                color="black",
                lw=2.2,
                ls="--",
                marker="o",
                label="all bins",
            )

    axes[0].axhline(1.0, color="0.45", lw=1.1, ls=":", label="ideal")
    axes[1].axhline(0.6827, color="0.45", lw=1.1, ls=":", label="ideal")
    axes[0].set_ylabel(r"RMSE of $(y_\mathrm{pred}-y_\mathrm{true})/\sigma_\mathrm{pred}$")
    axes[1].set_ylabel(r"fraction within $1\sigma_\mathrm{pred}$")
    axes[1].set_ylim(-0.03, 1.03)
    for axis in axes:
        axis.set_xlabel(plot_train_size_label)
        axis.set_xscale("log", base=2)
        positive_train_sizes = sorted(
            value for value in observable_metrics["_plot_train_size"].unique() if np.isfinite(value) and value > 0
        )
        axis.set_xticks(positive_train_sizes)
        axis.set_xticklabels([_format_train_size_tick(value) for value in positive_train_sizes])
        if positive_train_sizes:
            if len(positive_train_sizes) == 1:
                axis.set_xlim(positive_train_sizes[0] / 1.2, positive_train_sizes[0] * 1.2)
            else:
                axis.set_xlim(positive_train_sizes[0] / 1.15, positive_train_sizes[-1] * 1.15)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            legend = axis.legend(handles, labels, frameon=False, fontsize=8, loc="best")
            if compare_methods:
                axis.add_artist(legend)
                method_handles = [
                    Line2D([0], [0], color="0.2", lw=2.2, ls=_method_style(method), label=_method_label(method))
                    for method in methods
                ]
                axis.legend(handles=method_handles, frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"{observable_key} emulator uncertainty calibration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = campaign_root / "training_size_convergence" / "all_standard_observables"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_sizes = args.subset_size or [64, 128, 256, 512]
    observables = args.observable or ALL_STANDARD_OBSERVABLES
    configs = OBSERVABLE_CONFIGS

    samples = pd.read_csv(campaign_root / "samples.csv")
    input_columns = _input_columns_for_samples(samples)
    input_parameter_specs = _parameter_specs_for_columns(input_columns)
    n_total = len(samples)
    if max(subset_sizes) > n_total:
        raise ValueError(f"Requested subset size {max(subset_sizes)} exceeds campaign size {n_total}")

    all_metric_rows = []
    emulator_types = ["pca"]
    if args.compare_bin_by_bin:
        emulator_types.append("bin_by_bin")
    summary = {
        "campaign_root": str(campaign_root),
        "hdf5_filename": args.hdf5_filename,
        "subset_sizes": subset_sizes,
        "emulator_types": emulator_types,
        "n_folds": int(args.n_folds),
        "validation_mode": args.validation_mode,
        "random_state": int(args.random_state),
        "observables": observables,
        "note": (
            "kfold: each subset size uses the first N Sobol evaluations and K-fold CV within that subset. "
            "train_rest: each subset size trains on the first N Sobol evaluations and tests on the remaining campaign tail. "
            "train_rest_plus_full_kfold: use train_rest below the full campaign size, and K-fold CV for the full-campaign point."
        ),
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for observable_key in observables:
            config = configs[observable_key]
            print(f"Loading {observable_key} ({config['analysis']})", flush=True)
            (
                _samples,
                x_all,
                x_plot,
                _target_plot,
                _target_noise_plot,
                y_plot,
                y_noise,
                attrs,
                transform_metadata,
            ) = _load_observable_campaign(
                campaign_root,
                analysis=config["analysis"],
                hdf5_filename=args.hdf5_filename,
                min_log10_y=args.min_log10_y,
                input_columns=input_columns,
                input_parameter_specs=input_parameter_specs,
            )

            supported_bin_mask = np.ones(y_plot.shape[1], dtype=bool)
            if bool(config.get("drop_unsupported_bins", False)):
                supported_bin_mask = _supported_bin_mask(
                    y_plot,
                    bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
                )
                x_plot = x_plot[supported_bin_mask]
                y_plot = y_plot[:, supported_bin_mask]
                if y_noise is not None:
                    y_noise = y_noise[:, supported_bin_mask]

            training_bad_mask_all = training_bad_mask_override_from_transform(
                transform_metadata,
                bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
                bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
            )
            if training_bad_mask_all is not None and bool(config.get("drop_unsupported_bins", False)):
                training_bad_mask_all = training_bad_mask_all[:, supported_bin_mask]

            metric_bad_mask_all = None
            if (
                training_bad_mask_all is not None
                and str(config.get("bad_training_value_fill", "")) not in {"keep", "as_is"}
            ):
                # For log-transformed observables, good log10(Phi) values are often negative.
                # For relation-like observables with median-filled missing values, use the raw
                # pre-log bad mask instead of testing the plotted values.
                metric_bad_mask_all = np.asarray(training_bad_mask_all, dtype=bool)

            y_fit, alpha_sigma, training_metadata = _prepare_training_targets(
                y_plot,
                y_noise,
                analysis=config["analysis"],
                use_training_alpha=bool(config.get("use_training_alpha", True)),
                bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
                bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
                bad_training_sigma=float(config.get("bad_training_sigma", 5.0)),
                min_training_sigma=float(config.get("min_training_sigma", 1.0e-3)),
                bad_mask_override=training_bad_mask_all,
            )
            alpha = alpha_sigma**2 if alpha_sigma is not None else None

            for subset_size in subset_sizes:
                if args.validation_mode == "train_rest" and subset_size >= n_total:
                    print(
                        f"{observable_key}: subset {subset_size} has no remaining campaign tail; skipping train_rest.",
                        flush=True,
                    )
                    continue
                subset_indices = np.arange(subset_size, dtype=int)
                use_kfold = args.validation_mode == "kfold" or (
                    args.validation_mode == "train_rest_plus_full_kfold" and subset_size >= n_total
                )
                if use_kfold:
                    if subset_size < 2:
                        raise ValueError("K-fold validation needs at least two samples")
                    n_splits = min(args.n_folds, subset_size)
                    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
                    splits = [
                        (subset_indices[train_local], subset_indices[test_local])
                        for train_local, test_local in splitter.split(subset_indices)
                    ]
                else:
                    if subset_size >= n_total:
                        print(
                            f"{observable_key}: subset {subset_size} has no remaining campaign tail; skipping.",
                            flush=True,
                        )
                        continue
                    splits = [(subset_indices, np.arange(subset_size, n_total, dtype=int))]
                true_blocks = []
                bad_mask_blocks = []
                method_blocks = {
                    method: {
                        "pred": [],
                        "std": [],
                        "component_counts": [],
                        "explained_variances": [],
                    }
                    for method in emulator_types
                }
                n_train_values = []
                train_indices_by_split = []
                for fold_index, (train_index, test_index) in enumerate(splits, start=1):
                    print(
                        f"{observable_key}: subset {subset_size}, split {fold_index}/{len(splits)}, "
                        f"train {len(train_index)}, test {len(test_index)}",
                        flush=True,
                    )
                    true_blocks.append(y_plot[test_index])
                    if metric_bad_mask_all is not None:
                        bad_mask_blocks.append(metric_bad_mask_all[test_index])
                    n_train_values.append(len(train_index))
                    train_indices_by_split.append(train_index)
                    for method in emulator_types:
                        if method == "pca":
                            y_pred, y_std, fit_metadata = _predict_pca_gp(
                                x_train=x_all[train_index],
                                y_train=y_fit[train_index],
                                x_test=x_all[test_index],
                                alpha_train=alpha[train_index] if alpha is not None else None,
                                pca_scaling=args.pca_scaling,
                                pca_variance_threshold=args.pca_variance_threshold,
                                n_restarts_optimizer=args.n_restarts_optimizer,
                                optimize_hyperparameters=args.optimize_hyperparameters,
                                fit_white_noise=args.fit_white_noise,
                            )
                        elif method == "bin_by_bin":
                            y_pred, y_std, fit_metadata = _predict_bin_by_bin_gp(
                                x_train=x_all[train_index],
                                y_train=y_fit[train_index],
                                x_test=x_all[test_index],
                                alpha_train=alpha[train_index] if alpha is not None else None,
                                n_restarts_optimizer=args.n_restarts_optimizer,
                                optimize_hyperparameters=args.optimize_hyperparameters,
                                fit_white_noise=args.fit_white_noise,
                            )
                        else:
                            raise ValueError(f"Unknown emulator type: {method}")
                        method_blocks[method]["pred"].append(y_pred)
                        method_blocks[method]["std"].append(y_std)
                        method_blocks[method]["component_counts"].append(fit_metadata["pca_components"])
                        method_blocks[method]["explained_variances"].append(fit_metadata["pca_explained_variance"])

                training_sigma_rms_by_bin, training_sigma_rms_all = _training_sigma_rms(
                    alpha_sigma,
                    train_indices_by_split,
                )
                y_true_stack = np.vstack(true_blocks)
                bad_mask_stack = np.vstack(bad_mask_blocks) if bad_mask_blocks else None
                for method in emulator_types:
                    metric_rows = _metric_rows(
                        observable_key=observable_key,
                        emulator_type=method,
                        subset_size=subset_size,
                        validation_mode=args.validation_mode,
                        n_train_values=n_train_values,
                        x_plot=x_plot,
                        y_true=y_true_stack,
                        y_pred=np.vstack(method_blocks[method]["pred"]),
                        y_std=np.vstack(method_blocks[method]["std"]),
                        bad_training_condition=(
                            "nonfinite"
                            if str(config.get("bad_training_value_fill", "")) in {"keep", "as_is"}
                            else str(config.get("bad_training_condition", "nonfinite"))
                        ),
                        bad_mask=bad_mask_stack,
                        training_sigma_rms_by_bin=training_sigma_rms_by_bin,
                        training_sigma_rms_all=training_sigma_rms_all,
                    )
                    component_counts = np.asarray(method_blocks[method]["component_counts"], dtype=float)
                    explained_variances = np.asarray(method_blocks[method]["explained_variances"], dtype=float)
                    for row in metric_rows:
                        row["pca_components_mean"] = (
                            float(np.nanmean(component_counts)) if np.any(np.isfinite(component_counts)) else np.nan
                        )
                        row["pca_explained_variance_mean"] = (
                            float(np.nanmean(explained_variances))
                            if np.any(np.isfinite(explained_variances))
                            else np.nan
                        )
                        row["training_bad_points"] = int(training_metadata["n_bad_training_points"])
                    all_metric_rows.extend(metric_rows)

            observable_metrics = pd.DataFrame(
                [row for row in all_metric_rows if row["observable_key"] == observable_key]
            )
            observable_metrics.to_csv(output_dir / f"{observable_key}_metrics.csv", index=False)
            _plot_observable_metrics(
                observable_metrics,
                observable_key=observable_key,
                x_axis_label=str(attrs.get("xAxisLabel", "x")),
                output_path=output_dir / "figures" / f"{observable_key}_rmse_r2_vs_training_size.png",
                max_plotted_bins=args.max_plotted_bins,
                show_training_noise_floor=args.show_training_noise_floor,
                training_noise_floor_affects_ylim=args.training_noise_floor_affects_ylim,
                show_all_bins_line=args.show_all_bins_line,
            )
            _plot_observable_calibration(
                observable_metrics,
                observable_key=observable_key,
                x_axis_label=str(attrs.get("xAxisLabel", "x")),
                output_path=output_dir / "figures" / f"{observable_key}_calibration_vs_training_size.png",
                max_plotted_bins=args.max_plotted_bins,
                show_all_bins_line=args.show_all_bins_line,
            )

    metrics = pd.DataFrame(all_metric_rows)
    metrics_path = output_dir / "training_size_cv_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary_path = output_dir / "training_size_cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(metrics_path)
    print(summary_path)
    print(output_dir / "figures")


if __name__ == "__main__":
    main()
