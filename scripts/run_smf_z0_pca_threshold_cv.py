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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
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
    training_bad_mask_override_from_transform,
)
from galacticus_emu.plotting import set_ylim_from_values
from run_standard_observable_training_size_cv import _predict_bin_by_bin_gp


PCA99_COLOR = "#1f77b4"


def _set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-oriented SMF z~0 CV diagnostic comparing PCA explained-variance thresholds."
    )
    parser.add_argument(
        "campaign_root",
        type=Path,
        default=Path(
            "runs/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_freeYield_20p_moreHalos_512_hybrid_5fallback"
        ),
        nargs="?",
    )
    parser.add_argument("--hdf5-filename", default="galacticus.hdf5")
    parser.add_argument("--observable", default="smf_z0")
    parser.add_argument("--subset-size", type=int, action="append", default=None)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument(
        "--validation-mode",
        choices=["kfold_subset", "train_rest", "train_rest_plus_full_kfold"],
        default="kfold_subset",
        help=(
            "kfold_subset trains/tests within each Sobol prefix; train_rest trains on the prefix "
            "and tests on all remaining evaluations; train_rest_plus_full_kfold does train_rest "
            "for prefixes smaller than the full campaign and K-fold CV for the full prefix."
        ),
    )
    parser.add_argument("--random-state", type=int, default=12345)
    parser.add_argument("--pca-variance-threshold", type=float, action="append", default=None)
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="standardized")
    parser.add_argument("--highlight-x", type=float, action="append", default=None)
    parser.add_argument("--holdout-threshold", type=float, default=0.99)
    parser.add_argument("--n-example-curves", type=int, default=5)
    parser.add_argument(
        "--example-selection",
        choices=["mean_smf_quantiles", "input_space", "sample_index"],
        default="mean_smf_quantiles",
        help=(
            "How to choose the illustrative held-out curves: mean_smf_quantiles spreads examples "
            "by held-out SMF amplitude, input_space matches the old holdout-demo convention of "
            "spreading examples along the first emulator coordinate, and sample_index matches the "
            "original threshold-CV renderer."
        ),
    )
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--optimize-hyperparameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-white-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-log10-y", type=float, default=-7.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--paper-figures-only",
        action="store_true",
        help="Write only the three preferred paper-style figures, skipping exploratory variants.",
    )
    parser.add_argument("--force", action="store_true", help="Refit even if prediction CSV already exists.")
    return parser.parse_args()


def _threshold_label(threshold: float) -> str:
    pct = 100.0 * threshold
    if np.isclose(pct, round(pct)):
        text = f"{int(round(pct))}"
    else:
        text = f"{pct:g}".replace(".", "p")
    return f"pca{str(threshold).replace('.', 'p')}_{text}pct"


def _method_display(method: str) -> str:
    if method == "bin_by_bin":
        return "bin-by-bin"
    if method.startswith("pca_"):
        raw = method.removeprefix("pca_").replace("p", ".")
        return f"PCA {100.0 * float(raw):g}%"
    return method


def _decode_attr(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _choose_components(cumulative: np.ndarray, threshold: float) -> int:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("PCA thresholds must be in (0, 1]")
    return int(np.searchsorted(cumulative, threshold) + 1)


def _predict_pca_thresholds(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha_train: np.ndarray | None,
    thresholds: list[float],
    pca_scaling: str,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    fit_white_noise: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict]]:
    scaler = _make_preprocessor(pca_scaling)
    y_scaled = scaler.fit_transform(y_train)
    max_components = min(y_scaled.shape)
    pca = PCA(n_components=max_components)
    coefficients = pca.fit_transform(y_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    component_count_by_threshold = {
        threshold: min(_choose_components(cumulative, threshold), max_components) for threshold in thresholds
    }
    max_needed = max(component_count_by_threshold.values())

    coefficient_alpha = None
    if alpha_train is not None:
        alpha_scaled = np.asarray(alpha_train, dtype=float) / (scaler.scale_[None, :] ** 2)
        coefficient_alpha = alpha_scaled @ (pca.components_.T ** 2)
        coefficient_alpha = np.maximum(coefficient_alpha, 1.0e-12)

    coefficient_predictions = np.zeros((x_test.shape[0], max_needed), dtype=float)
    coefficient_stds = np.zeros_like(coefficient_predictions)
    kernels = []
    for component_index in range(max_needed):
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

    results = {}
    for threshold, n_components in component_count_by_threshold.items():
        components = pca.components_[:n_components]
        y_pred_scaled = coefficient_predictions[:, :n_components] @ components + pca.mean_[None, :]
        y_pred = y_pred_scaled * scaler.scale_[None, :] + scaler.mean_[None, :]
        y_var_scaled = (coefficient_stds[:, :n_components] ** 2) @ (components**2)
        y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * scaler.scale_[None, :]
        results[f"pca_{threshold:g}".replace(".", "p")] = (
            y_pred,
            y_std,
            {
                "pca_threshold": float(threshold),
                "pca_components": int(n_components),
                "pca_explained_variance": float(cumulative[n_components - 1]),
                "kernels": kernels[:n_components],
            },
        )
    return results


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2 or np.allclose(y_true[finite], y_true[finite][0]):
        return np.nan
    return float(r2_score(y_true[finite], y_pred[finite]))


def _save_figure(fig, output_path: Path, *, tight: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": 250}
    if tight:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
    fig.savefig(output_path, **save_kwargs)
    fig.savefig(output_path.with_suffix(".pdf"), **save_kwargs)


def _write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def _prediction_method_names(thresholds: list[float]) -> set[str]:
    methods = {f"pca_{threshold:g}".replace(".", "p") for threshold in thresholds}
    methods.add("bin_by_bin")
    return methods


def _completed_split_keys(predictions: pd.DataFrame, required_methods: set[str]) -> set[tuple[int, int]]:
    if predictions.empty or not {"subset_size", "fold_index", "method"}.issubset(predictions.columns):
        return set()
    completed: set[tuple[int, int]] = set()
    for (subset_size, fold_index), group in predictions.groupby(["subset_size", "fold_index"], sort=False):
        if required_methods.issubset(set(group["method"].astype(str))):
            completed.add((int(subset_size), int(fold_index)))
    return completed


def _replace_split_rows(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    *,
    subset_size: int,
    fold_index: int,
) -> pd.DataFrame:
    if existing.empty:
        return new_rows.reset_index(drop=True)
    keep = ~(
        (existing["subset_size"].astype(int) == int(subset_size))
        & (existing["fold_index"].astype(int) == int(fold_index))
    )
    return pd.concat([existing.loc[keep], new_rows], ignore_index=True)


def _replace_split_manifest_row(
    existing: pd.DataFrame,
    new_row: dict,
    *,
    subset_size: int,
    fold_index: int,
) -> pd.DataFrame:
    new_frame = pd.DataFrame([new_row])
    if existing.empty:
        return new_frame
    keep = ~(
        (existing["subset_size"].astype(int) == int(subset_size))
        & (existing["fold_index"].astype(int) == int(fold_index))
    )
    return pd.concat([existing.loc[keep], new_frame], ignore_index=True)


def _select_evenly_spaced_by_score(
    sample_indices: np.ndarray,
    scores: np.ndarray,
    *,
    n_examples: int,
    trim_extremes: bool,
) -> np.ndarray:
    if sample_indices.size <= n_examples:
        return sample_indices
    order = np.argsort(scores)
    if trim_extremes and n_examples > 1:
        positions = np.linspace(0.08, 0.92, n_examples) * (len(order) - 1)
    else:
        positions = np.linspace(0, len(order) - 1, n_examples)
    chosen_positions = np.asarray(sorted({int(round(value)) for value in positions}), dtype=int)
    return sample_indices[order[chosen_positions]]


def _select_example_indices(
    *,
    rows: pd.DataFrame,
    x_all: np.ndarray,
    n_examples: int,
    example_selection: str,
) -> np.ndarray:
    test_indices = np.asarray(sorted(rows["sample_index"].unique()), dtype=int)
    if test_indices.size <= n_examples:
        return test_indices
    if example_selection == "sample_index":
        positions = np.linspace(0, test_indices.size - 1, n_examples).round().astype(int)
        return test_indices[positions]
    if example_selection == "input_space":
        return _select_evenly_spaced_by_score(
            test_indices,
            x_all[test_indices, 0],
            n_examples=n_examples,
            trim_extremes=False,
        )
    if example_selection == "mean_smf_quantiles":
        mean_smf = rows.groupby("sample_index", sort=True)["y_true"].mean()
        return _select_evenly_spaced_by_score(
            mean_smf.index.to_numpy(dtype=int),
            mean_smf.to_numpy(dtype=float),
            n_examples=n_examples,
            trim_extremes=True,
        )
    raise ValueError(f"Unknown example selection mode: {example_selection}")


def _metrics_for_group(group: pd.DataFrame) -> dict:
    finite = np.isfinite(group["y_true"]) & np.isfinite(group["y_pred"])
    if not finite.any():
        return {"n_valid": 0, "rmse": np.nan, "bias": np.nan, "r2": np.nan, "normalized_rmse": np.nan}
    true = group.loc[finite, "y_true"].to_numpy(dtype=float)
    pred = group.loc[finite, "y_pred"].to_numpy(dtype=float)
    residual = pred - true
    std = group.loc[finite, "y_std"].to_numpy(dtype=float)
    finite_std = np.isfinite(std) & (std > 0.0)
    return {
        "n_valid": int(finite.sum()),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "bias": float(np.mean(residual)),
        "r2": _safe_r2(true, pred),
        "normalized_rmse": (
            float(np.sqrt(np.mean((residual[finite_std] / std[finite_std]) ** 2)))
            if np.any(finite_std)
            else np.nan
        ),
        "coverage_1sigma": (
            float(np.mean(np.abs(residual[finite_std]) <= std[finite_std])) if np.any(finite_std) else np.nan
        ),
    }


def _nearest_bins(x_plot: np.ndarray, requested: list[float]) -> list[int]:
    bins = []
    for value in requested:
        bins.append(int(np.argmin(np.abs(x_plot - value))))
    return list(dict.fromkeys(bins))


def _write_metrics(predictions: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    group_columns = ["observable_key", "method", "subset_size", "bin_index"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        first = group.iloc[0]
        row["x_plot"] = float(first["x_plot"])
        row["n_train_mean"] = float(group["n_train"].mean())
        row["n_test_total"] = int(len(group))
        row["pca_components_mean"] = float(group["pca_components"].mean(skipna=True))
        row["pca_explained_variance_mean"] = float(group["pca_explained_variance"].mean(skipna=True))
        row.update(_metrics_for_group(group))
        rows.append(row)
    for keys, group in predictions.groupby(["observable_key", "method", "subset_size"], sort=True):
        row = dict(zip(["observable_key", "method", "subset_size"], keys, strict=True))
        row["bin_index"] = "all"
        row["x_plot"] = np.nan
        row["n_train_mean"] = float(group["n_train"].mean())
        row["n_test_total"] = int(len(group))
        row["pca_components_mean"] = float(group["pca_components"].mean(skipna=True))
        row["pca_explained_variance_mean"] = float(group["pca_explained_variance"].mean(skipna=True))
        row.update(_metrics_for_group(group))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_path, index=False)
    return metrics


def _plot_training_size(
    metrics: pd.DataFrame,
    *,
    highlighted_bins: list[int],
    training_rms_sigma_by_bin: dict[int, float] | None,
    output_path: Path,
) -> None:
    bin_metrics = metrics.loc[metrics["bin_index"].astype(str) != "all"].copy()
    bin_metrics["bin_index"] = bin_metrics["bin_index"].astype(int)
    methods = [m for m in bin_metrics["method"].drop_duplicates() if m != "bin_by_bin"]
    if "bin_by_bin" in set(bin_metrics["method"]):
        methods.append("bin_by_bin")
    styles = {
        "pca_0p9": ("#d62728", "-"),
        "pca_0p99": (PCA99_COLOR, "-"),
        "pca_0p995": ("#9467bd", "-"),
        "pca_0p999": ("#2ca02c", "-"),
        "bin_by_bin": ("#222222", "--"),
    }
    fig, axes = plt.subplots(
        2,
        len(highlighted_bins),
        figsize=(12.0, 6.8),
        sharex=True,
        gridspec_kw={"wspace": 0.14, "hspace": 0.12},
    )
    axes = np.atleast_2d(axes)
    for column, bin_index in enumerate(highlighted_bins):
        rows_for_bin = bin_metrics.loc[bin_metrics["bin_index"] == bin_index]
        x_value = rows_for_bin["x_plot"].iloc[0]
        for method in methods:
            rows = rows_for_bin.loc[rows_for_bin["method"] == method].sort_values("subset_size")
            if rows.empty:
                continue
            color, linestyle = styles.get(method, ("0.35", "-"))
            label = _method_display(method)
            linewidth = 3.2 if method == "pca_0p99" else 1.8
            marker_size = 7.8
            if method == "pca_0p99":
                marker_size = 8.8
            elif method == "pca_0p999":
                marker_size = 6.6
            axes[0, column].plot(
                rows["subset_size"],
                rows["r2"],
                marker="o",
                ms=marker_size,
                color=color,
                ls=linestyle,
                lw=linewidth,
                label=label,
            )
            axes[1, column].plot(
                rows["subset_size"],
                rows["rmse"],
                marker="o",
                ms=marker_size,
                color=color,
                ls=linestyle,
                lw=linewidth,
                label=label,
            )
        if training_rms_sigma_by_bin is not None and bin_index in training_rms_sigma_by_bin:
            axes[1, column].axhline(
                training_rms_sigma_by_bin[bin_index],
                color="0.25",
                lw=1.1,
                ls=":",
                label="RMS training uncertainty" if column == 0 else None,
            )
        axes[0, column].set_title(
            rf"$\log_{{10}}(M_\star/\mathrm{{M}}_\odot)={x_value:.2f}$",
            pad=10,
        )
        axes[0, column].set_ylim(0.0, 1.0)
        axes[1, column].set_ylim(bottom=0.0)
        rmse_upper = axes[1, column].get_ylim()[1]
        axes[1, column].set_ylim(0.0, 1.2 * rmse_upper)
        if column == len(highlighted_bins) // 2:
            axes[1, column].set_xlabel("Number of Galacticus training runs")
        for axis in axes[:, column]:
            axis.set_xscale("log", base=2)
            ticks = sorted(rows_for_bin["subset_size"].dropna().unique())
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value))}"))
    axes[0, 0].set_ylabel(r"$R^2$")
    axes[1, 0].set_ylabel("RMSE [dex]")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(handles),
        )
    fig.subplots_adjust(top=0.88, left=0.08, right=0.98, bottom=0.10, wspace=0.14, hspace=0.12)
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_parity(
    predictions: pd.DataFrame,
    *,
    method: str,
    subset_size: int,
    highlighted_bins: list[int],
    target_plot: np.ndarray,
    target_label: str,
    point_fraction: float,
    point_alpha: float,
    errorbar_alpha: float,
    errorbar_lw: float,
    include_errorbars: bool,
    output_path: Path,
) -> None:
    subset = predictions.loc[(predictions["method"] == method) & (predictions["subset_size"] == subset_size)].copy()
    fig, axes = plt.subplots(
        1,
        len(highlighted_bins),
        figsize=(12.0, 4.5),
        gridspec_kw={"wspace": 0.10},
    )
    axes = np.atleast_1d(axes)
    fixed_limits = [(-3.1, -1.5), (-4.7, -1.5), (None, -1.8)]
    rng = np.random.default_rng(12345)
    for panel_index, (axis, bin_index) in enumerate(zip(axes, highlighted_bins, strict=True)):
        rows = subset.loc[subset["bin_index"] == bin_index]
        true = rows["y_true"].to_numpy(dtype=float)
        pred = rows["y_pred"].to_numpy(dtype=float)
        std = rows["y_std"].to_numpy(dtype=float)
        finite = np.isfinite(true) & np.isfinite(pred)
        draw = finite.copy()
        if point_fraction < 1.0 and finite.sum() > 0:
            finite_positions = np.where(finite)[0]
            n_draw = max(1, int(round(point_fraction * finite_positions.size)))
            selected = rng.choice(finite_positions, size=n_draw, replace=False)
            draw[:] = False
            draw[selected] = True
        if include_errorbars:
            errored = axis.errorbar(
                true[draw],
                pred[draw],
                yerr=std[draw],
                fmt="o",
                ms=2.7,
                alpha=point_alpha,
                color=PCA99_COLOR,
                ecolor=PCA99_COLOR,
                elinewidth=errorbar_lw,
                capsize=0,
                zorder=2,
            )
            for bar_collection in errored[2]:
                bar_collection.set_alpha(errorbar_alpha)
        else:
            axis.plot(
                true[draw],
                pred[draw],
                "o",
                ms=2.7,
                alpha=point_alpha,
                color=PCA99_COLOR,
                zorder=2,
            )
        if np.any(finite):
            lo = float(np.nanmin([true[finite].min(), pred[finite].min(), target_plot[bin_index]]))
            hi = float(np.nanmax([true[finite].max(), pred[finite].max(), target_plot[bin_index]]))
            pad = 0.08 * max(hi - lo, 0.1)
            label = target_label if panel_index == 0 else None
            axis.scatter(
                [target_plot[bin_index]],
                [target_plot[bin_index]],
                marker="*",
                s=120,
                color="black",
                zorder=5,
                label=label,
            )
            limits = fixed_limits[panel_index]
            if limits is None or limits[0] is None or limits[1] is None:
                lo_axis = np.floor(lo - pad)
                hi_axis = np.ceil(hi + pad)
                limits = (
                    lo_axis if limits is None or limits[0] is None else limits[0],
                    hi_axis if limits is None or limits[1] is None else limits[1],
                )
            axis.set_xlim(*limits)
            axis.set_ylim(*limits)
            axis.plot(limits, limits, color="0.25", ls="--", lw=1.0, zorder=1)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        metric = _metrics_for_group(rows)
        axis.set_title(
            rf"$\log_{{10}}(M_\star/\mathrm{{M}}_\odot)={rows['x_plot'].iloc[0]:.2f}$",
            pad=10,
        )
        axis.text(
            0.97,
            0.04,
            f"RMSE={metric['rmse']:.2f}\n" + rf"$R^2$={metric['r2']:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=16.5,
        )
        axis.set_xlabel(r"held-out $\log_{10}\Phi$")
    axes[0].set_ylabel(r"predicted $\log_{10}\Phi$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, fontsize=11, loc="upper left")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.18)
    _save_figure(fig, output_path, tight=True)
    plt.close(fig)


def _plot_heldout_curves(
    *,
    y_plot: np.ndarray,
    target_plot: np.ndarray,
    target_noise_plot: np.ndarray | None,
    x_plot: np.ndarray,
    x_all: np.ndarray,
    train_index: np.ndarray,
    predictions: pd.DataFrame,
    method: str,
    subset_size: int,
    fold_index: int,
    highlighted_bins: list[int],
    n_examples: int,
    example_selection: str,
    target_label: str,
    redshift_label: str,
    output_path: Path,
) -> None:
    rows = predictions.loc[
        (predictions["method"] == method)
        & (predictions["subset_size"] == subset_size)
        & (predictions["fold_index"] == fold_index)
    ].copy()
    chosen = _select_example_indices(
        rows=rows,
        x_all=x_all,
        n_examples=n_examples,
        example_selection=example_selection,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axis = plt.subplots(figsize=(8.0, 5.1), constrained_layout=True)
    alpha = min(0.10, 20.0 / max(len(train_index), 1))
    for index in train_index:
        axis.plot(x_plot, y_plot[index], color="0.55", lw=0.8, alpha=alpha, zorder=1)
    axis.plot([], [], color="0.55", lw=1.0, alpha=0.35, label="training-set model LFs")
    if target_noise_plot is None:
        axis.plot(x_plot, target_plot, "o-", color="black", lw=1.8, ms=4.0, label=target_label, zorder=5)
    else:
        axis.errorbar(
            x_plot,
            target_plot,
            yerr=target_noise_plot,
            fmt="o",
            color="black",
            ms=4.0,
            label=target_label,
            zorder=5,
        )
    example_index = chosen[1] if len(chosen) > 1 else chosen[0] if len(chosen) else None
    for color, sample_index in zip(colors, chosen, strict=False):
        sample_rows = rows.loc[rows["sample_index"] == sample_index].sort_values("bin_index")
        pred = sample_rows["y_pred"].to_numpy(dtype=float)
        std = sample_rows["y_std"].to_numpy(dtype=float)
        heldout_err = sample_rows["alpha_sigma"].to_numpy(dtype=float, copy=True)
        heldout_err[~np.isfinite(heldout_err)] = 0.0
        is_labeled_example = sample_index == example_index
        axis.errorbar(
            x_plot,
            y_plot[sample_index],
            yerr=heldout_err,
            fmt="o",
            ms=3.5,
            color=color,
            alpha=0.85,
            label="example hold-out" if is_labeled_example else None,
            zorder=4,
        )
        axis.plot(
            x_plot,
            pred,
            color=color,
            lw=2.0,
            alpha=0.95,
            label="example emulator prediction" if is_labeled_example else None,
        )
        axis.fill_between(x_plot, pred - std, pred + std, color=color, alpha=0.14, lw=0)
    for bin_index in highlighted_bins:
        axis.axvline(x_plot[bin_index], color="0.15", lw=1.0, ls=":", alpha=0.9)
        axis.scatter([x_plot[bin_index]], [target_plot[bin_index]], marker="*", s=90, color="black", zorder=6)
    axis.set_xlabel(r"$\log_{10}(M_\star/\mathrm{M}_\odot)$")
    axis.set_ylabel(r"$\log_{10}(\Phi\,/\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$")
    set_ylim_from_values(axis, y_plot[train_index], target_plot, rows["y_true"], rows["y_pred"])
    upper = axis.get_ylim()[1]
    axis.set_ylim(-6.2, upper)
    axis.set_xlim(float(np.nanmin(x_plot)), float(np.nanmax(x_plot)))
    if redshift_label:
        axis.text(
            0.97,
            0.96,
            redshift_label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=14,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 3.0},
        )
    handles, labels = axis.get_legend_handles_labels()
    order = [
        "training-set model LFs",
        "Tomczak et al. (2014)",
        "example hold-out",
        "example emulator prediction",
    ]
    handle_by_label = dict(zip(labels, handles, strict=False))
    ordered_labels = [label for label in order if label in handle_by_label]
    ordered_handles = [handle_by_label[label] for label in ordered_labels]
    legend = axis.legend(
        ordered_handles,
        ordered_labels,
        frameon=True,
        fontsize=11,
        ncol=1,
        loc="lower left",
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    legend.set_zorder(10)
    _save_figure(fig, output_path)
    plt.close(fig)


def _iter_splits(
    *,
    n_samples: int,
    subset_size: int,
    n_folds: int,
    random_state: int,
    validation_mode: str,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    subset_size = min(subset_size, n_samples)
    subset_indices = np.arange(subset_size, dtype=int)
    use_kfold = validation_mode == "kfold_subset" or (
        validation_mode == "train_rest_plus_full_kfold" and subset_size == n_samples
    )
    if use_kfold:
        if subset_size < 2:
            raise ValueError("K-fold validation needs at least two samples")
        n_splits = min(n_folds, subset_size)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return [
            (fold_index, subset_indices[train_local], subset_indices[test_local])
            for fold_index, (train_local, test_local) in enumerate(splitter.split(subset_indices), start=1)
        ]
    train_index = subset_indices
    test_index = np.arange(subset_size, n_samples, dtype=int)
    if test_index.size == 0:
        raise ValueError(
            f"Validation mode {validation_mode!r} gives no test samples for subset_size={subset_size}; "
            "use train_rest_plus_full_kfold for the full-campaign point."
        )
    return [(1, train_index, test_index)]


def main() -> None:
    _set_paper_style()
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else campaign_root / "cross_validation" / "pca_threshold_smf_z0"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "smf_z0_pca_threshold_cv_predictions.csv"
    metrics_path = output_dir / "smf_z0_pca_threshold_cv_metrics.csv"
    split_manifest_path = output_dir / "smf_z0_pca_threshold_cv_splits.csv"

    thresholds = args.pca_variance_threshold or [0.90, 0.99, 0.999]
    subset_sizes = args.subset_size or [64, 128, 256, 512]
    highlighted_x = args.highlight_x or [8.75, 9.75, 10.75]
    config = OBSERVABLE_CONFIGS[args.observable]

    samples = pd.read_csv(campaign_root / "samples.csv")
    input_columns = _input_columns_for_samples(samples)
    input_parameter_specs = _parameter_specs_for_columns(input_columns)
    (
        _samples,
        x_all,
        x_plot,
        target_plot,
        target_noise_plot,
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
    training_bad_mask = training_bad_mask_override_from_transform(
        transform_metadata,
        bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
        bad_training_value_fill=config.get("bad_training_value_fill", "keep"),
    )
    y_fit, alpha_sigma, training_metadata = _prepare_training_targets(
        y_plot,
        y_noise,
        analysis=config["analysis"],
        use_training_alpha=bool(config.get("use_training_alpha", True)),
        bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
        bad_training_value_fill=config.get("bad_training_value_fill", "keep"),
        bad_training_sigma=float(config.get("bad_training_sigma", 0.5)),
        min_training_sigma=float(config.get("min_training_sigma", 1.0e-3)),
        bad_mask_override=training_bad_mask,
    )
    alpha = alpha_sigma**2 if alpha_sigma is not None else None
    highlighted_bins = _nearest_bins(x_plot, highlighted_x)
    training_rms_sigma_by_bin = (
        {
            int(bin_index): float(np.sqrt(np.nanmean(alpha_sigma[:, bin_index] ** 2)))
            for bin_index in highlighted_bins
        }
        if alpha_sigma is not None
        else None
    )
    target_label = _decode_attr(attrs.get("targetLabel"), "Tomczak et al. (2014)")
    description = _decode_attr(attrs.get("description"), "")
    redshift_match = re.search(r"\$([^$]*<\s*z\s*<[^$]*)\$", description)
    redshift_label = redshift_match.group(1).replace("\\", "") if redshift_match else ""

    split_plan = []
    for subset_size in subset_sizes:
        split_cases = _iter_splits(
            n_samples=x_all.shape[0],
            subset_size=subset_size,
            n_folds=args.n_folds,
            random_state=args.random_state,
            validation_mode=args.validation_mode,
        )
        for fold_index, train_index, test_index in split_cases:
            split_plan.append(
                {
                    "subset_size": int(subset_size),
                    "fold_index": int(fold_index),
                    "fold_total": int(len(split_cases)),
                    "train_index": train_index,
                    "test_index": test_index,
                }
            )
    expected_keys = {(case["subset_size"], case["fold_index"]) for case in split_plan}
    required_methods = _prediction_method_names(thresholds)
    train_indices_for_holdout: np.ndarray | None = None
    for case in split_plan:
        if case["subset_size"] == max(subset_sizes) and case["fold_index"] == 1:
            train_indices_for_holdout = case["train_index"].copy()
            break

    if prediction_path.exists() and not args.force:
        cached_predictions = pd.read_csv(prediction_path)
        cached_predictions = cached_predictions.loc[
            cached_predictions["subset_size"].astype(int).isin(set(subset_sizes))
            & cached_predictions["method"].astype(str).isin(required_methods)
        ].copy()
        completed_keys = _completed_split_keys(cached_predictions, required_methods)
        missing_cases = [
            case for case in split_plan if (case["subset_size"], case["fold_index"]) not in completed_keys
        ]
        predictions = cached_predictions.reset_index(drop=True)
        if missing_cases:
            print(
                f"Resuming {prediction_path}: {len(completed_keys & expected_keys)}/"
                f"{len(expected_keys)} splits already cached.",
                flush=True,
            )
        else:
            print(f"Using existing complete {prediction_path}", flush=True)
    else:
        if prediction_path.exists() and args.force:
            print(f"Ignoring existing {prediction_path} because --force was supplied.", flush=True)
        predictions = pd.DataFrame()
        completed_keys = set()
        missing_cases = split_plan

    if not split_manifest_path.exists() or args.force:
        split_manifest = pd.DataFrame()
    else:
        split_manifest = pd.read_csv(split_manifest_path)

    if missing_cases:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for case in missing_cases:
                subset_size = case["subset_size"]
                fold_index = case["fold_index"]
                fold_total = case["fold_total"]
                train_index = case["train_index"]
                test_index = case["test_index"]
                print(
                    f"{args.observable}: subset={subset_size}, split={fold_index}/{fold_total}, "
                    f"train={len(train_index)}, test={len(test_index)}",
                    flush=True,
                )
                manifest_row = {
                    "subset_size": int(subset_size),
                    "fold_index": int(fold_index),
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "train_indices": " ".join(map(str, train_index)),
                    "test_indices": " ".join(map(str, test_index)),
                }
                method_results = _predict_pca_thresholds(
                    x_train=x_all[train_index],
                    y_train=y_fit[train_index],
                    x_test=x_all[test_index],
                    alpha_train=alpha[train_index] if alpha is not None else None,
                    thresholds=thresholds,
                    pca_scaling=args.pca_scaling,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    optimize_hyperparameters=args.optimize_hyperparameters,
                    fit_white_noise=args.fit_white_noise,
                )
                y_pred_bin, y_std_bin, bin_metadata = _predict_bin_by_bin_gp(
                    x_train=x_all[train_index],
                    y_train=y_fit[train_index],
                    x_test=x_all[test_index],
                    alpha_train=alpha[train_index] if alpha is not None else None,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    optimize_hyperparameters=args.optimize_hyperparameters,
                    fit_white_noise=args.fit_white_noise,
                )
                method_results["bin_by_bin"] = (y_pred_bin, y_std_bin, bin_metadata)
                split_prediction_rows = []
                for method, (y_pred, y_std, metadata) in method_results.items():
                    for test_position, sample_index in enumerate(test_index):
                        for bin_index, x_value in enumerate(x_plot):
                            split_prediction_rows.append(
                                {
                                    "observable_key": args.observable,
                                    "method": method,
                                    "subset_size": int(subset_size),
                                    "fold_index": int(fold_index),
                                    "sample_index": int(sample_index),
                                    "n_train": int(len(train_index)),
                                    "n_test": int(len(test_index)),
                                    "bin_index": int(bin_index),
                                    "x_plot": float(x_value),
                                    "y_true": float(y_plot[sample_index, bin_index]),
                                    "y_pred": float(y_pred[test_position, bin_index]),
                                    "y_std": float(y_std[test_position, bin_index]),
                                    "alpha_sigma": (
                                        np.nan
                                        if alpha_sigma is None
                                        else float(alpha_sigma[sample_index, bin_index])
                                    ),
                                    "pca_components": metadata.get("pca_components", np.nan),
                                    "pca_explained_variance": metadata.get("pca_explained_variance", np.nan),
                                }
                            )
                split_predictions = pd.DataFrame(split_prediction_rows)
                predictions = _replace_split_rows(
                    predictions,
                    split_predictions,
                    subset_size=subset_size,
                    fold_index=fold_index,
                )
                _write_csv_atomic(predictions, prediction_path)
                split_manifest = _replace_split_manifest_row(
                    split_manifest,
                    manifest_row,
                    subset_size=subset_size,
                    fold_index=fold_index,
                )
                _write_csv_atomic(split_manifest, split_manifest_path)
    elif not split_manifest_path.exists():
        split_manifest = pd.DataFrame(
            [
                {
                    "subset_size": int(case["subset_size"]),
                    "fold_index": int(case["fold_index"]),
                    "n_train": int(len(case["train_index"])),
                    "n_test": int(len(case["test_index"])),
                    "train_indices": " ".join(map(str, case["train_index"])),
                    "test_indices": " ".join(map(str, case["test_index"])),
                }
                for case in split_plan
            ]
        )
        _write_csv_atomic(split_manifest, split_manifest_path)

    metrics = _write_metrics(predictions, metrics_path)
    figures_dir = output_dir / "figures"
    _plot_training_size(
        metrics,
        highlighted_bins=highlighted_bins,
        training_rms_sigma_by_bin=None,
        output_path=figures_dir / "smf_z0_pca_threshold_rmse_r2_vs_training_size.png",
    )
    if not args.paper_figures_only:
        _plot_training_size(
            metrics,
            highlighted_bins=highlighted_bins,
            training_rms_sigma_by_bin=training_rms_sigma_by_bin,
            output_path=figures_dir / "smf_z0_pca_threshold_rmse_r2_vs_training_size_with_training_rms.png",
        )
    holdout_method = f"pca_{args.holdout_threshold:g}".replace(".", "p")
    subset_size_for_holdout = max(subset_sizes)
    if train_indices_for_holdout is None:
        split_path = output_dir / "smf_z0_pca_threshold_cv_splits.csv"
        if split_path.exists():
            split = pd.read_csv(split_path)
            row = split.loc[(split["subset_size"] == subset_size_for_holdout) & (split["fold_index"] == 1)].iloc[0]
            train_indices_for_holdout = np.fromstring(row["train_indices"], sep=" ", dtype=int)
        else:
            train_indices_for_holdout = np.arange(int(0.9 * subset_size_for_holdout), dtype=int)
    holdout_rows_for_examples = predictions.loc[
        (predictions["method"] == holdout_method)
        & (predictions["subset_size"] == subset_size_for_holdout)
        & (predictions["fold_index"] == 1)
    ].copy()
    selected_example_indices = _select_example_indices(
        rows=holdout_rows_for_examples,
        x_all=x_all,
        n_examples=args.n_example_curves,
        example_selection=args.example_selection,
    )
    _plot_heldout_curves(
        y_plot=y_plot,
        target_plot=target_plot,
        target_noise_plot=target_noise_plot,
        x_plot=x_plot,
        x_all=x_all,
        train_index=train_indices_for_holdout,
        predictions=predictions,
        method=holdout_method,
        subset_size=subset_size_for_holdout,
        fold_index=1,
        highlighted_bins=highlighted_bins,
        n_examples=args.n_example_curves,
        example_selection=args.example_selection,
        target_label=target_label,
        redshift_label=redshift_label,
        output_path=figures_dir / "smf_z0_pca99_holdout_curves_highlight_bins.png",
    )
    _plot_parity(
        predictions,
        method=holdout_method,
        subset_size=subset_size_for_holdout,
        highlighted_bins=highlighted_bins,
        target_plot=target_plot,
        target_label=target_label,
        point_fraction=0.25,
        point_alpha=0.48,
        errorbar_alpha=0.30,
        errorbar_lw=0.55,
        include_errorbars=True,
        output_path=figures_dir / "smf_z0_pca99_parity_highlight_bins_random_quarter.png",
    )
    if not args.paper_figures_only:
        _plot_parity(
            predictions,
            method=holdout_method,
            subset_size=subset_size_for_holdout,
            highlighted_bins=highlighted_bins,
            target_plot=target_plot,
            target_label=target_label,
            point_fraction=1.0,
            point_alpha=0.38,
            errorbar_alpha=0.22,
            errorbar_lw=0.55,
            include_errorbars=True,
            output_path=figures_dir / "smf_z0_pca99_parity_highlight_bins.png",
        )
        _plot_parity(
            predictions,
            method=holdout_method,
            subset_size=subset_size_for_holdout,
            highlighted_bins=highlighted_bins,
            target_plot=target_plot,
            target_label=target_label,
            point_fraction=1.0,
            point_alpha=0.28,
            errorbar_alpha=0.0,
            errorbar_lw=0.0,
            include_errorbars=False,
            output_path=figures_dir / "smf_z0_pca99_parity_highlight_bins_points_only.png",
        )
    summary = {
        "command": " ".join([sys.executable, *sys.argv]),
        "argv": [sys.executable, *sys.argv],
        "campaign_root": str(campaign_root),
        "hdf5_filename": args.hdf5_filename,
        "observable": args.observable,
        "analysis": config["analysis"],
        "subset_sizes": subset_sizes,
        "n_folds": args.n_folds,
        "validation_mode": args.validation_mode,
        "random_state": args.random_state,
        "pca_variance_thresholds": thresholds,
        "highlighted_x_requested": highlighted_x,
        "example_selection": args.example_selection,
        "n_example_curves": args.n_example_curves,
        "example_sample_indices": selected_example_indices.astype(int).tolist(),
        "example_evaluation_ids": samples.iloc[selected_example_indices]["evaluation_id"].tolist(),
        "highlighted_bins": [
            {"bin_index": int(bin_index), "x_plot": float(x_plot[bin_index])} for bin_index in highlighted_bins
        ],
        "highlighted_training_rms_sigma": (
            {
                str(bin_index): training_rms_sigma_by_bin[bin_index]
                for bin_index in highlighted_bins
            }
            if training_rms_sigma_by_bin is not None
            else None
        ),
        "training_metadata": training_metadata,
        "x_axis_label": str(attrs.get("xAxisLabel", "x")),
        "y_axis_label": str(attrs.get("yAxisLabel", "y")),
    }
    summary_path = output_dir / "smf_z0_pca_threshold_cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(prediction_path)
    print(metrics_path)
    print(summary_path)
    print(figures_dir)


if __name__ == "__main__":
    main()
