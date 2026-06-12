from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick PCA holdout demo for sidecar emission-line LF tables.")
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--table-filename", required=True)
    parser.add_argument("--long-filename", required=True)
    parser.add_argument("--observable", required=True)
    parser.add_argument(
        "--sample-label",
        action="append",
        default=None,
        help=(
            "Restrict outputs to one or more sidecar sample labels, for example Z1, z1.47, or z2.23. "
            "The label is matched case-insensitively against the label embedded in wide-table column names."
        ),
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--figures-dir-name", required=True)
    parser.add_argument("--emulator-dir-name", required=True)
    parser.add_argument("--dust-draw-index", type=int, action="append", default=None)
    parser.add_argument("--oii-boost-log10", type=float, action="append", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-restarts-optimizer", type=int, default=1)
    parser.add_argument("--pca-components", type=int, default=4)
    parser.add_argument(
        "--pca-variance-threshold",
        type=float,
        default=None,
        help="Choose enough PCA components to explain this variance fraction, e.g. 0.99. Overrides --pca-components.",
    )
    parser.add_argument("--min-log10-lf", type=float, default=-8.0)
    parser.add_argument("--max-example-curves", type=int, default=5)
    parser.add_argument("--use-training-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-training-log10-sigma", type=float, default=1.0e-4)
    parser.add_argument("--max-training-log10-sigma", type=float, default=2.0)
    return parser.parse_args()


def _read_sidecar_tables(campaign_root: Path, sidecar_dir: str, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted((campaign_root / "evaluations").glob(f"*/{sidecar_dir}/{filename}")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No {sidecar_dir}/{filename} files found below {campaign_root}")
    return pd.concat(frames, ignore_index=True)


def _read_first_sidecar_table(campaign_root: Path, sidecar_dir: str, filename: str) -> pd.DataFrame:
    for path in sorted((campaign_root / "evaluations").glob(f"*/{sidecar_dir}/{filename}")):
        return pd.read_csv(path)
    raise FileNotFoundError(f"No {sidecar_dir}/{filename} files found below {campaign_root}")


def _first_galacticus_file(campaign_root: Path) -> Path | None:
    for path in sorted((campaign_root / "evaluations").glob("*/galacticus.hdf5")):
        return path
    return None


def _bin_sort_key(column: str) -> tuple[str, int]:
    match = re.match(r"(.+)_bin(\d+)_phi_mpc3_dex$", column)
    if match is None:
        raise ValueError(f"Could not parse LF column name: {column}")
    return match.group(1), int(match.group(2))


def _normalize_sample_label(label: str) -> str:
    return label.strip().lower()


def _output_columns(table: pd.DataFrame, observable: str, sample_labels: list[str] | None) -> list[str]:
    allowed_sample_labels = None
    if sample_labels is not None:
        allowed_sample_labels = {_normalize_sample_label(label) for label in sample_labels}

    columns = []
    for column in table.columns:
        if not column.startswith(f"{observable}_") or not column.endswith("_phi_mpc3_dex"):
            continue
        stem = column.removesuffix("_phi_mpc3_dex")
        match = re.match(rf"{re.escape(observable)}_(.+)_bin\d+$", stem)
        if match is None:
            continue
        sample_label = _normalize_sample_label(match.group(1))
        if allowed_sample_labels is not None and sample_label not in allowed_sample_labels:
            continue
        columns.append(column)
    if not columns:
        label_note = "" if sample_labels is None else f" and sample label(s) {sample_labels}"
        raise ValueError(f"No output columns found for observable {observable!r}{label_note}.")
    return sorted(columns, key=_bin_sort_key)


def _input_columns(table: pd.DataFrame) -> list[str]:
    columns = [column for column in table.columns if column.endswith("_quantile")]
    for column in ["delta_0", "delta_z", "delta_M", "delta_Mz", "attenuation_scatter", "oii_boost_log10"]:
        if column in table.columns:
            columns.append(column)
    return columns


def _log10_with_floor(values: np.ndarray, min_log10_lf: float) -> tuple[np.ndarray, float]:
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("No positive LF values found.")
    floor = 0.5 * float(np.min(positive))
    logged = np.log10(np.where(values > 0.0, values, floor))
    return np.maximum(logged, min_log10_lf), floor


def _alpha_log10(table: pd.DataFrame, columns: list[str], *, min_sigma: float, max_sigma: float) -> np.ndarray:
    alpha = np.zeros((len(table), len(columns)), dtype=float)
    for index, column in enumerate(columns):
        std_column = column.replace("_phi_mpc3_dex", "_phi_shot_noise_std_conservative")
        values = table[column].to_numpy(dtype=float)
        if std_column not in table.columns:
            alpha[:, index] = max_sigma
            continue
        std = table[std_column].to_numpy(dtype=float)
        sigma = np.full_like(values, max_sigma, dtype=float)
        valid = np.isfinite(values) & np.isfinite(std) & (values > 0.0) & (std >= 0.0)
        sigma[valid] = std[valid] / (values[valid] * np.log(10.0))
        alpha[:, index] = np.clip(sigma, min_sigma, max_sigma)
    return alpha


def _sample_labels(columns: list[str]) -> list[str]:
    labels = []
    for column in columns:
        stem = column.removesuffix("_phi_mpc3_dex")
        match = re.match(r".+_(z[^_]+)_bin\d+$", stem)
        labels.append(match.group(1) if match else stem)
    return labels


def _bin_labels(columns: list[str]) -> list[str]:
    labels = []
    for column in columns:
        stem = column.removesuffix("_phi_mpc3_dex")
        match = re.match(r".+_bin(\d+)$", stem)
        labels.append(f"bin {match.group(1)}" if match else stem)
    return labels


def _bin_indices(columns: list[str]) -> list[int]:
    indices = []
    for column in columns:
        stem = column.removesuffix("_phi_mpc3_dex")
        match = re.match(r".+_bin(\d+)$", stem)
        if match is None:
            raise ValueError(f"Could not parse bin index from column name: {column}")
        indices.append(int(match.group(1)))
    return indices


def _bin_metadata(long_table: pd.DataFrame, observable: str, sample_label: str, columns: list[str]) -> pd.DataFrame:
    normalized_label = _normalize_sample_label(sample_label)
    subset = long_table.loc[
        (long_table["observable"] == observable)
        & (long_table["sample_label"].map(lambda value: _normalize_sample_label(str(value))) == normalized_label)
    ].copy()
    if subset.empty:
        raise ValueError(f"No long-table rows found for observable={observable!r}, sample_label={sample_label!r}.")
    subset = subset.sort_values("bin_index").drop_duplicates("bin_index")
    subset = subset.set_index("bin_index")
    return subset.loc[_bin_indices(columns)].reset_index()


def _hdf5_halpha_target(campaign_root: Path, sample_label: str, x_plot: np.ndarray, min_log10_lf: float) -> tuple[np.ndarray, np.ndarray | None]:
    path = _first_galacticus_file(campaign_root)
    if path is None:
        return np.full_like(x_plot, np.nan, dtype=float), None
    analysis = f"luminosityFunctionHalphaSobral2013HiZELS{sample_label.upper()}"
    with h5py.File(path, "r") as handle:
        group_path = f"/analyses/{analysis}"
        if group_path not in handle:
            return np.full_like(x_plot, np.nan, dtype=float), None
        group = handle[group_path]
        target_x = np.log10(np.asarray(group["luminosity"][...], dtype=float))
        target_phi = np.asarray(group["luminosityFunctionTarget"][...], dtype=float) * np.log(10.0)
        target_std = None
        if "luminosityFunctionCovarianceTarget" in group:
            target_std = np.sqrt(np.maximum(np.diag(np.asarray(group["luminosityFunctionCovarianceTarget"][...])), 0.0))
            target_std = target_std * np.log(10.0)

    order = [int(np.argmin(np.abs(target_x - value))) for value in x_plot]
    matched_x = target_x[order]
    if not np.allclose(matched_x, x_plot, rtol=0.0, atol=1.0e-6):
        return np.full_like(x_plot, np.nan, dtype=float), None
    return _linear_to_log10(target_phi[order], min_log10_lf), _linear_std_to_log10(target_phi[order], target_std[order], min_log10_lf) if target_std is not None else None


def _khostovan_lf_table(observable: str) -> pd.DataFrame:
    rows: list[tuple[float, float, float, float, float]] = []
    if observable == "hbeta_oiii_khostovan":
        rows = [
            (0.84, 41.10, 0.10, -1.82, 0.02),
            (0.84, 41.30, 0.10, -2.04, 0.03),
            (0.84, 41.50, 0.10, -2.35, 0.04),
            (0.84, 41.70, 0.10, -2.61, 0.06),
            (0.84, 41.90, 0.10, -2.94, 0.08),
            (0.84, 42.10, 0.10, -3.17, 0.13),
            (0.84, 42.30, 0.10, -3.52, 0.20),
            (0.84, 42.50, 0.10, -4.12, 0.39),
            (1.42, 41.95, 0.15, -2.49, 0.03),
            (1.42, 42.25, 0.15, -3.14, 0.07),
            (1.42, 42.55, 0.15, -3.89, 0.19),
            (1.42, 42.85, 0.15, -4.64, 0.48),
            (2.23, 42.60, 0.075, -3.08, 0.06),
            (2.23, 42.75, 0.075, -3.14, 0.07),
            (2.23, 42.90, 0.075, -3.65, 0.13),
            (2.23, 43.05, 0.075, -4.26, 0.29),
            (3.24, 42.65, 0.075, -3.17, 0.07),
            (3.24, 42.80, 0.075, -3.26, 0.09),
            (3.24, 42.95, 0.075, -3.55, 0.13),
            (3.24, 43.10, 0.075, -4.17, 0.27),
        ]
    elif observable == "oii_khostovan":
        rows = [
            (1.47, 41.65, 0.075, -2.08, 0.02),
            (1.47, 41.80, 0.075, -2.28, 0.03),
            (1.47, 41.95, 0.075, -2.46, 0.04),
            (1.47, 42.10, 0.075, -2.69, 0.06),
            (1.47, 42.25, 0.075, -3.05, 0.10),
            (1.47, 42.40, 0.075, -3.55, 0.15),
            (1.47, 42.55, 0.075, -4.23, 0.28),
            (2.25, 42.45, 0.10, -2.77, 0.05),
            (2.25, 42.65, 0.10, -3.15, 0.08),
            (2.25, 42.85, 0.10, -4.46, 0.35),
            (3.34, 43.05, 0.050, -3.86, 0.17),
            (3.34, 43.15, 0.075, -3.92, 0.24),
            (3.34, 43.30, 0.075, -4.87, 0.48),
        ]
    else:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["redshift", "log10_luminosity_center", "half_width", "log10_phi", "log10_phi_std"])


def _khostovan_target(observable: str, sample_label: str, x_plot: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    table = _khostovan_lf_table(observable)
    if table.empty:
        return np.full_like(x_plot, np.nan, dtype=float), None
    redshift = float(sample_label.removeprefix("z"))
    subset = table.loc[np.isclose(table["redshift"].to_numpy(dtype=float), redshift)].copy()
    if subset.empty:
        return np.full_like(x_plot, np.nan, dtype=float), None
    subset = subset.sort_values("log10_luminosity_center")
    target_x = subset["log10_luminosity_center"].to_numpy(dtype=float)
    order = [int(np.argmin(np.abs(target_x - value))) for value in x_plot]
    if not np.allclose(target_x[order], x_plot, rtol=0.0, atol=1.0e-6):
        return np.full_like(x_plot, np.nan, dtype=float), None
    return subset["log10_phi"].to_numpy(dtype=float)[order], subset["log10_phi_std"].to_numpy(dtype=float)[order]


def _target_curve(
    campaign_root: Path,
    observable: str,
    sample_label: str | None,
    x_plot: np.ndarray,
    min_log10_lf: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if sample_label is None:
        return np.full_like(x_plot, np.nan, dtype=float), None
    if observable == "halpha_sobral":
        return _hdf5_halpha_target(campaign_root, sample_label, x_plot, min_log10_lf)
    if observable in {"hbeta_oiii_khostovan", "oii_khostovan"}:
        return _khostovan_target(observable, sample_label, x_plot)
    return np.full_like(x_plot, np.nan, dtype=float), None


def _choose_pca_components(y_scaled: np.ndarray, requested: int, threshold: float | None) -> int:
    max_components = min(y_scaled.shape[0], y_scaled.shape[1])
    if threshold is None:
        return min(int(requested), max_components)
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"--pca-variance-threshold must be in (0, 1], got {threshold}")
    probe = PCA(n_components=max_components)
    probe.fit(y_scaled)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, threshold) + 1)


def _linear_to_log10(values: np.ndarray, min_log10_lf: float) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    floor = 0.5 * float(np.min(positive))
    return np.maximum(np.log10(np.where(values > 0.0, values, floor)), min_log10_lf)


def _linear_std_to_log10(values: np.ndarray, std: np.ndarray, min_log10_lf: float) -> np.ndarray:
    logged = _linear_to_log10(values, min_log10_lf)
    safe_values = 10.0**logged
    sigma = np.full_like(values, np.nan, dtype=float)
    valid = np.isfinite(std) & np.isfinite(safe_values) & (safe_values > 0.0) & (std >= 0.0)
    sigma[valid] = std[valid] / (safe_values[valid] * np.log(10.0))
    return sigma


def _plot_parity_by_bin(
    *,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    target_plot: np.ndarray | None,
    columns: list[str],
    metrics: pd.DataFrame,
    observable: str,
    path: Path,
) -> None:
    bin_labels = _bin_labels(columns)
    n_bins = y_test.shape[1]
    ncols = min(4, n_bins)
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for bin_index, (axis, column, bin_label) in enumerate(zip(axes_flat, columns, bin_labels, strict=False)):
        observed = y_test[:, bin_index]
        predicted = y_pred[:, bin_index]
        predicted_std = y_pred_std[:, bin_index]
        valid = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(predicted_std)
        if not np.any(valid):
            axis.text(0.5, 0.5, "no valid held-out data", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(bin_label)
            axis.set_axis_off()
            continue

        observed_valid = observed[valid]
        predicted_valid = predicted[valid]
        predicted_std_valid = predicted_std[valid]
        target_value = (
            float(target_plot[bin_index])
            if target_plot is not None and bin_index < len(target_plot) and np.isfinite(target_plot[bin_index])
            else None
        )
        lower_candidates = [float(observed_valid.min()), float(predicted_valid.min())]
        upper_candidates = [float(observed_valid.max()), float(predicted_valid.max())]
        if target_value is not None:
            lower_candidates.append(target_value)
            upper_candidates.append(target_value)
        lower = min(lower_candidates)
        upper = max(upper_candidates)
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(
            observed_valid,
            predicted_valid,
            yerr=predicted_std_valid,
            fmt="o",
            ms=3.0,
            alpha=0.6,
            elinewidth=0.8,
            capsize=0,
        )
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        if target_value is not None:
            axis.scatter(
                [target_value],
                [target_value],
                marker="*",
                s=110,
                color="k",
                edgecolor="white",
                linewidth=0.45,
                zorder=8,
            )
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        metric_row = metrics.loc[metrics["column"] == column]
        if metric_row.empty:
            axis.set_title(bin_label)
        else:
            rmse = float(metric_row["rmse"].iloc[0])
            r2 = float(metric_row["r2"].iloc[0])
            axis.set_title(f"{bin_label}\nRMSE={rmse:.3f}, R2={r2:.3f}")
        axis.set_xlabel("held-out log10 Phi")
        axis.set_ylabel("predicted log10 Phi")
        axis.grid(alpha=0.2)

    for axis in axes_flat[n_bins:]:
        axis.axis("off")

    fig.suptitle(observable)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pca_modes(
    *,
    x_plot: np.ndarray,
    pca: PCA,
    observable: str,
    pca_scaling: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    axes[0].plot(np.arange(1, len(cumulative) + 1), cumulative, "o-")
    axes[0].set_xlabel("PCA component")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.22)

    for component_index, component in enumerate(pca.components_, start=1):
        axes[1].plot(x_plot, component, label=f"PC{component_index}")
    axes[1].set_xlabel("log10 line luminosity [erg/s]")
    axes[1].set_ylabel("scaled PCA loading" if pca_scaling == "standardized" else "PCA loading")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle(f"PCA decomposition of {observable} ({pca_scaling})")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _robust_heldout_ylim(
    *,
    target_plot: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
) -> tuple[float, float] | None:
    values = [
        np.asarray(target_plot, dtype=float).ravel(),
        np.asarray(y_train, dtype=float).ravel(),
        np.asarray(y_test, dtype=float).ravel(),
        np.asarray(y_pred, dtype=float).ravel(),
        np.asarray(y_pred - y_pred_std, dtype=float).ravel(),
        np.asarray(y_pred + y_pred_std, dtype=float).ravel(),
    ]
    finite = np.concatenate([array[np.isfinite(array)] for array in values])
    if finite.size == 0:
        return None
    lower, upper = np.nanpercentile(finite, [0.5, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if upper <= lower:
        lower -= 0.5
        upper += 0.5
    margin = max(0.08 * (upper - lower), 0.15)
    return float(lower - margin), float(upper + margin)


def _plot_heldout_curves(
    *,
    x_plot: np.ndarray,
    target_plot: np.ndarray,
    target_std_plot: np.ndarray | None,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_test_std: np.ndarray | None,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    max_example_curves: int,
    observable: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    for curve in y_train:
        ax.plot(x_plot, curve, color="0.65", alpha=0.10, lw=1.0, zorder=1)

    n_examples = min(max_example_curves, y_test.shape[0])
    example_indices = np.linspace(0, y_test.shape[0] - 1, n_examples, dtype=int) if n_examples else np.array([], dtype=int)
    cmap = plt.get_cmap("tab10")
    x_spacing = np.diff(x_plot[np.isfinite(x_plot)])
    x_spacing = x_spacing[x_spacing > 0.0]
    offset_half_width = 0.15 * float(np.median(x_spacing)) if x_spacing.size else 0.0
    x_offsets = np.linspace(-offset_half_width, offset_half_width, len(example_indices)) if len(example_indices) else []

    for color_index, row_index in enumerate(example_indices):
        color = cmap(color_index % 10)
        lower = y_pred[row_index] - y_pred_std[row_index]
        upper = y_pred[row_index] + y_pred_std[row_index]
        ax.fill_between(x_plot, lower, upper, color=color, alpha=0.18, zorder=2)
        ax.plot(x_plot, y_pred[row_index], color=color, lw=2.0, zorder=3)

        observed = y_test[row_index]
        valid = np.isfinite(observed)
        x_observed = x_plot[valid] + x_offsets[color_index]
        if y_test_std is None:
            ax.scatter(x_observed, observed[valid], color=color, s=18, alpha=0.95, zorder=5)
        else:
            observed_std = y_test_std[row_index]
            valid = valid & np.isfinite(observed_std) & (observed_std >= 0.0)
            x_observed = x_plot[valid] + x_offsets[color_index]
            ax.errorbar(
                x_observed,
                observed[valid],
                yerr=observed_std[valid],
                fmt="o",
                ms=3.5,
                lw=1.0,
                capsize=2.0,
                color=color,
                alpha=0.95,
                zorder=5,
            )

    valid_target = np.isfinite(target_plot)
    if np.any(valid_target):
        ax.plot(x_plot[valid_target], target_plot[valid_target], color="k", lw=2.2, zorder=9)
        if target_std_plot is None:
            ax.scatter(x_plot[valid_target], target_plot[valid_target], color="k", s=22, zorder=10)
        else:
            valid_target = valid_target & np.isfinite(target_std_plot) & (target_std_plot >= 0.0)
            ax.errorbar(
                x_plot[valid_target],
                target_plot[valid_target],
                yerr=target_std_plot[valid_target],
                fmt="o",
                color="k",
                ms=4.3,
                lw=1.3,
                capsize=2.5,
                zorder=10,
            )

    ax.set_xlabel("log10 line luminosity [erg/s]")
    ax.set_ylabel("log10 Phi [Mpc^-3 dex^-1]")
    ax.set_title(f"{observable} held-out examples")
    ax.grid(alpha=0.2)
    y_limits = _robust_heldout_ylim(
        target_plot=target_plot,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_std=y_pred_std,
    )
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_dir = campaign_root / args.figures_dir_name
    emulator_dir = campaign_root / args.emulator_dir_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    emulator_dir.mkdir(parents=True, exist_ok=True)

    table = _read_sidecar_tables(campaign_root, args.sidecar_dir, args.table_filename)
    long_table = _read_first_sidecar_table(campaign_root, args.sidecar_dir, args.long_filename)
    if args.dust_draw_index is not None:
        table = table.loc[table["dust_draw_index"].isin(args.dust_draw_index)].reset_index(drop=True)
    if args.oii_boost_log10 is not None:
        boost_values = np.asarray(args.oii_boost_log10, dtype=float)
        table = table.loc[
            np.any(np.isclose(table["oii_boost_log10"].to_numpy(dtype=float)[:, None], boost_values[None, :]), axis=1)
        ].reset_index(drop=True)
    if table.empty:
        raise ValueError("No rows remain after filtering.")

    output_columns = _output_columns(table, args.observable, args.sample_label)
    sample_labels = _sample_labels(output_columns)
    sample_label = sample_labels[0] if sample_labels and all(label == sample_labels[0] for label in sample_labels) else None
    bin_metadata = _bin_metadata(long_table, args.observable, sample_label, output_columns) if sample_label is not None else None
    x_plot = (
        bin_metadata["log10_luminosity_center"].to_numpy(dtype=float)
        if bin_metadata is not None
        else np.arange(len(output_columns), dtype=float)
    )
    target_plot, target_std_plot = _target_curve(campaign_root, args.observable, sample_label, x_plot, args.min_log10_lf)
    input_columns = _input_columns(table)
    x = table[input_columns].to_numpy(dtype=float)
    y_linear = table[output_columns].to_numpy(dtype=float)
    y, floor = _log10_with_floor(y_linear, args.min_log10_lf)
    alpha = None
    if args.use_training_alpha:
        alpha_sigma = _alpha_log10(
            table,
            output_columns,
            min_sigma=args.min_training_log10_sigma,
            max_sigma=args.max_training_log10_sigma,
        )
        alpha = alpha_sigma**2

    groups = table["evaluation_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, train_size=args.train_fraction, random_state=args.random_state)
    train_index, test_index = next(splitter.split(x, y, groups=groups))
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    alpha_train = alpha[train_index] if alpha is not None else None

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    n_components = _choose_pca_components(y_train_scaled, args.pca_components, args.pca_variance_threshold)
    pca = PCA(n_components=n_components)
    coefficients = pca.fit_transform(y_train_scaled)

    coeff_pred = np.zeros((x_test.shape[0], n_components), dtype=float)
    coeff_var = np.zeros_like(coeff_pred)
    kernels = []
    component_alpha = None
    if alpha_train is not None:
        component_alpha = alpha_train @ (pca.components_[:n_components].T ** 2)
    for component in range(n_components):
        print(f"{args.observable}: fitting PCA component {component + 1}/{n_components}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model, y_mean, y_std = fit_scaled_gp(
                x_train,
                coefficients[:, component],
                n_restarts_optimizer=args.n_restarts_optimizer,
                optimize_hyperparameters=True,
                alpha=None if component_alpha is None else component_alpha[:, component],
                fit_white_noise=False,
            )
        pred, std = predict_scaled_gp(model, y_mean, y_std, x_test)
        coeff_pred[:, component] = pred
        coeff_var[:, component] = std**2
        kernels.append(str(model.kernel_))

    y_pred_scaled = pca.inverse_transform(coeff_pred)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_pred_var_scaled = coeff_var @ (pca.components_[:n_components] ** 2)
    y_pred_std = np.sqrt(np.maximum(y_pred_var_scaled, 0.0)) * scaler.scale_

    rows = []
    for j, column in enumerate(output_columns):
        finite = np.isfinite(y_test[:, j]) & np.isfinite(y_pred[:, j])
        rows.append(
            {
                "column": column,
                "rmse": float(np.sqrt(mean_squared_error(y_test[finite, j], y_pred[finite, j]))),
                "r2": float(r2_score(y_test[finite, j], y_pred[finite, j])) if np.count_nonzero(finite) > 1 else np.nan,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(emulator_dir / f"{args.output_prefix}_metrics.csv", index=False)

    prediction_rows = []
    for row_index, source_index in enumerate(test_index):
        base = {
            "evaluation_id": table.iloc[source_index]["evaluation_id"],
            "dust_draw_index": int(table.iloc[source_index]["dust_draw_index"]),
        }
        if "oii_boost_log10" in table.columns:
            base["oii_boost_log10"] = float(table.iloc[source_index]["oii_boost_log10"])
        for j, column in enumerate(output_columns):
            prediction_rows.append(
                {
                    **base,
                    "column": column,
                    "y_true": float(y_test[row_index, j]),
                    "y_pred": float(y_pred[row_index, j]),
                    "y_pred_std": float(y_pred_std[row_index, j]),
                }
            )
    pd.DataFrame(prediction_rows).to_csv(emulator_dir / f"{args.output_prefix}_predictions.csv", index=False)

    summary = {
        "observable": args.observable,
        "sample_labels": args.sample_label,
        "sidecar_dir": args.sidecar_dir,
        "n_rows": int(len(table)),
        "n_unique_evaluations": int(pd.Series(groups).nunique()),
        "n_train_rows": int(len(train_index)),
        "n_test_rows": int(len(test_index)),
        "input_columns": input_columns,
        "output_columns": output_columns,
        "x_plot": x_plot.tolist(),
        "target_log10_phi": target_plot.tolist(),
        "target_log10_phi_std": None if target_std_plot is None else target_std_plot.tolist(),
        "log10_floor_linear": floor,
        "pca_components_requested": int(args.pca_components),
        "pca_variance_threshold": None if args.pca_variance_threshold is None else float(args.pca_variance_threshold),
        "pca_components": int(n_components),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_ratio_cumulative": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "kernels": kernels,
    }
    (emulator_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _plot_parity_by_bin(
        y_test=y_test,
        y_pred=y_pred,
        y_pred_std=y_pred_std,
        target_plot=target_plot,
        columns=output_columns,
        metrics=metrics,
        observable=args.observable,
        path=figures_dir / f"{args.output_prefix}_parity.png",
    )

    _plot_pca_modes(
        x_plot=x_plot,
        pca=pca,
        observable=args.observable if sample_label is None else f"{args.observable} {sample_label}",
        pca_scaling="standardized",
        path=figures_dir / f"{args.output_prefix}_modes.png",
    )

    _plot_heldout_curves(
        x_plot=x_plot,
        target_plot=target_plot,
        target_std_plot=target_std_plot,
        y_train=y_train,
        y_test=y_test,
        y_test_std=alpha_sigma[test_index] if alpha is not None else None,
        y_pred=y_pred,
        y_pred_std=y_pred_std,
        max_example_curves=args.max_example_curves,
        observable=args.observable if sample_label is None else f"{args.observable} {sample_label}",
        path=figures_dir / f"{args.output_prefix}_heldout_curves.png",
    )

    print(emulator_dir / f"{args.output_prefix}_metrics.csv")
    print(emulator_dir / f"{args.output_prefix}_predictions.csv")
    print(emulator_dir / f"{args.output_prefix}_summary.json")
    print(figures_dir / f"{args.output_prefix}_parity.png")
    print(figures_dir / f"{args.output_prefix}_modes.png")
    print(figures_dir / f"{args.output_prefix}_heldout_curves.png")


if __name__ == "__main__":
    main()
