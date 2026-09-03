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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp


DEFAULT_ANALYSIS = "massFunctionStellarTomczak2014ZFOURGEz0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a multi-parameter GP emulator to one SMF analysis using an 80/20-style "
            "train/test split and make holdout diagnostics."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--min-log10-y", type=float, default=-7.0)
    parser.add_argument("--n-example-curves", type=int, default=10)
    parser.add_argument("--overlay-ymin", type=float, default=-6.0)
    parser.add_argument("--overlay-ymax", type=float, default=-1.0)
    parser.add_argument("--replot-only", action="store_true")
    parser.add_argument("--output-prefix", default="smf_zfourge_z0_holdout_demo")
    parser.add_argument("--figures-dir-name", default="figures_smf_zfourge_z0_holdout_demo")
    parser.add_argument("--emulator-dir-name", default="emulator_smf_zfourge_z0_holdout_demo")
    return parser.parse_args()


def _quantile_columns(samples: pd.DataFrame) -> list[str]:
    columns = [column for column in samples.columns if column.endswith("_quantile")]
    if not columns:
        raise ValueError("No quantile columns found in samples.csv")
    return columns


def _load_campaign(
    campaign_root: Path,
    *,
    analysis: str,
    hdf5_filename: str,
    min_log10_y: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    samples = pd.read_csv(campaign_root / "samples.csv")
    quantile_columns = _quantile_columns(samples)
    x = samples[quantile_columns].to_numpy(dtype=float)

    mass_bins: np.ndarray | None = None
    target: np.ndarray | None = None
    target_covariance: np.ndarray | None = None
    y_rows = []
    noise_rows = []
    for sample in samples.itertuples(index=False):
        evaluation_id = sample.evaluation_id
        path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            current_mass_bins = np.asarray(group["massStellar"][...], dtype=float)
            current_target = np.asarray(group["massFunctionTarget"][...], dtype=float)
            current_target_covariance = np.asarray(group["massFunctionCovarianceTarget"][...], dtype=float)
            current_y = np.asarray(group["massFunction"][...], dtype=float)
            current_covariance = np.asarray(group["massFunctionCovariance"][...], dtype=float)

        if mass_bins is None:
            mass_bins = current_mass_bins
            target = current_target
            target_covariance = current_target_covariance
        else:
            if not np.allclose(current_mass_bins, mass_bins):
                raise ValueError(f"Mass bins differ for {evaluation_id}")
            if not np.allclose(current_target, target):
                raise ValueError(f"Target SMF differs for {evaluation_id}")
            if not np.allclose(current_target_covariance, target_covariance):
                raise ValueError(f"Target SMF covariance differs for {evaluation_id}")

        y_rows.append(current_y)
        noise_rows.append(np.sqrt(np.maximum(np.diag(current_covariance), 0.0)))

    assert mass_bins is not None
    assert target is not None
    assert target_covariance is not None
    y_linear = np.vstack(y_rows)
    positive = y_linear[y_linear > 0.0]
    if positive.size == 0:
        raise ValueError("SMF contains no positive values.")
    effective_floor_linear = 10.0 ** min_log10_y
    n_floored = int(np.sum(y_linear <= 0.0))
    safe_y_linear = np.where(y_linear > 0.0, y_linear, effective_floor_linear)
    y_log10_raw = np.log10(safe_y_linear)
    n_clipped = int(np.sum(y_log10_raw < min_log10_y))
    y_log10 = np.maximum(y_log10_raw, min_log10_y)
    noise_linear = np.vstack(noise_rows)
    noise_log10 = np.maximum(noise_linear / (safe_y_linear * np.log(10.0)), 1.0e-6)
    # Once a model value has been floored/clipped, the local linear-to-log error propagation
    # stops being meaningful for plotting. Suppress those point error bars in the curve overlay.
    noise_log10 = np.where(y_log10_raw > min_log10_y, noise_log10, np.nan)
    target_sigma_log10 = np.maximum(
        np.sqrt(np.maximum(np.diag(target_covariance), 0.0)) / (target * np.log(10.0)),
        1.0e-6,
    )
    metadata = {
        "effective_floor_linear": float(effective_floor_linear),
        "min_log10_y": float(min_log10_y),
        "n_floored": n_floored,
        "n_clipped": n_clipped,
    }
    return samples, x, mass_bins, np.log10(target), target_sigma_log10, y_log10, noise_log10, metadata


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    n_restarts_optimizer: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pred = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=float)
    pred_std = np.zeros_like(pred)
    kernels: list[str] = []
    for bin_index in range(y_train.shape[1]):
        model, y_mean, y_std = fit_scaled_gp(
            x_train,
            y_train[:, bin_index],
            n_restarts_optimizer=n_restarts_optimizer,
        )
        pred[:, bin_index], pred_std[:, bin_index] = predict_scaled_gp(model, y_mean, y_std, x_test)
        kernels.append(str(model.kernel_))
    return pred, pred_std, kernels


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, mass_bins: np.ndarray) -> pd.DataFrame:
    rows = []
    for bin_index in range(y_true.shape[1]):
        rows.append(
            {
                "bin": int(bin_index),
                "mass_stellar": float(mass_bins[bin_index]),
                "mass_stellar_log10": float(np.log10(mass_bins[bin_index])),
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


def _space_filling_examples(x_test: np.ndarray, n_examples: int) -> np.ndarray:
    if x_test.shape[0] <= n_examples:
        return np.arange(x_test.shape[0], dtype=int)
    order = np.argsort(x_test[:, 0])
    values = np.linspace(0, len(order) - 1, n_examples)
    return np.asarray(sorted({int(order[int(round(value))]) for value in values}), dtype=int)


def _plot_overlay(
    *,
    mass_bins: np.ndarray,
    target_log10: np.ndarray,
    target_sigma_log10: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_test_std: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    example_indices: np.ndarray,
    ymin: float,
    ymax: float,
    path: Path,
) -> None:
    x = np.log10(mass_bins)
    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)

    for curve in y_train:
        ax.plot(x, curve, color="0.65", alpha=0.10, lw=1.0, zorder=1)

    cmap = plt.get_cmap("tab10")
    for color_index, row_index in enumerate(example_indices):
        color = cmap(color_index % 10)
        lower = y_pred[row_index] - y_pred_std[row_index]
        upper = y_pred[row_index] + y_pred_std[row_index]
        ax.fill_between(x, lower, upper, color=color, alpha=0.18, zorder=2)
        ax.plot(x, y_pred[row_index], color=color, lw=2.0, zorder=3)
        ax.errorbar(
            x,
            y_test[row_index],
            yerr=y_test_std[row_index],
            fmt="o",
            ms=3.5,
            lw=1.0,
            capsize=2.0,
            color=color,
            alpha=0.95,
            zorder=5,
        )

    ax.plot(x, target_log10, color="k", lw=2.2, zorder=9)
    ax.errorbar(
        x,
        target_log10,
        yerr=target_sigma_log10,
        fmt="o",
        color="k",
        ms=4.3,
        lw=1.3,
        capsize=2.5,
        label="Tomczak+14 target",
        zorder=10,
    )

    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(\Phi / \mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$")
    ax.set_title("Low-z Tomczak SMF: training family, target, and held-out emulator checks")
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_parity(
    *,
    mass_bins: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    path: Path,
) -> None:
    n_bins = y_test.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), constrained_layout=True)
    axes_flat = np.ravel(axes)

    for bin_index in range(n_bins):
        axis = axes_flat[bin_index]
        observed = y_test[:, bin_index]
        predicted = y_pred[:, bin_index]
        lower = float(min(observed.min(), predicted.min()))
        upper = float(max(observed.max(), predicted.max()))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(observed, predicted, yerr=y_pred_std[:, bin_index], fmt="o", ms=3.0, alpha=0.75)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(rf"$\log_{{10}} M_\star={np.log10(mass_bins[bin_index]):.2f}$")
        axis.set_xlabel("Held-out Galacticus")
        axis.set_ylabel("GP prediction")
        axis.grid(alpha=0.2)

    for axis in axes_flat[n_bins:]:
        axis.axis("off")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_saved_predictions(predictions_path: Path, n_bins: int) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(predictions_path)
    evaluation_ids = frame["evaluation_id"].tolist()
    y_true = np.zeros((len(frame), n_bins), dtype=float)
    y_true_std = np.full((len(frame), n_bins), np.nan, dtype=float)
    y_pred = np.zeros((len(frame), n_bins), dtype=float)
    y_pred_std = np.zeros((len(frame), n_bins), dtype=float)
    for bin_index in range(n_bins):
        prefix = f"bin{bin_index}"
        y_true[:, bin_index] = frame[f"{prefix}_true_log10"].to_numpy(dtype=float)
        y_true_std[:, bin_index] = frame[f"{prefix}_true_std_log10"].to_numpy(dtype=float)
        y_pred[:, bin_index] = frame[f"{prefix}_pred_log10"].to_numpy(dtype=float)
        y_pred_std[:, bin_index] = frame[f"{prefix}_pred_std_log10"].to_numpy(dtype=float)
    return evaluation_ids, y_true, y_true_std, y_pred, y_pred_std


def _replot_from_saved(
    *,
    campaign_root: Path,
    analysis: str,
    hdf5_filename: str,
    min_log10_y: float,
    figures_dir: Path,
    emulator_dir: Path,
    output_prefix: str,
    n_example_curves: int,
    overlay_ymin: float,
    overlay_ymax: float,
) -> None:
    summary_path = emulator_dir / f"{output_prefix}_summary.json"
    predictions_path = emulator_dir / f"{output_prefix}_predictions.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    summary = json.loads(summary_path.read_text())
    (
        samples,
        _x,
        mass_bins,
        target_log10,
        target_sigma_log10,
        y_log10,
        y_log10_std,
        _transform_metadata,
    ) = _load_campaign(
        campaign_root,
        analysis=analysis,
        hdf5_filename=hdf5_filename,
        min_log10_y=min_log10_y,
    )
    sample_index = {evaluation_id: index for index, evaluation_id in enumerate(samples["evaluation_id"])}
    train_index = np.asarray([sample_index[evaluation_id] for evaluation_id in summary["train_ids"]], dtype=int)
    test_ids, y_true, y_true_std, y_pred, y_pred_std = _load_saved_predictions(predictions_path, len(mass_bins))
    test_global_index = np.asarray([sample_index[evaluation_id] for evaluation_id in test_ids], dtype=int)
    example_local_indices = _space_filling_examples(_x[test_global_index], n_example_curves)
    example_indices = np.asarray(example_local_indices, dtype=int)

    overlay_path = figures_dir / f"{output_prefix}_heldout_curves.png"
    _plot_overlay(
        mass_bins=mass_bins,
        target_log10=target_log10,
        target_sigma_log10=target_sigma_log10,
        y_train=y_log10[train_index],
        y_test=y_true,
        y_test_std=y_true_std,
        y_pred=y_pred,
        y_pred_std=y_pred_std,
        example_indices=example_indices,
        ymin=overlay_ymin,
        ymax=overlay_ymax,
        path=overlay_path,
    )

    parity_path = figures_dir / f"{output_prefix}_parity.png"
    _plot_parity(
        mass_bins=mass_bins,
        y_test=y_true,
        y_pred=y_pred,
        y_pred_std=y_pred_std,
        path=parity_path,
    )

    print(overlay_path)
    print(parity_path)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_dir = campaign_root / args.figures_dir_name
    emulator_dir = campaign_root / args.emulator_dir_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    emulator_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_only:
        _replot_from_saved(
            campaign_root=campaign_root,
            analysis=args.analysis,
            hdf5_filename=args.hdf5_filename,
            min_log10_y=args.min_log10_y,
            figures_dir=figures_dir,
            emulator_dir=emulator_dir,
            output_prefix=args.output_prefix,
            n_example_curves=args.n_example_curves,
            overlay_ymin=args.overlay_ymin,
            overlay_ymax=args.overlay_ymax,
        )
        return

    (
        samples,
        x,
        mass_bins,
        target_log10,
        target_sigma_log10,
        y_log10,
        y_log10_std,
        transform_metadata,
    ) = _load_campaign(
        campaign_root,
        analysis=args.analysis,
        hdf5_filename=args.hdf5_filename,
        min_log10_y=args.min_log10_y,
    )

    n_samples = x.shape[0]
    n_splits = int(round(1.0 / (1.0 - args.train_fraction)))
    if n_splits < 2:
        raise ValueError("--train-fraction must be less than 1")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    train_index, test_index = next(splitter.split(x))

    print(
        f"{args.analysis}: train {len(train_index)} evals / test {len(test_index)} evals "
        f"({len(train_index)/n_samples:.3f}/{len(test_index)/n_samples:.3f})"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        y_pred, y_pred_std, kernels = _fit_predict(
            x[train_index],
            y_log10[train_index],
            x[test_index],
            n_restarts_optimizer=args.n_restarts_optimizer,
        )

    metrics = _metrics(y_log10[test_index], y_pred, y_pred_std, mass_bins)
    metrics_path = emulator_dir / f"{args.output_prefix}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    example_indices = _space_filling_examples(x[test_index], args.n_example_curves)

    overlay_path = figures_dir / f"{args.output_prefix}_heldout_curves.png"
    _plot_overlay(
        mass_bins=mass_bins,
        target_log10=target_log10,
        target_sigma_log10=target_sigma_log10,
        y_train=y_log10[train_index],
        y_test=y_log10[test_index],
        y_test_std=y_log10_std[test_index],
        y_pred=y_pred,
        y_pred_std=y_pred_std,
        example_indices=example_indices,
        ymin=args.overlay_ymin,
        ymax=args.overlay_ymax,
        path=overlay_path,
    )

    parity_path = figures_dir / f"{args.output_prefix}_parity.png"
    _plot_parity(
        mass_bins=mass_bins,
        y_test=y_log10[test_index],
        y_pred=y_pred,
        y_pred_std=y_pred_std,
        path=parity_path,
    )

    prediction_rows = []
    for local_row, global_row in enumerate(test_index):
        row = {
            "evaluation_id": samples.iloc[global_row]["evaluation_id"],
            "is_example_curve": bool(local_row in set(example_indices.tolist())),
        }
        for bin_index, mass_value in enumerate(mass_bins):
            prefix = f"bin{bin_index}"
            row[f"{prefix}_mass_stellar"] = float(mass_value)
            row[f"{prefix}_true_log10"] = float(y_log10[global_row, bin_index])
            row[f"{prefix}_true_std_log10"] = float(y_log10_std[global_row, bin_index])
            row[f"{prefix}_pred_log10"] = float(y_pred[local_row, bin_index])
            row[f"{prefix}_pred_std_log10"] = float(y_pred_std[local_row, bin_index])
        prediction_rows.append(row)
    predictions_path = emulator_dir / f"{args.output_prefix}_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    summary = {
        "campaign_root": str(campaign_root),
        "analysis": args.analysis,
        "n_total_evaluations": int(n_samples),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "train_fraction": float(len(train_index) / n_samples),
        "test_fraction": float(len(test_index) / n_samples),
        "train_ids": samples.iloc[train_index]["evaluation_id"].tolist(),
        "test_ids": samples.iloc[test_index]["evaluation_id"].tolist(),
        "example_curve_ids": samples.iloc[test_index[example_indices]]["evaluation_id"].tolist(),
        "quantile_columns": _quantile_columns(samples),
        "kernels": kernels,
        "transform_metadata": transform_metadata,
    }
    summary_path = emulator_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(overlay_path)
    print(parity_path)


if __name__ == "__main__":
    main()
