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
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit direct grouped-CV GP emulators for all Halpha LF bins at one Sobral redshift "
            "and write parity/metrics similar to the PCA workflow."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--sobral-label", default="Z1", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--train-draws-per-eval", type=int, default=8)
    parser.add_argument("--n-restarts-optimizer", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--figures-dir-name", default="figures_halpha_direct")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_direct")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


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
        kernel_summaries = []
        for bin_index in range(y.shape[1]):
            if fold_index == 1:
                print(
                    f"starting {args.sobral_label} bin {bin_index}",
                    flush=True,
                )
            print(
                f"{args.sobral_label} bin {bin_index}: fold {fold_index}/{args.n_splits}, "
                f"hold out {len(heldout)} evals / {len(test_index)} rows",
                flush=True,
            )
            model, y_mean, y_std = _fit_scaled_gp(
                x[train_index],
                y[train_index, bin_index],
                n_restarts_optimizer=args.n_restarts_optimizer,
                random_state=args.random_state,
            )
            pred[test_index, bin_index], pred_std[test_index, bin_index] = _predict_scaled_gp(
                model,
                y_mean,
                y_std,
                x[test_index],
            )
            kernel_summaries.append(str(model.kernel_))
            if fold_index == args.n_splits:
                print(
                    f"finished {args.sobral_label} bin {bin_index}",
                    flush=True,
                )
        fold_metadata.append(
            {
                "fold_index": fold_index,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "heldout_evaluations": heldout.tolist(),
                "kernel_summaries": kernel_summaries,
            }
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

    summary = {
        "campaign_root": str(campaign_root),
        "sobral_label": args.sobral_label,
        "n_splits": args.n_splits,
        "train_draws_per_eval": args.train_draws_per_eval,
        "n_restarts_optimizer": args.n_restarts_optimizer,
        "min_log10_lf": args.min_log10_lf,
        "max_abs_delta": args.max_abs_delta,
        "max_attenuation_scatter": args.max_attenuation_scatter,
        "log10_floor_linear": floor,
        "n_floored_values": n_floored,
        "n_clipped_values": n_clipped,
        "input_transform": "prior_quantiles",
        "fold_metadata": fold_metadata,
        "luminosity_centers": centers.tolist(),
    }
    summary_path = emulator_root / f"gp_cv_metadata_{output_prefix}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    parity_path = figures_root / f"gp_cv_parity_{output_prefix}.png"
    _plot_parity(y, pred, pred_std, centers, parity_path)

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(parity_path)


if __name__ == "__main__":
    main()
