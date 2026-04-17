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

from galacticus_emu import (
    compute_metrics,
    fit_cv_predictions,
    fit_scaled_gp,
    sanitize_output_name,
    save_emulator_bundle,
    transform_to_prior_quantiles,
    trinity_parameter_specs,
)


DEFAULT_OUTPUT_COLUMNS = [
    "z0_mass_metallicity_abundance_oxygen_12logoh_residual_1",
    "z0_mass_metallicity_abundance_oxygen_12logoh_residual_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-validated GP diagnostics for the approximate z=0 MZR residual outputs."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-restarts-optimizer", type=int, default=1)
    parser.add_argument("--optimize-hyperparameters", action="store_true")
    parser.add_argument("--output-suffix", default="_mzr_residual_upper_cv")
    parser.add_argument("--output-column", action="append", dest="output_columns")
    return parser.parse_args()


def _load_mass_metallicity_table(campaign_root: Path) -> pd.DataFrame:
    table_path = campaign_root / "mass_metallicity_emulator_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Expected {table_path}; run extract_trinity_mass_metallicity_summary.py first."
        )
    return pd.read_csv(table_path)


def _make_parity_plot(
    observed: np.ndarray,
    predicted: np.ndarray,
    predicted_std: np.ndarray,
    output_names: list[str],
    output_path: Path,
) -> None:
    n_outputs = len(output_names)
    ncols = min(3, n_outputs)
    nrows = int(np.ceil(n_outputs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    flat_axes = list(axes.flat)
    for axis, output_name, y_true, y_pred, y_std in zip(flat_axes[:n_outputs], output_names, observed.T, predicted.T, predicted_std.T, strict=True):
        lower = min(float(np.min(y_true)), float(np.min(y_pred)))
        upper = max(float(np.max(y_true)), float(np.max(y_pred)))
        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=3.0, alpha=0.65, color="#2a6f97", ecolor="#9fbcd1")
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="#d1495b", linewidth=1.2)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_xlabel("Derived Galacticus quantity")
        axis.set_ylabel("GP CV prediction")
        axis.set_title(output_name, fontsize=9)
        axis.grid(alpha=0.2, linewidth=0.5)
    for axis in flat_axes[n_outputs:]:
        axis.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _make_residual_histogram(predicted: np.ndarray, observed: np.ndarray, output_names: list[str], output_path: Path) -> None:
    residuals = predicted - observed
    fig, axes = plt.subplots(1, len(output_names), figsize=(5.0 * len(output_names), 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, name, resid in zip(axes, output_names, residuals.T, strict=True):
        axis.hist(resid, bins=24, color="#457b9d", alpha=0.8)
        axis.axvline(0.0, color="#d1495b", linestyle="--", linewidth=1.2)
        axis.set_title(name, fontsize=9)
        axis.set_xlabel("CV residual")
        axis.set_ylabel("Count")
        axis.grid(alpha=0.2, linewidth=0.5)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")

    campaign_root = Path(args.campaign_root).resolve()
    emulator_root = campaign_root / "emulator"
    figures_root = campaign_root / "figures"
    models_root = emulator_root / "models"
    emulator_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    table = _load_mass_metallicity_table(campaign_root)
    parameter_specs = trinity_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    output_columns = args.output_columns or list(DEFAULT_OUTPUT_COLUMNS)

    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)

    cv_predictions = []
    cv_std_predictions = []
    metrics_rows = []
    model_summaries = []

    for output_name in output_columns:
        y = table[output_name].to_numpy(dtype=float)
        pred, pred_std = fit_cv_predictions(
            x=x,
            y=y,
            n_restarts_optimizer=args.n_restarts_optimizer,
            cv_folds=args.cv_folds,
            optimize_hyperparameters=args.optimize_hyperparameters,
        )
        cv_predictions.append(pred)
        cv_std_predictions.append(pred_std)
        metric_row = {"output": output_name, **compute_metrics(y, pred, pred_std)}
        metrics_rows.append(metric_row)

        model, y_mean, y_std = fit_scaled_gp(
            x=x,
            y=y,
            n_restarts_optimizer=args.n_restarts_optimizer,
            optimize_hyperparameters=args.optimize_hyperparameters,
        )
        model_summaries.append(
            {
                "output": output_name,
                "kernel_optimized": str(model.kernel_),
                "log_marginal_likelihood": float(model.log_marginal_likelihood(model.kernel_.theta)),
                "y_mean": y_mean,
                "y_std": y_std,
            }
        )
        model_filename = f"{sanitize_output_name(output_name)}{args.output_suffix}.joblib"
        save_emulator_bundle(
            models_root / model_filename,
            model=model,
            input_columns=input_columns,
            output_column=output_name,
            y_mean=y_mean,
            y_std=y_std,
            kernel_optimized=str(model.kernel_),
            log_marginal_likelihood=float(model.log_marginal_likelihood(model.kernel_.theta)),
            optimize_hyperparameters=args.optimize_hyperparameters,
            n_restarts_optimizer=args.n_restarts_optimizer,
            cv_folds=args.cv_folds,
        )

    observed_array = table[output_columns].to_numpy(dtype=float)
    cv_predictions_array = np.column_stack(cv_predictions)
    cv_std_predictions_array = np.column_stack(cv_std_predictions)

    predictions_table = table[["evaluation_id", *input_columns, *output_columns]].copy()
    for index, output_name in enumerate(output_columns):
        predictions_table[f"{output_name}_gp_cv_mean"] = cv_predictions_array[:, index]
        predictions_table[f"{output_name}_gp_cv_std"] = cv_std_predictions_array[:, index]
        predictions_table[f"{output_name}_gp_cv_residual"] = cv_predictions_array[:, index] - observed_array[:, index]

    metrics_path = emulator_root / f"gp_metrics_mass_metallicity{args.output_suffix}.csv"
    predictions_path = emulator_root / f"gp_cv_predictions_mass_metallicity{args.output_suffix}.csv"
    summary_path = emulator_root / f"gp_fit_summary_mass_metallicity{args.output_suffix}.json"
    parity_path = figures_root / f"gp_cv_parity_mass_metallicity{args.output_suffix}.png"
    residual_hist_path = figures_root / f"gp_cv_residual_hist_mass_metallicity{args.output_suffix}.png"

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    predictions_table.to_csv(predictions_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "input_columns": input_columns,
                "output_columns": output_columns,
                "cv_folds": args.cv_folds,
                "n_restarts_optimizer": args.n_restarts_optimizer,
                "optimize_hyperparameters": args.optimize_hyperparameters,
                "models_root": str(models_root),
                "models": model_summaries,
            },
            indent=2,
        )
        + "\n"
    )

    _make_parity_plot(observed_array, cv_predictions_array, cv_std_predictions_array, output_columns, parity_path)
    _make_residual_histogram(cv_predictions_array, observed_array, output_columns, residual_hist_path)

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(parity_path)
    print(residual_hist_path)


if __name__ == "__main__":
    main()
