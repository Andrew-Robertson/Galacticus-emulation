from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu import (
    compute_metrics,
    default_parameter_specs,
    fit_cv_predictions,
    fit_scaled_gp,
    transform_to_prior_quantiles,
)


MEAN_OUTPUT_COLUMNS = [
    "mass_stellar_log10_0",
    "mass_stellar_log10_1",
    "mass_stellar_log10_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a rough scikit-learn GP emulator to the mean SHMR outputs."
    )
    parser.add_argument("campaign_root", help="Path to a completed campaign directory.")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation. Use a value >= n_samples for leave-one-out.",
    )
    parser.add_argument(
        "--n-restarts-optimizer",
        type=int,
        default=3,
        help="Number of optimizer restarts for each GP fit.",
    )
    return parser.parse_args()


def _load_emulator_table(campaign_root: Path) -> pd.DataFrame:
    emulator_table_path = campaign_root / "emulator_table.csv"
    if emulator_table_path.exists():
        return pd.read_csv(emulator_table_path)
    samples = pd.read_csv(campaign_root / "samples.csv")
    summary = pd.read_csv(campaign_root / "summary.csv")
    merged = samples.merge(summary, on="evaluation_id", how="inner", validate="one_to_one")
    merged.to_csv(emulator_table_path, index=False)
    return merged


def _make_parity_plot(
    observed: np.ndarray,
    predicted: np.ndarray,
    predicted_std: np.ndarray,
    output_names: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(output_names), figsize=(5.0 * len(output_names), 4.4), constrained_layout=True)
    if len(output_names) == 1:
        axes = [axes]

    for axis, output_name, y_true, y_pred, y_std in zip(axes, output_names, observed.T, predicted.T, predicted_std.T, strict=True):
        lower = min(float(np.min(y_true)), float(np.min(y_pred)))
        upper = max(float(np.max(y_true)), float(np.max(y_pred)))
        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=5, alpha=0.8, color="#2a6f97", ecolor="#9fbcd1")
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="#d1495b", linewidth=1.5)
        axis.set_xlabel("Galacticus")
        axis.set_ylabel("GP CV prediction")
        axis.set_title(output_name)
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / "figures"
    emulator_root = campaign_root / "emulator"
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    table = _load_emulator_table(campaign_root)
    parameter_specs = default_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)

    cv_predictions = []
    cv_std_predictions = []
    metrics_rows = []
    model_summaries = []

    for output_name in MEAN_OUTPUT_COLUMNS:
        y = table[output_name].to_numpy(dtype=float)
        pred, pred_std = fit_cv_predictions(
            x=x,
            y=y,
            n_restarts_optimizer=args.n_restarts_optimizer,
            cv_folds=args.cv_folds,
        )
        cv_predictions.append(pred)
        cv_std_predictions.append(pred_std)

        metric_values = compute_metrics(y, pred, pred_std)
        metrics_rows.append(
            {
                "output": output_name,
                **metric_values,
            }
        )

        model, y_mean, y_std = fit_scaled_gp(
            x=x,
            y=y,
            n_restarts_optimizer=args.n_restarts_optimizer,
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

    cv_predictions_array = np.column_stack(cv_predictions)
    cv_std_predictions_array = np.column_stack(cv_std_predictions)
    observed_array = table[MEAN_OUTPUT_COLUMNS].to_numpy(dtype=float)

    predictions_table = table[["evaluation_id", *input_columns, *MEAN_OUTPUT_COLUMNS]].copy()
    for index, output_name in enumerate(MEAN_OUTPUT_COLUMNS):
        predictions_table[f"{output_name}_gp_cv_mean"] = cv_predictions_array[:, index]
        predictions_table[f"{output_name}_gp_cv_std"] = cv_std_predictions_array[:, index]
        predictions_table[f"{output_name}_gp_cv_residual"] = cv_predictions_array[:, index] - observed_array[:, index]
    predictions_path = emulator_root / "gp_cv_predictions_mean.csv"
    predictions_table.to_csv(predictions_path, index=False)

    metrics_path = emulator_root / "gp_metrics_mean.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    summary_path = emulator_root / "gp_fit_summary_mean.json"
    summary_path.write_text(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "input_columns": input_columns,
                "output_columns": MEAN_OUTPUT_COLUMNS,
                "cv_folds": args.cv_folds,
                "n_restarts_optimizer": args.n_restarts_optimizer,
                "models": model_summaries,
            },
            indent=2,
        )
        + "\n"
    )

    parity_path = figures_root / "gp_cv_parity_mean.png"
    _make_parity_plot(
        observed=observed_array,
        predicted=cv_predictions_array,
        predicted_std=cv_std_predictions_array,
        output_names=MEAN_OUTPUT_COLUMNS,
        output_path=parity_path,
    )

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(parity_path)


if __name__ == "__main__":
    main()
