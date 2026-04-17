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
    predict_scaled_gp,
    transform_to_prior_quantiles,
)


MEAN_OUTPUT_COLUMNS = [
    "mass_stellar_log10_0",
    "mass_stellar_log10_1",
    "mass_stellar_log10_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nested GP subset sizes using a larger completed campaign."
    )
    parser.add_argument("campaign_root", help="Path to a completed campaign directory.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[128, 256, 512],
        help="Nested training subset sizes to compare.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-restarts-optimizer", type=int, default=1)
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
    ylabel: str,
    axis_limits: list[tuple[float, float]],
) -> None:
    fig, axes = plt.subplots(1, len(output_names), figsize=(5.0 * len(output_names), 4.4), constrained_layout=True)
    if len(output_names) == 1:
        axes = [axes]

    for axis, output_name, y_true, y_pred, y_std, (lower, upper) in zip(
        axes,
        output_names,
        observed.T,
        predicted.T,
        predicted_std.T,
        axis_limits,
        strict=True,
    ):
        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=4, alpha=0.65, color="#2a6f97", ecolor="#9fbcd1")
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="#d1495b", linewidth=1.5)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_xlabel("Galacticus")
        axis.set_ylabel(ylabel)
        axis.set_title(output_name)
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _greedy_maximin_order(x: np.ndarray) -> np.ndarray:
    n_samples = x.shape[0]
    remaining = np.ones(n_samples, dtype=bool)
    center = np.full(x.shape[1], 0.5)
    first_index = int(np.argmin(np.sum((x - center) ** 2, axis=1)))
    order = [first_index]
    remaining[first_index] = False
    min_dist_sq = np.sum((x - x[first_index]) ** 2, axis=1)

    for _ in range(1, n_samples):
        candidate_indices = np.where(remaining)[0]
        next_index = int(candidate_indices[np.argmax(min_dist_sq[candidate_indices])])
        order.append(next_index)
        remaining[next_index] = False
        dist_sq = np.sum((x - x[next_index]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    return np.asarray(order, dtype=int)


def _plot_metric_vs_size(metrics_df: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for output_name, group in metrics_df.groupby("output"):
        ordered = group.sort_values("train_size")
        ax.plot(ordered["train_size"], ordered[metric_name], marker="o", linewidth=2.0, label=output_name)
    ax.set_xlabel("Training subset size")
    ax.set_ylabel(metric_name)
    ax.set_xscale("log", base=2)
    train_sizes = sorted(metrics_df["train_size"].unique())
    ax.set_xticks(train_sizes, [str(size) for size in train_sizes])
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    emulator_root = campaign_root / "emulator"
    figures_root = campaign_root / "figures"
    emulator_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    table = _load_emulator_table(campaign_root)
    parameter_specs = default_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)

    order = _greedy_maximin_order(x)
    ordered_table = table.iloc[order].reset_index(drop=True)
    order_path = emulator_root / "nested_subset_order.csv"
    pd.DataFrame(
        {
            "rank": np.arange(len(order)),
            "ordered_index": order,
            "evaluation_id": table.iloc[order]["evaluation_id"].to_numpy(),
        }
    ).to_csv(order_path, index=False)

    requested_sizes = sorted(set(args.sizes))
    if requested_sizes[-1] != len(table):
        requested_sizes.append(len(table))
    requested_sizes = [size for size in requested_sizes if 0 < size <= len(table)]
    axis_limits = [
        (
            float(table[output_name].min()),
            float(table[output_name].max()),
        )
        for output_name in MEAN_OUTPUT_COLUMNS
    ]

    metrics_rows: list[dict[str, float | int | str]] = []
    prediction_tables: list[pd.DataFrame] = []
    subset_summary: list[dict[str, object]] = []

    for train_size in requested_sizes:
        train_indices = order[:train_size]
        train_table = table.iloc[train_indices].reset_index(drop=True)
        x_train = x[train_indices]

        if train_size < len(table):
            test_indices = order[train_size:]
            test_table = table.iloc[test_indices].reset_index(drop=True)
            x_test = x[test_indices]
            prediction_mode = "holdout"
        else:
            test_indices = order[:train_size]
            test_table = train_table.copy()
            x_test = x_train
            prediction_mode = "cross_validation"

        output_predictions = []
        output_std_predictions = []
        prediction_frame = test_table[["evaluation_id", *input_columns, *MEAN_OUTPUT_COLUMNS]].copy()

        for output_name in MEAN_OUTPUT_COLUMNS:
            y_train = train_table[output_name].to_numpy(dtype=float)
            if prediction_mode == "holdout":
                y_test = test_table[output_name].to_numpy(dtype=float)
                model, y_mean, y_std = fit_scaled_gp(
                    x=x_train,
                    y=y_train,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                )
                pred, pred_std = predict_scaled_gp(model, y_mean, y_std, x_test)
            else:
                y_test = test_table[output_name].to_numpy(dtype=float)
                pred, pred_std = fit_cv_predictions(
                    x=x_train,
                    y=y_train,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    cv_folds=args.cv_folds,
                )

            output_predictions.append(pred)
            output_std_predictions.append(pred_std)
            metrics_rows.append(
                {
                    "train_size": train_size,
                    "test_size": len(test_table),
                    "mode": prediction_mode,
                    "output": output_name,
                    **compute_metrics(y_test, pred, pred_std),
                }
            )
            prediction_frame[f"{output_name}_gp_mean"] = pred
            prediction_frame[f"{output_name}_gp_std"] = pred_std
            prediction_frame[f"{output_name}_gp_residual"] = pred - y_test

        prediction_frame["train_size"] = train_size
        prediction_frame["mode"] = prediction_mode
        prediction_tables.append(prediction_frame)

        predictions_array = np.column_stack(output_predictions)
        predicted_std_array = np.column_stack(output_std_predictions)
        observed_array = test_table[MEAN_OUTPUT_COLUMNS].to_numpy(dtype=float)
        parity_path = figures_root / f"gp_subset_parity_mean_n{train_size}.png"
        ylabel = "GP holdout prediction" if prediction_mode == "holdout" else "GP CV prediction"
        _make_parity_plot(
            observed=observed_array,
            predicted=predictions_array,
            predicted_std=predicted_std_array,
            output_names=MEAN_OUTPUT_COLUMNS,
            output_path=parity_path,
            ylabel=ylabel,
            axis_limits=axis_limits,
        )

        subset_ids_path = emulator_root / f"subset_train_ids_n{train_size}.txt"
        subset_ids_path.write_text("\n".join(train_table["evaluation_id"].tolist()) + "\n")
        subset_summary.append(
            {
                "train_size": train_size,
                "mode": prediction_mode,
                "train_ids_file": str(subset_ids_path),
                "parity_plot": str(parity_path),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = emulator_root / "gp_subset_comparison_metrics_mean.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_path = emulator_root / "gp_subset_comparison_predictions_mean.csv"
    pd.concat(prediction_tables, ignore_index=True).to_csv(predictions_path, index=False)

    summary_path = emulator_root / "gp_subset_comparison_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "sizes": requested_sizes,
                "cv_folds_for_full_size": args.cv_folds,
                "n_restarts_optimizer": args.n_restarts_optimizer,
                "subset_order_file": str(order_path),
                "subsets": subset_summary,
            },
            indent=2,
        )
        + "\n"
    )

    _plot_metric_vs_size(metrics_df, "rmse", figures_root / "gp_subset_metric_rmse_mean.png")
    _plot_metric_vs_size(metrics_df, "r2", figures_root / "gp_subset_metric_r2_mean.png")

    print(order_path)
    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(figures_root / "gp_subset_metric_rmse_mean.png")
    print(figures_root / "gp_subset_metric_r2_mean.png")


if __name__ == "__main__":
    main()
