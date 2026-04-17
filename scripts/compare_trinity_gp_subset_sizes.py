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
    TRINITY_MEAN_OUTPUT_COLUMNS,
    TRINITY_RUN_CONFIGS,
    compute_metrics,
    fit_cv_predictions,
    fit_scaled_gp,
    load_or_build_emulator_table,
    predict_scaled_gp,
    transform_to_prior_quantiles,
    trinity_parameter_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare nested GP subset sizes for the Trinity campaign.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-restarts-optimizer", type=int, default=1)
    parser.add_argument("--output-suffix", default="", help="Suffix to append before output file extensions.")
    parser.add_argument("--optimize-hyperparameters", action="store_true", help="Optimize GP hyperparameters instead of using the fixed starting kernel.")
    parser.add_argument("--run-name", choices=sorted(TRINITY_RUN_CONFIGS.keys()))
    parser.add_argument("--family", choices=["stellar", "black_hole"])
    parser.add_argument("--output-column", help="Compare subset sizes for only a single explicit output column.")
    return parser.parse_args()


def _select_outputs(run_name: str | None, family: str | None, output_column: str | None) -> list[str]:
    if output_column is not None:
        return [output_column]
    outputs = TRINITY_MEAN_OUTPUT_COLUMNS
    if run_name is not None:
        outputs = [column for column in outputs if column.startswith(f"{run_name}_")]
    if family is not None:
        token = "mass_stellar" if family == "stellar" else "mass_black_hole"
        outputs = [column for column in outputs if token in column]
    return outputs


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


def _make_parity_plot(observed: np.ndarray, predicted: np.ndarray, predicted_std: np.ndarray, output_names: list[str], output_path: Path, ylabel: str, axis_limits: list[tuple[float, float]]) -> None:
    n_outputs = len(output_names)
    ncols = 3
    nrows = int(np.ceil(n_outputs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    flat_axes = list(axes.flat)
    for axis, output_name, y_true, y_pred, y_std, (lower, upper) in zip(flat_axes[:n_outputs], output_names, observed.T, predicted.T, predicted_std.T, axis_limits, strict=True):
        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=2.6, alpha=0.55, color="#2a6f97", ecolor="#9fbcd1")
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="#d1495b", linewidth=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_xlabel("Galacticus")
        axis.set_ylabel(ylabel)
        axis.set_title(output_name, fontsize=9)
        axis.grid(alpha=0.2, linewidth=0.5)
    for axis in flat_axes[n_outputs:]:
        axis.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_metric_heatmap(metrics_df: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    pivot = metrics_df.pivot(index="output", columns="train_size", values=metric_name)
    fig, ax = plt.subplots(figsize=(1.2 * len(pivot.columns) + 2, 0.45 * len(pivot.index) + 2), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), [str(v) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=8)
    ax.set_xlabel("Training subset size")
    ax.set_ylabel("Output")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(metric_name)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_metric_grouped(metrics_df: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    unique_outputs = metrics_df["output"].nunique()
    if unique_outputs <= 6:
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for output_name, group in metrics_df.groupby("output"):
            ordered = group.sort_values("train_size")
            ax.plot(ordered["train_size"], ordered[metric_name], marker="o", linewidth=1.8, label=output_name)
        ax.set_xscale("log", base=2)
        sizes = sorted(metrics_df["train_size"].unique())
        ax.set_xticks(sizes, [str(size) for size in sizes])
        ax.set_ylabel(metric_name)
        ax.set_xlabel("Training subset size")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2), constrained_layout=True, sharex=True)
    groups = {
        "Stellar mass means": [col for col in metrics_df["output"].unique() if "mass_stellar" in col],
        "Black-hole mass means": [col for col in metrics_df["output"].unique() if "mass_black_hole" in col],
    }
    for axis, (title, outputs) in zip(axes, groups.items(), strict=True):
        subset = metrics_df[metrics_df["output"].isin(outputs)]
        for output_name, group in subset.groupby("output"):
            ordered = group.sort_values("train_size")
            axis.plot(ordered["train_size"], ordered[metric_name], marker="o", linewidth=1.6, label=output_name)
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        sizes = sorted(metrics_df["train_size"].unique())
        axis.set_xticks(sizes, [str(size) for size in sizes])
        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Training subset size")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")
    campaign_root = Path(args.campaign_root).resolve()
    emulator_root = campaign_root / "emulator"
    figures_root = campaign_root / "figures"
    emulator_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    table = load_or_build_emulator_table(campaign_root)
    input_columns = [spec.short_name for spec in trinity_parameter_specs()]
    output_columns = _select_outputs(args.run_name, args.family, args.output_column)
    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(trinity_parameter_specs(), x_raw)

    order = _greedy_maximin_order(x)
    pd.DataFrame(
        {
            "rank": np.arange(len(order)),
            "ordered_index": order,
            "evaluation_id": table.iloc[order]["evaluation_id"].to_numpy(),
        }
    ).to_csv(emulator_root / f"nested_subset_order{args.output_suffix}.csv", index=False)

    requested_sizes = sorted(set(args.sizes))
    if requested_sizes[-1] != len(table):
        requested_sizes.append(len(table))
    requested_sizes = [size for size in requested_sizes if 0 < size <= len(table)]
    axis_limits = [(float(table[col].min()), float(table[col].max())) for col in output_columns]

    metrics_rows = []
    prediction_tables = []
    subset_summary = []

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
        prediction_frame = test_table[["evaluation_id", *input_columns, *output_columns]].copy()

        for output_name in output_columns:
            y_train = train_table[output_name].to_numpy(dtype=float)
            y_test = test_table[output_name].to_numpy(dtype=float)
            if prediction_mode == "holdout":
                model, y_mean, y_std = fit_scaled_gp(
                    x=x_train,
                    y=y_train,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    optimize_hyperparameters=args.optimize_hyperparameters,
                )
                pred, pred_std = predict_scaled_gp(model, y_mean, y_std, x_test)
            else:
                pred, pred_std = fit_cv_predictions(
                    x=x_train,
                    y=y_train,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    cv_folds=args.cv_folds,
                    optimize_hyperparameters=args.optimize_hyperparameters,
                )
                model, y_mean, y_std = fit_scaled_gp(
                    x=x_train,
                    y=y_train,
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    optimize_hyperparameters=args.optimize_hyperparameters,
                )
            output_predictions.append(pred)
            output_std_predictions.append(pred_std)
            metrics_rows.append(
                {
                    "train_size": train_size,
                    "test_size": len(test_table),
                    "mode": prediction_mode,
                    "output": output_name,
                    "kernel_optimized": str(model.kernel_),
                    "log_marginal_likelihood": float(model.log_marginal_likelihood(model.kernel_.theta)),
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
        observed_array = test_table[output_columns].to_numpy(dtype=float)
        parity_path = figures_root / f"gp_subset_parity_mean_n{train_size}{args.output_suffix}.png"
        ylabel = "GP holdout prediction" if prediction_mode == "holdout" else "GP CV prediction"
        _make_parity_plot(observed_array, predictions_array, predicted_std_array, output_columns, parity_path, ylabel, axis_limits)

        subset_ids_path = emulator_root / f"subset_train_ids_n{train_size}{args.output_suffix}.txt"
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
    metrics_df.to_csv(emulator_root / f"gp_subset_comparison_metrics_mean{args.output_suffix}.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(emulator_root / f"gp_subset_predictions_mean{args.output_suffix}.csv", index=False)
    (emulator_root / f"gp_subset_summary_mean{args.output_suffix}.json").write_text(json.dumps(subset_summary, indent=2) + "\n")

    _plot_metric_grouped(metrics_df, "rmse", figures_root / f"gp_subset_metric_rmse_mean{args.output_suffix}.png")
    _plot_metric_grouped(metrics_df, "r2", figures_root / f"gp_subset_metric_r2_mean{args.output_suffix}.png")
    _plot_metric_heatmap(metrics_df, "rmse", figures_root / f"gp_subset_metric_rmse_mean_heatmap{args.output_suffix}.png")
    _plot_metric_heatmap(metrics_df, "r2", figures_root / f"gp_subset_metric_r2_mean_heatmap{args.output_suffix}.png")

    print(emulator_root / f"gp_subset_comparison_metrics_mean{args.output_suffix}.csv")
    print(figures_root / f"gp_subset_metric_rmse_mean{args.output_suffix}.png")
    print(figures_root / f"gp_subset_metric_r2_mean{args.output_suffix}.png")
    print(figures_root / f"gp_subset_metric_rmse_mean_heatmap{args.output_suffix}.png")
    print(figures_root / f"gp_subset_metric_r2_mean_heatmap{args.output_suffix}.png")


if __name__ == "__main__":
    main()
