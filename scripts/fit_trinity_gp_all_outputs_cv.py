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
    TRINITY_TRAINABLE_OUTPUT_COLUMNS,
    compute_metrics,
    fit_cv_predictions,
    load_or_build_emulator_table,
    transform_to_prior_quantiles,
    trinity_parameter_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-validated GP diagnostics for all 36 Trinity trainable outputs."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-restarts-optimizer", type=int, default=1)
    parser.add_argument(
        "--optimize-hyperparameters",
        action="store_true",
        help="Optimize GP hyperparameters during each CV fit.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_all36_cv_gw",
        help="Suffix appended to output file names.",
    )
    return parser.parse_args()


def _target_error_column(output_name: str) -> str:
    prefix, bin_index = output_name.rsplit("_", 1)
    return f"{prefix}_target_error_{bin_index}"


def _output_group(output_name: str) -> str:
    if "mass_stellar" in output_name and "scatter" not in output_name:
        return "stellar_mean"
    if "mass_black_hole" in output_name and "scatter" not in output_name:
        return "black_hole_mean"
    if "mass_stellar" in output_name and "scatter" in output_name:
        return "stellar_scatter"
    return "black_hole_scatter"


def _output_redshift(output_name: str) -> str:
    return output_name.split("_", 1)[0]


def _output_bin(output_name: str) -> int:
    return int(output_name.rsplit("_", 1)[1])


def _plot_metric_heatmap(metrics_df: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    ordered = metrics_df.sort_values(["group", "redshift", "bin_index", "output"]).reset_index(drop=True)
    values = ordered[[metric_name]].to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 0.35 * len(ordered) + 1.6), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks([0], [metric_name])
    ax.set_yticks(np.arange(len(ordered)), ordered["output"], fontsize=8)
    ax.set_title(f"All-output {metric_name}")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label(metric_name)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_group_summary(metrics_df: pd.DataFrame, metric_name: str, output_path: Path) -> None:
    groups = ["stellar_mean", "black_hole_mean", "stellar_scatter", "black_hole_scatter"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    axes = list(axes.flat)
    redshift_order = ["z0", "z1", "z2"]
    colors = {"z0": "#1b9e77", "z1": "#d95f02", "z2": "#7570b3"}
    for axis, group in zip(axes, groups, strict=True):
        subset = metrics_df[metrics_df["group"] == group].copy()
        subset["x"] = subset["redshift"].map({name: i for i, name in enumerate(redshift_order)})
        offsets = np.linspace(-0.18, 0.18, 3)
        for offset, bin_index in zip(offsets, sorted(subset["bin_index"].unique()), strict=True):
            bin_subset = subset[subset["bin_index"] == bin_index]
            axis.scatter(
                bin_subset["x"] + offset,
                bin_subset[metric_name],
                s=42,
                label=f"bin {bin_index + 1}" if group == groups[0] else None,
                c=[colors[z] for z in bin_subset["redshift"]],
                alpha=0.9,
            )
        axis.set_title(group.replace("_", " ").title())
        axis.set_xticks(range(len(redshift_order)), redshift_order)
        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_worst_outputs(metrics_df: pd.DataFrame, metric_name: str, output_path: Path, n_show: int = 12) -> None:
    worst = metrics_df.sort_values(metric_name, ascending=False).head(n_show).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(worst) + 1.8), constrained_layout=True)
    ax.barh(worst["output"], worst[metric_name], color="#d1495b", alpha=0.8)
    ax.set_xlabel(metric_name)
    ax.set_title(f"Worst {n_show} outputs by {metric_name}")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
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
    parameter_specs = trinity_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)

    predictions_table = table[["evaluation_id", *input_columns, *TRINITY_TRAINABLE_OUTPUT_COLUMNS]].copy()
    metrics_rows = []

    for output_name in TRINITY_TRAINABLE_OUTPUT_COLUMNS:
        y = table[output_name].to_numpy(dtype=float)
        pred, pred_std = fit_cv_predictions(
            x=x,
            y=y,
            n_restarts_optimizer=args.n_restarts_optimizer,
            cv_folds=args.cv_folds,
            optimize_hyperparameters=args.optimize_hyperparameters,
        )
        metric_row = compute_metrics(y, pred, pred_std)
        target_error_col = _target_error_column(output_name)
        target_sigma = float(table[target_error_col].iloc[0]) if target_error_col in table.columns else np.nan
        dynamic_range = float(np.max(y) - np.min(y))
        metric_row.update(
            {
                "output": output_name,
                "group": _output_group(output_name),
                "redshift": _output_redshift(output_name),
                "bin_index": _output_bin(output_name),
                "target_sigma": target_sigma,
                "rmse_over_target_sigma": float(metric_row["rmse"] / target_sigma) if np.isfinite(target_sigma) and target_sigma > 0.0 else np.nan,
                "rmse_over_range": float(metric_row["rmse"] / dynamic_range) if dynamic_range > 0.0 else np.nan,
            }
        )
        metrics_rows.append(metric_row)
        predictions_table[f"{output_name}_gp_cv_mean"] = pred
        predictions_table[f"{output_name}_gp_cv_std"] = pred_std
        predictions_table[f"{output_name}_gp_cv_residual"] = pred - y

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["group", "redshift", "bin_index", "output"]).reset_index(drop=True)

    metrics_path = emulator_root / f"gp_metrics_all_outputs{args.output_suffix}.csv"
    predictions_path = emulator_root / f"gp_cv_predictions_all_outputs{args.output_suffix}.csv"
    summary_path = emulator_root / f"gp_cv_summary_all_outputs{args.output_suffix}.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_table.to_csv(predictions_path, index=False)

    summary = {
        "campaign_root": str(campaign_root),
        "cv_folds": args.cv_folds,
        "n_restarts_optimizer": args.n_restarts_optimizer,
        "optimize_hyperparameters": args.optimize_hyperparameters,
        "n_outputs": len(TRINITY_TRAINABLE_OUTPUT_COLUMNS),
        "best_r2_output": metrics_df.sort_values("r2", ascending=False).iloc[0]["output"],
        "worst_r2_output": metrics_df.sort_values("r2", ascending=True).iloc[0]["output"],
        "worst_rmse_over_target_sigma_output": metrics_df.sort_values("rmse_over_target_sigma", ascending=False).iloc[0]["output"],
        "group_summary": (
            metrics_df.groupby("group")[["rmse", "r2", "rmse_over_target_sigma", "coverage_1sigma", "coverage_2sigma"]]
            .agg(["mean", "median", "min", "max"])
            .to_dict()
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    _plot_metric_heatmap(
        metrics_df,
        "r2",
        figures_root / f"gp_metric_heatmap_r2_all_outputs{args.output_suffix}.png",
    )
    _plot_metric_heatmap(
        metrics_df,
        "rmse_over_target_sigma",
        figures_root / f"gp_metric_heatmap_rmse_over_target_sigma_all_outputs{args.output_suffix}.png",
    )
    _plot_group_summary(
        metrics_df,
        "r2",
        figures_root / f"gp_metric_group_summary_r2_all_outputs{args.output_suffix}.png",
    )
    _plot_group_summary(
        metrics_df,
        "rmse_over_target_sigma",
        figures_root / f"gp_metric_group_summary_rmse_over_target_sigma_all_outputs{args.output_suffix}.png",
    )
    _plot_worst_outputs(
        metrics_df,
        "rmse_over_target_sigma",
        figures_root / f"gp_metric_worst_outputs_rmse_over_target_sigma_all_outputs{args.output_suffix}.png",
    )

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(figures_root / f"gp_metric_heatmap_r2_all_outputs{args.output_suffix}.png")
    print(figures_root / f"gp_metric_heatmap_rmse_over_target_sigma_all_outputs{args.output_suffix}.png")


if __name__ == "__main__":
    main()
