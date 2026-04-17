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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary JSON and figures from an existing all-output Trinity GP CV metrics CSV."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Path to the metrics CSV. Defaults to emulator/gp_metrics_all_outputs_all36_cv_gw.csv.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_all36_cv_gw",
        help="Suffix used in the output file names.",
    )
    return parser.parse_args()


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


def _serializable_group_summary(metrics_df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    metrics = ["rmse", "r2", "rmse_over_target_sigma", "coverage_1sigma", "coverage_2sigma"]
    stats = ["mean", "median", "min", "max"]
    grouped = metrics_df.groupby("group")[metrics].agg(stats)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group_name in grouped.index:
        summary[group_name] = {}
        for metric in metrics:
            summary[group_name][metric] = {}
            for stat in stats:
                summary[group_name][metric][stat] = float(grouped.loc[group_name, (metric, stat)])
    return summary


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    emulator_root = campaign_root / "emulator"
    figures_root = campaign_root / "figures"
    emulator_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        Path(args.metrics_csv).resolve()
        if args.metrics_csv is not None
        else emulator_root / f"gp_metrics_all_outputs{args.output_suffix}.csv"
    )
    metrics_df = pd.read_csv(metrics_path).sort_values(["group", "redshift", "bin_index", "output"]).reset_index(drop=True)

    summary = {
        "campaign_root": str(campaign_root),
        "metrics_csv": str(metrics_path),
        "n_outputs": int(len(metrics_df)),
        "best_r2_output": str(metrics_df.sort_values("r2", ascending=False).iloc[0]["output"]),
        "worst_r2_output": str(metrics_df.sort_values("r2", ascending=True).iloc[0]["output"]),
        "worst_rmse_over_target_sigma_output": str(
            metrics_df.sort_values("rmse_over_target_sigma", ascending=False).iloc[0]["output"]
        ),
        "group_summary": _serializable_group_summary(metrics_df),
    }
    summary_path = emulator_root / f"gp_cv_summary_all_outputs{args.output_suffix}.json"
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

    print(summary_path)
    print(figures_root / f"gp_metric_heatmap_r2_all_outputs{args.output_suffix}.png")
    print(figures_root / f"gp_metric_heatmap_rmse_over_target_sigma_all_outputs{args.output_suffix}.png")
    print(figures_root / f"gp_metric_group_summary_r2_all_outputs{args.output_suffix}.png")
    print(figures_root / f"gp_metric_group_summary_rmse_over_target_sigma_all_outputs{args.output_suffix}.png")
    print(figures_root / f"gp_metric_worst_outputs_rmse_over_target_sigma_all_outputs{args.output_suffix}.png")


if __name__ == "__main__":
    main()
