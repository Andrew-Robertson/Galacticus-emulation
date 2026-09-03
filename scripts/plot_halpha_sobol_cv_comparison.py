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
        description="Compare two Halpha Sobol PCA CV runs with overlaid parity and held-out example plots."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--sobral-label", required=True, choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--run-a", required=True, help="Output prefix for the first run, e.g. halpha_z1_draws1")
    parser.add_argument("--run-b", required=True, help="Output prefix for the second run, e.g. halpha_z1_draws2")
    parser.add_argument("--label-a", default="draws1")
    parser.add_argument("--label-b", default="draws2")
    parser.add_argument("--figures-dir-name", default="figures_halpha_sobol_pca_cv")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_sobol_pca_cv")
    parser.add_argument("--n-example-curves", type=int, default=5)
    parser.add_argument("--example-ymin", type=float, default=None)
    parser.add_argument("--example-ymax", type=float, default=None)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def _read_inputs(emulator_root: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics = pd.read_csv(emulator_root / f"gp_cv_metrics_{prefix}.csv")
    predictions = pd.read_csv(emulator_root / f"gp_cv_predictions_{prefix}.csv")
    metadata = json.loads((emulator_root / f"gp_cv_metadata_{prefix}.json").read_text())
    return metrics, predictions, metadata


def _common_rows(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["evaluation_id", "dust_draw_index"]
    common = pred_a[key_cols].merge(pred_b[key_cols], on=key_cols, how="inner")
    common = common.drop_duplicates().sort_values(key_cols).reset_index(drop=True)
    a = common.merge(pred_a, on=key_cols, how="left", validate="one_to_one")
    b = common.merge(pred_b, on=key_cols, how="left", validate="one_to_one")
    return a, b


def _bin_columns(predictions: pd.DataFrame, suffix: str) -> list[str]:
    return sorted(
        [column for column in predictions.columns if column.endswith(suffix)],
        key=lambda col: int(col.split("_")[0].replace("bin", "")),
    )


def _luminosity_centers(predictions: pd.DataFrame, true_columns: list[str]) -> np.ndarray:
    centers = []
    for column in true_columns:
        token = column.split("_")[1]
        centers.append(float(token.replace("log10L", "")))
    return np.asarray(centers, dtype=float)


def _target_from_table(campaign_root: Path, sobral_label: str, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(campaign_root / "halpha_dust_lf_emulator_table.csv", nrows=1)
    target = []
    target_std = []
    for bin_index in range(n_bins):
        base = f"halpha_sobral_{sobral_label.lower()}_bin{bin_index}"
        target.append(float(table[f"{base}_target"].iloc[0]))
        target_std.append(float(table[f"{base}_target_std"].iloc[0]))
    return np.asarray(target, dtype=float), np.asarray(target_std, dtype=float)


def _example_indices(predictions: pd.DataFrame, n_examples: int) -> np.ndarray:
    if len(predictions) <= n_examples:
        return np.arange(len(predictions), dtype=int)
    first_quantile_col = next(column for column in predictions.columns if column.endswith("_quantile"))
    order = np.argsort(predictions[first_quantile_col].to_numpy(dtype=float))
    values = np.linspace(0, len(order) - 1, n_examples)
    return np.asarray(sorted({int(order[int(round(v))]) for v in values}), dtype=int)


def _plot_metrics_overlay(
    metrics_a: pd.DataFrame,
    metrics_b: pd.DataFrame,
    output_path: Path,
    *,
    label_a: str,
    label_b: str,
) -> None:
    a = metrics_a[metrics_a["bin_index"].astype(str) != "all"].copy()
    b = metrics_b[metrics_b["bin_index"].astype(str) != "all"].copy()
    x = a["log10_luminosity_center"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for axis, metric, ylabel in [
        (axes[0], "rmse_dex", "RMSE [dex]"),
        (axes[1], "mae_dex", "MAE [dex]"),
        (axes[2], "r2", r"$R^2$"),
    ]:
        axis.plot(x, a[metric].to_numpy(dtype=float), "o--", color="tab:blue", label=label_a, lw=1.6, ms=4)
        axis.plot(x, b[metric].to_numpy(dtype=float), "o-", color="tab:orange", label=label_b, lw=1.8, ms=4)
        axis.set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_parity_overlay(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    output_path: Path,
    *,
    label_a: str,
    label_b: str,
) -> None:
    true_columns = _bin_columns(pred_a, "_true")
    pred_cols_a = [column.replace("_true", "_pred") for column in true_columns]
    std_cols_a = [column.replace("_true", "_std") for column in true_columns]
    pred_cols_b = [column.replace("_true", "_pred") for column in true_columns]
    std_cols_b = [column.replace("_true", "_std") for column in true_columns]
    centers = _luminosity_centers(pred_a, true_columns)

    y_true = pred_a[true_columns].to_numpy(dtype=float)
    y_pred_a = pred_a[pred_cols_a].to_numpy(dtype=float)
    y_std_a = pred_a[std_cols_a].to_numpy(dtype=float)
    y_pred_b = pred_b[pred_cols_b].to_numpy(dtype=float)
    y_std_b = pred_b[std_cols_b].to_numpy(dtype=float)

    n_bins = y_true.shape[1]
    ncols = min(4, n_bins)
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, bin_index in zip(axes_flat, range(n_bins), strict=False):
        observed = y_true[:, bin_index]
        pred1 = y_pred_a[:, bin_index]
        pred2 = y_pred_b[:, bin_index]
        lower = float(min(observed.min(), pred1.min(), pred2.min()))
        upper = float(max(observed.max(), pred1.max(), pred2.max()))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(observed, pred1, yerr=y_std_a[:, bin_index], fmt="o", ms=2.3, alpha=0.45, color="tab:blue", zorder=2)
        axis.errorbar(observed, pred2, yerr=y_std_b[:, bin_index], fmt="o", ms=2.5, alpha=0.65, color="tab:orange", zorder=3)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0, zorder=1)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(rf"$\log_{{10}}L={centers[bin_index]:.2f}$")
        axis.set_xlabel("True log10 LF")
        axis.set_ylabel("Predicted log10 LF")
        axis.grid(alpha=0.2)
    for axis in axes_flat[n_bins:]:
        axis.axis("off")
    handles = [
        plt.Line2D([], [], color="tab:blue", marker="o", linestyle="--", label=label_a, alpha=0.7),
        plt.Line2D([], [], color="tab:orange", marker="o", linestyle="-", label=label_b, alpha=0.8),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_example_curves_overlay(
    campaign_root: Path,
    sobral_label: str,
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    output_path: Path,
    *,
    label_a: str,
    label_b: str,
    n_examples: int,
    ymin: float | None,
    ymax: float | None,
) -> None:
    true_columns = _bin_columns(pred_a, "_true")
    pred_cols = [column.replace("_true", "_pred") for column in true_columns]
    std_cols = [column.replace("_true", "_std") for column in true_columns]
    x_plot = _luminosity_centers(pred_a, true_columns)
    target, target_std = _target_from_table(campaign_root, sobral_label, len(true_columns))
    example_indices = _example_indices(pred_a, n_examples)

    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for color_index, row_index in enumerate(example_indices):
        color = cmap(color_index % 10)
        y_true = pred_a.iloc[row_index][true_columns].to_numpy(dtype=float)
        y_pred_a = pred_a.iloc[row_index][pred_cols].to_numpy(dtype=float)
        y_std_a = pred_a.iloc[row_index][std_cols].to_numpy(dtype=float)
        y_pred_b = pred_b.iloc[row_index][pred_cols].to_numpy(dtype=float)
        y_std_b = pred_b.iloc[row_index][std_cols].to_numpy(dtype=float)

        ax.errorbar(x_plot, y_true, fmt="o", ms=3.3, color=color, alpha=0.95, zorder=5)
        ax.plot(x_plot, y_pred_a, color=color, lw=1.7, ls="--", zorder=3)
        ax.fill_between(x_plot, y_pred_a - y_std_a, y_pred_a + y_std_a, color=color, alpha=0.08, zorder=2)
        ax.plot(x_plot, y_pred_b, color=color, lw=2.0, ls="-", zorder=4)
        ax.fill_between(x_plot, y_pred_b - y_std_b, y_pred_b + y_std_b, color=color, alpha=0.16, zorder=3)

    target_plot = np.log10(np.maximum(target, 1.0e-30))
    lower_target = np.maximum(target - target_std, 1.0e-30)
    upper_target = target + target_std
    yerr_lower = target_plot - np.log10(lower_target)
    yerr_upper = np.log10(upper_target) - target_plot
    ax.plot(x_plot, target_plot, color="k", lw=2.2, zorder=9)
    ax.errorbar(x_plot, target_plot, yerr=np.vstack([yerr_lower, yerr_upper]), fmt="o", color="k", ms=4.0, zorder=10)

    legend_handles = [
        plt.Line2D([], [], color="k", marker="o", linestyle="-", label="Target"),
        plt.Line2D([], [], color="0.2", marker="o", linestyle="None", label="Held-out Galacticus"),
        plt.Line2D([], [], color="0.2", linestyle="--", label=label_a),
        plt.Line2D([], [], color="0.2", linestyle="-", label=label_b),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")
    ax.set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    ax.set_ylabel(r"$\log_{10}\Phi$")
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.22)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    metrics_a, pred_a, meta_a = _read_inputs(emulator_root, args.run_a)
    metrics_b, pred_b, meta_b = _read_inputs(emulator_root, args.run_b)
    if meta_a["sobral_label"] != args.sobral_label or meta_b["sobral_label"] != args.sobral_label:
        raise ValueError("Requested sobral label does not match one of the run metadata files.")

    common_a, common_b = _common_rows(pred_a, pred_b)
    output_prefix = args.output_prefix or f"{args.sobral_label.lower()}_{args.label_a}_vs_{args.label_b}".replace(" ", "_")

    _plot_metrics_overlay(
        metrics_a,
        metrics_b,
        figures_root / f"gp_cv_metrics_overlay_{output_prefix}.png",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    _plot_parity_overlay(
        common_a,
        common_b,
        figures_root / f"gp_cv_parity_overlay_{output_prefix}.png",
        label_a=args.label_a,
        label_b=args.label_b,
    )
    _plot_example_curves_overlay(
        campaign_root,
        args.sobral_label,
        common_a,
        common_b,
        figures_root / f"gp_cv_examples_overlay_{output_prefix}.png",
        label_a=args.label_a,
        label_b=args.label_b,
        n_examples=args.n_example_curves,
        ymin=args.example_ymin,
        ymax=args.example_ymax,
    )

    print(figures_root / f"gp_cv_metrics_overlay_{output_prefix}.png")
    print(figures_root / f"gp_cv_parity_overlay_{output_prefix}.png")
    print(figures_root / f"gp_cv_examples_overlay_{output_prefix}.png")


if __name__ == "__main__":
    main()
