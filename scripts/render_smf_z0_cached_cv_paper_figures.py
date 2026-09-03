from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from galacticus_emu.interactive_observables import (
    OBSERVABLE_CONFIGS,
    _input_columns_for_samples,
    _load_observable_campaign,
    _parameter_specs_for_columns,
)
from galacticus_emu.plotting import set_ylim_from_values


PCA_COLOR = "#1f77b4"


def _set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paper-style SMF z~0 CV figures from cached 1024-campaign CV products."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--hdf5-filename", default="galacticus.hdf5")
    parser.add_argument("--observable", default="smf_z0")
    parser.add_argument("--holdout-key", default="smf_zfourge_z0")
    parser.add_argument("--highlight-x", type=float, action="append", default=None)
    parser.add_argument("--point-fraction", type=float, default=0.25)
    parser.add_argument("--min-log10-y", type=float, default=-7.0)
    parser.add_argument("--n-example-curves", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _decode_attr(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _save_figure(fig: plt.Figure, output_path: Path, *, tight: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": 250}
    if tight:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
    fig.savefig(output_path, **save_kwargs)
    fig.savefig(output_path.with_suffix(".pdf"), **save_kwargs)


def _nearest_bins(x_plot: np.ndarray, requested: list[float]) -> list[int]:
    bins = [int(np.argmin(np.abs(x_plot - value))) for value in requested]
    return list(dict.fromkeys(bins))


def _prediction_columns(predictions: pd.DataFrame, suffix: str) -> list[str]:
    return sorted(
        [column for column in predictions.columns if column.endswith(suffix)],
        key=lambda column: int(column.split("_")[0].replace("bin", "")),
    )


def _prediction_matrix(predictions: pd.DataFrame, suffix: str) -> np.ndarray:
    return predictions[_prediction_columns(predictions, suffix)].to_numpy(dtype=float)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2 or np.allclose(y_true[finite], y_true[finite][0]):
        return np.nan
    return float(r2_score(y_true[finite], y_pred[finite]))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        return np.nan, np.nan
    return (
        float(np.sqrt(mean_squared_error(y_true[finite], y_pred[finite]))),
        _safe_r2(y_true[finite], y_pred[finite]),
    )


def _load_campaign_observable(
    campaign_root: Path,
    *,
    observable_key: str,
    hdf5_filename: str,
    min_log10_y: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict]:
    config = OBSERVABLE_CONFIGS[observable_key]
    samples = pd.read_csv(campaign_root / "samples.csv")
    input_columns = _input_columns_for_samples(samples)
    input_parameter_specs = _parameter_specs_for_columns(input_columns)
    (
        _samples,
        _x_all,
        x_plot,
        target_plot,
        target_noise_plot,
        y_plot,
        _y_noise,
        attrs,
        _transform_metadata,
    ) = _load_observable_campaign(
        campaign_root,
        analysis=config["analysis"],
        hdf5_filename=hdf5_filename,
        min_log10_y=min_log10_y,
        input_columns=input_columns,
        input_parameter_specs=input_parameter_specs,
    )
    return samples, x_plot, target_plot, target_noise_plot, y_plot, attrs


def _plot_heldout_curves(
    *,
    predictions: pd.DataFrame,
    summary: dict,
    samples: pd.DataFrame,
    y_plot: np.ndarray,
    x_plot: np.ndarray,
    target_plot: np.ndarray,
    target_noise_plot: np.ndarray | None,
    highlighted_bins: list[int],
    n_examples: int,
    attrs: dict,
    output_path: Path,
) -> None:
    x_from_predictions = _prediction_matrix(predictions, "_x_plot")[0]
    y_true = _prediction_matrix(predictions, "_true")
    y_true_std = _prediction_matrix(predictions, "_true_std")
    y_pred = _prediction_matrix(predictions, "_pred")
    y_pred_std = _prediction_matrix(predictions, "_pred_std")
    if not np.allclose(x_plot, x_from_predictions, equal_nan=True):
        raise ValueError("Campaign x bins do not match cached holdout prediction x bins")

    example_rows = np.flatnonzero(predictions["is_example_curve"].to_numpy(dtype=bool))
    if example_rows.size > n_examples:
        example_rows = example_rows[:n_examples]
    train_ids = set(summary["train_ids"])
    train_mask = samples["evaluation_id"].isin(train_ids).to_numpy()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    target_label = _decode_attr(attrs.get("targetLabel"), "Tomczak et al. (2014)")
    description = _decode_attr(attrs.get("description"), "")
    redshift_match = re.search(r"\$([^$]*<\s*z\s*<[^$]*)\$", description)
    redshift_label = redshift_match.group(1).replace("\\", "") if redshift_match else ""

    fig, axis = plt.subplots(figsize=(8.0, 5.1), constrained_layout=True)
    y_train = y_plot[train_mask]
    alpha = min(0.10, 20.0 / max(y_train.shape[0], 1))
    for row in y_train:
        axis.plot(x_plot, row, color="0.55", lw=0.8, alpha=alpha, zorder=1)
    axis.plot([], [], color="0.55", lw=1.0, alpha=0.35, label="training-set model LFs")

    if target_noise_plot is None:
        axis.plot(x_plot, target_plot, "o", color="black", ms=4.0, label=target_label, zorder=5)
    else:
        axis.errorbar(
            x_plot,
            target_plot,
            yerr=target_noise_plot,
            fmt="o",
            color="black",
            ms=4.0,
            label=target_label,
            zorder=5,
        )

    labeled_example = example_rows[1] if example_rows.size > 1 else example_rows[0]
    for color, row_index in zip(colors, example_rows, strict=False):
        is_labeled = row_index == labeled_example
        heldout_err = y_true_std[row_index].copy()
        heldout_err[~np.isfinite(heldout_err)] = 0.0
        axis.errorbar(
            x_plot,
            y_true[row_index],
            yerr=heldout_err,
            fmt="o",
            ms=3.5,
            color=color,
            alpha=0.85,
            label="example hold-out" if is_labeled else None,
            zorder=4,
        )
        axis.plot(
            x_plot,
            y_pred[row_index],
            color=color,
            lw=2.0,
            alpha=0.95,
            label="example emulator prediction" if is_labeled else None,
        )
        axis.fill_between(
            x_plot,
            y_pred[row_index] - y_pred_std[row_index],
            y_pred[row_index] + y_pred_std[row_index],
            color=color,
            alpha=0.14,
            lw=0,
        )

    for bin_index in highlighted_bins:
        axis.axvline(x_plot[bin_index], color="0.15", lw=1.0, ls=":", alpha=0.9)
        axis.scatter([x_plot[bin_index]], [target_plot[bin_index]], marker="*", s=90, color="black", zorder=6)

    axis.set_xlabel(r"$\log_{10}(M_\star/\mathrm{M}_\odot)$")
    axis.set_ylabel(r"$\log_{10}(\Phi\,/\,\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$")
    set_ylim_from_values(axis, y_train, target_plot, y_true, y_pred)
    upper = axis.get_ylim()[1]
    axis.set_ylim(-6.2, upper)
    axis.set_xlim(float(np.nanmin(x_plot)), float(np.nanmax(x_plot)))
    if redshift_label:
        axis.text(0.97, 0.96, redshift_label, transform=axis.transAxes, ha="right", va="top", fontsize=14)

    handles, labels = axis.get_legend_handles_labels()
    order = [
        "training-set model LFs",
        target_label,
        "example hold-out",
        "example emulator prediction",
    ]
    handle_by_label = dict(zip(labels, handles, strict=False))
    ordered_labels = [label for label in order if label in handle_by_label]
    ordered_handles = [handle_by_label[label] for label in ordered_labels]
    axis.legend(
        ordered_handles,
        ordered_labels,
        frameon=True,
        fontsize=11,
        ncol=1,
        loc="lower left",
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_parity(
    *,
    predictions: pd.DataFrame,
    target_plot: np.ndarray,
    target_label: str,
    highlighted_bins: list[int],
    point_fraction: float,
    output_path: Path,
) -> None:
    x_plot = _prediction_matrix(predictions, "_x_plot")[0]
    y_true = _prediction_matrix(predictions, "_true")
    y_pred = _prediction_matrix(predictions, "_pred")
    y_pred_std = _prediction_matrix(predictions, "_pred_std")

    fig, axes = plt.subplots(
        1,
        len(highlighted_bins),
        figsize=(12.0, 4.5),
        gridspec_kw={"wspace": 0.10},
    )
    axes = np.atleast_1d(axes)
    fixed_limits = [(-3.1, -1.5), (-4.7, -1.5), (None, -1.8)]
    rng = np.random.default_rng(12345)

    for panel_index, (axis, bin_index) in enumerate(zip(axes, highlighted_bins, strict=True)):
        true = y_true[:, bin_index]
        pred = y_pred[:, bin_index]
        std = y_pred_std[:, bin_index]
        finite = np.isfinite(true) & np.isfinite(pred)
        draw = finite.copy()
        if point_fraction < 1.0 and finite.sum() > 0:
            finite_positions = np.where(finite)[0]
            n_draw = max(1, int(round(point_fraction * finite_positions.size)))
            selected = rng.choice(finite_positions, size=n_draw, replace=False)
            draw[:] = False
            draw[selected] = True
        errored = axis.errorbar(
            true[draw],
            pred[draw],
            yerr=std[draw],
            fmt="o",
            ms=2.7,
            alpha=0.48,
            color=PCA_COLOR,
            ecolor=PCA_COLOR,
            elinewidth=0.55,
            capsize=0,
            zorder=2,
        )
        for bar_collection in errored[2]:
            bar_collection.set_alpha(0.30)
        if np.any(finite):
            lo = float(np.nanmin([true[finite].min(), pred[finite].min(), target_plot[bin_index]]))
            hi = float(np.nanmax([true[finite].max(), pred[finite].max(), target_plot[bin_index]]))
            pad = 0.08 * max(hi - lo, 0.1)
            axis.scatter(
                [target_plot[bin_index]],
                [target_plot[bin_index]],
                marker="*",
                s=120,
                color="black",
                zorder=5,
                label=target_label if panel_index == 0 else None,
            )
            limits = fixed_limits[panel_index]
            if limits is None or limits[0] is None or limits[1] is None:
                lo_axis = np.floor(lo - pad)
                hi_axis = np.ceil(hi + pad)
                limits = (
                    lo_axis if limits is None or limits[0] is None else limits[0],
                    hi_axis if limits is None or limits[1] is None else limits[1],
                )
            axis.set_xlim(*limits)
            axis.set_ylim(*limits)
            axis.plot(limits, limits, color="0.25", ls="--", lw=1.0, zorder=1)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        rmse, r2 = _metrics(true, pred)
        axis.set_title(rf"$\log_{{10}}(M_\star/\mathrm{{M}}_\odot)={x_plot[bin_index]:.2f}$", pad=10)
        axis.text(
            0.97,
            0.04,
            f"RMSE={rmse:.2f}\n" + rf"$R^2$={r2:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=16.5,
        )
        axis.set_xlabel(r"held-out $\log_{10}\Phi$")
    axes[0].set_ylabel(r"predicted $\log_{10}\Phi$")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, frameon=False, fontsize=11, loc="upper left")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.18)
    _save_figure(fig, output_path, tight=True)
    plt.close(fig)


def _plot_training_size(
    *,
    metrics: pd.DataFrame,
    highlighted_bins: list[int],
    output_path: Path,
) -> None:
    bin_metrics = metrics.loc[metrics["bin"].astype(str) != "all"].copy()
    bin_metrics["bin"] = bin_metrics["bin"].astype(int)
    fig, axes = plt.subplots(
        2,
        len(highlighted_bins),
        figsize=(12.0, 6.8),
        sharex=True,
        gridspec_kw={"wspace": 0.14, "hspace": 0.12},
    )
    axes = np.atleast_2d(axes)
    for column, bin_index in enumerate(highlighted_bins):
        rows = bin_metrics.loc[bin_metrics["bin"] == bin_index].sort_values("subset_size")
        if rows.empty:
            continue
        x_value = rows["x_plot"].iloc[0]
        axes[0, column].plot(
            rows["subset_size"],
            rows["r2"],
            marker="o",
            ms=8.8,
            color=PCA_COLOR,
            lw=3.2,
            label="PCA-GP",
        )
        axes[1, column].plot(
            rows["subset_size"],
            rows["rmse"],
            marker="o",
            ms=8.8,
            color=PCA_COLOR,
            lw=3.2,
            label="PCA-GP",
        )
        axes[0, column].set_title(rf"$\log_{{10}}(M_\star/\mathrm{{M}}_\odot)={x_value:.2f}$", pad=10)
        axes[0, column].set_ylim(0.0, 1.0)
        axes[1, column].set_ylim(bottom=0.0)
        rmse_upper = axes[1, column].get_ylim()[1]
        axes[1, column].set_ylim(0.0, 1.2 * rmse_upper)
        axes[1, column].set_xlabel("training subset size")
        for axis in axes[:, column]:
            axis.set_xscale("log", base=2)
            ticks = sorted(rows["subset_size"].dropna().unique())
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value))}"))
    axes[0, 0].set_ylabel(r"$R^2$")
    axes[1, 0].set_ylabel("RMSE [dex]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.subplots_adjust(top=0.88, left=0.08, right=0.98, bottom=0.10, wspace=0.14, hspace=0.12)
    _save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    _set_paper_style()
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else campaign_root / "cross_validation" / "paper_style_smf_z0_cached_cv" / "figures"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_root = campaign_root / "cross_validation" / "holdout_80_20" / args.holdout_key
    holdout_emulator_dir = holdout_root / "emulator"
    predictions_path = holdout_emulator_dir / f"{args.holdout_key}_holdout_demo_pca_predictions.csv"
    holdout_summary_path = holdout_emulator_dir / f"{args.holdout_key}_holdout_demo_pca_summary.json"
    training_metrics_path = (
        campaign_root / "cross_validation" / "training_size_convergence" / "kfold_5" / f"{args.observable}_metrics.csv"
    )

    predictions = pd.read_csv(predictions_path)
    holdout_summary = json.loads(holdout_summary_path.read_text())
    training_metrics = pd.read_csv(training_metrics_path)
    samples, x_plot, target_plot, target_noise_plot, y_plot, attrs = _load_campaign_observable(
        campaign_root,
        observable_key=args.observable,
        hdf5_filename=args.hdf5_filename,
        min_log10_y=args.min_log10_y,
    )

    highlighted_x = args.highlight_x or [8.75, 9.75, 10.75]
    highlighted_bins = _nearest_bins(x_plot, highlighted_x)
    target_label = _decode_attr(attrs.get("targetLabel"), "Tomczak et al. (2014)")

    heldout_path = output_dir / "smf_z0_1024_holdout_curves_highlight_bins.png"
    parity_path = output_dir / "smf_z0_1024_parity_highlight_bins_random_quarter.png"
    training_size_path = output_dir / "smf_z0_1024_rmse_r2_vs_training_size.png"
    _plot_heldout_curves(
        predictions=predictions,
        summary=holdout_summary,
        samples=samples,
        y_plot=y_plot,
        x_plot=x_plot,
        target_plot=target_plot,
        target_noise_plot=target_noise_plot,
        highlighted_bins=highlighted_bins,
        n_examples=args.n_example_curves,
        attrs=attrs,
        output_path=heldout_path,
    )
    _plot_parity(
        predictions=predictions,
        target_plot=target_plot,
        target_label=target_label,
        highlighted_bins=highlighted_bins,
        point_fraction=args.point_fraction,
        output_path=parity_path,
    )
    _plot_training_size(
        metrics=training_metrics,
        highlighted_bins=highlighted_bins,
        output_path=training_size_path,
    )

    provenance = f"""# SMF z~0 Cached-CV Paper Figures

Generated UTC: {datetime.now(timezone.utc).isoformat()}

These figures are paper-style renderings of cached cross-validation products for:

`{campaign_root}`

They are intended as the 1024-campaign replacement for the older polished figures in:

`runs/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_freeYield_20p_moreHalos_512_hybrid_5fallback/cross_validation/pca_threshold_smf_z0_train_rest_plus_full_kfold/figures`

## Source Products

- Holdout predictions: `{predictions_path}`
- Holdout summary: `{holdout_summary_path}`
- Training-size metrics: `{training_metrics_path}`

## Output Files

- `smf_z0_1024_holdout_curves_highlight_bins.{{png,pdf}}`
- `smf_z0_1024_parity_highlight_bins_random_quarter.{{png,pdf}}`
- `smf_z0_1024_rmse_r2_vs_training_size.{{png,pdf}}`

## Command

Run from the repository root:

```bash
python scripts/render_smf_z0_cached_cv_paper_figures.py \\
  runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced
```

## Notes

- The held-out-curve and parity figures use the cached 80/20 holdout PCA-GP predictions for `smf_zfourge_z0`.
- The training-size figure uses cached 5-fold training-size metrics for `smf_z0`.
- The older 512-campaign figures came from `scripts/run_smf_z0_pca_threshold_cv.py`, which compared PCA variance thresholds and a bin-by-bin GP. Re-running that exact threshold/bin-by-bin workflow at the full 1024 point is substantially more expensive locally because the bin-by-bin GP optimizer scales poorly with the number of training evaluations.
- Highlighted bins are `{', '.join(f'{x_plot[index]:.2f}' for index in highlighted_bins)}` in `log10(M_star/M_sun)`.
"""
    (output_dir / "PROVENANCE.md").write_text(provenance)

    for path in [heldout_path, parity_path, training_size_path, output_dir / "PROVENANCE.md"]:
        print(path)


if __name__ == "__main__":
    main()
