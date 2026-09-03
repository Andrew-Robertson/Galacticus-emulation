from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


EVAL_INDEX_RE = re.compile(r"eval-(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay saved 80/20 holdout parity predictions from two campaigns. "
            "The two prediction tables are matched by eval index, so campaign name "
            "prefixes may differ."
        )
    )
    parser.add_argument("--small-campaign", type=Path, required=True)
    parser.add_argument("--large-campaign", type=Path, required=True)
    parser.add_argument("--small-label", default="smaller run")
    parser.add_argument("--large-label", default="more halos")
    parser.add_argument("--cv-subdir", default="cross_validation/holdout_80_20")
    parser.add_argument("--observable", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _eval_index(evaluation_id: str) -> int:
    match = EVAL_INDEX_RE.search(str(evaluation_id))
    if match is None:
        raise ValueError(f"Could not parse eval index from evaluation_id={evaluation_id!r}")
    return int(match.group(1))


def _common_observables(small_root: Path, large_root: Path) -> list[str]:
    small = {path.name for path in small_root.iterdir() if path.is_dir()}
    large = {path.name for path in large_root.iterdir() if path.is_dir()}
    return sorted(small & large)


def _output_prefix(observable: str) -> str:
    return f"{observable}_holdout_demo_pca"


def _paths(root: Path, observable: str) -> tuple[Path, Path]:
    prefix = _output_prefix(observable)
    obs_root = root / observable
    return (
        obs_root / "emulator" / f"{prefix}_predictions.csv",
        obs_root / "emulator" / f"{prefix}_summary.json",
    )


def _load_predictions(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    predictions = pd.read_csv(path)
    predictions["eval_index"] = predictions["evaluation_id"].map(_eval_index)
    x_columns = sorted(
        [column for column in predictions.columns if column.endswith("_x_plot")],
        key=lambda column: int(column.split("_")[0].replace("bin", "")),
    )
    x_plot = predictions.loc[0, x_columns].to_numpy(dtype=float)
    return predictions, x_plot


def _bin_columns(frame: pd.DataFrame, suffix: str) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.endswith(suffix)],
        key=lambda column: int(column.split("_")[0].replace("bin", "")),
    )


def _common_bin_indices(small_x_plot: np.ndarray, large_x_plot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    small_by_x = {round(float(value), 8): index for index, value in enumerate(small_x_plot)}
    large_by_x = {round(float(value), 8): index for index, value in enumerate(large_x_plot)}
    common_x = sorted(set(small_by_x) & set(large_by_x))
    if not common_x:
        raise ValueError("No common x-bins between prediction tables")
    small_indices = np.asarray([small_by_x[value] for value in common_x], dtype=int)
    large_indices = np.asarray([large_by_x[value] for value in common_x], dtype=int)
    return small_indices, large_indices, np.asarray(common_x, dtype=float)


def _valid_mask(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    bad_training_condition: str,
    bad_training_value_fill: str,
) -> np.ndarray:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if bad_training_value_fill in {"keep", "as_is"}:
        return valid
    if bad_training_condition == "nonpositive":
        return valid & (observed > 0.0)
    if bad_training_condition == "zero_only":
        return valid & (observed != 0.0)
    return valid


def _safe_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    if observed.size < 2 or np.allclose(observed, observed[0]):
        return np.nan
    return float(r2_score(observed, predicted))


def _plot_overlay_parity(
    *,
    observable: str,
    small: pd.DataFrame,
    large: pd.DataFrame,
    x_plot: np.ndarray,
    small_bin_indices: np.ndarray,
    large_bin_indices: np.ndarray,
    small_label: str,
    large_label: str,
    summary: dict,
    path: Path,
    dpi: int,
) -> pd.DataFrame:
    small_true_columns = _bin_columns(small, "_true")
    small_pred_columns = _bin_columns(small, "_pred")
    small_std_columns = _bin_columns(small, "_pred_std")
    large_true_columns = _bin_columns(large, "_true")
    large_pred_columns = _bin_columns(large, "_pred")
    large_std_columns = _bin_columns(large, "_pred_std")

    n_bins = len(x_plot)
    ncols = min(4, n_bins)
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    training_metadata = summary.get("training_target_metadata", {})
    bad_training_condition = str(training_metadata.get("bad_training_condition", "nonfinite"))
    bad_training_value_fill = str(training_metadata.get("bad_training_value_fill", "bin_median"))
    metric_rows = []

    for bin_index in range(n_bins):
        axis = axes_flat[bin_index]
        small_source_bin = int(small_bin_indices[bin_index])
        large_source_bin = int(large_bin_indices[bin_index])
        small_observed = small[small_true_columns[small_source_bin]].to_numpy(dtype=float)
        small_predicted = small[small_pred_columns[small_source_bin]].to_numpy(dtype=float)
        small_std = small[small_std_columns[small_source_bin]].to_numpy(dtype=float)
        large_observed = large[large_true_columns[large_source_bin]].to_numpy(dtype=float)
        large_predicted = large[large_pred_columns[large_source_bin]].to_numpy(dtype=float)
        large_std = large[large_std_columns[large_source_bin]].to_numpy(dtype=float)

        small_valid = _valid_mask(
            small_observed,
            small_predicted,
            bad_training_condition=bad_training_condition,
            bad_training_value_fill=bad_training_value_fill,
        )
        large_valid = _valid_mask(
            large_observed,
            large_predicted,
            bad_training_condition=bad_training_condition,
            bad_training_value_fill=bad_training_value_fill,
        )

        candidates = []
        for observed, predicted, valid in [
            (small_observed, small_predicted, small_valid),
            (large_observed, large_predicted, large_valid),
        ]:
            if np.any(valid):
                candidates.extend([observed[valid], predicted[valid]])
        if not candidates:
            axis.set_title(f"x={x_plot[bin_index]:.2f}")
            axis.text(0.5, 0.5, "no valid held-out data", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue

        lower = min(float(np.nanmin(values)) for values in candidates)
        upper = max(float(np.nanmax(values)) for values in candidates)
        margin = 0.08 * (upper - lower if upper > lower else 1.0)

        axis.errorbar(
            small_observed[small_valid],
            small_predicted[small_valid],
            yerr=small_std[small_valid],
            fmt="o",
            ms=3.0,
            alpha=0.32,
            color="#7b7b7b",
            ecolor="#b5b5b5",
            elinewidth=0.8,
            capsize=0.0,
            zorder=2,
        )
        axis.errorbar(
            large_observed[large_valid],
            large_predicted[large_valid],
            yerr=large_std[large_valid],
            fmt="o",
            ms=3.2,
            alpha=0.78,
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=0.8,
            capsize=0.0,
            zorder=4,
        )
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)

        small_rmse = (
            float(np.sqrt(mean_squared_error(small_observed[small_valid], small_predicted[small_valid])))
            if np.any(small_valid)
            else np.nan
        )
        large_rmse = (
            float(np.sqrt(mean_squared_error(large_observed[large_valid], large_predicted[large_valid])))
            if np.any(large_valid)
            else np.nan
        )
        small_r2 = _safe_r2(small_observed[small_valid], small_predicted[small_valid]) if np.any(small_valid) else np.nan
        large_r2 = _safe_r2(large_observed[large_valid], large_predicted[large_valid]) if np.any(large_valid) else np.nan

        axis.set_title(
            f"x={x_plot[bin_index]:.2f}\nRMSE {small_rmse:.3g} -> {large_rmse:.3g}",
            fontsize=9,
        )
        axis.set_xlabel("Held-out Galacticus")
        axis.set_ylabel("GP prediction")
        axis.grid(alpha=0.2)

        metric_rows.extend(
            [
                {
                    "observable": observable,
                    "run_label": small_label,
                    "bin": bin_index,
                    "source_bin": small_source_bin,
                    "x_plot": float(x_plot[bin_index]),
                    "n_valid": int(np.count_nonzero(small_valid)),
                    "rmse": small_rmse,
                    "r2": small_r2,
                },
                {
                    "observable": observable,
                    "run_label": large_label,
                    "bin": bin_index,
                    "source_bin": large_source_bin,
                    "x_plot": float(x_plot[bin_index]),
                    "n_valid": int(np.count_nonzero(large_valid)),
                    "rmse": large_rmse,
                    "r2": large_r2,
                },
            ]
        )

    for axis in axes_flat[n_bins:]:
        axis.axis("off")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7b7b7b", alpha=0.55, label=small_label),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", alpha=0.85, label=large_label),
        Line2D([0], [0], color="0.25", ls="--", lw=1.0, label="1:1"),
    ]
    axes_flat[0].legend(handles=handles, loc="best", frameon=False, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(metric_rows)


def main() -> None:
    args = parse_args()
    small_root = (args.small_campaign / args.cv_subdir).resolve()
    large_root = (args.large_campaign / args.cv_subdir).resolve()
    observables = args.observable or _common_observables(small_root, large_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    summary_rows = []
    for observable in observables:
        small_predictions_path, small_summary_path = _paths(small_root, observable)
        large_predictions_path, large_summary_path = _paths(large_root, observable)
        if not small_predictions_path.exists():
            raise FileNotFoundError(small_predictions_path)
        if not large_predictions_path.exists():
            raise FileNotFoundError(large_predictions_path)
        small_predictions, small_x_plot = _load_predictions(small_predictions_path)
        large_predictions, large_x_plot = _load_predictions(large_predictions_path)
        small_bin_indices, large_bin_indices, common_x_plot = _common_bin_indices(small_x_plot, large_x_plot)

        common_eval_indices = sorted(set(small_predictions["eval_index"]) & set(large_predictions["eval_index"]))
        if not common_eval_indices:
            raise ValueError(f"{observable}: no common held-out eval indices")
        small_matched = (
            small_predictions.loc[small_predictions["eval_index"].isin(common_eval_indices)]
            .sort_values("eval_index")
            .reset_index(drop=True)
        )
        large_matched = (
            large_predictions.loc[large_predictions["eval_index"].isin(common_eval_indices)]
            .sort_values("eval_index")
            .reset_index(drop=True)
        )
        if not np.array_equal(small_matched["eval_index"].to_numpy(), large_matched["eval_index"].to_numpy()):
            raise ValueError(f"{observable}: matched eval index ordering failed")

        summary = json.loads(large_summary_path.read_text())
        output_path = (
            args.output_dir
            / observable
            / "figures"
            / f"{observable}_holdout_demo_pca_parity_moreHalos_vs_smaller_run.png"
        )
        metrics = _plot_overlay_parity(
            observable=observable,
            small=small_matched,
            large=large_matched,
            x_plot=common_x_plot,
            small_bin_indices=small_bin_indices,
            large_bin_indices=large_bin_indices,
            small_label=args.small_label,
            large_label=args.large_label,
            summary=summary,
            path=output_path,
            dpi=args.dpi,
        )
        metrics.to_csv(args.output_dir / observable / f"{observable}_parity_comparison_metrics.csv", index=False)
        all_metrics.append(metrics)
        summary_rows.append(
            {
                "observable": observable,
                "n_common_heldout": int(len(common_eval_indices)),
                "n_common_bins": int(len(common_x_plot)),
                "n_small_bins": int(len(small_x_plot)),
                "n_large_bins": int(len(large_x_plot)),
                "output_path": str(output_path),
                "small_predictions": str(small_predictions_path),
                "large_predictions": str(large_predictions_path),
            }
        )
        print(output_path)

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "comparison_summary.csv", index=False)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(args.output_dir / "comparison_metrics.csv", index=False)


if __name__ == "__main__":
    main()
