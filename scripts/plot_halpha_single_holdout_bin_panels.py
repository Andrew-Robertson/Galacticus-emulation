from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot truth-vs-prediction panels for selected Halpha LF bins from a single-holdout GP run, "
            "including a training-mean baseline."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--holdout-evaluation-id", required=True)
    parser.add_argument("--sobral-label", default="Z1")
    parser.add_argument("--bin-indices", type=int, nargs="*", default=[4, 8, 12, 16])
    parser.add_argument("--figures-dir-name", default="figures_halpha_dust")
    return parser.parse_args()


def _extract_true_columns(predictions: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in predictions.columns if column.endswith("_true_log10")],
        key=lambda column: int(re.search(r"_bin(\d+)_", column).group(1)),
    )


def _column_for_bin(true_columns: list[str], bin_index: int) -> str:
    for column in true_columns:
        if f"_bin{bin_index}_" in column:
            return column
    raise KeyError(f"Could not find true column for bin {bin_index}")


def _pred_column(true_column: str) -> str:
    return true_column.replace("_true_log10", "_pred_log10")


def _std_column(true_column: str) -> str:
    return true_column.replace("_true_log10", "_pred_std_log10")


def _linear_column_from_true(true_column: str) -> str:
    return true_column.replace("_true_log10", "")


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    predictions = pd.read_csv(campaign_root / args.predictions_file)
    table = pd.read_csv(campaign_root / args.table_filename)
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    holdout = predictions.loc[predictions["evaluation_id"] == args.holdout_evaluation_id].copy()
    training = table.loc[table["evaluation_id"] != args.holdout_evaluation_id].copy()

    true_columns = _extract_true_columns(predictions)
    selected_true_columns = [_column_for_bin(true_columns, bin_index) for bin_index in args.bin_indices]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()

    for axis, true_column in zip(axes_flat, selected_true_columns, strict=False):
        pred_column = _pred_column(true_column)
        std_column = _std_column(true_column)
        linear_column = _linear_column_from_true(true_column)

        mean_log10 = float(np.mean(np.log10(np.clip(training[linear_column].to_numpy(dtype=float), 1.0e-30, None))))
        observed = holdout[true_column].to_numpy(dtype=float)
        predicted = holdout[pred_column].to_numpy(dtype=float)
        predicted_std = holdout[std_column].to_numpy(dtype=float)

        lower = float(min(observed.min(), predicted.min(), mean_log10))
        upper = float(max(observed.max(), predicted.max(), mean_log10))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)

        axis.errorbar(observed, predicted, yerr=predicted_std, fmt="o", ms=4.0, alpha=0.75, label="GP")
        axis.axhline(mean_log10, color="#c44e52", lw=1.4, ls="--", label="Training mean")
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(true_column.replace("_true_log10", "").replace("halpha_sobral_", "").replace("_", " "))
        axis.set_xlabel("True log10 LF")
        axis.set_ylabel("Predicted log10 LF")
        axis.grid(alpha=0.2)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    output_path = figures_root / (
        f"gp_holdout_bin_panels_{args.sobral_label.lower()}_"
        f"{Path(args.predictions_file).stem.replace('gp_holdout_predictions_', '')}.png"
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
