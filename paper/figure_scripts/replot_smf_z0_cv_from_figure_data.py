from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd

from run_smf_z0_pca_threshold_cv import (
    _plot_heldout_curves,
    _plot_parity,
    _plot_training_size,
    _set_paper_style,
)


DEFAULT_DATA_DIR = REPO_ROOT / "paper/figure_data/smf_z0_cv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper/tmp/rebuilt_figures"
METHOD = "pca_0p99"
SUBSET_SIZE = 1024
FOLD_INDEX = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replot the three paper SMF z~0 emulator-validation figures from "
            "the compact CSV and JSON products tracked in the repository."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the checked-in SMF z~0 CV products.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the rebuilt PNG and PDF figures.",
    )
    return parser.parse_args()


def _reconstruct_training_curves(
    predictions: pd.DataFrame,
    *,
    method: str,
    subset_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = predictions.loc[
        (predictions["method"] == method)
        & (predictions["subset_size"] == subset_size)
    ].copy()
    if rows.empty:
        raise ValueError(f"No cached predictions found for {method}, Ntrain={subset_size}")

    unique_truth = rows.drop_duplicates(["sample_index", "bin_index"])
    truth = unique_truth.pivot(index="sample_index", columns="bin_index", values="y_true")
    x_values = unique_truth.pivot(index="sample_index", columns="bin_index", values="x_plot")
    expected_samples = np.arange(int(truth.index.max()) + 1)
    expected_bins = np.arange(int(truth.columns.max()) + 1)
    truth = truth.reindex(index=expected_samples, columns=expected_bins)
    x_values = x_values.reindex(index=expected_samples, columns=expected_bins)
    if truth.isna().all(axis=1).any() or truth.isna().all(axis=0).any():
        raise ValueError("Cached predictions do not cover every training sample and output bin")

    x_plot = x_values.iloc[0].to_numpy(dtype=float)
    if not np.allclose(x_values.to_numpy(dtype=float), x_plot[None, :], equal_nan=True):
        raise ValueError("Cached prediction rows contain inconsistent stellar-mass bins")
    return truth.to_numpy(dtype=float), x_plot


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(data_dir / "smf_z0_pca_threshold_cv_predictions.csv")
    metrics = pd.read_csv(data_dir / "smf_z0_pca_threshold_cv_metrics.csv")
    splits = pd.read_csv(data_dir / "smf_z0_pca_threshold_cv_splits.csv")
    target = pd.read_csv(data_dir / "smf_z0_target.csv").sort_values("bin_index")
    summary = json.loads((data_dir / "smf_z0_pca_threshold_cv_summary.json").read_text())

    y_plot, x_plot = _reconstruct_training_curves(
        predictions,
        method=METHOD,
        subset_size=SUBSET_SIZE,
    )
    if not np.allclose(target["x_plot"].to_numpy(dtype=float), x_plot):
        raise ValueError("The target table and cached predictions use different stellar-mass bins")

    split_rows = splits.loc[
        (splits["subset_size"] == SUBSET_SIZE)
        & (splits["fold_index"] == FOLD_INDEX)
    ]
    if len(split_rows) != 1:
        raise ValueError("Expected one cached split for Ntrain=1024, fold 1")
    train_index = np.fromstring(split_rows.iloc[0]["train_indices"], sep=" ", dtype=int)
    highlighted_bins = [int(item["bin_index"]) for item in summary["highlighted_bins"]]
    target_plot = target["target_plot"].to_numpy(dtype=float)
    target_sigma_plot = target["target_sigma_plot"].to_numpy(dtype=float)

    _set_paper_style()
    _plot_training_size(
        metrics,
        highlighted_bins=highlighted_bins,
        training_rms_sigma_by_bin=None,
        output_path=output_dir / "smf_z0_pca_threshold_rmse_r2_vs_training_size.png",
    )
    _plot_heldout_curves(
        y_plot=y_plot,
        target_plot=target_plot,
        target_noise_plot=target_sigma_plot,
        x_plot=x_plot,
        x_all=np.zeros((len(y_plot), 1), dtype=float),
        train_index=train_index,
        predictions=predictions,
        method=METHOD,
        subset_size=SUBSET_SIZE,
        fold_index=FOLD_INDEX,
        highlighted_bins=highlighted_bins,
        n_examples=int(summary["n_example_curves"]),
        example_selection=str(summary["example_selection"]),
        target_label="Tomczak et al. (2014)",
        redshift_label=r"0.20 < z < 0.50",
        output_path=output_dir / "smf_z0_pca99_holdout_curves_highlight_bins.png",
    )
    _plot_parity(
        predictions,
        method=METHOD,
        subset_size=SUBSET_SIZE,
        highlighted_bins=highlighted_bins,
        target_plot=target_plot,
        target_label="Tomczak et al. (2014)",
        point_fraction=0.25,
        point_alpha=0.48,
        errorbar_alpha=0.30,
        errorbar_lw=0.55,
        include_errorbars=True,
        output_path=output_dir / "smf_z0_pca99_parity_highlight_bins_random_quarter.png",
    )

    for filename in (
        "smf_z0_pca99_holdout_curves_highlight_bins.pdf",
        "smf_z0_pca99_parity_highlight_bins_random_quarter.pdf",
        "smf_z0_pca_threshold_rmse_r2_vs_training_size.pdf",
    ):
        print(output_dir / filename)


if __name__ == "__main__":
    main()
