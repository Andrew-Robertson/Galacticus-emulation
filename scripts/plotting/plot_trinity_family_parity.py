from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_DEFINITIONS = {
    "mstar_mean": {
        "title": "GP CV Parity: Mean Stellar Mass",
        "stem": "mass_stellar_log10",
        "filename": "gp_cv_parity_family_mstar_mean.png",
    },
    "mstar_scatter": {
        "title": "GP CV Parity: Stellar-Mass Scatter",
        "stem": "mass_stellar_log10_scatter",
        "filename": "gp_cv_parity_family_mstar_scatter.png",
    },
    "mbh_mean": {
        "title": "GP CV Parity: Mean Black-Hole Mass",
        "stem": "mass_black_hole_log10",
        "filename": "gp_cv_parity_family_mbh_mean.png",
    },
    "mbh_scatter": {
        "title": "GP CV Parity: Black-Hole-Mass Scatter",
        "stem": "mass_black_hole_log10_scatter",
        "filename": "gp_cv_parity_family_mbh_scatter.png",
    },
}

RUN_NAMES = ["z0", "z1", "z2"]
BIN_INDICES = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3x3 family parity figures from the all-output Trinity GP CV predictions."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--predictions-csv",
        default=None,
        help="Path to the all-output CV predictions CSV. Defaults to emulator/gp_cv_predictions_all_outputs_all36_cv_gw.csv.",
    )
    return parser.parse_args()


def _make_family_plot(predictions_df: pd.DataFrame, family_name: str, figures_root: Path) -> Path:
    family = FAMILY_DEFINITIONS[family_name]
    stem = family["stem"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for row_index, run_name in enumerate(RUN_NAMES):
        for col_index, bin_index in enumerate(BIN_INDICES):
            axis = axes[row_index, col_index]
            output_name = f"{run_name}_{stem}_{bin_index}"
            observed = predictions_df[output_name].to_numpy(dtype=float)
            predicted = predictions_df[f"{output_name}_gp_cv_mean"].to_numpy(dtype=float)
            predicted_std = predictions_df[f"{output_name}_gp_cv_std"].to_numpy(dtype=float)

            lower = min(float(np.min(observed)), float(np.min(predicted)))
            upper = max(float(np.max(observed)), float(np.max(predicted)))
            margin = 0.05 * (upper - lower if upper > lower else 1.0)

            axis.errorbar(
                observed,
                predicted,
                yerr=predicted_std,
                fmt="o",
                ms=2.8,
                alpha=0.55,
                color="#2a6f97",
                ecolor="#9fbcd1",
            )
            axis.plot(
                [lower - margin, upper + margin],
                [lower - margin, upper + margin],
                "--",
                color="#d1495b",
                linewidth=1.1,
            )
            axis.set_xlim(lower - margin, upper + margin)
            axis.set_ylim(lower - margin, upper + margin)
            axis.set_title(f"{run_name}, bin {bin_index + 1}", fontsize=10)
            axis.set_xlabel("Galacticus")
            axis.set_ylabel("GP CV prediction")
            axis.grid(alpha=0.2, linewidth=0.5)

    fig.suptitle(family["title"], fontsize=14)
    output_path = figures_root / family["filename"]
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    emulator_root = campaign_root / "emulator"
    figures_root = campaign_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        Path(args.predictions_csv).resolve()
        if args.predictions_csv is not None
        else emulator_root / "gp_cv_predictions_all_outputs_all36_cv_gw.csv"
    )
    predictions_df = pd.read_csv(predictions_path)

    output_paths = []
    for family_name in FAMILY_DEFINITIONS:
        output_paths.append(_make_family_plot(predictions_df, family_name, figures_root))

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
