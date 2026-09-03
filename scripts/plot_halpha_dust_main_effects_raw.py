from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_COLUMNS = [
    "diskVelocityCharacteristic",
    "delta_0",
    "delta_z",
    "delta_M",
    "delta_Mz",
    "attenuation_scatter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw-data main-effect style trends for Halpha dust-emulator tables."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--sobral-label", default="Z1", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--bins", type=int, nargs="*", default=None)
    parser.add_argument(
        "--log10-l-values",
        type=float,
        nargs="*",
        default=None,
        help="Requested log10(L_Halpha) values; nearest available Sobral bins will be used.",
    )
    parser.add_argument("--figures-dir-name", default="figures_halpha_dust")
    parser.add_argument("--window", type=int, default=101, help="Rolling window size in sorted-point order.")
    return parser.parse_args()


def _choose_bins(columns: list[str], explicit: list[int] | None) -> list[int]:
    if explicit:
        return explicit
    n = len(columns)
    if n <= 2:
        return list(range(n))
    return [n // 3, (2 * n) // 3]


def _bin_sort_key(column: str) -> int:
    match = re.search(r"_bin(\d+)$", column)
    if match is None:
        raise ValueError(f"Could not parse bin index from column '{column}'")
    return int(match.group(1))


def _choose_bins_from_log10_l(
    long_table: pd.DataFrame,
    sobral_label: str,
    requested: list[float],
) -> tuple[list[int], list[float]]:
    subset = (
        long_table.loc[long_table["sobral_label"] == sobral_label, ["bin_index", "log10_luminosity_center"]]
        .drop_duplicates()
        .sort_values("bin_index")
        .reset_index(drop=True)
    )
    if subset.empty:
        raise ValueError(f"No long-table rows found for sobral label {sobral_label}")
    centers = subset["log10_luminosity_center"].to_numpy(dtype=float)
    bin_indices = subset["bin_index"].to_numpy(dtype=int)
    selected_bins: list[int] = []
    selected_centers: list[float] = []
    for value in requested:
        index = int(np.argmin(np.abs(centers - value)))
        selected_bins.append(int(bin_indices[index]))
        selected_centers.append(float(centers[index]))
    return selected_bins, selected_centers


def _rolling_mean_sorted(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if window < 3:
        return x_sorted, y_sorted
    window = min(window, len(x_sorted))
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_sorted, kernel, mode="valid")
    half = window // 2
    x_mid = x_sorted[half : len(x_sorted) - half]
    return x_mid, y_smooth


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(campaign_root / args.table_filename)
    output_columns = sorted(
        (
            column
            for column in table.columns
            if column.startswith(f"halpha_sobral_{args.sobral_label.lower()}_bin")
            and not column.endswith("_target")
            and not column.endswith("_target_std")
        ),
        key=_bin_sort_key,
    )
    selected_centers = None
    if args.log10_l_values:
        long_table = pd.read_csv(campaign_root / args.long_filename)
        selected_bins, selected_centers = _choose_bins_from_log10_l(long_table, args.sobral_label, args.log10_l_values)
    else:
        selected_bins = _choose_bins(output_columns, args.bins)

    fig, axes = plt.subplots(
        len(selected_bins),
        len(INPUT_COLUMNS),
        figsize=(3.1 * len(INPUT_COLUMNS), 2.8 * len(selected_bins)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, bin_index in enumerate(selected_bins):
        y = np.log10(np.clip(table[output_columns[bin_index]].to_numpy(dtype=float), 1.0e-12, None))
        for col_index, input_name in enumerate(INPUT_COLUMNS):
            axis = axes[row_index, col_index]
            x = table[input_name].to_numpy(dtype=float)
            axis.scatter(x, y, s=8, alpha=0.15, color="#345995")
            x_mid, y_smooth = _rolling_mean_sorted(x, y, args.window)
            axis.plot(x_mid, y_smooth, color="#d1495b", lw=2.0)
            axis.set_xlabel(input_name)
            if col_index == 0:
                if selected_centers is not None:
                    axis.set_ylabel(rf"$\log_{{10}} L_{{\mathrm{{H}}\alpha}}={selected_centers[row_index]:.1f}$")
                else:
                    axis.set_ylabel(output_columns[bin_index].replace("halpha_sobral_", "").replace("_", " "))
            axis.grid(alpha=0.2)

    suffix = "_raw"
    if selected_centers is not None:
        suffix = "_raw_log10L"
    path = figures_root / f"main_effects_halpha_{args.sobral_label.lower()}{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
