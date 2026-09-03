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
            "Plot random dust-attenuated Halpha luminosity functions from a campaign, "
            "colored by a chosen input parameter."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--sobral-label", default="Z1", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--n-curves", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-by", default="diskVelocityCharacteristic")
    parser.add_argument("--figures-dir-name", default="figures_halpha_dust")
    return parser.parse_args()


def _bin_sort_key(column: str) -> int:
    match = re.search(r"_bin(\d+)$", column)
    if match is None:
        raise ValueError(f"Could not parse bin index from column '{column}'")
    return int(match.group(1))


def _output_columns(table: pd.DataFrame, sobral_label: str) -> list[str]:
    return sorted(
        [
            column
            for column in table.columns
            if column.startswith(f"halpha_sobral_{sobral_label.lower()}_bin")
            and not column.endswith("_target")
            and not column.endswith("_target_std")
        ],
        key=_bin_sort_key,
    )


def _log10_with_floor(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("No positive values available for log10 transform.")
    floor = 0.5 * float(np.min(positive))
    safe = np.where(values > 0.0, values, floor)
    return np.log10(safe)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(campaign_root / args.table_filename)
    long_table = pd.read_csv(campaign_root / args.long_filename)
    output_columns = _output_columns(table, args.sobral_label)

    if args.color_by not in table.columns:
        raise KeyError(f"{args.color_by} not present in {args.table_filename}")

    long_subset = long_table.loc[long_table["sobral_label"] == args.sobral_label].copy()
    centers = (
        long_subset[["bin_index", "log10_luminosity_center"]]
        .drop_duplicates()
        .sort_values("bin_index")
    )
    x = centers["log10_luminosity_center"].to_numpy(dtype=float)
    if len(x) != len(output_columns):
        raise ValueError("Mismatch between wide-table output columns and long-table luminosity centers.")

    rng = np.random.default_rng(args.seed)
    sample_size = min(args.n_curves, len(table))
    chosen = np.sort(rng.choice(len(table), size=sample_size, replace=False))
    subset = table.iloc[chosen].copy()

    y = _log10_with_floor(subset[output_columns].to_numpy(dtype=float))
    target = _log10_with_floor(subset[[f"{col}_target" for col in output_columns]].iloc[[0]].to_numpy(dtype=float).ravel())
    target_std_linear = subset[[f"{col}_target_std" for col in output_columns]].iloc[[0]].to_numpy(dtype=float).ravel()
    target_linear = subset[[f"{col}_target" for col in output_columns]].iloc[[0]].to_numpy(dtype=float).ravel()
    target_std_log10 = target_std_linear / (np.maximum(target_linear, 1.0e-30) * np.log(10.0))

    color_values = subset[args.color_by].to_numpy(dtype=float)
    norm = plt.Normalize(vmin=float(np.min(color_values)), vmax=float(np.max(color_values)))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
    for curve, color_value in zip(y, color_values, strict=True):
        ax.plot(x, curve, color=cmap(norm(color_value)), alpha=0.32, lw=1.2)

    ax.plot(x, target, color="black", lw=2.2, label="Sobral target")
    ax.fill_between(x, target - target_std_log10, target + target_std_log10, color="black", alpha=0.12)
    ax.set_xlabel(r"$\log_{10} L_{\mathrm{H}\alpha}$")
    ax.set_ylabel(r"$\log_{10}\,\mathrm{d}n/\mathrm{d}\ln L$")
    ax.set_title(f"{sample_size} random z={long_subset['redshift'].iloc[0]:.2f} Halpha LFs")
    ax.set_ylim(-7.0, -1.0)
    ax.grid(alpha=0.2)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(args.color_by)

    output_path = figures_root / f"random_halpha_lfs_{args.sobral_label.lower()}_colorby_{args.color_by}.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
