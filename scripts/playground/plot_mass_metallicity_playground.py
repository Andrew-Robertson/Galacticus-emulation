from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HALO_BIN_TARGETS = np.array([1.0e11, 1.0e12, 1.0e13], dtype=float)
HALO_BIN_EDGES = np.array(
    [0.0, np.sqrt(HALO_BIN_TARGETS[0] * HALO_BIN_TARGETS[1]), np.sqrt(HALO_BIN_TARGETS[1] * HALO_BIN_TARGETS[2]), np.inf],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Galacticus stellar-mass versus gas-phase metallicity diagnostics for a single output file."
    )
    parser.add_argument("galacticus_output", help="Path to a Galacticus HDF5 output file.")
    parser.add_argument("observation_file", help="Path to the Blanc 2019 mass-metallicity HDF5 file.")
    parser.add_argument("output_dir", help="Directory where plots and tables will be written.")
    parser.add_argument(
        "--hydrogen-mass-fraction",
        type=float,
        default=0.70,
        help="Assumed hydrogen mass fraction X used when approximately converting total gas metallicity to O/H.",
    )
    parser.add_argument(
        "--oxygen-metal-mass-fraction",
        type=float,
        default=0.44,
        help="Assumed fraction of total metal mass that is oxygen, M_O / M_Z.",
    )
    return parser.parse_args()


def _running_median(x: np.ndarray, y: np.ndarray, n_bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    edges = np.linspace(0, len(x_sorted), n_bins + 1, dtype=int)
    x_mid = []
    y_mid = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        if end <= start:
            continue
        x_chunk = x_sorted[start:end]
        y_chunk = y_sorted[start:end]
        if len(x_chunk) == 0:
            continue
        x_mid.append(float(np.median(x_chunk)))
        y_mid.append(float(np.median(y_chunk)))
    return np.asarray(x_mid), np.asarray(y_mid)


def _approximate_oxygen_abundance_12logoh(
    metallicity_fraction: np.ndarray,
    hydrogen_mass_fraction: float,
    oxygen_metal_mass_fraction: float,
) -> np.ndarray:
    return 12.0 + np.log10((oxygen_metal_mass_fraction * metallicity_fraction) / (16.0 * hydrogen_mass_fraction))


def _halo_bin_summary(
    halo_mass: np.ndarray,
    log10_stellar_mass: np.ndarray,
    approx_oxygen_abundance: np.ndarray,
) -> pd.DataFrame:
    bin_index = np.digitize(halo_mass, HALO_BIN_EDGES[1:-1])
    rows: list[dict[str, float | int]] = []
    for index, target_mass in enumerate(HALO_BIN_TARGETS):
        mask = bin_index == index
        if not np.any(mask):
            rows.append(
                {
                    "halo_bin_index": index,
                    "halo_mass_target": target_mass,
                    "n_galaxies": 0,
                    "mean_log10_stellar_mass": np.nan,
                    "mean_abundance_oxygen_12_plus_log10_oh": np.nan,
                }
            )
            continue
        rows.append(
            {
                "halo_bin_index": index,
                "halo_mass_target": target_mass,
                "n_galaxies": int(np.count_nonzero(mask)),
                "mean_log10_stellar_mass": float(np.mean(log10_stellar_mass[mask])),
                "mean_abundance_oxygen_12_plus_log10_oh": float(np.mean(approx_oxygen_abundance[mask])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.galacticus_output, "r") as handle:
        node_data = handle["Outputs/Output1/nodeData"]
        disk_mass_stellar = node_data["diskMassStellar"][:]
        spheroid_mass_stellar = node_data["spheroidMassStellar"][:]
        disk_mass_gas = node_data["diskMassGas"][:]
        spheroid_mass_gas = node_data["spheroidMassGas"][:]
        disk_metals_gas = node_data["diskAbundancesGasMetals"][:]
        spheroid_metals_gas = node_data["spheroidAbundancesGasMetals"][:]
        basic_mass = node_data["basicMass"][:]
        node_is_isolated = node_data["nodeIsIsolated"][:].astype(bool)

    stellar_mass = disk_mass_stellar + spheroid_mass_stellar
    gas_mass = disk_mass_gas + spheroid_mass_gas
    gas_metals = disk_metals_gas + spheroid_metals_gas

    valid = (
        node_is_isolated
        & (stellar_mass > 0.0)
        & (gas_mass > 0.0)
        & np.isfinite(stellar_mass)
        & np.isfinite(gas_mass)
        & np.isfinite(gas_metals)
    )
    metallicity = np.full_like(stellar_mass, np.nan, dtype=float)
    metallicity[valid] = gas_metals[valid] / gas_mass[valid]

    log10_stellar_mass = np.log10(stellar_mass[valid])
    metallicity_valid = metallicity[valid]
    positive_metallicity = metallicity_valid > 0.0
    log10_metallicity = np.log10(metallicity_valid[positive_metallicity])
    log10_stellar_mass_for_logz = log10_stellar_mass[positive_metallicity]
    approx_oxygen_abundance = _approximate_oxygen_abundance_12logoh(
        metallicity_valid[positive_metallicity],
        hydrogen_mass_fraction=args.hydrogen_mass_fraction,
        oxygen_metal_mass_fraction=args.oxygen_metal_mass_fraction,
    )
    basic_mass_valid = basic_mass[valid][positive_metallicity]
    halo_bin_summary = _halo_bin_summary(
        halo_mass=basic_mass_valid,
        log10_stellar_mass=log10_stellar_mass_for_logz,
        approx_oxygen_abundance=approx_oxygen_abundance,
    )

    galaxy_table = pd.DataFrame(
        {
            "log10_stellar_mass": log10_stellar_mass,
            "gas_metallicity_fraction": metallicity_valid,
        }
    )
    galaxy_table.to_csv(output_dir / "galacticus_central_mass_metallicity.csv", index=False)

    converted_table = pd.DataFrame(
        {
            "log10_stellar_mass": log10_stellar_mass_for_logz,
            "gas_metallicity_fraction": metallicity_valid[positive_metallicity],
            "approx_abundance_oxygen_12_plus_log10_oh": approx_oxygen_abundance,
            "assumed_hydrogen_mass_fraction": args.hydrogen_mass_fraction,
            "assumed_oxygen_metal_mass_fraction": args.oxygen_metal_mass_fraction,
        }
    )
    converted_table.to_csv(output_dir / "galacticus_central_mass_metallicity_approx_oxygen.csv", index=False)
    halo_bin_summary.to_csv(output_dir / "galacticus_mass_metallicity_halo_bin_means.csv", index=False)

    with h5py.File(args.observation_file, "r") as handle:
        obs_mass_stellar = handle["massStellar"][:]
        obs_abundance_mean = handle["abundanceOxygenMean"][:]
        obs_abundance_lo = handle["abundanceOxygen16PercentCI"][:]
        obs_abundance_hi = handle["abundanceOxygen84PercentCI"][:]

    obs_table = pd.DataFrame(
        {
            "log10_stellar_mass": np.log10(obs_mass_stellar),
            "abundance_oxygen_mean": obs_abundance_mean,
            "abundance_oxygen_16pct_ci": obs_abundance_lo,
            "abundance_oxygen_84pct_ci": obs_abundance_hi,
        }
    )
    obs_table.to_csv(output_dir / "blanc2019_mass_metallicity.csv", index=False)

    median_x, median_y = _running_median(log10_stellar_mass, metallicity_valid)
    median_logx, median_logy = _running_median(log10_stellar_mass_for_logz, log10_metallicity)
    median_ox_x, median_ox_y = _running_median(log10_stellar_mass_for_logz, approx_oxygen_abundance)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 10), sharex=True, constrained_layout=True)

    axes[0].scatter(log10_stellar_mass, metallicity_valid, s=10, alpha=0.35, color="#1f77b4")
    if len(median_x) > 0:
        axes[0].plot(median_x, median_y, color="#d62728", linewidth=2.0, label="Running median")
        axes[0].legend(frameon=False)
    axes[0].set_ylabel("Gas Metallicity Fraction")
    axes[0].set_title("Galacticus z=0 central galaxies")

    axes[1].scatter(log10_stellar_mass_for_logz, log10_metallicity, s=10, alpha=0.35, color="#1f77b4")
    if len(median_logx) > 0:
        axes[1].plot(median_logx, median_logy, color="#d62728", linewidth=2.0, label="Running median")
    axes[1].errorbar(
        np.log10(obs_mass_stellar),
        obs_abundance_mean,
        yerr=np.vstack([obs_abundance_lo, obs_abundance_hi]),
        fmt="o",
        color="#2ca02c",
        markersize=3,
        linewidth=1.0,
        alpha=0.8,
        label="Blanc 2019 (12+log10(O/H))",
    )
    axes[1].legend(frameon=False)
    axes[1].set_xlabel("log10(Mstar / Msun)")
    axes[1].set_ylabel("log10(Zgas) / Blanc 2019 oxygen abundance")
    axes[1].set_title("Not directly comparable units; shown together only for trend inspection")

    figure_path = output_dir / "mass_metallicity_diagnostic.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    converted_fig, converted_ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    converted_ax.scatter(
        log10_stellar_mass_for_logz,
        approx_oxygen_abundance,
        s=10,
        alpha=0.35,
        color="#1f77b4",
        label="Galacticus central galaxies",
    )
    if len(median_ox_x) > 0:
        converted_ax.plot(median_ox_x, median_ox_y, color="#d62728", linewidth=2.0, label="Running median")
    converted_ax.scatter(
        halo_bin_summary["mean_log10_stellar_mass"],
        halo_bin_summary["mean_abundance_oxygen_12_plus_log10_oh"],
        s=80,
        marker="D",
        color="#ff7f0e",
        edgecolors="black",
        linewidths=0.7,
        zorder=5,
        label="Halo-bin means",
    )
    for row in halo_bin_summary.itertuples(index=False):
        if np.isfinite(row.mean_log10_stellar_mass) and np.isfinite(row.mean_abundance_oxygen_12_plus_log10_oh):
            converted_ax.annotate(
                f"$10^{{{int(np.log10(row.halo_mass_target))}}} M_\\odot$",
                (row.mean_log10_stellar_mass, row.mean_abundance_oxygen_12_plus_log10_oh),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    converted_ax.errorbar(
        np.log10(obs_mass_stellar),
        obs_abundance_mean,
        yerr=np.vstack([obs_abundance_lo, obs_abundance_hi]),
        fmt="o",
        color="#2ca02c",
        markersize=3,
        linewidth=1.0,
        alpha=0.8,
        label="Blanc 2019",
    )
    converted_ax.legend(frameon=False)
    converted_ax.set_xlabel("log10(Mstar / Msun)")
    converted_ax.set_ylabel("12 + log10(O/H)")
    converted_ax.set_title(
        "Approximate oxygen abundance from gas-phase total Z\n"
        f"(assuming X={args.hydrogen_mass_fraction:.2f}, M_O/M_Z={args.oxygen_metal_mass_fraction:.2f})"
    )
    converted_figure_path = output_dir / "mass_metallicity_approx_oxygen_comparison.png"
    converted_fig.savefig(converted_figure_path, dpi=180)
    plt.close(converted_fig)

    print(output_dir / "galacticus_central_mass_metallicity.csv")
    print(output_dir / "galacticus_central_mass_metallicity_approx_oxygen.csv")
    print(output_dir / "galacticus_mass_metallicity_halo_bin_means.csv")
    print(output_dir / "blanc2019_mass_metallicity.csv")
    print(figure_path)
    print(converted_figure_path)


if __name__ == "__main__":
    main()
