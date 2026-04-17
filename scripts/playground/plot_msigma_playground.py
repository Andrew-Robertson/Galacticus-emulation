from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OBSERVATION_FILE = (
    "/Users/arobertson/Documents/Projects/GalacticusEmu/Galacticus/"
    "datasets/static/observations/blackHoles/blackHoleMassVsVelocityDispersion_McConnellMa2013.hdf5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a Galacticus M-sigma diagnostic with observational overlays.")
    parser.add_argument("galacticus_output", help="Path to a Galacticus HDF5 output file.")
    parser.add_argument("output_dir", help="Directory where plot and table will be written.")
    parser.add_argument(
        "--observation-file",
        default=DEFAULT_OBSERVATION_FILE,
        help="Path to the McConnell & Ma 2013 HDF5 observation file.",
    )
    parser.add_argument(
        "--include-satellites",
        action="store_true",
        help="Plot all galaxies instead of only central galaxies.",
    )
    return parser.parse_args()


def _load_galacticus(path: Path, include_satellites: bool) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        node_data = handle["Outputs/Output1/nodeData"]
        spheroid_velocity = node_data["spheroidVelocity"][:]
        black_hole_mass = node_data["blackHoleMass"][:]
        basic_mass = node_data["basicMass"][:] if "basicMass" in node_data else np.full_like(black_hole_mass, np.nan)
        stellar_mass = (
            node_data["diskMassStellar"][:] + node_data["spheroidMassStellar"][:]
            if "diskMassStellar" in node_data and "spheroidMassStellar" in node_data
            else np.full_like(black_hole_mass, np.nan)
        )
        node_is_isolated = (
            node_data["nodeIsIsolated"][:].astype(bool)
            if "nodeIsIsolated" in node_data
            else np.ones_like(black_hole_mass, dtype=bool)
        )

    valid = (
        (spheroid_velocity > 0.0)
        & (black_hole_mass > 0.0)
        & np.isfinite(spheroid_velocity)
        & np.isfinite(black_hole_mass)
    )
    if not include_satellites:
        valid &= node_is_isolated

    return pd.DataFrame(
        {
            "spheroid_velocity": spheroid_velocity[valid],
            "black_hole_mass": black_hole_mass[valid],
            "basic_mass": basic_mass[valid],
            "stellar_mass": stellar_mass[valid],
            "node_is_isolated": node_is_isolated[valid],
            "log10_spheroid_velocity": np.log10(spheroid_velocity[valid]),
            "log10_black_hole_mass": np.log10(black_hole_mass[valid]),
            "log10_basic_mass": np.log10(basic_mass[valid]),
            "log10_stellar_mass": np.log10(stellar_mass[valid]),
        }
    )


def _load_observations(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "velocity": handle["velocityDispersion"][:],
            "mass": handle["massBlackHole"][:],
            "mass_err_low": handle["massBlackHoleError_lower68"][:],
            "mass_err_high": handle["massBlackHoleError_upper68"][:],
            "velocity_binned": handle["velocityDispersionBinned"][:],
            "velocity_binned_err": handle["velocityDispersionError"][:],
            "mass_mean": handle["massBlackHoleMean"][:],
            "mass_mean_err": handle["massBlackHoleMeanError"][:],
        }


def _mcconnell_ma_2013_relation(sigma: np.ndarray) -> np.ndarray:
    # McConnell & Ma (2013), all-galaxy fit: log10(M_BH/Msun) = 8.32 + 5.64 log10(sigma / 200 km/s).
    return 10.0 ** (8.32 + 5.64 * np.log10(sigma / 200.0))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    galacticus_table = _load_galacticus(Path(args.galacticus_output).resolve(), args.include_satellites)
    observations = _load_observations(Path(args.observation_file).resolve())

    galacticus_table.to_csv(output_dir / "galacticus_msigma.csv", index=False)

    obs_table = pd.DataFrame(
        {
            "log10_velocity_dispersion": observations["velocity"],
            "log10_black_hole_mass": observations["mass"],
            "log10_black_hole_mass_error_low": observations["mass_err_low"],
            "log10_black_hole_mass_error_high": observations["mass_err_high"],
        }
    )
    obs_table.to_csv(output_dir / "mcconnell_ma_2013_msigma.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 6.6), constrained_layout=True)

    ax.scatter(
        10.0 ** observations["velocity"],
        10.0 ** observations["mass"],
        color="#9a9a9a",
        s=20,
        alpha=0.65,
        label="McConnell & Ma 2013 galaxies",
        zorder=1,
    )
    ax.errorbar(
        10.0 ** observations["velocity_binned"],
        10.0 ** observations["mass_mean"],
        xerr=np.vstack(
            [
                10.0 ** observations["velocity_binned"] - 10.0 ** (observations["velocity_binned"] - observations["velocity_binned_err"]),
                10.0 ** (observations["velocity_binned"] + observations["velocity_binned_err"]) - 10.0 ** observations["velocity_binned"],
            ]
        ),
        yerr=np.vstack(
            [
                10.0 ** observations["mass_mean"] - 10.0 ** (observations["mass_mean"] - observations["mass_mean_err"]),
                10.0 ** (observations["mass_mean"] + observations["mass_mean_err"]) - 10.0 ** observations["mass_mean"],
            ]
        ),
        fmt="s",
        color="#111111",
        ecolor="#111111",
        markersize=5,
        linewidth=1.1,
        label="McConnell & Ma 2013 binned means",
        zorder=3,
    )

    sigma_grid = np.logspace(np.log10(30.0), np.log10(500.0), 200)
    ax.plot(
        sigma_grid,
        _mcconnell_ma_2013_relation(sigma_grid),
        color="#d1495b",
        linewidth=2.0,
        label=r"McConnell & Ma 2013 fit",
        zorder=2,
    )

    scatter = ax.scatter(
        galacticus_table["spheroid_velocity"],
        galacticus_table["black_hole_mass"],
        c=galacticus_table["log10_basic_mass"],
        cmap="viridis",
        s=46,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.9,
        label="Galacticus centrals" if not args.include_satellites else "Galacticus galaxies",
        zorder=4,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label(r"$\log_{10}(M_{\rm halo}/M_\odot)$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Spheroid velocity, $\sigma$ proxy [km s$^{-1}$]")
    ax.set_ylabel(r"Black hole mass, $M_{\rm BH}$ [$M_\odot$]")
    ax.set_title("z=0 Galacticus M-sigma diagnostic")
    ax.grid(alpha=0.25, linewidth=0.5, which="both")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    figure_path = output_dir / "msigma_diagnostic.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(output_dir / "galacticus_msigma.csv")
    print(output_dir / "mcconnell_ma_2013_msigma.csv")
    print(figure_path)


if __name__ == "__main__":
    main()
