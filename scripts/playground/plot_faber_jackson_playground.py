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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a Faber-Jackson-like diagnostic using Galacticus spheroidVelocity.")
    parser.add_argument("galacticus_output", help="Path to a Galacticus HDF5 output file.")
    parser.add_argument("output_dir", help="Directory where plot and table will be written.")
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
        spheroid_mass_stellar = node_data["spheroidMassStellar"][:]
        disk_mass_stellar = node_data["diskMassStellar"][:]
        basic_mass = node_data["basicMass"][:] if "basicMass" in node_data else np.full_like(spheroid_velocity, np.nan)
        node_is_isolated = (
            node_data["nodeIsIsolated"][:].astype(bool)
            if "nodeIsIsolated" in node_data
            else np.ones_like(spheroid_velocity, dtype=bool)
        )

    total_mass_stellar = disk_mass_stellar + spheroid_mass_stellar
    spheroid_fraction = np.divide(
        spheroid_mass_stellar,
        total_mass_stellar,
        out=np.full_like(spheroid_mass_stellar, np.nan, dtype=float),
        where=total_mass_stellar > 0.0,
    )
    valid = (
        (spheroid_velocity > 0.0)
        & (spheroid_mass_stellar > 0.0)
        & np.isfinite(spheroid_velocity)
        & np.isfinite(spheroid_mass_stellar)
    )
    if not include_satellites:
        valid &= node_is_isolated

    return pd.DataFrame(
        {
            "spheroid_velocity": spheroid_velocity[valid],
            "spheroid_mass_stellar": spheroid_mass_stellar[valid],
            "disk_mass_stellar": disk_mass_stellar[valid],
            "total_mass_stellar": total_mass_stellar[valid],
            "spheroid_fraction": spheroid_fraction[valid],
            "basic_mass": basic_mass[valid],
            "node_is_isolated": node_is_isolated[valid],
            "log10_spheroid_velocity": np.log10(spheroid_velocity[valid]),
            "log10_spheroid_mass_stellar": np.log10(spheroid_mass_stellar[valid]),
            "log10_total_mass_stellar": np.log10(total_mass_stellar[valid]),
            "log10_basic_mass": np.log10(basic_mass[valid]),
        }
    )


def _reference_mass(sigma: np.ndarray, mass_at_200: float) -> np.ndarray:
    return mass_at_200 * (sigma / 200.0) ** 4


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = _load_galacticus(Path(args.galacticus_output).resolve(), args.include_satellites)
    table.to_csv(output_dir / "galacticus_faber_jackson.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True, sharex=True)
    sigma_grid = np.logspace(np.log10(20.0), np.log10(500.0), 200)

    for axis, y_column, ylabel, title in [
        (axes[0], "spheroid_mass_stellar", r"Spheroid stellar mass [$M_\odot$]", "Spheroid-mass Faber-Jackson-like"),
        (axes[1], "total_mass_stellar", r"Total stellar mass [$M_\odot$]", "Total stellar mass for context"),
    ]:
        scatter = axis.scatter(
            table["spheroid_velocity"],
            table[y_column],
            c=table["spheroid_fraction"],
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            s=48,
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
        )
        for mass_at_200, linestyle in [(1.0e10, ":"), (3.0e10, "--"), (1.0e11, "-.")]:
            axis.plot(
                sigma_grid,
                _reference_mass(sigma_grid, mass_at_200),
                linestyle=linestyle,
                linewidth=1.4,
                color="#555555",
                alpha=0.75,
                label=rf"$M(200\,\mathrm{{km\,s^{{-1}}}})={mass_at_200:.0e} M_\odot$",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"Spheroid velocity, $\sigma$ proxy [km s$^{-1}$]")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25, linewidth=0.5, which="both")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    colorbar = fig.colorbar(scatter, ax=axes, pad=0.015)
    colorbar.set_label(r"$M_{\star,\rm spheroid}/M_{\star,\rm total}$")

    figure_path = output_dir / "faber_jackson_diagnostic.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(output_dir / "galacticus_faber_jackson.csv")
    print(figure_path)


if __name__ == "__main__":
    main()
