from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import h5py
import matplotlib.pyplot as plt
import numpy as np

from calculate_halpha_dust_lf_grid import _combined_node_weights  # type: ignore
from plot_comparat_oii_example import _lf_per_dex, _oii_luminosity  # type: ignore


DEFAULT_GALACTICUS_FILE = (
    REPO_ROOT
    / "runs/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_19p_512/evaluations/"
    / "sobol_mass_function_emissionlines_dust_simpleSizes_19p_512-eval-0000/galacticus.hdf5"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "playing/emission_line_lf_comparisons/comparat_oii_eval0000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot intrinsic Galacticus [OII] luminosity functions for all output times on common bins."
    )
    parser.add_argument("--galacticus-file", type=Path, default=DEFAULT_GALACTICUS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log10-lmin", type=float, default=40.75)
    parser.add_argument("--log10-lmax", type=float, default=43.25)
    parser.add_argument("--bin-width", type=float, default=0.25)
    parser.add_argument("--include-agn", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _output_sort_key(output_name: str) -> int:
    return int(output_name.replace("Output", ""))


def _output_redshift(output_group: h5py.Group) -> float:
    expansion_factor = float(output_group.attrs["outputExpansionFactor"])
    return 1.0 / expansion_factor - 1.0


def _common_edges(log10_lmin: float, log10_lmax: float, bin_width: float) -> np.ndarray:
    edge_count = int(round((log10_lmax - log10_lmin) / bin_width))
    return log10_lmin + bin_width * np.arange(edge_count + 1, dtype=float)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = _common_edges(args.log10_lmin, args.log10_lmax, args.bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows: list[dict[str, float | int | str | bool]] = []
    curves: list[tuple[str, float, np.ndarray]] = []

    with h5py.File(args.galacticus_file.resolve(), "r") as handle:
        for output_name in sorted(handle["Outputs"], key=_output_sort_key):
            output_group = handle[f"Outputs/{output_name}"]
            redshift = _output_redshift(output_group)
            luminosity = _oii_luminosity(output_group, include_agn=args.include_agn)
            weights = _combined_node_weights(output_group)
            phi, phi_std = _lf_per_dex(luminosity, weights, edges)
            curves.append((output_name, redshift, phi))
            for bin_index, (center, value, sigma) in enumerate(zip(centers, phi, phi_std, strict=True)):
                rows.append(
                    {
                        "output_name": output_name,
                        "output_index": _output_sort_key(output_name),
                        "redshift": redshift,
                        "include_agn": args.include_agn,
                        "bin_index": bin_index,
                        "log10_luminosity_center": float(center),
                        "phi_mpc3_dex": float(value),
                        "phi_shot_noise_std_mpc3_dex": float(sigma),
                    }
                )

    csv_path = output_dir / "galacticus_eval0000_intrinsic_oii_all_outputs_common_bins.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(redshift for _, redshift, _ in curves), max(redshift for _, redshift, _ in curves))
    for output_name, redshift, phi in curves:
        color = cmap(norm(redshift))
        axis.plot(
            centers,
            phi,
            marker="o",
            ms=3.0,
            lw=1.3,
            color=color,
            alpha=0.9,
            label=f"{output_name}, z={redshift:.2f}",
        )
    axis.set_yscale("log")
    axis.set_ylim(1.0e-8, 3.0e-2)
    axis.set_xlim(edges[0], edges[-1])
    axis.set_xlabel(r"$\log_{10}(L_{\mathrm{[OII]}}/\mathrm{erg\,s^{-1}})$")
    axis.set_ylabel(r"$\phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    axis.set_title(r"Intrinsic Galacticus [OII] luminosity functions on common 0.25 dex bins")
    axis.grid(alpha=0.18)
    axis.legend(ncol=2, fontsize=7, frameon=False, loc="lower left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axis, pad=0.015)
    cbar.set_label("redshift")

    fig_path = output_dir / "galacticus_eval0000_intrinsic_oii_all_outputs_common_bins.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(csv_path)
    print(fig_path)


if __name__ == "__main__":
    main()
