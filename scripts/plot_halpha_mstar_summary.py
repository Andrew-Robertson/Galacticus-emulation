from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np


GROUPS = ["halphaSobralZ1", "halphaSobralZ2", "halphaSobralZ3", "halphaSobralZ4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2D Halpha-Mstar summaries and marginalized luminosity functions."
    )
    parser.add_argument("summary_hdf5", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--group-name", default="emissionLineSummaries")
    return parser.parse_args()


def _density_per_dex2(counts: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    dx = np.diff(x_edges)[:, None]
    dy = np.diff(y_edges)[None, :]
    return counts / (dx * dy)


def _lf_per_dln(counts: np.ndarray, log10_edges: np.ndarray) -> np.ndarray:
    dlog10 = np.diff(log10_edges)
    return counts / (dlog10 * np.log(10.0))


def _plot_2d(root: h5py.Group, output_path: Path) -> None:
    x_edges = np.asarray(root["log10_luminosity_edges"][...], dtype=float)
    y_edges = np.asarray(root["log10_stellar_mass_edges"][...], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), sharex=True, sharey=True, constrained_layout=True)

    densities = []
    labels = []
    for name in GROUPS:
        group = root[name]
        densities.append(_density_per_dex2(np.asarray(group["counts_2d"][...], dtype=float), x_edges, y_edges))
        labels.append(f"z={float(group.attrs['redshift']):.2f}")
    positive = np.concatenate([density[density > 0.0] for density in densities if np.any(density > 0.0)])
    vmin = np.log10(np.nanmin(positive)) if positive.size else -8.0
    vmax = np.log10(np.nanmax(positive)) if positive.size else 0.0

    mesh = None
    for axis, name, density, label in zip(axes.ravel(), GROUPS, densities, labels, strict=True):
        plot_values = np.full_like(density.T, np.nan, dtype=float)
        mask = density.T > 0.0
        plot_values[mask] = np.log10(density.T[mask])
        mesh = axis.pcolormesh(x_edges, y_edges, plot_values, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        group = root[name]
        axis.set_title(f"{label} ({int(group['n_objects_used'][...])} galaxies)")
        axis.set_xlabel(r"$\log_{10}(L_{\mathrm{H}\alpha,\mathrm{int}}/\mathrm{erg\,s^{-1}})$")
        axis.set_ylabel(r"$\log_{10}(M_\star/M_\odot)$")
        axis.grid(alpha=0.12)
    fig.colorbar(mesh, ax=axes, label=r"$\log_{10}[\mathrm{d}n/(\mathrm{d}\log_{10}L\,\mathrm{d}\log_{10}M_\star)]$")
    fig.suptitle(r"Intrinsic H$\alpha$-stellar mass distribution", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_lf(root: h5py.Group, output_path: Path) -> None:
    fine_edges = np.asarray(root["log10_luminosity_edges"][...], dtype=float)
    fine_centers = np.asarray(root["log10_luminosity_centers"][...], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), sharey=True, constrained_layout=True)

    for axis, name in zip(axes.ravel(), GROUPS, strict=True):
        group = root[name]
        z = float(group.attrs["redshift"])
        fine_counts = np.asarray(group["intrinsic_lf_fine"][...], dtype=float)
        fine_lf = _lf_per_dln(fine_counts, fine_edges)

        sobral_centers = np.asarray(group["sobral_bin_centers_log10_luminosity"][...], dtype=float)
        sobral_edges = np.asarray(group["sobral_bin_edges_log10_luminosity"][...], dtype=float)
        sobral_counts = np.asarray(group["intrinsic_lf_sobral_bins"][...], dtype=float)
        sobral_lf_from_counts = _lf_per_dln(sobral_counts, sobral_edges)
        sobral_analysis_lf = np.asarray(group["sobral_analysis_lf"][...], dtype=float)
        sobral_target_lf = np.asarray(group["sobral_target_lf"][...], dtype=float)

        axis.plot(fine_centers, fine_lf, color="#2a6f97", lw=1.8, label="Marginalized 2D histogram")
        axis.plot(sobral_centers, sobral_lf_from_counts, "o", color="#2a6f97", ms=4.5, label="Same, rebinned to Sobral")
        axis.plot(sobral_centers, sobral_analysis_lf, color="#d1495b", lw=1.4, ls="--", label="Galacticus Sobral analysis")
        axis.plot(sobral_centers, sobral_target_lf, color="0.35", lw=1.2, ls=":", label="Sobral target")
        axis.set_yscale("log")
        axis.set_xlabel(r"$\log_{10}(L_{\mathrm{H}\alpha,\mathrm{int}}/\mathrm{erg\,s^{-1}})$")
        axis.set_ylabel(r"$\mathrm{d}n/\mathrm{d}\ln L\ [\mathrm{Mpc}^{-3}]$")
        axis.set_title(f"z={z:.2f}")
        axis.grid(alpha=0.18)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(r"Intrinsic H$\alpha$ luminosity function from the 2D summary", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_hdf5 = args.summary_hdf5.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else summary_hdf5.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(summary_hdf5, "r") as handle:
        root = handle[args.group_name]
        _plot_2d(root, output_dir / "halpha_mstar_2d_histograms.png")
        _plot_lf(root, output_dir / "halpha_lf_marginal_comparison.png")

    print(output_dir / "halpha_mstar_2d_histograms.png")
    print(output_dir / "halpha_lf_marginal_comparison.png")


if __name__ == "__main__":
    main()
