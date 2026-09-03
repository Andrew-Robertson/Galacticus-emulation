from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning

from fit_1d_smf_gp_cv import DEFAULT_ANALYSIS, _fit_scaled_gp, _load_campaign, _predict_scaled_gp, _space_filling_subset_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize how a 1D GP prediction for one SMF bin changes as the "
            "number of training points is increased."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--input-column", default="diskVelocityCharacteristic")
    parser.add_argument("--quantile-column", default="prior_quantile")
    parser.add_argument("--bin-index", type=int, default=12)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--n-restarts-optimizer", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--uncertainty-sigma", type=float, default=3.0)
    parser.add_argument("--known-noise", action="store_true", help="Use the SMF covariance diagonal as known GP alpha instead of an optimized WhiteKernel.")
    parser.add_argument("--show-data-errorbars", action="store_true", help="Plot all Galacticus points with covariance-derived log-space error bars.")
    parser.add_argument("--noise-floor-dex", type=float, default=1.0e-4)
    parser.add_argument("--figures-dir-name", default="figures_Mstar1e11bin")
    parser.add_argument("--output-name", default="gp_learning_curve_mstar_bin12.png")
    return parser.parse_args()


def _training_indices(n_samples: int, train_size: int) -> np.ndarray:
    if train_size >= n_samples:
        return np.arange(n_samples)
    return _space_filling_subset_indices(n_samples, train_size)


def _load_log10_noise(campaign_root: Path, samples: pd.DataFrame, analysis: str, hdf5_filename: str, bin_index: int, noise_floor_dex: float) -> np.ndarray:
    noise = []
    for evaluation_id in samples["evaluation_id"]:
        path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            phi = float(group["massFunction"][bin_index])
            variance_phi = float(group["massFunctionCovariance"][bin_index, bin_index])
        sigma = np.sqrt(max(variance_phi, 0.0)) / (phi * np.log(10.0))
        noise.append(max(sigma, noise_floor_dex))
    return np.asarray(noise, dtype=float)


def _fit_known_noise_gp(
    x: np.ndarray,
    y: np.ndarray,
    noise_dex: np.ndarray,
    *,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[GaussianProcessRegressor, float, float]:
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std == 0.0:
        y_std = 1.0
    kernel = ConstantKernel(1.0, (1.0e-3, 1.0e3)) * Matern(
        length_scale=np.array([0.25]),
        length_scale_bounds=(1.0e-2, 10.0),
        nu=2.5,
    )
    alpha = np.maximum((noise_dex / y_std) ** 2, 1.0e-10)
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    model.fit(x, (y - y_mean) / y_std)
    return model, y_mean, y_std


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    samples, mass_bins, _target, y = _load_campaign(campaign_root, args.analysis, args.hdf5_filename)
    if args.bin_index < 0 or args.bin_index >= len(mass_bins):
        raise ValueError(f"bin-index must be in [0, {len(mass_bins) - 1}]")

    x_quantile = samples[[args.quantile_column]].to_numpy(dtype=float)
    x_velocity = samples[args.input_column].to_numpy(dtype=float)
    y_bin = y[:, args.bin_index]
    noise_dex = _load_log10_noise(campaign_root, samples, args.analysis, args.hdf5_filename, args.bin_index, args.noise_floor_dex)
    grid_velocity = np.linspace(float(np.min(x_velocity)), float(np.max(x_velocity)), 400)
    grid_quantile = np.interp(grid_velocity, x_velocity, x_quantile[:, 0])[:, None]

    colors = {
        8: "#edae49",
        16: "#2a9d8f",
        32: "#2a6f97",
    }
    markers = {
        8: "+",
        16: "x",
    }
    fallback_colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(args.train_sizes)))

    fig, axis = plt.subplots(figsize=(8.2, 5.3), constrained_layout=True)
    if args.show_data_errorbars:
        axis.errorbar(
            x_velocity,
            y_bin,
            yerr=noise_dex,
            fmt="o",
            ms=4.5,
            color="black",
            ecolor="0.45",
            elinewidth=0.9,
            capsize=1.8,
            alpha=0.78,
            label="all Galacticus evaluations",
            zorder=4,
        )
    else:
        axis.scatter(
            x_velocity,
            y_bin,
            s=34,
            color="black",
            alpha=0.76,
            label="all Galacticus evaluations",
            zorder=4,
        )

    for color_index, train_size in enumerate(args.train_sizes):
        train_index = _training_indices(len(samples), train_size)
        color = colors.get(train_size, fallback_colors[color_index])
        if args.known_noise:
            model, y_mean, y_std = _fit_known_noise_gp(
                x_quantile[train_index],
                y_bin[train_index],
                noise_dex[train_index],
                n_restarts_optimizer=args.n_restarts_optimizer,
                random_state=args.random_state,
            )
        else:
            model, y_mean, y_std = _fit_scaled_gp(
                x_quantile[train_index],
                y_bin[train_index],
                n_restarts_optimizer=args.n_restarts_optimizer,
                random_state=args.random_state,
            )
        pred, pred_std = _predict_scaled_gp(model, y_mean, y_std, grid_quantile)
        label = f"GP trained on {len(train_index)} points"
        axis.plot(grid_velocity, pred, color=color, lw=2.2, label=label)
        band = args.uncertainty_sigma * pred_std
        axis.fill_between(
            grid_velocity,
            pred - band,
            pred + band,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        axis.plot(grid_velocity, pred - band, color=color, lw=0.65, alpha=0.45)
        axis.plot(grid_velocity, pred + band, color=color, lw=0.65, alpha=0.45)
        if train_size in markers:
            axis.scatter(
                x_velocity[train_index],
                y_bin[train_index],
                marker=markers[train_size],
                s=95 if train_size == 8 else 70,
                color=color,
                linewidth=2.0,
                zorder=5,
                label=f"{len(train_index)} training points",
            )

    axis.set_xlabel("diskVelocityCharacteristic [km/s]")
    axis.set_ylabel(r"$\log_{10}\Phi$")
    axis.set_title(
        rf"GP learning curve for SMF bin $\log_{{10}} M_\star={np.log10(mass_bins[args.bin_index]):.2f}$"
        + rf" (bands: $\pm {args.uncertainty_sigma:g}\sigma$)"
    )
    if args.known_noise:
        axis.text(
            0.02,
            0.03,
            "GP alpha from Galacticus SMF covariance",
            transform=axis.transAxes,
            fontsize=8,
            color="0.25",
        )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    output_path = figures_root / args.output_name
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
