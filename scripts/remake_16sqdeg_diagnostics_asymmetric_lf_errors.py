from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu.plotting import finite_limits_from_values
from plot_comparat_oii_example import _load_comparat_lf
from plot_khostovan_hbeta_oiii_example import _load_khostovan_lf


DEFAULT_DIAGNOSTICS_DIR = Path.home() / "Downloads" / "16sqDeg_diagnostics"
DEFAULT_HALPHA_TARGET_HDF5 = (
    REPO_ROOT
    / "runs/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_freeYield_20p_moreHalos_512_hybrid_5fallback"
    / "pipeline/MCMCs/standard_observables_plus_emission_line_lfs"
    / "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99"
    / "bestFitModel/halpha_sobral_1dustdraw_pca99_romanEPS_massFunction.hdf5"
)
DEFAULT_COMPARAT_DIR = REPO_ROOT / "data/observations/emission_line_lfs/comparat_oii_2015/J_A+A_575_A40"

FIGURES = {
    "halpha_sobral": {
        "path": "emission_line_lfs_halpha.png",
        "figsize": (10.8, 7.6),
        "nrows": 2,
        "ncols": 2,
    },
    "oii_khostovan": {
        "path": "emission_line_lfs_oii_khostovan.png",
        "figsize": (10.8, 7.6),
        "nrows": 2,
        "ncols": 2,
    },
    "hbeta_oiii_khostovan": {
        "path": "emission_line_lfs_oiii_hbeta.png",
        "figsize": (10.8, 7.6),
        "nrows": 2,
        "ncols": 2,
    },
    "oii_comparat": {
        "path": "emission_line_lfs_oii_comparat.png",
        "figsize": (10.8, 15.2),
        "nrows": 4,
        "ncols": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw 16 sq. deg. diagnostic emission-line LF plots with asymmetric log-space error bars."
    )
    parser.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_DIAGNOSTICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--halpha-target-hdf5", type=Path, default=DEFAULT_HALPHA_TARGET_HDF5)
    parser.add_argument("--comparat-vizier-dir", type=Path, default=DEFAULT_COMPARAT_DIR)
    parser.add_argument("--min-log10-phi", type=float, default=-8.0)
    parser.add_argument("--lower-phi-floor-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-log10-yerr-upper",
        type=float,
        default=np.nan,
        help=(
            "Presentation cap on the upward log-space errorbar extension after converting "
            "linear-Phi errors. Use NaN or a negative value to disable."
        ),
    )
    parser.add_argument(
        "--halpha-target-error-mode",
        choices=["poisson-proxy", "covariance"],
        default="poisson-proxy",
        help=(
            "Use a monotonic Poisson-like proxy for Sobral target errors, or use the stored "
            "target covariance directly."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _asymmetric_log10_yerr(
    phi: np.ndarray,
    sigma_phi: np.ndarray,
    *,
    min_log10_phi: float,
    lower_phi_floor_fraction: float,
    max_log10_yerr_upper: float | None,
) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    sigma_phi = np.asarray(sigma_phi, dtype=float)
    yerr = np.full((2, phi.size), np.nan, dtype=float)
    valid = np.isfinite(phi) & np.isfinite(sigma_phi) & (phi > 0.0) & (sigma_phi >= 0.0)
    if not np.any(valid):
        return yerr

    floor_phi = 10.0**min_log10_phi
    center = np.log10(np.maximum(phi[valid], floor_phi))
    lower_floor = np.maximum(floor_phi, lower_phi_floor_fraction * phi[valid])
    lower_phi = np.maximum(phi[valid] - sigma_phi[valid], lower_floor)
    upper_phi = np.maximum(phi[valid] + sigma_phi[valid], floor_phi)
    yerr[0, valid] = center - np.log10(lower_phi)
    yerr[1, valid] = np.log10(upper_phi) - center
    if max_log10_yerr_upper is not None and np.isfinite(max_log10_yerr_upper) and max_log10_yerr_upper >= 0.0:
        yerr[1, valid] = np.minimum(yerr[1, valid], float(max_log10_yerr_upper))
    return yerr


def _finite_y_extents(y: np.ndarray, yerr: np.ndarray | None = None) -> list[np.ndarray]:
    values = [np.asarray(y, dtype=float)]
    if yerr is not None:
        y = np.asarray(y, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        if yerr.shape == (2, y.size):
            values.extend([y - yerr[0], y + yerr[1]])
    return values


def _model_for_panel(
    rows: pd.DataFrame,
    *,
    min_log10_phi: float,
    lower_phi_floor_fraction: float,
    max_log10_yerr_upper: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rows["log10_luminosity_center"].to_numpy(dtype=float)
    phi = rows["phi_mpc3_dex"].to_numpy(dtype=float)
    sigma = rows["phi_mpc3_dex_shot_noise_std"].to_numpy(dtype=float)
    y = np.log10(np.maximum(phi, 10.0**min_log10_phi))
    yerr = _asymmetric_log10_yerr(
        phi,
        sigma,
        min_log10_phi=min_log10_phi,
        lower_phi_floor_fraction=lower_phi_floor_fraction,
        max_log10_yerr_upper=max_log10_yerr_upper,
    )
    return x, y, yerr


def _halpha_target(
    hdf5_path: Path,
    sample_label: str,
    *,
    min_log10_phi: float,
    lower_phi_floor_fraction: float,
    max_log10_yerr_upper: float | None,
    error_mode: str,
) -> pd.DataFrame:
    analysis = f"luminosityFunctionHalphaSobral2013HiZELS{sample_label.upper()}"
    with h5py.File(hdf5_path, "r") as handle:
        group_path = f"/analyses/{analysis}"
        if group_path not in handle:
            return pd.DataFrame()
        group = handle[group_path]
        luminosity = np.asarray(group["luminosity"][...], dtype=float)
        phi = np.asarray(group["luminosityFunctionTarget"][...], dtype=float) * np.log(10.0)
        if "luminosityFunctionCovarianceTarget" in group:
            covariance = np.asarray(group["luminosityFunctionCovarianceTarget"][...], dtype=float)
            sigma_phi = np.sqrt(np.maximum(np.diag(covariance), 0.0)) * np.log(10.0)
        else:
            sigma_phi = np.zeros_like(phi)

    if error_mode == "poisson-proxy":
        relative = np.divide(sigma_phi, phi, out=np.full_like(phi, np.nan), where=phi > 0.0)
        scale_candidates = np.divide(
            sigma_phi**2,
            phi,
            out=np.full_like(phi, np.nan),
            where=(phi > 0.0) & np.isfinite(relative) & (relative > 0.0) & (relative < 1.0),
        )
        finite_scale = scale_candidates[np.isfinite(scale_candidates) & (scale_candidates > 0.0)]
        if finite_scale.size:
            sigma_phi = np.sqrt(np.median(finite_scale) * phi)

    y = np.log10(np.maximum(phi, 10.0**min_log10_phi))
    yerr = _asymmetric_log10_yerr(
        phi,
        sigma_phi,
        min_log10_phi=min_log10_phi,
        lower_phi_floor_fraction=lower_phi_floor_fraction,
        max_log10_yerr_upper=max_log10_yerr_upper,
    )
    return pd.DataFrame(
        {
            "x": np.log10(luminosity),
            "log10_phi": y,
            "yerr_lower": yerr[0],
            "yerr_upper": yerr[1],
        }
    )


def _khostovan_target(
    observable: str,
    sample_label: str,
    *,
    min_log10_phi: float,
    lower_phi_floor_fraction: float,
    max_log10_yerr_upper: float | None,
) -> pd.DataFrame:
    line_set = "hbeta_oiii" if observable == "hbeta_oiii_khostovan" else "oii"
    target = _load_khostovan_lf(line_set)
    redshift = float(sample_label.removeprefix("z"))
    target = target.loc[np.isclose(target["redshift"].to_numpy(dtype=float), redshift)].copy()
    if target.empty:
        return pd.DataFrame()

    phi = target["phi_mpc3_dex"].to_numpy(dtype=float)
    n_emitters = target["n_emitters"].to_numpy(dtype=float)
    sigma_phi = np.where(n_emitters > 0.0, phi / np.sqrt(n_emitters), np.nan)
    yerr = _asymmetric_log10_yerr(
        phi,
        sigma_phi,
        min_log10_phi=min_log10_phi,
        lower_phi_floor_fraction=lower_phi_floor_fraction,
        max_log10_yerr_upper=max_log10_yerr_upper,
    )
    return pd.DataFrame(
        {
            "x": target["log10_luminosity_erg_s"].to_numpy(dtype=float),
            "xerr": target["half_width_dex"].to_numpy(dtype=float),
            "log10_phi": target["log10_phi_final"].to_numpy(dtype=float),
            "yerr_lower": yerr[0],
            "yerr_upper": yerr[1],
        }
    )


def _comparat_target(
    vizier_dir: Path,
    sample_label: str,
    *,
    min_log10_phi: float,
    lower_phi_floor_fraction: float,
    max_log10_yerr_upper: float | None,
) -> pd.DataFrame:
    target = _load_comparat_lf(vizier_dir)
    z_min, z_max = [float(value) for value in sample_label.removeprefix("z").split("_")]
    target = target.loc[np.isclose(target["z_min"], z_min) & np.isclose(target["z_max"], z_max)].copy()
    if target.empty:
        return pd.DataFrame()

    phi = target["phi"].to_numpy(dtype=float)
    sigma_phi = target["e_phi"].to_numpy(dtype=float)
    yerr = _asymmetric_log10_yerr(
        phi,
        sigma_phi,
        min_log10_phi=min_log10_phi,
        lower_phi_floor_fraction=lower_phi_floor_fraction,
        max_log10_yerr_upper=max_log10_yerr_upper,
    )
    return pd.DataFrame(
        {
            "x": target["logL_plot"].to_numpy(dtype=float),
            "xerr": 0.5 * (target["logL_max"].to_numpy(dtype=float) - target["logL_min"].to_numpy(dtype=float)),
            "log10_phi": np.log10(np.maximum(phi, 10.0**min_log10_phi)),
            "yerr_lower": yerr[0],
            "yerr_upper": yerr[1],
        }
    )


def _target_for_panel(
    observable: str,
    sample_label: str,
    *,
    halpha_target_hdf5: Path,
    comparat_vizier_dir: Path,
    min_log10_phi: float,
    lower_phi_floor_fraction: float,
    max_log10_yerr_upper: float | None,
    halpha_target_error_mode: str,
) -> pd.DataFrame:
    if observable == "halpha_sobral":
        return _halpha_target(
            halpha_target_hdf5,
            sample_label,
            min_log10_phi=min_log10_phi,
            lower_phi_floor_fraction=lower_phi_floor_fraction,
            max_log10_yerr_upper=max_log10_yerr_upper,
            error_mode=halpha_target_error_mode,
        )
    if observable in {"hbeta_oiii_khostovan", "oii_khostovan"}:
        return _khostovan_target(
            observable,
            sample_label,
            min_log10_phi=min_log10_phi,
            lower_phi_floor_fraction=lower_phi_floor_fraction,
            max_log10_yerr_upper=max_log10_yerr_upper,
        )
    if observable == "oii_comparat":
        return _comparat_target(
            comparat_vizier_dir,
            sample_label,
            min_log10_phi=min_log10_phi,
            lower_phi_floor_fraction=lower_phi_floor_fraction,
            max_log10_yerr_upper=max_log10_yerr_upper,
        )
    return pd.DataFrame()


def _sample_label_sort_value(value: str) -> float:
    if "_" in value:
        return float(value.removeprefix("z").split("_")[0])
    return float(value.removeprefix("z"))


def _panel_title(rows: pd.DataFrame, sample_label: str) -> str:
    first = rows.iloc[0]
    return (
        f"{first['observable_label']} {sample_label} "
        f"({float(first['z_min']):.2f}<z<{float(first['z_max']):.2f})"
    )


def _plot_observable(
    frame: pd.DataFrame,
    observable: str,
    output_path: Path,
    *,
    halpha_target_hdf5: Path,
    comparat_vizier_dir: Path,
    min_log10_phi: float,
    lower_phi_floor_fraction: float,
    max_log10_yerr_upper: float | None,
    halpha_target_error_mode: str,
    dpi: int,
) -> None:
    config = FIGURES[observable]
    subset = frame.loc[frame["observable"] == observable].copy()
    sample_labels = sorted(
        list(dict.fromkeys(subset["sample_label"].astype(str))),
        key=_sample_label_sort_value,
    )
    fig, axes = plt.subplots(
        config["nrows"],
        config["ncols"],
        figsize=config["figsize"],
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for axis, sample_label in zip(axes_flat, sample_labels, strict=False):
        rows = subset.loc[subset["sample_label"].astype(str) == sample_label].copy().sort_values("bin_index")
        x_model, y_model, yerr_model = _model_for_panel(
            rows,
            min_log10_phi=min_log10_phi,
            lower_phi_floor_fraction=lower_phi_floor_fraction,
            max_log10_yerr_upper=max_log10_yerr_upper,
        )
        y_values = _finite_y_extents(y_model, yerr_model)
        x_values = [x_model]

        axis.errorbar(
            x_model,
            y_model,
            yerr=yerr_model,
            fmt="s-",
            color="tab:blue",
            ecolor="tab:blue",
            lw=2.0,
            elinewidth=0.9,
            ms=3.5,
            capsize=2.0,
            label="UNIT lightcone",
            zorder=3,
        )

        target = _target_for_panel(
            observable,
            sample_label,
            halpha_target_hdf5=halpha_target_hdf5,
            comparat_vizier_dir=comparat_vizier_dir,
            min_log10_phi=min_log10_phi,
            lower_phi_floor_fraction=lower_phi_floor_fraction,
            max_log10_yerr_upper=max_log10_yerr_upper,
            halpha_target_error_mode=halpha_target_error_mode,
        )
        if not target.empty:
            target_y = target["log10_phi"].to_numpy(dtype=float)
            target_yerr = target[["yerr_lower", "yerr_upper"]].to_numpy(dtype=float).T
            target_x = target["x"].to_numpy(dtype=float)
            target_xerr = target["xerr"].to_numpy(dtype=float) if "xerr" in target else None
            axis.errorbar(
                target_x,
                target_y,
                xerr=target_xerr,
                yerr=target_yerr,
                fmt="o",
                color="0.15",
                ecolor="0.35",
                elinewidth=0.95,
                ms=4.0,
                capsize=2.0,
                label="target",
                zorder=5,
            )
            y_values.extend(_finite_y_extents(target_y, target_yerr))
            x_values.append(target_x if target_xerr is None else np.concatenate([target_x - target_xerr, target_x + target_xerr]))

        x_limits = finite_limits_from_values(*x_values, margin_fraction=0.04, min_margin=0.05)
        y_limits = finite_limits_from_values(*y_values, margin_fraction=0.08, min_margin=0.15)
        if x_limits is not None:
            axis.set_xlim(*x_limits)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        axis.set_title(_panel_title(rows, sample_label))
        axis.set_xlabel(r"$\log_{10}(L/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(r"$\log_{10}\Phi\ [\mathrm{Mpc}^{-3}\mathrm{dex}^{-1}]$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)

    for axis in axes_flat[len(sample_labels) :]:
        axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    diagnostics_dir = args.diagnostics_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else diagnostics_dir
    frame = pd.read_csv(diagnostics_dir / "data" / "emission_line_lfs.csv")
    for observable, config in FIGURES.items():
        output_path = output_dir / config["path"]
        _plot_observable(
            frame,
            observable,
            output_path,
            halpha_target_hdf5=args.halpha_target_hdf5.expanduser().resolve(),
            comparat_vizier_dir=args.comparat_vizier_dir.expanduser().resolve(),
            min_log10_phi=float(args.min_log10_phi),
            lower_phi_floor_fraction=float(args.lower_phi_floor_fraction),
            max_log10_yerr_upper=float(args.max_log10_yerr_upper),
            halpha_target_error_mode=str(args.halpha_target_error_mode),
            dpi=int(args.dpi),
        )
        print(output_path)


if __name__ == "__main__":
    main()
