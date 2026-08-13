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

from galacticus_emu.plotting import set_ylim_from_values
from plot_comparat_oii_example import _load_comparat_lf
from fit_sidecar_lf_pca_holdout_demo import (
    TARGET_ERROR_MODE_LEGACY,
    TARGET_ERROR_MODES,
    _linear_upper_excursion_to_log10_error,
)
from plot_khostovan_hbeta_oiii_example import _load_khostovan_lf


DEFAULT_OBSERVABLES = ["halpha_sobral", "hbeta_oiii_khostovan", "oii_khostovan", "oii_comparat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot actual Galacticus emission-line LF sidecars against target data."
    )
    parser.add_argument(
        "--sidecar",
        action="append",
        required=True,
        help="Actual sidecar to plot, as LABEL=PATH_TO_emission_line_dust. Can be repeated.",
    )
    parser.add_argument(
        "--observable",
        action="append",
        default=[],
        help=f"Observable to plot. Defaults to: {', '.join(DEFAULT_OBSERVABLES)}.",
    )
    parser.add_argument(
        "--halpha-target-hdf5",
        type=Path,
        default=None,
        help="Reduced Galacticus HDF5 file containing /analyses H-alpha Sobral target data.",
    )
    parser.add_argument(
        "--halpha-target-error-mode",
        choices=TARGET_ERROR_MODES,
        default=TARGET_ERROR_MODE_LEGACY,
        help=(
            "How to convert H-alpha Sobral target LF errors into log10(Phi) space. "
            "The default preserves legacy plots; use sobral_log_table for the "
            "Sobral table's log-space errors."
        ),
    )
    parser.add_argument(
        "--comparat-vizier-dir",
        type=Path,
        default=REPO_ROOT / "data/observations/emission_line_lfs/comparat_oii_2015/J_A+A_575_A40",
    )
    parser.add_argument("--dust-case", default=None, help="Optional dust_case label to select.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="actual_map_eps_vs_unit")
    parser.add_argument("--min-log10-phi", type=float, default=-8.0)
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Expected non-empty LABEL in {value!r}")
    return label, Path(path).expanduser().resolve()


def _read_sidecar(path: Path, *, dust_case: str | None, min_log10_phi: float) -> pd.DataFrame:
    table_path = path / "emission_line_dust_lf_long.csv"
    frame = pd.read_csv(table_path)
    if "dust_case" in frame.columns:
        cases = sorted(frame["dust_case"].astype(str).unique())
        if dust_case is None:
            if len(cases) != 1:
                raise ValueError(f"{table_path} has multiple dust_case values {cases}; pass --dust-case")
            dust_case = cases[0]
        frame = frame.loc[frame["dust_case"].astype(str) == str(dust_case)].copy()
        if frame.empty:
            raise ValueError(f"{table_path} contains no rows for dust_case={dust_case!r}")

    phi = frame["phi_mpc3_dex"].to_numpy(dtype=float)
    frame["log10_phi"] = np.log10(np.maximum(phi, 10.0**min_log10_phi))
    if "log10_phi_shot_noise_std_conservative" in frame.columns:
        sigma = pd.to_numeric(frame["log10_phi_shot_noise_std_conservative"], errors="coerce").to_numpy(dtype=float)
        frame["log10_phi_sigma"] = np.where(np.isfinite(sigma), sigma, np.nan)
    else:
        frame["log10_phi_sigma"] = np.nan
    return frame


def _linear_std_to_log10(value: np.ndarray, sigma: np.ndarray, min_log10_phi: float) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    clipped = np.maximum(value, 10.0**min_log10_phi)
    return np.where((clipped > 0.0) & np.isfinite(sigma), sigma / (clipped * np.log(10.0)), np.nan)


def _halpha_target(
    path: Path | None,
    sample_label: str,
    x_plot: np.ndarray,
    min_log10_phi: float,
    *,
    target_error_mode: str,
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    analysis = f"luminosityFunctionHalphaSobral2013HiZELS{sample_label.upper()}"
    with h5py.File(path, "r") as handle:
        group_path = f"/analyses/{analysis}"
        if group_path not in handle:
            return pd.DataFrame()
        group = handle[group_path]
        target_x = np.log10(np.asarray(group["luminosity"][...], dtype=float))
        target_phi = np.asarray(group["luminosityFunctionTarget"][...], dtype=float) * np.log(10.0)
        target_sigma = None
        if "luminosityFunctionCovarianceTarget" in group:
            covariance = np.asarray(group["luminosityFunctionCovarianceTarget"][...], dtype=float)
            target_sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0)) * np.log(10.0)
    order = [int(np.argmin(np.abs(target_x - value))) for value in x_plot]
    if not np.allclose(target_x[order], x_plot, rtol=0.0, atol=1.0e-6):
        return pd.DataFrame()
    target_phi = target_phi[order]
    rows = {
        "x": target_x[order],
        "log10_phi": np.log10(np.maximum(target_phi, 10.0**min_log10_phi)),
    }
    if target_sigma is not None:
        if target_error_mode == TARGET_ERROR_MODE_SOBRAL_LOG:
            rows["log10_phi_sigma"] = _linear_upper_excursion_to_log10_error(target_phi, target_sigma[order])
        else:
            rows["log10_phi_sigma"] = _linear_std_to_log10(target_phi, target_sigma[order], min_log10_phi)
    return pd.DataFrame(rows)


def _khostovan_target(observable: str, sample_label: str) -> pd.DataFrame:
    line_set = "hbeta_oiii" if observable == "hbeta_oiii_khostovan" else "oii"
    target = _load_khostovan_lf(line_set)
    redshift = float(sample_label.removeprefix("z"))
    target = target.loc[np.isclose(target["redshift"].to_numpy(dtype=float), redshift)].copy()
    if target.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "x": target["log10_luminosity_erg_s"].to_numpy(dtype=float),
            "xerr": target["half_width_dex"].to_numpy(dtype=float),
            "log10_phi": target["log10_phi_final"].to_numpy(dtype=float),
            "log10_phi_sigma": target["log10_phi_final_error"].to_numpy(dtype=float),
        }
    )


def _comparat_target(vizier_dir: Path, sample_label: str, min_log10_phi: float) -> pd.DataFrame:
    target = _load_comparat_lf(vizier_dir)
    z_min, z_max = [float(value) for value in sample_label.removeprefix("z").split("_")]
    target = target.loc[np.isclose(target["z_min"], z_min) & np.isclose(target["z_max"], z_max)].copy()
    if target.empty:
        return pd.DataFrame()
    phi = target["phi"].to_numpy(dtype=float)
    sigma = target["e_phi"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "x": target["logL_plot"].to_numpy(dtype=float),
            "xerr": 0.5 * (target["logL_max"].to_numpy(dtype=float) - target["logL_min"].to_numpy(dtype=float)),
            "log10_phi": np.log10(np.maximum(phi, 10.0**min_log10_phi)),
            "log10_phi_sigma": _linear_std_to_log10(phi, sigma, min_log10_phi),
        }
    )


def _target_for(
    observable: str,
    sample_label: str,
    x_plot: np.ndarray,
    *,
    halpha_target_hdf5: Path | None,
    halpha_target_error_mode: str,
    comparat_vizier_dir: Path,
    min_log10_phi: float,
) -> pd.DataFrame:
    if observable == "halpha_sobral":
        return _halpha_target(
            halpha_target_hdf5,
            sample_label,
            x_plot,
            min_log10_phi,
            target_error_mode=halpha_target_error_mode,
        )
    if observable in {"hbeta_oiii_khostovan", "oii_khostovan"}:
        return _khostovan_target(observable, sample_label)
    if observable == "oii_comparat":
        return _comparat_target(comparat_vizier_dir, sample_label, min_log10_phi)
    return pd.DataFrame()


def _panel_title(observable: str, sample_label: str, rows: pd.DataFrame) -> str:
    if observable == "halpha_sobral":
        prefix = "Halpha Sobral"
    elif observable == "hbeta_oiii_khostovan":
        prefix = "Hbeta+[OIII] Khostovan"
    elif observable == "oii_khostovan":
        prefix = "[OII] Khostovan"
    elif observable == "oii_comparat":
        prefix = "[OII] Comparat"
    else:
        prefix = observable
    if len(rows) and "output_name" in rows and "redshift" in rows:
        first = rows.iloc[0]
        return f"{prefix} {sample_label} ({first['output_name']}, z={float(first['redshift']):.2f})"
    return f"{prefix} {sample_label}"


def _plot_observable(
    sidecars: list[tuple[str, pd.DataFrame]],
    observable: str,
    output_path: Path,
    *,
    halpha_target_hdf5: Path | None,
    halpha_target_error_mode: str,
    comparat_vizier_dir: Path,
    min_log10_phi: float,
    dpi: int,
) -> None:
    first = sidecars[0][1]
    subset = first.loc[first["observable"] == observable]
    sample_labels = list(dict.fromkeys(subset.sort_values(["redshift", "sample_label"])["sample_label"].astype(str)))
    if not sample_labels:
        return
    ncols = 2 if len(sample_labels) > 1 else 1
    nrows = int(np.ceil(len(sample_labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.8 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    linestyles = ["--", ":", "-.", (0, (3, 1, 1, 1))]
    markers = ["s", "D", "^", "v"]

    for axis, sample_label in zip(axes_flat, sample_labels, strict=False):
        panel_reference = pd.DataFrame()
        y_values: list[np.ndarray] = []
        for index, (label, frame) in enumerate(sidecars):
            rows = frame.loc[
                (frame["observable"] == observable) & (frame["sample_label"].astype(str) == sample_label)
            ].sort_values("bin_index" if "bin_index" in frame.columns else "log10_luminosity_center")
            if rows.empty:
                continue
            if panel_reference.empty:
                panel_reference = rows
            x = rows["log10_luminosity_center"].to_numpy(dtype=float)
            y = rows["log10_phi"].to_numpy(dtype=float)
            y_values.append(y)
            axis.plot(
                x,
                y,
                color="black",
                lw=2.0,
                ls=linestyles[index % len(linestyles)],
                marker=markers[index % len(markers)],
                ms=3.2,
                label=label,
            )

        if not panel_reference.empty:
            target = _target_for(
                observable,
                sample_label,
                panel_reference["log10_luminosity_center"].to_numpy(dtype=float),
                halpha_target_hdf5=halpha_target_hdf5,
                halpha_target_error_mode=halpha_target_error_mode,
                comparat_vizier_dir=comparat_vizier_dir,
                min_log10_phi=min_log10_phi,
            )
            if not target.empty:
                xerr = target["xerr"].to_numpy(dtype=float) if "xerr" in target.columns else None
                yerr = (
                    target["log10_phi_sigma"].to_numpy(dtype=float)
                    if "log10_phi_sigma" in target.columns
                    else None
                )
                axis.errorbar(
                    target["x"],
                    target["log10_phi"],
                    xerr=xerr,
                    yerr=yerr,
                    fmt="o",
                    color="0.15",
                    ecolor="0.35",
                    ms=4.0,
                    capsize=2.0,
                    label="target",
                    zorder=5,
                )
                y_values.append(target["log10_phi"].to_numpy(dtype=float))

        set_ylim_from_values(axis, *y_values)
        axis.set_title(_panel_title(observable, sample_label, panel_reference))
        axis.set_xlabel(r"$\log_{10}(L/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(r"$\log_{10}\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)

    for axis in axes_flat[len(sample_labels) :]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    observables = args.observable or DEFAULT_OBSERVABLES
    sidecars = [
        (label, _read_sidecar(path, dust_case=args.dust_case, min_log10_phi=args.min_log10_phi))
        for label, path in (_parse_label_path(value) for value in args.sidecar)
    ]
    output_dir = args.output_dir.expanduser().resolve()
    for observable in observables:
        output_path = output_dir / f"{args.output_prefix}_{observable}.png"
        _plot_observable(
            sidecars,
            observable,
            output_path,
            halpha_target_hdf5=args.halpha_target_hdf5.expanduser().resolve() if args.halpha_target_hdf5 else None,
            halpha_target_error_mode=args.halpha_target_error_mode,
            comparat_vizier_dir=args.comparat_vizier_dir.expanduser().resolve(),
            min_log10_phi=args.min_log10_phi,
            dpi=args.dpi,
        )
        print(output_path)


if __name__ == "__main__":
    main()
