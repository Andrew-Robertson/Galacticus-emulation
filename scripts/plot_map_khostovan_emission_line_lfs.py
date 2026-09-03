from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_khostovan_hbeta_oiii_example import _load_khostovan_lf
from process_emission_line_dust_evaluation import _compute_case_lfs
from process_oii_boost_grid_evaluation import _compute_oii_lfs


DUST_PARAMETER_NAMES = ["delta_0", "delta_z", "delta_M", "delta_Mz", "attenuation_scatter"]
OII_DEFAULT_BOOSTS = [0.0, 0.15, 0.30, 0.45, 0.60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MAP-dust Galacticus Hbeta+[OIII] and [OII] luminosity functions "
            "from one HDF5 file against Khostovan et al. (2015) HiZELS data."
        )
    )
    parser.add_argument("--actual-hdf5", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="map_khostovan")
    parser.add_argument(
        "--oii-boost-log10",
        type=float,
        action="append",
        default=None,
        help="log10 luminosity boost applied to [OII] before binning. Repeat for a grid.",
    )
    parser.add_argument("--min-log10-phi", type=float, default=-8.0)
    parser.add_argument("--z-pivot", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _map_dust_case(run_summary: dict[str, Any], *, z_pivot: float) -> dict[str, Any]:
    best_theta = run_summary["best_theta"]
    missing = [name for name in DUST_PARAMETER_NAMES if name not in best_theta]
    if missing:
        raise ValueError(f"Run summary is missing MAP dust parameter(s): {missing}")
    dust_params = {name: float(best_theta[name]) for name in DUST_PARAMETER_NAMES}
    dust_params["z_pivot"] = float(z_pivot)
    return {
        "label": "map_dust",
        "dust_model": "gb10_generalised",
        "dust_params": dust_params,
        "dust_law": "calzetti",
        "random_uniform_index": None,
        "scatter_mode": "expected",
        "dust_draw_index": 0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _log10_phi(phi: pd.Series | np.ndarray, min_log10_phi: float) -> np.ndarray:
    values = np.asarray(phi, dtype=float)
    return np.log10(np.maximum(values, 10.0**min_log10_phi))


def _target_by_redshift(line_set: str) -> pd.DataFrame:
    data = _load_khostovan_lf(line_set).copy()
    data["log10_phi"] = data["log10_phi_final"]
    data["log10_phi_sigma"] = data["log10_phi_final_error"]
    return data


def _model_rows_hbeta_oiii(
    actual_hdf5: Path,
    dust_case: dict[str, Any],
    min_log10_phi: float,
    *,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for row in _compute_case_lfs(actual_hdf5, dust_case, base_seed=base_seed):
        if row["observable"] != "hbeta_oiii_khostovan":
            continue
        phi = float(row["phi_mpc3_dex"])
        rows.append(
            {
                "line_set": "hbeta_oiii",
                "observable": row["observable"],
                "sample_label": row["sample_label"],
                "case": "MAP dust",
                "oii_boost_log10": np.nan,
                "output_name": row["output_name"],
                "output_redshift": float(row["redshift"]),
                "bin_index": int(row["bin_index"]),
                "log10_luminosity_center": float(row["log10_luminosity_center"]),
                "log10_luminosity_min": float(row["log10_luminosity_min"]),
                "log10_luminosity_max": float(row["log10_luminosity_max"]),
                "phi_mpc3_dex": phi,
                "log10_phi": float(np.log10(max(phi, 10.0**min_log10_phi))),
                "target_redshift": float(row["target_redshift"]),
                **dust_case["dust_params"],
            }
        )
    return rows


def _model_rows_oii(
    actual_hdf5: Path,
    dust_case: dict[str, Any],
    boosts: list[float],
    min_log10_phi: float,
) -> list[dict[str, Any]]:
    rows = []
    for boost_log10 in boosts:
        for row in _compute_oii_lfs(actual_hdf5, dust_case, boost_log10=float(boost_log10), base_seed=12345):
            if row["observable"] != "oii_khostovan":
                continue
            phi = float(row["phi_mpc3_dex"])
            rows.append(
                {
                    "line_set": "oii",
                    "observable": row["observable"],
                    "sample_label": row["sample_label"],
                    "case": f"OII boost {boost_log10:.2f} dex",
                    "oii_boost_log10": float(boost_log10),
                    "output_name": row["output_name"],
                    "output_redshift": float(row["redshift"]),
                    "bin_index": int(row["bin_index"]),
                    "log10_luminosity_center": float(row["log10_luminosity_center"]),
                    "log10_luminosity_min": float(row["log10_luminosity_min"]),
                    "log10_luminosity_max": float(row["log10_luminosity_max"]),
                    "phi_mpc3_dex": phi,
                    "log10_phi": float(np.log10(max(phi, 10.0**min_log10_phi))),
                    "target_redshift": float(row["target_redshift"]),
                    **dust_case["dust_params"],
                }
            )
    return rows


def _sample_label_to_redshift(label: str) -> float:
    return float(str(label).strip().lower().removeprefix("z"))


def _set_centered_ylim(axis, values: list[np.ndarray], *, min_margin: float = 0.35) -> None:
    finite = []
    for value in values:
        array = np.asarray(value, dtype=float)
        finite.append(array[np.isfinite(array)])
    finite = [array for array in finite if array.size]
    if not finite:
        return
    all_values = np.concatenate(finite)
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    if upper <= lower:
        lower -= min_margin
        upper += min_margin
    else:
        margin = max(0.08 * (upper - lower), min_margin)
        lower -= margin
        upper += margin
    axis.set_ylim(lower, upper)


def _plot_hbeta_oiii(target: pd.DataFrame, model: pd.DataFrame, output_path: Path, *, dpi: int) -> None:
    redshifts = sorted(target["redshift"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, redshift in zip(axes_flat, redshifts, strict=False):
        obs = target[target["redshift"] == redshift].sort_values("log10_luminosity_erg_s")
        label = f"z{redshift:.2f}"
        curve = model[np.isclose(model["target_redshift"].to_numpy(dtype=float), redshift)].sort_values(
            "log10_luminosity_center"
        )
        axis.errorbar(
            obs["log10_luminosity_erg_s"],
            obs["log10_phi"],
            xerr=obs["half_width_dex"],
            yerr=obs["log10_phi_sigma"],
            fmt="o",
            color="black",
            ecolor="0.35",
            capsize=2.0,
            ms=4.0,
            label="Khostovan et al. 2015",
        )
        axis.plot(
            curve["log10_luminosity_center"],
            curve["log10_phi"],
            color="#1f77b4",
            lw=2.0,
            marker="s",
            ms=3.5,
            label="Galacticus MAP dust",
        )
        if len(curve):
            example = curve.iloc[0]
            title_suffix = f"{example['output_name']}, z={example['output_redshift']:.2f}"
        else:
            title_suffix = "no matching model rows"
        axis.set_title(f"Hbeta+[OIII] {label} ({title_suffix})")
        axis.set_xlabel(r"$\log_{10}(L/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(r"$\log_{10}\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
        _set_centered_ylim(axis, [obs["log10_phi"].to_numpy(dtype=float), curve["log10_phi"].to_numpy(dtype=float)])
    for axis in axes_flat[len(redshifts) :]:
        axis.axis("off")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_oii(target: pd.DataFrame, model: pd.DataFrame, output_path: Path, *, dpi: int) -> None:
    redshifts = sorted(target["redshift"].unique())
    boosts = sorted(model["oii_boost_log10"].dropna().unique())
    colors = plt.get_cmap("viridis")(np.linspace(0.10, 0.90, max(len(boosts), 2)))[: len(boosts)]
    color_by_boost = {float(boost): color for boost, color in zip(boosts, colors, strict=True)}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, redshift in zip(axes_flat, redshifts, strict=False):
        obs = target[target["redshift"] == redshift].sort_values("log10_luminosity_erg_s")
        selection = np.isclose(model["target_redshift"].to_numpy(dtype=float), redshift)
        panel_model = model[selection]
        axis.errorbar(
            obs["log10_luminosity_erg_s"],
            obs["log10_phi"],
            xerr=obs["half_width_dex"],
            yerr=obs["log10_phi_sigma"],
            fmt="o",
            color="black",
            ecolor="0.35",
            capsize=2.0,
            ms=4.0,
            label="Khostovan et al. 2015",
        )
        ylim_values = [obs["log10_phi"].to_numpy(dtype=float)]
        for boost, group in panel_model.groupby("oii_boost_log10", sort=True):
            group = group.sort_values("log10_luminosity_center")
            label = f"{float(boost):.2f} dex"
            axis.plot(
                group["log10_luminosity_center"],
                group["log10_phi"],
                color=color_by_boost[float(boost)],
                lw=2.0 if np.isclose(float(boost), 0.30) else 1.5,
                marker="s",
                ms=3.2,
                label=label,
            )
            ylim_values.append(group["log10_phi"].to_numpy(dtype=float))
        if len(panel_model):
            example = panel_model.iloc[0]
            title_suffix = f"{example['output_name']}, z={example['output_redshift']:.2f}"
        else:
            title_suffix = "no matching model rows"
        axis.set_title(f"[OII] z{redshift:.2f} ({title_suffix})")
        axis.set_xlabel(r"$\log_{10}(L/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(r"$\log_{10}\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
        axis.grid(alpha=0.22)
        _set_centered_ylim(axis, ylim_values)
    for axis in axes_flat[len(redshifts) :]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].legend(handles, labels, title=r"[OII] boost", frameon=False, loc="center")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_hdf5 = args.actual_hdf5.expanduser().resolve()
    run_summary = json.loads(args.run_summary.expanduser().resolve().read_text())
    dust_case = _map_dust_case(run_summary, z_pivot=args.z_pivot)
    boosts = args.oii_boost_log10 if args.oii_boost_log10 is not None else OII_DEFAULT_BOOSTS

    hbeta_target = _target_by_redshift("hbeta_oiii")
    oii_target = _target_by_redshift("oii")
    hbeta_rows = _model_rows_hbeta_oiii(actual_hdf5, dust_case, args.min_log10_phi, base_seed=12345)
    oii_rows = _model_rows_oii(actual_hdf5, dust_case, boosts, args.min_log10_phi)
    hbeta_model = pd.DataFrame(hbeta_rows)
    oii_model = pd.DataFrame(oii_rows)

    target_path = output_dir / f"{args.output_prefix}_khostovan_targets.csv"
    model_path = output_dir / f"{args.output_prefix}_galacticus_model_lfs.csv"
    dust_path = output_dir / f"{args.output_prefix}_map_dust_params.json"
    hbeta_path = output_dir / f"{args.output_prefix}_hbeta_oiii_khostovan.png"
    oii_path = output_dir / f"{args.output_prefix}_oii_khostovan_boost_grid.png"

    targets = pd.concat(
        [hbeta_target.assign(line_set="hbeta_oiii"), oii_target.assign(line_set="oii")],
        ignore_index=True,
    )
    targets.to_csv(target_path, index=False)
    _write_csv(model_path, hbeta_rows + oii_rows)
    dust_path.write_text(json.dumps(dust_case["dust_params"], indent=2) + "\n")
    _plot_hbeta_oiii(hbeta_target, hbeta_model, hbeta_path, dpi=args.dpi)
    _plot_oii(oii_target, oii_model, oii_path, dpi=args.dpi)

    print(target_path)
    print(model_path)
    print(dust_path)
    print(hbeta_path)
    print(oii_path)


if __name__ == "__main__":
    main()
