from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calculate_halpha_dust_lf_grid import _combined_node_weights  # type: ignore
from process_halpha_dust_campaign import _load_dust_priors, _sample_dust_params  # type: ignore

if "GALACTICUS_SED_CALC_PATH" in os.environ:
    SED_REPO_ROOT = Path(os.environ["GALACTICUS_SED_CALC_PATH"]).expanduser().resolve()
else:
    SED_REPO_ROOT = (REPO_ROOT.parent / "galacticus_sed_calculator").resolve()
SED_PACKAGE_ROOT = SED_REPO_ROOT / "galacticus_sed_calculator"
if not SED_PACKAGE_ROOT.exists():
    raise FileNotFoundError(
        "Could not find galacticus_sed_calculator package directory. "
        "Set GALACTICUS_SED_CALC_PATH to the repo root containing galacticus_sed_calculator/."
    )
sys.path.insert(0, str(SED_PACKAGE_ROOT))

from dust_attenuation import apply_dust_attenuation_to_line, dust_attenuation_gb10_generalised  # type: ignore


DEFAULT_GALACTICUS_FILE = (
    REPO_ROOT
    / "runs/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_19p_512/evaluations/"
    / "sobol_mass_function_emissionlines_dust_simpleSizes_19p_512-eval-0000/galacticus.hdf5"
)
DEFAULT_VIZIER_DIR = REPO_ROOT / "data/observations/emission_line_lfs/comparat_oii_2015/J_A+A_575_A40"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "playing/emission_line_lf_comparisons/comparat_oii_eval0000"

OII_LINES = ["oxygenII3727", "oxygenII3730"]
COMPONENTS_NO_AGN = ["Disk", "Spheroid"]
COMPONENTS_WITH_AGN = ["Disk", "Spheroid", "AGN"]
OII_EFFECTIVE_WAVELENGTH_ANGSTROM = 3728.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a Galacticus [OII] luminosity function from one full output against Comparat et al. (2015)."
    )
    parser.add_argument("--galacticus-file", type=Path, default=DEFAULT_GALACTICUS_FILE)
    parser.add_argument("--vizier-dir", type=Path, default=DEFAULT_VIZIER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-agn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include luminosityEmissionLineAGN:* in addition to disk+spheroid.",
    )
    parser.add_argument(
        "--skip-dust",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only plot the intrinsic LF, without sampled GB10+Calzetti attenuated curves.",
    )
    parser.add_argument("--n-dust-draws", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--z-pivot", type=float, default=1.0)
    parser.add_argument(
        "--dust-law",
        choices=["calzetti"],
        default="calzetti",
        help="Fixed attenuation law used to scale A_Halpha to the [OII] wavelength.",
    )
    parser.add_argument(
        "--dust-prior-json",
        action="append",
        default=[],
        help=(
            "Override one dust-parameter prior using the same syntax as process_halpha_dust_campaign.py, "
            "for example label='{\"distribution\":\"uniform\",\"lower\":...,\"upper\":...}'."
        ),
    )
    return parser.parse_args()


def _load_comparat_lf(vizier_dir: Path) -> pd.DataFrame:
    path = vizier_dir / "lf.dat"
    names = ["z_range", "logL_min", "logL_max", "logL", "phi", "e_phi", "jkES", "galES", "wES", "Ngal"]
    data = pd.read_csv(path, sep=r"\s+", names=names, comment="#", skip_blank_lines=True, engine="python")
    data = data.dropna(subset=["z_range", "logL", "phi", "e_phi"]).copy()
    z_parts = data["z_range"].str.split("-", expand=True).astype(float)
    data["z_min"] = z_parts[0]
    data["z_max"] = z_parts[1]
    for column in ["logL_min", "logL_max", "logL", "phi", "e_phi", "jkES", "galES", "wES", "Ngal"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # Some high-redshift rows in the CDS ASCII table have corrupted left-edge
    # fields. The right edges remain the useful bin boundary, and the LF bins
    # are 0.25 dex wide, so infer plotting/bin centers from the right edge.
    inferred_logL_min = data["logL_max"] - 0.25
    corrupted_left_edge = ~np.isclose(data["logL_min"], inferred_logL_min, rtol=0.0, atol=1.0e-6)
    data["logL_min_raw"] = data["logL_min"]
    data["logL_min"] = inferred_logL_min
    data["logL_center_from_edges"] = 0.5 * (data["logL_min"] + data["logL_max"])
    data["logL_plot"] = data["logL_center_from_edges"]
    data["corrected_corrupted_logL_min"] = corrupted_left_edge
    data["z_label"] = data["z_min"].map(lambda value: f"{value:.3f}") + "-" + data["z_max"].map(lambda value: f"{value:.3f}")
    return data


def _bin_edges_from_centers(log10_centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(log10_centers, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.125, centers[0] + 0.125], dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def _output_redshifts(handle: h5py.File) -> dict[str, float]:
    redshifts: dict[str, float] = {}
    for output_name in handle["Outputs"]:
        output_group = handle[f"Outputs/{output_name}"]
        expansion_factor = float(output_group.attrs["outputExpansionFactor"])
        redshifts[output_name] = 1.0 / expansion_factor - 1.0
    return redshifts


def _select_output(redshifts: dict[str, float], z_min: float, z_max: float) -> tuple[str, float]:
    z_mid = 0.5 * (z_min + z_max)
    in_bin = {name: z for name, z in redshifts.items() if z_min <= z <= z_max}
    candidates = in_bin or redshifts
    output_name, redshift = min(candidates.items(), key=lambda item: abs(item[1] - z_mid))
    return output_name, redshift


def _oii_luminosity(output_group: h5py.Group, include_agn: bool) -> np.ndarray:
    nd = output_group["nodeData"]
    components = COMPONENTS_WITH_AGN if include_agn else COMPONENTS_NO_AGN
    luminosity = None
    for component in components:
        for line_name in OII_LINES:
            dataset_name = f"luminosityEmissionLine{component}:{line_name}"
            values = np.asarray(nd[dataset_name][...], dtype=float)
            luminosity = values.copy() if luminosity is None else luminosity + values
    if luminosity is None:
        raise RuntimeError("No [OII] luminosity datasets were summed")
    return luminosity


def _stellar_mass(output_group: h5py.Group) -> np.ndarray:
    nd = output_group["nodeData"]
    return np.asarray(nd["diskMassStellar"][...], dtype=float) + np.asarray(nd["spheroidMassStellar"][...], dtype=float)


def _gb10_attenuated_oii_luminosity(
    output_group: h5py.Group,
    redshift: float,
    intrinsic_luminosity: np.ndarray,
    dust_params: dict[str, float],
    *,
    dust_law: str,
    rng: np.random.Generator,
) -> np.ndarray:
    mstar = _stellar_mass(output_group)
    a_halpha = dust_attenuation_gb10_generalised(
        mstar,
        np.full_like(mstar, redshift, dtype=float),
        rng=rng,
        **dust_params,
    )
    return np.asarray(
        apply_dust_attenuation_to_line(
            intrinsic_luminosity,
            OII_EFFECTIVE_WAVELENGTH_ANGSTROM,
            a_halpha,
            dust_law=dust_law,
        ),
        dtype=float,
    )


def _gb10_baseline_oii_attenuation_scale(
    *,
    stellar_mass: float = 1.0e10,
    redshift: float = 1.0,
    dust_law: str = "calzetti",
) -> tuple[float, float]:
    a_halpha = dust_attenuation_gb10_generalised(
        np.asarray([stellar_mass], dtype=float),
        np.asarray([redshift], dtype=float),
        delta_0=0.0,
        delta_z=0.0,
        delta_M=0.0,
        delta_Mz=0.0,
        attenuation_scatter=0.0,
        z_pivot=1.0,
    )
    attenuated = np.asarray(
        apply_dust_attenuation_to_line(
            np.ones(1, dtype=float),
            OII_EFFECTIVE_WAVELENGTH_ANGSTROM,
            a_halpha,
            dust_law=dust_law,
        ),
        dtype=float,
    )
    delta_log10_luminosity = -float(np.log10(attenuated[0]))
    a_oii = delta_log10_luminosity / 0.4
    return delta_log10_luminosity, a_oii


def _lf_per_dex(luminosity: np.ndarray, weights: np.ndarray, log10_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(luminosity) & np.isfinite(weights) & (luminosity > 0.0)
    log10_luminosity = np.log10(luminosity[valid])
    counts, _ = np.histogram(log10_luminosity, bins=log10_edges, weights=weights[valid])
    variance_counts, _ = np.histogram(log10_luminosity, bins=log10_edges, weights=weights[valid] ** 2)
    bin_widths = np.diff(log10_edges)
    return counts / bin_widths, np.sqrt(np.clip(variance_counts, 0.0, None)) / bin_widths


def _model_rows(
    galacticus_file: Path,
    comparat: pd.DataFrame,
    *,
    include_agn: bool,
    include_dust: bool,
    dust_params_by_draw: list[dict[str, float]],
    dust_law: str,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(galacticus_file, "r") as handle:
        redshifts = _output_redshifts(handle)
        for (z_min, z_max), group in comparat.groupby(["z_min", "z_max"], sort=True):
            output_name, output_redshift = _select_output(redshifts, float(z_min), float(z_max))
            output_group = handle[f"Outputs/{output_name}"]
            weights = _combined_node_weights(output_group)
            centers = group["logL_plot"].to_numpy(dtype=float)
            edges = _bin_edges_from_centers(centers)
            intrinsic = _oii_luminosity(output_group, include_agn=include_agn)
            cases = {"intrinsic": intrinsic}
            if include_dust:
                baseline_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 0, int(output_name[6:]), 10_000]))
                cases["gb10_baseline_calzetti"] = _gb10_attenuated_oii_luminosity(
                    output_group,
                    output_redshift,
                    intrinsic,
                    {
                        "delta_0": 0.0,
                        "delta_z": 0.0,
                        "delta_M": 0.0,
                        "delta_Mz": 0.0,
                        "attenuation_scatter": 0.0,
                        "z_pivot": 1.0,
                    },
                    dust_law=dust_law,
                    rng=baseline_rng,
                )
                for dust_draw_index, dust_params in enumerate(dust_params_by_draw):
                    rng = np.random.default_rng(np.random.SeedSequence([base_seed, dust_draw_index, int(output_name[6:])]))
                    cases[f"dust_draw_{dust_draw_index:04d}"] = _gb10_attenuated_oii_luminosity(
                        output_group,
                        output_redshift,
                        intrinsic,
                        dust_params,
                        dust_law=dust_law,
                        rng=rng,
                    )
            for case_label, luminosity in cases.items():
                phi, phi_std = _lf_per_dex(luminosity, weights, edges)
                for bin_index, (center, value, sigma) in enumerate(zip(centers, phi, phi_std, strict=True)):
                    dust_draw_index = None
                    dust_params: dict[str, float] = {}
                    if case_label.startswith("dust_draw_"):
                        dust_draw_index = int(case_label.rsplit("_", 1)[-1])
                        dust_params = dust_params_by_draw[dust_draw_index]
                    rows.append(
                        {
                            "z_min": float(z_min),
                            "z_max": float(z_max),
                            "z_mid": 0.5 * (float(z_min) + float(z_max)),
                            "output_name": output_name,
                            "output_redshift": output_redshift,
                            "case": case_label,
                            "dust_draw_index": dust_draw_index,
                            "dust_law": dust_law if case_label.startswith("dust_draw_") or case_label == "gb10_baseline_calzetti" else None,
                            "include_agn": include_agn,
                            "bin_index": int(bin_index),
                            "log10_luminosity_erg_s": float(center),
                            "phi_mpc3_dex": float(value),
                            "phi_shot_noise_std_mpc3_dex": float(sigma),
                            **dust_params,
                        }
                    )
    return rows


def _write_model_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(comparat: pd.DataFrame, model: pd.DataFrame, output_path: Path) -> None:
    z_bins = list(comparat.groupby(["z_min", "z_max"], sort=True).groups)
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 13.0), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    dust_color = "#d1495b"
    baseline_attenuation_scales = [
        (1.0e9, *_gb10_baseline_oii_attenuation_scale(stellar_mass=1.0e9)),
        (1.0e10, *_gb10_baseline_oii_attenuation_scale(stellar_mass=1.0e10)),
    ]

    for axis, (z_min, z_max) in zip(axes_flat, z_bins, strict=False):
        obs = comparat[(comparat["z_min"] == z_min) & (comparat["z_max"] == z_max)]
        axis.errorbar(
            obs["logL_plot"],
            obs["phi"],
            yerr=obs["e_phi"],
            fmt="o",
            ms=4,
            color="black",
            ecolor="0.35",
            elinewidth=0.9,
            capsize=2,
            label="Comparat et al. 2015",
        )
        selection = (model["z_min"] == z_min) & (model["z_max"] == z_max)
        for case, case_model in model[selection].groupby("case", sort=False):
            is_dust_draw = str(case).startswith("dust_draw_")
            is_baseline_dust = case == "gb10_baseline_calzetti"
            label = (
                "16 GB10+Calzetti dust draws"
                if is_dust_draw and case == "dust_draw_0000"
                else (
                    "GB10 baseline + Calzetti"
                    if is_baseline_dust
                    else ("Galacticus intrinsic" if case == "intrinsic" else None)
                )
            )
            color = dust_color if is_dust_draw or is_baseline_dust else "#1f77b4"
            if is_dust_draw:
                axis.plot(
                    case_model["log10_luminosity_erg_s"],
                    case_model["phi_mpc3_dex"],
                    lw=0.9,
                    alpha=0.32,
                    color=color,
                    label=label,
                )
            else:
                y = case_model["phi_mpc3_dex"].to_numpy(dtype=float)
                yerr = case_model["phi_shot_noise_std_mpc3_dex"].to_numpy(dtype=float)
                yerr = np.where(y > 0.0, np.minimum(yerr, 0.98 * y), 0.0)
                axis.errorbar(
                    case_model["log10_luminosity_erg_s"],
                    y,
                    yerr=yerr,
                    fmt="s-",
                    ms=3.5,
                    lw=2.0,
                    elinewidth=1.0,
                    capsize=2.0,
                    color=color,
                    ecolor=color,
                    alpha=1.0,
                    label=label,
                )
        example = model[selection].iloc[0]
        axis.set_title(f"{z_min:.3f}<z<{z_max:.3f}  ({example['output_name']}, z={example['output_redshift']:.2f})")
        axis.set_yscale("log")
        axis.set_ylim(1.0e-8, 3.0e-2)
        axis.set_xlim(40.3, 43.8)
        axis.grid(alpha=0.18)
        if example["output_name"] == "Output9":
            label_x = 40.72 + 0.5 * baseline_attenuation_scales[-1][1]
            for row_index, (stellar_mass, delta_log10_luminosity, a_oii) in enumerate(baseline_attenuation_scales):
                marker_start = 40.72
                marker_x = marker_start + 0.5 * delta_log10_luminosity
                marker_y = 1.0e-6 if row_index == 0 else 2.0e-8
                axis.errorbar(
                    marker_x,
                    marker_y,
                    xerr=0.5 * delta_log10_luminosity,
                    fmt="none",
                    color=dust_color,
                    ecolor=dust_color,
                    elinewidth=2.0,
                    capsize=4.0,
                    zorder=5,
                )
                axis.text(
                    label_x,
                    marker_y * 1.35,
                    (
                        rf"GB10 $10^{{{int(np.log10(stellar_mass))}}}\,M_\odot$"
                        "\n"
                        rf"$\Delta\log L={delta_log10_luminosity:.2f}$ dex"
                        "\n"
                        rf"$A_{{[OII]}}={a_oii:.2f}$ mag"
                    ),
                    color=dust_color,
                    fontsize=7.5,
                    ha="center",
                    va="bottom",
                )

    for axis in axes_flat[len(z_bins) :]:
        axis.axis("off")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\log_{10}(L_{\mathrm{[OII]}}/\mathrm{erg\,s^{-1}})$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower right", frameon=False, bbox_to_anchor=(0.98, 0.04))
    fig.suptitle(r"[OII] luminosity function: full eval-0000 Galacticus output vs. Comparat et al. (2015)", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparat = _load_comparat_lf(args.vizier_dir.resolve())
    priors = _load_dust_priors(args)
    rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, 10_002]))
    dust_params_by_draw = [
        _sample_dust_params(priors, z_pivot=args.z_pivot, rng=rng)
        for _ in range(args.n_dust_draws)
    ]
    model_rows = _model_rows(
        args.galacticus_file.resolve(),
        comparat,
        include_agn=args.include_agn,
        include_dust=not args.skip_dust,
        dust_params_by_draw=dust_params_by_draw,
        dust_law=args.dust_law,
        base_seed=args.base_seed,
    )
    model = pd.DataFrame(model_rows)

    comparat_csv = output_dir / "comparat_oii_2015_lf.csv"
    model_csv = output_dir / "galacticus_eval0000_oii_lf.csv"
    figure_path = output_dir / "galacticus_eval0000_vs_comparat_oii_2015.png"
    dust_draws_path = output_dir / "oii_dust_draws.csv"
    comparat.to_csv(comparat_csv, index=False)
    _write_model_csv(model_rows, model_csv)
    pd.DataFrame(
        [{"dust_draw_index": index, "dust_law": args.dust_law, **params} for index, params in enumerate(dust_params_by_draw)]
    ).to_csv(dust_draws_path, index=False)
    _plot(comparat, model, figure_path)

    print(comparat_csv)
    print(model_csv)
    print(dust_draws_path)
    print(figure_path)


if __name__ == "__main__":
    main()
