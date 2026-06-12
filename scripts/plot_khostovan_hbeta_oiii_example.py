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

COMPONENTS_NO_AGN = ["Disk", "Spheroid"]
COMPONENTS_WITH_AGN = ["Disk", "Spheroid", "AGN"]
LINE_SETS = {
    "hbeta_oiii": {
        "lines": [
            ("balmerBeta4863", 4862.68),
            ("oxygenIII4960", 4960.30),
            ("oxygenIII5008", 5008.24),
        ],
        "output_dir_name": "khostovan_hbeta_oiii_eval0000",
        "csv_stem": "hbeta_oiii",
        "axis_label": r"H\beta + [OIII]",
        "title_label": r"H$\beta$ + [OIII]",
        "plain_label": "Hbeta+[OIII]",
    },
    "oii": {
        "lines": [
            ("oxygenII3727", 3727.09),
            ("oxygenII3730", 3729.88),
        ],
        "output_dir_name": "khostovan_oii_eval0000",
        "csv_stem": "oii",
        "axis_label": r"[OII]",
        "title_label": r"[OII]",
        "plain_label": "[OII]",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a Galacticus Hbeta+[OIII] luminosity function from one full output "
            "against Khostovan et al. (2015) HiZELS binned LF data."
        )
    )
    parser.add_argument("--galacticus-file", type=Path, default=DEFAULT_GALACTICUS_FILE)
    parser.add_argument("--line-set", choices=sorted(LINE_SETS), default="hbeta_oiii")
    parser.add_argument("--output-dir", type=Path, default=None)
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
        help="Fixed attenuation law used to scale A_Halpha to each line wavelength.",
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


def _lf_dataframe(table: list[tuple[float, float, float, int, float, float, float, float]]) -> pd.DataFrame:
    data = pd.DataFrame(
        table,
        columns=[
            "redshift",
            "log10_luminosity_erg_s",
            "half_width_dex",
            "n_emitters",
            "log10_phi_observed",
            "log10_phi_final",
            "log10_phi_final_error",
            "volume_1e5_mpc3",
        ],
    )
    data["z_min"] = data["redshift"]
    data["z_max"] = data["redshift"]
    data["log10_luminosity_min"] = data["log10_luminosity_erg_s"] - data["half_width_dex"]
    data["log10_luminosity_max"] = data["log10_luminosity_erg_s"] + data["half_width_dex"]
    data["phi_mpc3_dex"] = 10.0 ** data["log10_phi_final"]
    data["phi_lower_mpc3_dex"] = 10.0 ** (data["log10_phi_final"] - data["log10_phi_final_error"])
    data["phi_upper_mpc3_dex"] = 10.0 ** (data["log10_phi_final"] + data["log10_phi_final_error"])
    data["phi_err_lower_mpc3_dex"] = data["phi_mpc3_dex"] - data["phi_lower_mpc3_dex"]
    data["phi_err_upper_mpc3_dex"] = data["phi_upper_mpc3_dex"] - data["phi_mpc3_dex"]
    return data


def _load_khostovan_lf(line_set: str) -> pd.DataFrame:
    # Khostovan et al. 2015, MNRAS 452, 3948, Appendix B.
    # Table B1 is the narrow-band Hbeta + [OIII] blend.
    if line_set == "hbeta_oiii":
        return _lf_dataframe([
        (0.84, 41.10, 0.10, 703, -1.97, -1.82, 0.02, 3.25),
        (0.84, 41.30, 0.10, 465, -2.15, -2.04, 0.03, 3.25),
        (0.84, 41.50, 0.10, 262, -2.39, -2.35, 0.04, 3.25),
        (0.84, 41.70, 0.10, 128, -2.71, -2.61, 0.06, 3.25),
        (0.84, 41.90, 0.10, 68, -2.98, -2.94, 0.08, 3.25),
        (0.84, 42.10, 0.10, 28, -3.37, -3.17, 0.13, 3.25),
        (0.84, 42.30, 0.10, 12, -3.73, -3.52, 0.20, 3.25),
        (0.84, 42.50, 0.10, 3, -4.34, -4.12, 0.39, 3.25),
        (1.42, 41.95, 0.15, 284, -2.63, -2.49, 0.03, 4.06),
        (1.42, 42.25, 0.15, 73, -3.22, -3.14, 0.07, 4.06),
        (1.42, 42.55, 0.15, 12, -4.01, -3.89, 0.19, 4.06),
        (1.42, 42.85, 0.15, 2, -4.78, -4.64, 0.48, 4.06),
        (2.23, 42.60, 0.075, 84, -3.27, -3.08, 0.06, 10.46),
        (2.23, 42.75, 0.075, 70, -3.36, -3.14, 0.07, 10.69),
        (2.23, 42.90, 0.075, 22, -3.86, -3.65, 0.13, 10.69),
        (2.23, 43.05, 0.075, 5, -4.51, -4.26, 0.29, 10.69),
        (3.24, 42.65, 0.075, 70, -3.33, -3.17, 0.07, 9.99),
        (3.24, 42.80, 0.075, 52, -3.48, -3.26, 0.09, 10.48),
        (3.24, 42.95, 0.075, 25, -3.80, -3.55, 0.13, 10.48),
        (3.24, 43.10, 0.075, 6, -4.42, -4.17, 0.27, 10.48),
    ])
    if line_set == "oii":
        # Table B2. The z=4.69 rows are intentionally omitted because this
        # campaign has no matching output and the Roman-mock use case is z~1-3.
        return _lf_dataframe([
            (1.47, 41.65, 0.075, 590, -2.24, -2.08, 0.02, 6.80),
            (1.47, 41.80, 0.075, 425, -2.38, -2.28, 0.03, 6.80),
            (1.47, 41.95, 0.075, 257, -2.60, -2.46, 0.04, 6.80),
            (1.47, 42.10, 0.075, 127, -2.90, -2.69, 0.06, 6.80),
            (1.47, 42.25, 0.075, 42, -3.39, -3.05, 0.10, 6.80),
            (1.47, 42.40, 0.075, 19, -3.73, -3.55, 0.15, 6.80),
            (1.47, 42.55, 0.075, 6, -4.23, -4.23, 0.28, 6.80),
            (2.25, 42.45, 0.10, 92, -3.14, -2.77, 0.05, 6.29),
            (2.25, 42.65, 0.10, 37, -3.53, -3.15, 0.08, 6.29),
            (2.25, 42.85, 0.10, 3, -4.62, -4.46, 0.35, 6.29),
            (3.34, 43.05, 0.050, 12, -4.12, -3.86, 0.17, 15.88),
            (3.34, 43.15, 0.075, 7, -4.37, -3.92, 0.24, 16.52),
            (3.34, 43.30, 0.075, 2, -5.22, -4.87, 0.48, 16.52),
        ])
    raise ValueError(f"Unknown line_set={line_set!r}")


def _output_redshifts(handle: h5py.File) -> dict[str, float]:
    redshifts: dict[str, float] = {}
    for output_name in handle["Outputs"]:
        output_group = handle[f"Outputs/{output_name}"]
        expansion_factor = float(output_group.attrs["outputExpansionFactor"])
        redshifts[output_name] = 1.0 / expansion_factor - 1.0
    return redshifts


def _select_output(redshifts: dict[str, float], redshift: float) -> tuple[str, float]:
    output_name, output_redshift = min(redshifts.items(), key=lambda item: abs(item[1] - redshift))
    return output_name, output_redshift


def _line_luminosities(output_group: h5py.Group, line_set: str, include_agn: bool) -> dict[str, np.ndarray]:
    nd = output_group["nodeData"]
    components = COMPONENTS_WITH_AGN if include_agn else COMPONENTS_NO_AGN
    line_luminosities: dict[str, np.ndarray] = {}
    for line_name, _wavelength in LINE_SETS[line_set]["lines"]:
        luminosity = None
        for component in components:
            dataset_name = f"luminosityEmissionLine{component}:{line_name}"
            values = np.asarray(nd[dataset_name][...], dtype=float)
            luminosity = values.copy() if luminosity is None else luminosity + values
        if luminosity is None:
            raise RuntimeError(f"No luminosity datasets were summed for {line_name}")
        line_luminosities[line_name] = luminosity
    return line_luminosities


def _total_luminosity(line_luminosities: dict[str, np.ndarray], line_set: str) -> np.ndarray:
    luminosity = None
    for line_name, _wavelength in LINE_SETS[line_set]["lines"]:
        values = line_luminosities[line_name]
        luminosity = values.copy() if luminosity is None else luminosity + values
    if luminosity is None:
        raise RuntimeError("No line luminosities were summed")
    return luminosity


def _stellar_mass(output_group: h5py.Group) -> np.ndarray:
    nd = output_group["nodeData"]
    return np.asarray(nd["diskMassStellar"][...], dtype=float) + np.asarray(nd["spheroidMassStellar"][...], dtype=float)


def _gb10_attenuated_luminosity(
    output_group: h5py.Group,
    redshift: float,
    line_set: str,
    line_luminosities: dict[str, np.ndarray],
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
    attenuated = None
    for line_name, wavelength in LINE_SETS[line_set]["lines"]:
        values = np.asarray(
            apply_dust_attenuation_to_line(
                line_luminosities[line_name],
                wavelength,
                a_halpha,
                dust_law=dust_law,
            ),
            dtype=float,
        )
        attenuated = values.copy() if attenuated is None else attenuated + values
    if attenuated is None:
        raise RuntimeError("No attenuated luminosities were summed")
    return attenuated


def _lf_per_dex(luminosity: np.ndarray, weights: np.ndarray, log10_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(luminosity) & np.isfinite(weights) & (luminosity > 0.0)
    log10_luminosity = np.log10(luminosity[valid])
    counts, _ = np.histogram(log10_luminosity, bins=log10_edges, weights=weights[valid])
    variance_counts, _ = np.histogram(log10_luminosity, bins=log10_edges, weights=weights[valid] ** 2)
    bin_widths = np.diff(log10_edges)
    return counts / bin_widths, np.sqrt(np.clip(variance_counts, 0.0, None)) / bin_widths


def _model_rows(
    galacticus_file: Path,
    observations: pd.DataFrame,
    *,
    line_set: str,
    include_agn: bool,
    include_dust: bool,
    dust_params_by_draw: list[dict[str, float]],
    dust_law: str,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(galacticus_file, "r") as handle:
        redshifts = _output_redshifts(handle)
        for redshift, group in observations.groupby("redshift", sort=True):
            sorted_group = group.sort_values("log10_luminosity_erg_s").reset_index(drop=True)
            output_name, output_redshift = _select_output(redshifts, float(redshift))
            output_group = handle[f"Outputs/{output_name}"]
            weights = _combined_node_weights(output_group)
            edges = np.round(
                np.concatenate(
                    [
                        [float(sorted_group["log10_luminosity_min"].iloc[0])],
                        sorted_group["log10_luminosity_max"].to_numpy(dtype=float),
                    ]
                ),
                decimals=6,
            )
            line_luminosities = _line_luminosities(output_group, line_set=line_set, include_agn=include_agn)
            cases = {"intrinsic": _total_luminosity(line_luminosities, line_set)}
            if include_dust:
                baseline_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 0, int(output_name[6:]), 10_000]))
                cases["gb10_baseline_calzetti"] = _gb10_attenuated_luminosity(
                    output_group,
                    output_redshift,
                    line_set,
                    line_luminosities,
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
                    cases[f"dust_draw_{dust_draw_index:04d}"] = _gb10_attenuated_luminosity(
                        output_group,
                        output_redshift,
                        line_set,
                        line_luminosities,
                        dust_params,
                        dust_law=dust_law,
                        rng=rng,
                    )
            for case_label, luminosity in cases.items():
                phi, phi_std = _lf_per_dex(luminosity, weights, edges)
                for bin_index, row in sorted_group.iterrows():
                    dust_draw_index = None
                    dust_params: dict[str, float] = {}
                    if case_label.startswith("dust_draw_"):
                        dust_draw_index = int(case_label.rsplit("_", 1)[-1])
                        dust_params = dust_params_by_draw[dust_draw_index]
                    rows.append(
                        {
                            "redshift": float(redshift),
                            "output_name": output_name,
                            "output_redshift": output_redshift,
                            "case": case_label,
                            "dust_draw_index": dust_draw_index,
                            "dust_law": dust_law if case_label.startswith("dust_draw_") or case_label == "gb10_baseline_calzetti" else None,
                            "include_agn": include_agn,
                            "bin_index": int(bin_index),
                            "log10_luminosity_erg_s": float(row["log10_luminosity_erg_s"]),
                            "log10_luminosity_min": float(row["log10_luminosity_min"]),
                            "log10_luminosity_max": float(row["log10_luminosity_max"]),
                            "phi_mpc3_dex": float(phi[int(bin_index)]),
                            "phi_shot_noise_std_mpc3_dex": float(phi_std[int(bin_index)]),
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


def _plot(observations: pd.DataFrame, model: pd.DataFrame, line_set: str, output_path: Path) -> None:
    line_config = LINE_SETS[line_set]
    redshifts = sorted(observations["redshift"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    dust_color = "#d1495b"

    for axis, redshift in zip(axes_flat, redshifts, strict=False):
        obs = observations[observations["redshift"] == redshift]
        axis.errorbar(
            obs["log10_luminosity_erg_s"],
            obs["phi_mpc3_dex"],
            xerr=obs["half_width_dex"],
            yerr=[
                obs["phi_err_lower_mpc3_dex"],
                obs["phi_err_upper_mpc3_dex"],
            ],
            fmt="o",
            ms=4,
            color="black",
            ecolor="0.35",
            elinewidth=0.9,
            capsize=2,
            label="Khostovan et al. 2015",
        )
        selection = model["redshift"] == redshift
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
                positive = case_model["phi_mpc3_dex"].to_numpy(dtype=float) > 0.0
                axis.plot(
                    case_model["log10_luminosity_erg_s"].to_numpy(dtype=float)[positive],
                    case_model["phi_mpc3_dex"].to_numpy(dtype=float)[positive],
                    lw=0.9,
                    alpha=0.32,
                    color=color,
                    label=label,
                )
            else:
                x = case_model["log10_luminosity_erg_s"].to_numpy(dtype=float)
                y = case_model["phi_mpc3_dex"].to_numpy(dtype=float)
                yerr = case_model["phi_shot_noise_std_mpc3_dex"].to_numpy(dtype=float)
                positive = y > 0.0
                x = x[positive]
                y = y[positive]
                yerr = yerr[positive]
                yerr = np.where(y > 0.0, np.minimum(yerr, 0.98 * y), 0.0)
                axis.errorbar(
                    x,
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
        axis.set_title(f"z={redshift:.2f}  ({example['output_name']}, z={example['output_redshift']:.2f})")
        axis.set_yscale("log")
        axis.set_ylim(1.0e-8, 3.0e-2)
        axis.grid(alpha=0.18)
    for axis in axes_flat[len(redshifts) :]:
        axis.axis("off")

    x_min = float(observations["log10_luminosity_min"].min()) - 0.25
    x_max = float(observations["log10_luminosity_max"].max()) + 0.25
    for axis in axes_flat[: len(redshifts)]:
        axis.set_xlim(x_min, x_max)

    for axis in axes[-1, :]:
        axis.set_xlabel(rf"$\log_{{10}}(L_{{\mathrm{{{line_config['axis_label']}}}}}/\mathrm{{erg\,s^{{-1}}}})$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    legend_axis = axes_flat[len(redshifts) - 1]
    legend_axis.legend(handles, legend_labels, loc="lower right", frameon=False, fontsize=10)
    fig.suptitle(
        rf"{line_config['title_label']} luminosity function: full eval-0000 Galacticus output vs. Khostovan et al. (2015)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    line_config = LINE_SETS[args.line_set]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / "playing/emission_line_lf_comparisons" / str(line_config["output_dir_name"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = _load_khostovan_lf(args.line_set)
    priors = _load_dust_priors(args)
    rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, 10_002]))
    dust_params_by_draw = [
        _sample_dust_params(priors, z_pivot=args.z_pivot, rng=rng)
        for _ in range(args.n_dust_draws)
    ]
    model_rows = _model_rows(
        args.galacticus_file.resolve(),
        observations,
        line_set=args.line_set,
        include_agn=args.include_agn,
        include_dust=not args.skip_dust,
        dust_params_by_draw=dust_params_by_draw,
        dust_law=args.dust_law,
        base_seed=args.base_seed,
    )
    model = pd.DataFrame(model_rows)

    stem = str(line_config["csv_stem"])
    observations_csv = output_dir / f"khostovan_2015_{stem}_lf.csv"
    model_csv = output_dir / f"galacticus_eval0000_{stem}_lf.csv"
    figure_path = output_dir / f"galacticus_eval0000_vs_khostovan_{stem}_2015.png"
    dust_draws_path = output_dir / f"{stem}_dust_draws.csv"
    observations.to_csv(observations_csv, index=False)
    _write_model_csv(model_rows, model_csv)
    pd.DataFrame(
        [{"dust_draw_index": index, "dust_law": args.dust_law, **params} for index, params in enumerate(dust_params_by_draw)]
    ).to_csv(dust_draws_path, index=False)
    _plot(observations, model, args.line_set, figure_path)

    print(observations_csv)
    print(model_csv)
    print(dust_draws_path)
    print(figure_path)


if __name__ == "__main__":
    main()
