from __future__ import annotations

import argparse
import csv
import json
import os
from math import erf
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import h5py
import numpy as np

from calculate_halpha_dust_lf_grid import (  # type: ignore
    SOBRAL_CASES,
    _combined_node_weights,
    _log10_std_from_linear,
    _sobral_bin_edges,
)
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


LINE_GROUPS = {
    "halpha_sobral": {
        "label": "Halpha Sobral",
        "lines": [("balmerAlpha6565", 6564.61)],
        "components": ["AGN", "Disk", "Spheroid"],
        "effective_wavelength": 6562.8,
    },
    "oii_comparat": {
        "label": "[OII] Comparat",
        "lines": [("oxygenII3727", 3727.09), ("oxygenII3730", 3729.88)],
        "components": ["Disk", "Spheroid"],
        "effective_wavelength": 3728.5,
    },
    "oii_khostovan": {
        "label": "[OII] Khostovan",
        "lines": [("oxygenII3727", 3727.09), ("oxygenII3730", 3729.88)],
        "components": ["Disk", "Spheroid"],
        "effective_wavelength": 3728.5,
    },
    "hbeta_oiii_khostovan": {
        "label": "Hbeta+[OIII] Khostovan",
        "lines": [("balmerBeta4863", 4862.68), ("oxygenIII4960", 4960.30), ("oxygenIII5008", 5008.24)],
        "components": ["Disk", "Spheroid"],
        "effective_wavelength": 5000.0,
    },
}

COMPARAT_OII_CASES = [
    (0.100, 0.240, [40.625, 40.875, 41.125, 41.375, 41.625, 41.875, 42.125]),
    (0.240, 0.400, [42.125, 42.375, 42.625]),
    (0.500, 0.695, [41.125, 41.375, 41.625, 41.875, 42.125, 42.375, 42.625, 42.875, 43.125]),
    (0.695, 0.880, [40.875, 41.125, 41.375, 41.625, 41.875, 42.125, 42.375, 42.625, 42.875, 43.125]),
    (0.880, 1.090, [41.125, 41.375, 41.625, 41.875, 42.125, 42.375, 42.625, 42.875, 43.125, 43.375]),
    (1.090, 1.340, [41.875, 42.125, 42.375, 42.625, 42.875, 43.125, 43.375]),
    (1.340, 1.650, [42.375, 42.625, 42.875, 43.125, 43.625]),
]

KHOSTOVAN_HBETA_OIII_CASES = [
    (0.84, [41.10, 41.30, 41.50, 41.70, 41.90, 42.10, 42.30, 42.50], [0.10] * 8),
    (1.42, [41.95, 42.25, 42.55, 42.85], [0.15] * 4),
    (2.23, [42.60, 42.75, 42.90, 43.05], [0.075] * 4),
    (3.24, [42.65, 42.80, 42.95, 43.10], [0.075] * 4),
]

KHOSTOVAN_OII_CASES = [
    (1.47, [41.65, 41.80, 41.95, 42.10, 42.25, 42.40, 42.55], [0.075] * 7),
    (2.25, [42.45, 42.65, 42.85], [0.10] * 3),
    (3.34, [43.05, 43.15, 43.30], [0.050, 0.075, 0.075]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process one Galacticus evaluation to compute dust-attenuated "
            "Halpha, [OII], and Hbeta+[OIII] luminosity functions for common dust draws."
        )
    )
    parser.add_argument("galacticus_file", type=Path)
    parser.add_argument("--evaluation-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-dust-draws", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--evaluation-index", type=int, default=0)
    parser.add_argument("--scatter-mode", choices=["draw", "expected"], default="expected")
    parser.add_argument("--z-pivot", type=float, default=1.0)
    parser.add_argument("--random-uniform-index", type=int, default=None)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--dust-prior-json", action="append", default=[])
    parser.add_argument(
        "--fill-missing-node-weights-with-median",
        action="store_true",
        help=(
            "Salvage Galacticus files affected by the merger-tree weight race condition by filling "
            "unassigned node weights with the median finite mergerTreeWeight. Default is strict/error."
        ),
    )
    return parser.parse_args()


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def _output_redshifts(handle: h5py.File) -> dict[str, float]:
    redshifts: dict[str, float] = {}
    for output_name in handle["Outputs"]:
        expansion_factor = float(handle[f"Outputs/{output_name}"].attrs["outputExpansionFactor"])
        redshifts[output_name] = 1.0 / expansion_factor - 1.0
    return redshifts


def _select_output(redshifts: dict[str, float], z_min: float, z_max: float) -> tuple[str, float]:
    z_mid = 0.5 * (z_min + z_max)
    in_bin = {name: z for name, z in redshifts.items() if z_min <= z <= z_max}
    candidates = in_bin or redshifts
    return min(candidates.items(), key=lambda item: abs(item[1] - z_mid))


def _centers_to_edges(log10_centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(log10_centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def _centers_half_widths_to_edges(centers: np.ndarray, half_widths: np.ndarray) -> np.ndarray:
    return np.round(np.concatenate([[centers[0] - half_widths[0]], centers + half_widths]), decimals=6)


def _stellar_mass(output_group: h5py.Group) -> np.ndarray:
    nd = output_group["nodeData"]
    return np.asarray(nd["diskMassStellar"][...], dtype=float) + np.asarray(nd["spheroidMassStellar"][...], dtype=float)


def _line_group_luminosities(output_group: h5py.Group, group_name: str) -> dict[str, np.ndarray]:
    spec = LINE_GROUPS[group_name]
    nd = output_group["nodeData"]
    components = spec["components"]
    line_luminosities: dict[str, np.ndarray] = {}
    for line_name, _wavelength in spec["lines"]:
        luminosity = None
        for component in components:
            dataset_name = f"luminosityEmissionLine{component}:{line_name}"
            values = np.asarray(nd[dataset_name][...], dtype=float)
            luminosity = values.copy() if luminosity is None else luminosity + values
        if luminosity is None:
            raise RuntimeError(f"No luminosity datasets were summed for {line_name}")
        line_luminosities[line_name] = luminosity
    return line_luminosities


def _total_luminosity(line_luminosities: dict[str, np.ndarray], group_name: str) -> np.ndarray:
    total = None
    for line_name, _wavelength in LINE_GROUPS[group_name]["lines"]:
        values = line_luminosities[line_name]
        total = values.copy() if total is None else total + values
    if total is None:
        raise RuntimeError(f"No luminosities were summed for {group_name}")
    return total


def _attenuation_coefficient(wavelength: float) -> float:
    attenuated = np.asarray(
        apply_dust_attenuation_to_line(
            np.ones(1, dtype=float),
            wavelength,
            np.ones(1, dtype=float),
            dust_law="calzetti",
        ),
        dtype=float,
    )
    return -float(np.log10(attenuated[0])) / 0.4


def _dust_attenuated_luminosity(
    output_group: h5py.Group,
    redshift: float,
    group_name: str,
    line_luminosities: dict[str, np.ndarray],
    dust_case: dict[str, Any],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    mstar = _stellar_mass(output_group)
    nd = output_group["nodeData"]
    random_uniform = None
    if dust_case["random_uniform_index"] is not None and "randomUniform" in nd:
        random_uniform_array = np.asarray(nd["randomUniform"][...], dtype=float)
        random_uniform = random_uniform_array[:, int(dust_case["random_uniform_index"])]
    a_halpha = dust_attenuation_gb10_generalised(
        mstar,
        np.full_like(mstar, redshift, dtype=float),
        rng=rng,
        random_uniform=random_uniform,
        **dust_case["dust_params"],
    )
    attenuated = None
    for line_name, wavelength in LINE_GROUPS[group_name]["lines"]:
        values = np.asarray(
            apply_dust_attenuation_to_line(
                line_luminosities[line_name],
                wavelength,
                a_halpha,
                dust_law=dust_case["dust_law"],
            ),
            dtype=float,
        )
        attenuated = values.copy() if attenuated is None else attenuated + values
    if attenuated is None:
        raise RuntimeError(f"No attenuated luminosities were summed for {group_name}")
    return attenuated


def _lf_from_luminosities(luminosities: np.ndarray, weights: np.ndarray, log10_edges: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(luminosities) & np.isfinite(weights) & (luminosities > 0.0)
    log10_luminosity = np.log10(luminosities[valid])
    counts, _ = np.histogram(log10_luminosity, bins=log10_edges, weights=weights[valid])
    variance_counts, _ = np.histogram(log10_luminosity, bins=log10_edges, weights=weights[valid] ** 2)
    dlog10 = np.diff(log10_edges)
    dln = dlog10 * np.log(10.0)
    return {
        "phi_mpc3_dex": counts / dlog10,
        "phi_variance_mpc3_dex2": variance_counts / dlog10**2,
        "dn_dlnL_mpc3": counts / dln,
        "dn_dlnL_variance_mpc3": variance_counts / dln**2,
    }


def _lf_from_expected_scatter(
    luminosities: np.ndarray,
    stellar_mass: np.ndarray,
    weights: np.ndarray,
    redshift: float,
    dust_params: dict[str, Any],
    log10_edges: np.ndarray,
    attenuation_coefficient: float,
) -> dict[str, np.ndarray]:
    valid = (
        np.isfinite(luminosities)
        & np.isfinite(stellar_mass)
        & np.isfinite(weights)
        & (luminosities > 0.0)
        & (stellar_mass > 0.0)
    )
    n_bins = log10_edges.size - 1
    if not np.any(valid):
        zeros = np.zeros(n_bins, dtype=float)
        return {
            "phi_mpc3_dex": zeros,
            "phi_variance_conservative_mpc3_dex2": zeros,
            "phi_variance_smoothed_mpc3_dex2": zeros,
            "dn_dlnL_mpc3": zeros,
            "dn_dlnL_variance_conservative_mpc3": zeros,
            "dn_dlnL_variance_smoothed_mpc3": zeros,
        }

    lum = luminosities[valid]
    mstar = stellar_mass[valid]
    w = weights[valid]
    sigma_a = float(dust_params.get("attenuation_scatter", 0.0))
    mean_params = dict(dust_params)
    mean_params["attenuation_scatter"] = 0.0
    a0 = dust_attenuation_gb10_generalised(
        mstar,
        np.full_like(mstar, redshift, dtype=float),
        **mean_params,
    )
    log10_lum = np.log10(lum)
    if sigma_a <= 0.0:
        shifted = log10_lum - 0.4 * attenuation_coefficient * a0
        return _lf_from_luminosities(10.0**shifted, w, log10_edges)

    counts = np.zeros(n_bins, dtype=float)
    variance_conservative_counts = np.zeros(n_bins, dtype=float)
    variance_smoothed_counts = np.zeros(n_bins, dtype=float)
    p_nonpos = _normal_cdf((-a0) / sigma_a)
    shift_per_mag = 0.4 * attenuation_coefficient

    for bin_index, (lo, hi) in enumerate(zip(log10_edges[:-1], log10_edges[1:], strict=True)):
        point_mask = (log10_lum >= lo) & (log10_lum < hi)
        p_bin = np.zeros_like(w)
        p_bin[point_mask] += p_nonpos[point_mask]
        lower_a = np.maximum(0.0, (log10_lum - hi) / shift_per_mag)
        upper_a = np.maximum(0.0, (log10_lum - lo) / shift_per_mag)
        cdf_upper = _normal_cdf((upper_a - a0) / sigma_a)
        cdf_lower = _normal_cdf((lower_a - a0) / sigma_a)
        p_bin += np.maximum(cdf_upper - cdf_lower, 0.0)
        p_bin = np.clip(p_bin, 0.0, 1.0)
        weighted_probabilities = w * p_bin
        counts[bin_index] = np.sum(weighted_probabilities)
        variance_conservative_counts[bin_index] = np.sum((w**2) * p_bin)
        variance_smoothed_counts[bin_index] = np.sum(weighted_probabilities**2)

    dlog10 = np.diff(log10_edges)
    dln = dlog10 * np.log(10.0)
    return {
        "phi_mpc3_dex": counts / dlog10,
        "phi_variance_conservative_mpc3_dex2": variance_conservative_counts / dlog10**2,
        "phi_variance_smoothed_mpc3_dex2": variance_smoothed_counts / dlog10**2,
        "dn_dlnL_mpc3": counts / dln,
        "dn_dlnL_variance_conservative_mpc3": variance_conservative_counts / dln**2,
        "dn_dlnL_variance_smoothed_mpc3": variance_smoothed_counts / dln**2,
    }


def _case_definitions(handle: h5py.File) -> list[dict[str, Any]]:
    redshifts = _output_redshifts(handle)
    cases: list[dict[str, Any]] = []
    for output_name, redshift, sobral_label, analysis_name in SOBRAL_CASES:
        analysis_group = handle[f"/analyses/{analysis_name}"]
        centers = np.log10(np.asarray(analysis_group["luminosity"][...], dtype=float))
        cases.append(
            {
                "observable": "halpha_sobral",
                "sample_label": sobral_label,
                "output_name": output_name,
                "redshift": float(redshift),
                "analysis_name": analysis_name,
                "log10_centers": centers,
                "log10_edges": _sobral_bin_edges(centers),
            }
        )
    for z_min, z_max, centers_text in COMPARAT_OII_CASES:
        output_name, output_redshift = _select_output(redshifts, z_min, z_max)
        centers = np.asarray(centers_text, dtype=float)
        cases.append(
            {
                "observable": "oii_comparat",
                "sample_label": f"z{z_min:.3f}_{z_max:.3f}",
                "output_name": output_name,
                "redshift": float(output_redshift),
                "z_min": float(z_min),
                "z_max": float(z_max),
                "log10_centers": centers,
                "log10_edges": _centers_to_edges(centers),
            }
        )
    for redshift, centers_text, half_widths_text in KHOSTOVAN_OII_CASES:
        output_name, output_redshift = _select_output(redshifts, redshift, redshift)
        centers = np.asarray(centers_text, dtype=float)
        half_widths = np.asarray(half_widths_text, dtype=float)
        cases.append(
            {
                "observable": "oii_khostovan",
                "sample_label": f"z{redshift:.2f}",
                "output_name": output_name,
                "redshift": float(output_redshift),
                "target_redshift": float(redshift),
                "log10_centers": centers,
                "log10_edges": _centers_half_widths_to_edges(centers, half_widths),
            }
        )
    for redshift, centers_text, half_widths_text in KHOSTOVAN_HBETA_OIII_CASES:
        output_name, output_redshift = _select_output(redshifts, redshift, redshift)
        centers = np.asarray(centers_text, dtype=float)
        half_widths = np.asarray(half_widths_text, dtype=float)
        cases.append(
            {
                "observable": "hbeta_oiii_khostovan",
                "sample_label": f"z{redshift:.2f}",
                "output_name": output_name,
                "redshift": float(output_redshift),
                "target_redshift": float(redshift),
                "log10_centers": centers,
                "log10_edges": _centers_half_widths_to_edges(centers, half_widths),
            }
        )
    return cases


def _compute_case_lfs(
    galacticus_file: Path,
    dust_case: dict[str, Any],
    *,
    base_seed: int,
    fill_missing_node_weights_with_median: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(galacticus_file, "r") as handle:
        for case in _case_definitions(handle):
            observable = case["observable"]
            output_group = handle[f"/Outputs/{case['output_name']}"]
            weights = _combined_node_weights(
                output_group,
                fill_missing_with_median=fill_missing_node_weights_with_median,
            )
            line_luminosities = _line_group_luminosities(output_group, observable)
            intrinsic = _total_luminosity(line_luminosities, observable)
            if dust_case["scatter_mode"] == "expected":
                lf = _lf_from_expected_scatter(
                    intrinsic,
                    _stellar_mass(output_group),
                    weights,
                    float(case["redshift"]),
                    dust_case["dust_params"],
                    np.asarray(case["log10_edges"], dtype=float),
                    _attenuation_coefficient(float(LINE_GROUPS[observable]["effective_wavelength"])),
                )
                phi_var_conservative = lf["phi_variance_conservative_mpc3_dex2"]
                phi_var_smoothed = lf["phi_variance_smoothed_mpc3_dex2"]
                dn_var_conservative = lf["dn_dlnL_variance_conservative_mpc3"]
                dn_var_smoothed = lf["dn_dlnL_variance_smoothed_mpc3"]
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([base_seed, dust_case["dust_draw_index"], int(case["output_name"][6:])])
                )
                luminosity = _dust_attenuated_luminosity(
                    output_group,
                    float(case["redshift"]),
                    observable,
                    line_luminosities,
                    dust_case,
                    rng=rng,
                )
                lf = _lf_from_luminosities(luminosity, weights, np.asarray(case["log10_edges"], dtype=float))
                phi_var_conservative = lf["phi_variance_mpc3_dex2"]
                phi_var_smoothed = lf["phi_variance_mpc3_dex2"]
                dn_var_conservative = lf["dn_dlnL_variance_mpc3"]
                dn_var_smoothed = lf["dn_dlnL_variance_mpc3"]

            phi_std_conservative = np.sqrt(np.clip(phi_var_conservative, 0.0, None))
            phi_std_smoothed = np.sqrt(np.clip(phi_var_smoothed, 0.0, None))
            dn_std_conservative = np.sqrt(np.clip(dn_var_conservative, 0.0, None))
            dn_std_smoothed = np.sqrt(np.clip(dn_var_smoothed, 0.0, None))
            centers = np.asarray(case["log10_centers"], dtype=float)
            edges = np.asarray(case["log10_edges"], dtype=float)
            for bin_index, center in enumerate(centers):
                value_phi = float(lf["phi_mpc3_dex"][bin_index])
                value_dn = float(lf["dn_dlnL_mpc3"][bin_index])
                row = {
                    "observable": observable,
                    "observable_label": LINE_GROUPS[observable]["label"],
                    "sample_label": case["sample_label"],
                    "output_name": case["output_name"],
                    "redshift": float(case["redshift"]),
                    "bin_index": int(bin_index),
                    "log10_luminosity_min": float(edges[bin_index]),
                    "log10_luminosity_center": float(center),
                    "log10_luminosity_max": float(edges[bin_index + 1]),
                    "phi_mpc3_dex": value_phi,
                    "phi_mpc3_dex_shot_noise_std_conservative": float(phi_std_conservative[bin_index]),
                    "phi_mpc3_dex_shot_noise_std_smoothed_expectation": float(phi_std_smoothed[bin_index]),
                    "log10_phi_shot_noise_std_conservative": _log10_std_from_linear(
                        value_phi,
                        float(phi_std_conservative[bin_index]),
                    ),
                    "log10_phi_shot_noise_std_smoothed_expectation": _log10_std_from_linear(
                        value_phi,
                        float(phi_std_smoothed[bin_index]),
                    ),
                    "dn_dlnL_mpc3": value_dn,
                    "dn_dlnL_mpc3_shot_noise_std_conservative": float(dn_std_conservative[bin_index]),
                    "dn_dlnL_mpc3_shot_noise_std_smoothed_expectation": float(dn_std_smoothed[bin_index]),
                    "log10_dn_dlnL_shot_noise_std_conservative": _log10_std_from_linear(
                        value_dn,
                        float(dn_std_conservative[bin_index]),
                    ),
                    "log10_dn_dlnL_shot_noise_std_smoothed_expectation": _log10_std_from_linear(
                        value_dn,
                        float(dn_std_smoothed[bin_index]),
                    ),
                }
                if "z_min" in case:
                    row["z_min"] = case["z_min"]
                    row["z_max"] = case["z_max"]
                if "target_redshift" in case:
                    row["target_redshift"] = case["target_redshift"]
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    galacticus_file = args.galacticus_file.resolve()
    evaluation_id = args.evaluation_id or galacticus_file.parent.name
    output_dir = args.output_dir.resolve() if args.output_dir is not None else galacticus_file.parent / "emission_line_dust"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_values: dict[str, Any] = {}
    if args.input_json is not None:
        input_values = json.loads(args.input_json.read_text())

    priors = _load_dust_priors(args)
    rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, args.evaluation_index]))
    dust_draw_rows: list[dict[str, Any]] = []
    lf_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    print(f"starting emission-line dust post-processing for {evaluation_id}")
    for dust_draw_index in range(args.n_dust_draws):
        print(f"{evaluation_id} dust draw {dust_draw_index + 1}/{args.n_dust_draws}")
        dust_params = _sample_dust_params(priors, z_pivot=args.z_pivot, rng=rng)
        dust_case = {
            "label": f"dust_draw_{dust_draw_index:04d}",
            "dust_model": "gb10_generalised",
            "dust_params": dust_params,
            "dust_law": "calzetti",
            "random_uniform_index": args.random_uniform_index,
            "scatter_mode": args.scatter_mode,
            "dust_draw_index": int(dust_draw_index),
        }
        draw_metadata = {
            "evaluation_id": evaluation_id,
            "dust_draw_index": int(dust_draw_index),
            "dust_case": dust_case["label"],
            "scatter_mode": args.scatter_mode,
            **dust_params,
        }
        dust_draw_rows.append(draw_metadata)
        wide_row = {
            "evaluation_id": evaluation_id,
            "dust_draw_index": int(dust_draw_index),
            "dust_case": dust_case["label"],
            "scatter_mode": args.scatter_mode,
            **input_values,
            **dust_params,
        }
        for row in _compute_case_lfs(
            galacticus_file,
            dust_case,
            base_seed=args.base_seed,
            fill_missing_node_weights_with_median=args.fill_missing_node_weights_with_median,
        ):
            merged = dict(draw_metadata)
            merged.update(row)
            lf_rows.append(merged)
            prefix = f"{row['observable']}_{row['sample_label'].lower()}_bin{row['bin_index']}"
            wide_row[prefix] = row["dn_dlnL_mpc3"]
            wide_row[f"{prefix}_shot_noise_std_conservative"] = row["dn_dlnL_mpc3_shot_noise_std_conservative"]
            wide_row[f"{prefix}_shot_noise_std_smoothed_expectation"] = row[
                "dn_dlnL_mpc3_shot_noise_std_smoothed_expectation"
            ]
            wide_row[f"{prefix}_shot_noise_log10_std_conservative"] = row[
                "log10_dn_dlnL_shot_noise_std_conservative"
            ]
            wide_row[f"{prefix}_shot_noise_log10_std_smoothed_expectation"] = row[
                "log10_dn_dlnL_shot_noise_std_smoothed_expectation"
            ]
            wide_row[f"{prefix}_phi_mpc3_dex"] = row["phi_mpc3_dex"]
            wide_row[f"{prefix}_phi_shot_noise_std_conservative"] = row["phi_mpc3_dex_shot_noise_std_conservative"]
            wide_row[f"{prefix}_phi_shot_noise_std_smoothed_expectation"] = row[
                "phi_mpc3_dex_shot_noise_std_smoothed_expectation"
            ]
        table_rows.append(wide_row)

    dust_draws_path = output_dir / "dust_draws.csv"
    lf_long_path = output_dir / "emission_line_dust_lf_long.csv"
    emulator_table_path = output_dir / "emission_line_dust_lf_emulator_table.csv"
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "manifest.csv"
    _write_csv(dust_draws_path, dust_draw_rows)
    _write_csv(lf_long_path, lf_rows)
    _write_csv(emulator_table_path, table_rows)
    metadata_path.write_text(
        json.dumps(
            {
                "evaluation_id": evaluation_id,
                "galacticus_file": str(galacticus_file),
                "n_dust_draws": args.n_dust_draws,
                "scatter_mode": args.scatter_mode,
                "z_pivot": args.z_pivot,
                "random_uniform_index": args.random_uniform_index,
                "fill_missing_node_weights_with_median": args.fill_missing_node_weights_with_median,
                "dust_priors": priors,
                "input_values": input_values,
                "line_groups": LINE_GROUPS,
            },
            indent=2,
        )
        + "\n"
    )
    _write_csv(
        manifest_path,
        [
            {"file": str(dust_draws_path), "description": "Common dust parameter draws for this evaluation."},
            {"file": str(lf_long_path), "description": "Long-form emission-line LF values for this evaluation."},
            {"file": str(emulator_table_path), "description": "Wide-form emulator table rows for this evaluation."},
            {"file": str(metadata_path), "description": "Processing metadata for this evaluation."},
        ],
    )
    print(f"finished emission-line dust post-processing for {evaluation_id}")
    print(dust_draws_path)
    print(lf_long_path)
    print(emulator_table_path)
    print(metadata_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
