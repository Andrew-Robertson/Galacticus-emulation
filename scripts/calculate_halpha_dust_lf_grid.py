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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

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

from dust_attenuation import dust_attenuation_gb10_generalised  # type: ignore


SOBRAL_CASES = [
    ("Output16", 0.40, "Z1", "luminosityFunctionHalphaSobral2013HiZELSZ1"),
    ("Output13", 0.84, "Z2", "luminosityFunctionHalphaSobral2013HiZELSZ2"),
    ("Output10", 1.47, "Z3", "luminosityFunctionHalphaSobral2013HiZELSZ3"),
    ("Output6", 2.23, "Z4", "luminosityFunctionHalphaSobral2013HiZELSZ4"),
]

HALPHA_DATASETS = [
    "luminosityEmissionLineAGN:balmerAlpha6565",
    "luminosityEmissionLineDisk:balmerAlpha6565",
    "luminosityEmissionLineSpheroid:balmerAlpha6565",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate intrinsic and dust-attenuated Halpha luminosity functions "
            "for Galacticus fixed-time outputs using galacticus_sed_calculator code."
        )
    )
    parser.add_argument("galacticus_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-no-dust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dust-yaml",
        action="append",
        default=[],
        help="Dust case in the form label=/path/to/config.yaml. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--dust-json",
        action="append",
        default=[],
        help=(
            "Dust case in the form label='{\"dust_model\":...,\"dust_params\":...,\"dust_law\":...}'. "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--allow-missing-random-uniform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a dust config requests random_uniform_index but the file lacks randomUniform, drop the index and continue.",
    )
    parser.add_argument(
        "--scatter-mode",
        choices=["draw", "expected", "both"],
        default="both",
        help=(
            "How to treat attenuation_scatter when present. 'draw' uses one random realization per galaxy; "
            "'expected' computes the expected LF by integrating over the clipped attenuation distribution; "
            "'both' includes both curves."
        ),
    )
    return parser.parse_args()


def _read_tree_weights(output_group: h5py.Group, n_nodes: int) -> np.ndarray:
    tree_weights = np.asarray(output_group["mergerTreeWeight"][...], dtype=float)
    tree_counts = np.asarray(output_group["mergerTreeCount"][...], dtype=int)
    tree_starts = np.asarray(output_group["mergerTreeStartIndex"][...], dtype=int)
    node_weights = np.zeros(n_nodes, dtype=float)
    for start, count, weight in zip(tree_starts, tree_counts, tree_weights, strict=True):
        node_weights[start : start + count] = weight
    if np.any(node_weights == 0.0):
        raise ValueError("Some nodes were not assigned a merger-tree weight")
    return node_weights


def _combined_node_weights(output_group: h5py.Group) -> np.ndarray:
    nd = output_group["nodeData"]
    first_dataset = next(iter(nd.values()))
    n_nodes = first_dataset.shape[0]
    weights = _read_tree_weights(output_group, n_nodes)
    if "nodeSubsamplingWeight" in nd:
        weights = weights * np.asarray(nd["nodeSubsamplingWeight"][...], dtype=float)
    return weights


def _parse_labelled_spec(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Expected 'label=value' format, got: {text}")
    label, value = text.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise ValueError(f"Expected non-empty label and value, got: {text}")
    return label, value


def _load_dust_cases(args: argparse.Namespace, galacticus_file: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if args.include_no_dust:
        cases.append(
            {
                "label": "no_dust",
                "dust_model": None,
                "dust_params": None,
                "dust_law": None,
                "random_uniform_index": None,
            }
        )

    for spec in args.dust_yaml:
        label, path_text = _parse_labelled_spec(spec)
        with open(Path(path_text).expanduser(), "r") as handle:
            config = yaml.safe_load(handle)
        cases.append(
            {
                "label": label,
                "dust_model": config["dust_model"],
                "dust_params": config["dust_params"],
                "dust_law": config.get("dust_law", "calzetti"),
                "random_uniform_index": config.get("random_uniform_index"),
            }
        )

    for spec in args.dust_json:
        label, json_text = _parse_labelled_spec(spec)
        config = json.loads(json_text)
        cases.append(
            {
                "label": label,
                "dust_model": config["dust_model"],
                "dust_params": config["dust_params"],
                "dust_law": config.get("dust_law", "calzetti"),
                "random_uniform_index": config.get("random_uniform_index"),
            }
        )

    with h5py.File(galacticus_file, "r") as handle:
        for case in cases:
            if case["dust_model"] is None or case["random_uniform_index"] is None:
                continue
            all_outputs_have_random = all(
                "randomUniform" in handle[f"/Outputs/{output_name}/nodeData"] for output_name, *_ in SOBRAL_CASES
            )
            if not all_outputs_have_random:
                if not args.allow_missing_random_uniform:
                    raise ValueError(
                        f"Dust case '{case['label']}' requested random_uniform_index={case['random_uniform_index']}, "
                        "but the Galacticus file does not contain nodeData/randomUniform for all Sobral outputs."
                    )
                case["random_uniform_index"] = None
                case["label"] = case["label"] + "_fresh_scatter"
    expanded_cases: list[dict[str, Any]] = []
    for case in cases:
        has_scatter = (
            case["dust_model"] is not None
            and case["dust_params"] is not None
            and float(case["dust_params"].get("attenuation_scatter", 0.0)) > 0.0
        )
        if not has_scatter:
            case["scatter_mode"] = "deterministic"
            expanded_cases.append(case)
            continue
        if args.scatter_mode in ("draw", "both"):
            draw_case = dict(case)
            draw_case["scatter_mode"] = "draw"
            draw_case["label"] = case["label"] + "_draw"
            expanded_cases.append(draw_case)
        if args.scatter_mode in ("expected", "both"):
            expected_case = dict(case)
            expected_case["scatter_mode"] = "expected"
            expected_case["label"] = case["label"] + "_expected"
            expanded_cases.append(expected_case)
    return expanded_cases


def _intrinsic_halpha(output_group: h5py.Group) -> np.ndarray:
    nd = output_group["nodeData"]
    return sum(np.asarray(nd[name][...], dtype=float) for name in HALPHA_DATASETS)


def _attenuated_halpha(output_group: h5py.Group, redshift: float, case: dict[str, Any]) -> np.ndarray:
    intrinsic = _intrinsic_halpha(output_group)
    if case["dust_model"] is None:
        return intrinsic
    if case["dust_model"] != "gb10_generalised":
        raise ValueError(f"Unsupported dust model: {case['dust_model']}")
    nd = output_group["nodeData"]
    stellar_mass = (
        np.asarray(nd["diskMassStellar"][...], dtype=float)
        + np.asarray(nd["spheroidMassStellar"][...], dtype=float)
    )
    random_uniform = None
    if case["random_uniform_index"] is not None and "randomUniform" in nd:
        ru = np.asarray(nd["randomUniform"][...], dtype=float)
        if ru.ndim != 2:
            raise ValueError(f"randomUniform has unexpected shape {ru.shape}; expected 2D array")
        random_uniform = ru[:, int(case["random_uniform_index"])]
    A_halpha = dust_attenuation_gb10_generalised(
        stellar_mass,
        np.full_like(stellar_mass, redshift, dtype=float),
        **case["dust_params"],
        random_uniform=random_uniform,
    )
    attenuation_factor = 10.0 ** (-0.4 * A_halpha)
    return intrinsic * attenuation_factor


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def _lf_from_expected_scatter(
    luminosities: np.ndarray,
    stellar_mass: np.ndarray,
    weights: np.ndarray,
    redshift: float,
    dust_params: dict[str, Any],
    log10_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(luminosities) & np.isfinite(stellar_mass) & np.isfinite(weights) & (luminosities > 0.0) & (stellar_mass > 0.0)
    if not np.any(valid):
        zeros = np.zeros(log10_edges.size - 1, dtype=float)
        return zeros, zeros, zeros

    lum = luminosities[valid]
    mstar = stellar_mass[valid]
    w = weights[valid]
    sigma_A = float(dust_params.get("attenuation_scatter", 0.0))
    mean_params = dict(dust_params)
    mean_params["attenuation_scatter"] = 0.0
    A0 = dust_attenuation_gb10_generalised(
        mstar,
        np.full_like(mstar, redshift, dtype=float),
        **mean_params,
    )
    log10_lum = np.log10(lum)

    if sigma_A <= 0.0:
        shifted = log10_lum - 0.4 * A0
        return _lf_from_luminosities(10.0 ** shifted, w, log10_edges)

    normalization = np.diff(log10_edges) * np.log(10.0)
    counts = np.zeros(log10_edges.size - 1, dtype=float)
    variance_conservative_counts = np.zeros_like(counts)
    variance_smoothed_counts = np.zeros_like(counts)
    p_nonpos = _normal_cdf((-A0) / sigma_A)

    for bin_index, (lo, hi) in enumerate(zip(log10_edges[:-1], log10_edges[1:], strict=True)):
        # Point mass from the clipped A<=0 part: no attenuation, so luminosity remains intrinsic.
        point_mask = (log10_lum >= lo) & (log10_lum < hi)
        p_bin = np.zeros_like(w)
        p_bin[point_mask] += p_nonpos[point_mask]

        # Continuous part from A>0, mapped from attenuation magnitudes into observed-logL bins.
        lower_A = np.maximum(0.0, (log10_lum - hi) / 0.4)
        upper_A = np.maximum(0.0, (log10_lum - lo) / 0.4)
        cdf_upper = _normal_cdf((upper_A - A0) / sigma_A)
        cdf_lower = _normal_cdf((lower_A - A0) / sigma_A)
        p_bin += np.maximum(cdf_upper - cdf_lower, 0.0)
        p_bin = np.clip(p_bin, 0.0, 1.0)

        weighted_probabilities = w * p_bin
        counts[bin_index] = np.sum(weighted_probabilities)
        variance_conservative_counts[bin_index] = np.sum((w**2) * p_bin)
        variance_smoothed_counts[bin_index] = np.sum(weighted_probabilities**2)

    return (
        counts / normalization,
        variance_conservative_counts / normalization**2,
        variance_smoothed_counts / normalization**2,
    )


def _log10_std_from_linear(value: float, std: float) -> float:
    if not np.isfinite(value) or not np.isfinite(std) or value <= 0.0:
        return float("nan")
    return float(std / (value * np.log(10.0)))


def _sobral_bin_edges(log10_centers: np.ndarray) -> np.ndarray:
    edges = np.empty(log10_centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (log10_centers[:-1] + log10_centers[1:])
    edges[0] = log10_centers[0] - (edges[1] - log10_centers[0])
    edges[-1] = log10_centers[-1] + (log10_centers[-1] - edges[-2])
    return edges


def _lf_from_luminosities(luminosities: np.ndarray, weights: np.ndarray, log10_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(luminosities) & np.isfinite(weights) & (luminosities > 0.0)
    counts, _ = np.histogram(np.log10(luminosities[valid]), bins=log10_edges, weights=weights[valid])
    variance_counts, _ = np.histogram(np.log10(luminosities[valid]), bins=log10_edges, weights=weights[valid] ** 2)
    normalization = np.diff(log10_edges) * np.log(10.0)
    variance = variance_counts / normalization**2
    return counts / normalization, variance, variance


def _plot(results: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), sharey=True, constrained_layout=True)
    case_labels = []
    for axis, case_result in zip(axes.ravel(), results, strict=True):
        z = case_result["redshift"]
        log10_centers = case_result["sobral_bin_centers"]
        for i, (label, lf) in enumerate(case_result["case_lfs"].items()):
            color = plt.get_cmap("tab10")(i % 10)
            axis.plot(log10_centers, lf, marker="o", ms=4, lw=1.7, color=color, label=label)
            if label not in case_labels:
                case_labels.append(label)
        axis.plot(
            log10_centers,
            case_result["sobral_analysis_lf"],
            color="#d1495b",
            lw=1.6,
            ls="--",
            label="Galacticus Sobral analysis",
        )
        axis.plot(
            log10_centers,
            case_result["sobral_target_lf"],
            color="0.35",
            lw=1.2,
            ls=":",
            label="Sobral target",
        )
        axis.set_title(f"z={z:.2f}")
        axis.set_xlabel(r"$\log_{10}(L_{\mathrm{H}\alpha}/\mathrm{erg\,s^{-1}})$")
        axis.set_ylabel(r"$\mathrm{d}n/\mathrm{d}\ln L\ [\mathrm{Mpc}^{-3}]$")
        axis.set_yscale("log")
        axis.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(r"H$\alpha$ luminosity functions from galacticus_sed_calculator dust attenuation", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    galacticus_file = args.galacticus_file.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else galacticus_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_dust_cases(args, galacticus_file)
    if not cases:
        raise ValueError("No dust cases were specified")

    rows: list[dict[str, Any]] = []
    plot_results: list[dict[str, Any]] = []

    with h5py.File(galacticus_file, "r") as handle:
        for output_name, redshift, label, analysis_name in SOBRAL_CASES:
            output_group = handle[f"/Outputs/{output_name}"]
            weights = _combined_node_weights(output_group)
            nd = output_group["nodeData"]
            stellar_mass = (
                np.asarray(nd["diskMassStellar"][...], dtype=float)
                + np.asarray(nd["spheroidMassStellar"][...], dtype=float)
            )
            intrinsic_luminosity = _intrinsic_halpha(output_group)
            analysis_group = handle[f"/analyses/{analysis_name}"]
            log10_centers = np.log10(np.asarray(analysis_group["luminosity"][...], dtype=float))
            log10_edges = _sobral_bin_edges(log10_centers)
            sobral_analysis_lf = np.asarray(analysis_group["luminosityFunction"][...], dtype=float)
            sobral_target_lf = np.asarray(analysis_group["luminosityFunctionTarget"][...], dtype=float)

            case_lfs: dict[str, np.ndarray] = {}
            for case in cases:
                if case.get("scatter_mode") == "expected":
                    lf, variance_conservative, variance_smoothed = _lf_from_expected_scatter(
                        intrinsic_luminosity,
                        stellar_mass,
                        weights,
                        redshift,
                        case["dust_params"],
                        log10_edges,
                    )
                else:
                    lum = _attenuated_halpha(
                        output_group,
                        redshift,
                        case,
                    )
                    lf, variance_conservative, variance_smoothed = _lf_from_luminosities(lum, weights, log10_edges)
                case_lfs[case["label"]] = lf
                std_conservative = np.sqrt(np.clip(variance_conservative, 0.0, None))
                std_smoothed = np.sqrt(np.clip(variance_smoothed, 0.0, None))
                for bin_index, (center, value, sigma_conservative, sigma_smoothed) in enumerate(
                    zip(log10_centers, lf, std_conservative, std_smoothed, strict=True)
                ):
                    rows.append(
                        {
                            "output_name": output_name,
                            "sobral_label": label,
                            "redshift": redshift,
                            "analysis_name": analysis_name,
                            "dust_case": case["label"],
                            "bin_index": int(bin_index),
                            "log10_luminosity_center": float(center),
                            "dn_dlnL_mpc3": float(value),
                            "dn_dlnL_mpc3_shot_noise_std_conservative": float(sigma_conservative),
                            "dn_dlnL_mpc3_shot_noise_std_smoothed_expectation": float(sigma_smoothed),
                            "log10_dn_dlnL_shot_noise_std_conservative": _log10_std_from_linear(
                                float(value),
                                float(sigma_conservative),
                            ),
                            "log10_dn_dlnL_shot_noise_std_smoothed_expectation": _log10_std_from_linear(
                                float(value),
                                float(sigma_smoothed),
                            ),
                        }
                    )
            plot_results.append(
                {
                    "redshift": redshift,
                    "sobral_bin_centers": log10_centers,
                    "sobral_analysis_lf": sobral_analysis_lf,
                    "sobral_target_lf": sobral_target_lf,
                    "case_lfs": case_lfs,
                }
            )

    csv_path = output_dir / "halpha_dust_lf_cases.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "galacticus_file": str(galacticus_file),
        "cases": cases,
        "sobral_cases": [
            {
                "output_name": output_name,
                "redshift": redshift,
                "label": label,
                "analysis_name": analysis_name,
            }
            for output_name, redshift, label, analysis_name in SOBRAL_CASES
        ],
    }
    metadata_path = output_dir / "halpha_dust_lf_cases.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    fig_path = output_dir / "halpha_dust_lf_cases.png"
    _plot(plot_results, fig_path)

    print(csv_path)
    print(metadata_path)
    print(fig_path)


if __name__ == "__main__":
    main()
