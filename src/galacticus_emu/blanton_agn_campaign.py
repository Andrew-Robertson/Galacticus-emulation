"""After-the-fact Blanton AGN observables for Galacticus campaigns.

The extractor in this module intentionally stores literal fractions and integer
counts.  It does not apply a logarithm, a floor, or missing-value replacement;
those are emulator-training choices that should be made after inspecting the
complete campaign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from .agn_demographics import (
    BLANTON_LEDD_COEFFICIENT,
    BLANTON_SIGMA_MINIMUM_KM_S,
    GALACTICUS_MDOT_EDD_PER_MBH_MSUN_GYR,
    PAPER_II_MASS_EDGES,
    SwitchedDiskParameters,
    accretion_quantities,
    combine_host_properties,
    first_vlen_element,
    mock_observed_agn_quantities,
    node_weights,
    output_redshift,
    select_outputs,
    switched_disk_adaf_fraction,
    switched_disk_parameters_from_xml,
)


SCHEMA_VERSION = 1
DEFAULT_REDSHIFTS = (0.0, 0.1)
DEFAULT_SSFR_CUTS = (-11.5, -11.0, -10.5)
DEFAULT_LAMBDA_THRESHOLDS = (1.0e-3, 1.0e-2, 1.0e-1)
DEFAULT_MODES = ("inclusive", "thin_disk_dominated")
DEFAULT_WEIGHT_TOLERANCE = 0.10
HALPHA_LOG10_SFR_CALIBRATION = 41.27


@dataclass(frozen=True)
class AGNCase:
    """One Eddington-ratio definition and its allowed parent selection."""

    name: str
    lambda_key: str
    parent_selection: str


# The sigma-restricted raw and H-beta/true-BH cases allow exact decomposition
# on the same parent sample as the fully mock-inferred quantity.
DEFAULT_AGN_CASES = (
    AGNCase("raw_catalog_eta_true_bh", "raw_catalog_eta_true_bh", "unrestricted"),
    AGNCase("raw_catalog_eta_true_bh", "raw_catalog_eta_true_bh", "sigma60"),
    AGNCase("raw_fixed_eta_0p1_true_bh", "raw_fixed_eta_0p1_true_bh", "unrestricted"),
    AGNCase("hbeta_true_bh", "hbeta_true_bh", "unrestricted"),
    AGNCase("hbeta_true_bh", "hbeta_true_bh", "sigma60"),
    AGNCase("raw_catalog_eta_sigma_bh", "raw_catalog_eta_sigma_bh", "sigma60"),
    AGNCase("hbeta_sigma_bh", "hbeta_sigma_bh", "sigma60"),
)


def extraction_config(
    *,
    redshifts: Sequence[float] = DEFAULT_REDSHIFTS,
    redshift_tolerance: float = 0.02,
    mass_edges: Sequence[float] = PAPER_II_MASS_EDGES,
    ssfr_cuts: Sequence[float] = DEFAULT_SSFR_CUTS,
    lambda_thresholds: Sequence[float] = DEFAULT_LAMBDA_THRESHOLDS,
    modes: Sequence[str] = DEFAULT_MODES,
    weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
) -> dict[str, Any]:
    """Return the serializable observable contract used by the extractor."""

    config = {
        "schema_version": SCHEMA_VERSION,
        "redshifts": [float(value) for value in redshifts],
        "redshift_tolerance": float(redshift_tolerance),
        "mass_edges_log10_msun": [float(value) for value in mass_edges],
        "ssfr_cuts_log10_per_year": [float(value) for value in ssfr_cuts],
        "sfr_definitions": {
            "instantaneous": "diskStarFormationRate + spheroidStarFormationRate",
            "halpha_ke12": (
                "log10(SFR/Msun yr^-1) = log10[(L_Halpha,disk + L_Halpha,spheroid)/erg s^-1] "
                f"- {HALPHA_LOG10_SFR_CALIBRATION:g}"
            ),
        },
        "lambda_thresholds": [float(value) for value in lambda_thresholds],
        "modes": list(modes),
        "agn_cases": [asdict(case) for case in DEFAULT_AGN_CASES],
        "ledd_coefficient_erg_s_per_msun": BLANTON_LEDD_COEFFICIENT,
        "sigma_minimum_km_s": BLANTON_SIGMA_MINIMUM_KM_S,
        "weight_check": {
            "quantity": "mergerTreeWeight * nodeSubsamplingWeight",
            "maximum_fractional_deviation_from_median": float(weight_tolerance),
            "action_on_failure": "error",
            "use_after_check": "none; fractions use integer counts",
        },
        "population_rule": "star_forming if log10(sSFR/yr^-1) > cut; otherwise quiescent",
        "mass_bin_rule": "left-inclusive/right-exclusive except final bin is inclusive at both ends",
        "activity_rule": "strict lambda > threshold",
        "galaxy_selection": "all galaxies (centrals and satellites)",
        "fraction_representation": "literal fraction; zero is retained; empty denominator is NaN",
    }
    validate_extraction_config(config)
    return config


def validate_extraction_config(config: Mapping[str, Any]) -> None:
    """Validate the dimensions and numerical ranges in an extraction config."""

    edges = np.asarray(config["mass_edges_log10_msun"], dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("mass edges must be finite and strictly increasing")
    for name in ("redshifts", "ssfr_cuts_log10_per_year", "lambda_thresholds"):
        values = np.asarray(config[name], dtype=float)
        if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must contain finite values")
    if np.any(np.asarray(config["lambda_thresholds"], dtype=float) < 0.0):
        raise ValueError("lambda thresholds must be non-negative")
    if float(config["redshift_tolerance"]) < 0.0:
        raise ValueError("redshift tolerance must be non-negative")
    tolerance = float(config["weight_check"]["maximum_fractional_deviation_from_median"])
    if tolerance < 0.0:
        raise ValueError("weight tolerance must be non-negative")
    invalid_modes = set(config["modes"]) - set(DEFAULT_MODES)
    if invalid_modes:
        raise ValueError(f"unsupported modes: {sorted(invalid_modes)}")


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable hash identifying an exact observable contract."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def observable_contract_hash(config: Mapping[str, Any]) -> str:
    """Hash choices that affect values, excluding validation-only settings."""

    contract = dict(config)
    contract.pop("weight_check", None)
    return config_hash(contract)


def check_uniform_weights(
    weights: Sequence[float], *, tolerance: float = DEFAULT_WEIGHT_TOLERANCE
) -> dict[str, float | int]:
    """Require all combined node weights to agree with their median within tolerance."""

    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("node weights must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("node weights must all be finite and strictly positive")
    median = float(np.median(values))
    maximum_deviation = float(np.max(np.abs(values / median - 1.0)))
    if maximum_deviation > tolerance and not np.isclose(maximum_deviation, tolerance):
        raise ValueError(
            "combined node weights are not uniform: maximum fractional deviation "
            f"from the median is {maximum_deviation:.6g}, exceeding {tolerance:.6g}; "
            f"min={np.min(values):.12g}, median={median:.12g}, max={np.max(values):.12g}"
        )
    return {
        "n_weights": int(values.size),
        "minimum": float(np.min(values)),
        "median": median,
        "maximum": float(np.max(values)),
        "maximum_fractional_deviation_from_median": maximum_deviation,
        "tolerance": float(tolerance),
    }


def _line_luminosity_erg_s(dataset: h5py.Dataset) -> np.ndarray:
    values = first_vlen_element(dataset[:])
    units_in_si = float(dataset.attrs.get("unitsInSI", 1.0e-7))
    return values * units_in_si * 1.0e7


def halpha_sfr_msun_per_year(
    luminosity_erg_s: Sequence[float],
    *,
    log10_calibration: float = HALPHA_LOG10_SFR_CALIBRATION,
) -> np.ndarray:
    """Convert intrinsic star-formation H-alpha luminosity to recent SFR."""

    luminosity = np.asarray(luminosity_erg_s, dtype=float)
    sfr = np.full(luminosity.shape, np.nan, dtype=float)
    valid = np.isfinite(luminosity) & (luminosity >= 0.0)
    sfr[valid] = luminosity[valid] * 10.0 ** (-float(log10_calibration))
    return sfr


def log10_ssfr_from_sfr(
    sfr_msun_per_year: Sequence[float], stellar_mass_msun: Sequence[float]
) -> np.ndarray:
    """Return log10(sSFR/yr^-1), mapping a valid zero SFR to negative infinity."""

    sfr = np.asarray(sfr_msun_per_year, dtype=float)
    mass = np.asarray(stellar_mass_msun, dtype=float)
    if sfr.shape != mass.shape:
        raise ValueError("SFR and stellar mass must have matching shapes")
    result = np.full(sfr.shape, np.nan, dtype=float)
    zero = np.isfinite(sfr) & (sfr == 0.0) & np.isfinite(mass) & (mass > 0.0)
    positive = np.isfinite(sfr) & (sfr > 0.0) & np.isfinite(mass) & (mass > 0.0)
    result[zero] = -np.inf
    result[positive] = np.log10(sfr[positive] / mass[positive])
    return result


def _mass_bin_mask(log_mass: np.ndarray, edges: np.ndarray, index: int) -> np.ndarray:
    low, high = edges[index], edges[index + 1]
    upper = log_mass <= high if index == edges.size - 2 else log_mass < high
    return np.isfinite(log_mass) & (log_mass >= low) & upper


def _observable_id(row: Mapping[str, Any], *, kind: str) -> str:
    common = (
        f"z={float(row['target_redshift']):g}|sfr={row['sfr_definition']}|"
        f"ssfr={float(row['ssfr_cut']):g}|parent={row['parent_selection']}|"
        f"population={row.get('population', 'all')}|"
        f"mass={float(row['mass_low']):g}:{float(row['mass_high']):g}"
    )
    if kind == "quiescent":
        return f"fquiescent|{common}"
    return (
        f"fagn|{common}|agn={row['agn_definition']}|mode={row['mode']}|"
        f"lambda={float(row['lambda_threshold']):g}"
    )


def _load_snapshot_arrays(
    output_group: h5py.Group,
    *,
    disk_parameters: SwitchedDiskParameters,
    weight_tolerance: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    node_data = output_group["nodeData"]
    required = (
        "nodeIndex",
        "diskMassStellar",
        "spheroidMassStellar",
        "diskStarFormationRate",
        "spheroidStarFormationRate",
        "blackHoleMass",
        "massAccretionRateBlackHoles",
        "radiativeEfficiencyBlackHoles",
        "luminosityEmissionLineAGN:balmerBeta4863",
        "luminosityEmissionLineDisk:balmerAlpha6565",
        "luminosityEmissionLineSpheroid:balmerAlpha6565",
        "velocityDispersion",
    )
    missing = [name for name in required if name not in node_data]
    if missing:
        raise KeyError(f"{node_data.name} is missing required datasets {missing}")

    weights = node_weights(output_group)
    weight_diagnostics = check_uniform_weights(weights, tolerance=weight_tolerance)
    host = combine_host_properties(
        node_data["diskMassStellar"],
        node_data["spheroidMassStellar"],
        node_data["diskStarFormationRate"],
        node_data["spheroidStarFormationRate"],
    )
    mass = host.stellar_mass_msun
    log_mass = np.full(mass.shape, np.nan, dtype=float)
    positive_mass = np.isfinite(mass) & (mass > 0.0)
    log_mass[positive_mass] = np.log10(mass[positive_mass])

    mdot_dataset = node_data["massAccretionRateBlackHoles"]
    mdot = first_vlen_element(mdot_dataset[:])
    eta = first_vlen_element(node_data["radiativeEfficiencyBlackHoles"][:])
    bh_mass = np.asarray(node_data["blackHoleMass"], dtype=float)
    mdot_units_in_si = float(mdot_dataset.attrs.get("unitsInSI", 6.302397367723697e13))
    raw = accretion_quantities(
        bh_mass,
        mdot,
        eta,
        ledd_coefficient=BLANTON_LEDD_COEFFICIENT,
        galacticus_mdot_edd_per_mbh=GALACTICUS_MDOT_EDD_PER_MBH_MSUN_GYR,
        mdot_units_in_si=mdot_units_in_si,
    )
    fixed = accretion_quantities(
        bh_mass,
        mdot,
        eta,
        ledd_coefficient=BLANTON_LEDD_COEFFICIENT,
        fixed_radiative_efficiency=0.1,
        galacticus_mdot_edd_per_mbh=GALACTICUS_MDOT_EDD_PER_MBH_MSUN_GYR,
        mdot_units_in_si=mdot_units_in_si,
    )
    fraction_adaf = switched_disk_adaf_fraction(
        raw.accretion_rate_eddington_galacticus,
        thin_disk_minimum=disk_parameters.thin_disk_minimum,
        thin_disk_maximum=disk_parameters.thin_disk_maximum,
        transition_width=disk_parameters.transition_width,
        valid_accreting_black_hole=raw.valid_accreting_black_hole,
    )

    hbeta = _line_luminosity_erg_s(
        node_data["luminosityEmissionLineAGN:balmerBeta4863"]
    )
    sigma_dataset = node_data["velocityDispersion"]
    sigma = first_vlen_element(sigma_dataset[:])
    sigma *= float(sigma_dataset.attrs.get("unitsInSI", 1.0e3)) / 1.0e3
    mock = mock_observed_agn_quantities(
        hbeta_agn_intrinsic_erg_s=hbeta,
        velocity_dispersion_km_s=sigma,
        black_hole_mass_true_msun=bh_mass,
        bolometric_luminosity_raw_erg_s=raw.bolometric_luminosity_catalog_erg_s,
        ledd_coefficient=BLANTON_LEDD_COEFFICIENT,
        sigma_minimum_km_s=BLANTON_SIGMA_MINIMUM_KM_S,
    )

    halpha = _line_luminosity_erg_s(
        node_data["luminosityEmissionLineDisk:balmerAlpha6565"]
    ) + _line_luminosity_erg_s(
        node_data["luminosityEmissionLineSpheroid:balmerAlpha6565"]
    )
    halpha_sfr = halpha_sfr_msun_per_year(halpha)
    arrays = {
        "log10_stellar_mass_msun": log_mass,
        "log10_ssfr_instantaneous_per_year": host.log10_ssfr_per_year,
        "log10_ssfr_halpha_ke12_per_year": log10_ssfr_from_sfr(halpha_sfr, mass),
        "fraction_adaf": fraction_adaf,
        "valid_accreting_black_hole": raw.valid_accreting_black_hole,
        "valid_sigma60": mock.valid_sigma_sample,
        "raw_catalog_eta_true_bh": raw.eddington_ratio_catalog,
        "raw_fixed_eta_0p1_true_bh": fixed.eddington_ratio,
        "hbeta_true_bh": mock.eddington_ratio_hbeta_true_bh,
        "raw_catalog_eta_sigma_bh": mock.eddington_ratio_raw_sigma_bh,
        "hbeta_sigma_bh": mock.eddington_ratio_hbeta_sigma_bh,
    }
    diagnostics = {
        "output_name": output_group.name.rsplit("/", 1)[-1],
        "redshift": output_redshift(output_group),
        "n_nodes": int(log_mass.size),
        "weight_check": weight_diagnostics,
        "negative_instantaneous_sfr_count": host.negative_sfr_count,
        "nonpositive_instantaneous_sfr_count": host.nonpositive_sfr_count,
        "nonpositive_halpha_sfr_count": int(np.count_nonzero(np.isfinite(halpha_sfr) & (halpha_sfr <= 0.0))),
        "disk_parameters": asdict(disk_parameters),
    }
    return arrays, diagnostics


def summarize_snapshot(
    arrays: Mapping[str, np.ndarray],
    *,
    evaluation_id: str,
    output_name: str,
    target_redshift: float,
    redshift: float,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate all configured F_AGN and quiescent-fraction rows."""

    edges = np.asarray(config["mass_edges_log10_msun"], dtype=float)
    log_mass = np.asarray(arrays["log10_stellar_mass_msun"], dtype=float)
    parent_masks = {
        "unrestricted": np.ones(log_mass.shape, dtype=bool),
        "sigma60": np.asarray(arrays["valid_sigma60"], dtype=bool),
    }
    sfr_arrays = {
        "instantaneous": np.asarray(arrays["log10_ssfr_instantaneous_per_year"], dtype=float),
        "halpha_ke12": np.asarray(arrays["log10_ssfr_halpha_ke12_per_year"], dtype=float),
    }
    thin_disk = (
        np.asarray(arrays["valid_accreting_black_hole"], dtype=bool)
        & np.isfinite(arrays["fraction_adaf"])
        & (np.asarray(arrays["fraction_adaf"], dtype=float) < 0.5)
    )

    fraction_rows: list[dict[str, Any]] = []
    quiescent_rows: list[dict[str, Any]] = []
    for sfr_name, log_ssfr in sfr_arrays.items():
        valid_ssfr = ~np.isnan(log_ssfr)
        for ssfr_cut in config["ssfr_cuts_log10_per_year"]:
            populations = {
                "star_forming": valid_ssfr & (log_ssfr > float(ssfr_cut)),
                "quiescent": valid_ssfr & (log_ssfr <= float(ssfr_cut)),
            }
            for parent_name, parent_mask in parent_masks.items():
                for bin_index in range(edges.size - 1):
                    mass_mask = _mass_bin_mask(log_mass, edges, bin_index)
                    denominator = mass_mask & valid_ssfr & parent_mask
                    n_total = int(np.count_nonzero(denominator))
                    n_quiescent = int(np.count_nonzero(denominator & populations["quiescent"]))
                    row = {
                        "evaluation_id": evaluation_id,
                        "output_name": output_name,
                        "target_redshift": float(target_redshift),
                        "redshift": float(redshift),
                        "sfr_definition": sfr_name,
                        "ssfr_cut": float(ssfr_cut),
                        "parent_selection": parent_name,
                        "mass_low": float(edges[bin_index]),
                        "mass_high": float(edges[bin_index + 1]),
                        "n_total": n_total,
                        "n_quiescent": n_quiescent,
                        "f_quiescent": n_quiescent / n_total if n_total else np.nan,
                        "is_missing": n_total == 0,
                    }
                    row["observable_id"] = _observable_id(row, kind="quiescent")
                    quiescent_rows.append(row)

            for case in DEFAULT_AGN_CASES:
                ratio = np.asarray(arrays[case.lambda_key], dtype=float)
                parent_mask = parent_masks[case.parent_selection]
                for mode in config["modes"]:
                    mode_mask = np.ones(log_mass.shape, dtype=bool) if mode == "inclusive" else thin_disk
                    for threshold in config["lambda_thresholds"]:
                        active = np.isfinite(ratio) & (ratio > float(threshold)) & mode_mask
                        for population_name, population_mask in populations.items():
                            for bin_index in range(edges.size - 1):
                                denominator = (
                                    _mass_bin_mask(log_mass, edges, bin_index)
                                    & population_mask
                                    & parent_mask
                                )
                                n_galaxies = int(np.count_nonzero(denominator))
                                n_agn = int(np.count_nonzero(denominator & active))
                                row = {
                                    "evaluation_id": evaluation_id,
                                    "output_name": output_name,
                                    "target_redshift": float(target_redshift),
                                    "redshift": float(redshift),
                                    "sfr_definition": sfr_name,
                                    "ssfr_cut": float(ssfr_cut),
                                    "agn_definition": case.name,
                                    "parent_selection": case.parent_selection,
                                    "mode": mode,
                                    "lambda_threshold": float(threshold),
                                    "population": population_name,
                                    "mass_low": float(edges[bin_index]),
                                    "mass_high": float(edges[bin_index + 1]),
                                    "n_galaxies": n_galaxies,
                                    "n_agn": n_agn,
                                    "f_agn": n_agn / n_galaxies if n_galaxies else np.nan,
                                    "is_zero": n_galaxies > 0 and n_agn == 0,
                                    "is_missing": n_galaxies == 0,
                                }
                                row["observable_id"] = _observable_id(row, kind="fagn")
                                fraction_rows.append(row)
    return fraction_rows, quiescent_rows


def extract_evaluation(
    galacticus_file: str | Path,
    *,
    evaluation_id: str,
    params_xml: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Extract every configured observable from one Galacticus evaluation."""

    path = Path(galacticus_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    parameter_path = (
        Path(params_xml).expanduser().resolve() if params_xml is not None else path.with_name("params.xml")
    )
    if not parameter_path.is_file():
        raise FileNotFoundError(f"params.xml is required for accretion-state diagnostics: {parameter_path}")
    selected_config = dict(config) if config is not None else extraction_config()
    validate_extraction_config(selected_config)
    disk_parameters = switched_disk_parameters_from_xml(parameter_path)

    fraction_rows: list[dict[str, Any]] = []
    quiescent_rows: list[dict[str, Any]] = []
    snapshot_metadata: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        if "Outputs" not in handle:
            raise KeyError(f"{path} has no Outputs group; the unreduced Galacticus file is required")
        selected_outputs = select_outputs(
            handle["Outputs"],
            selected_config["redshifts"],
            tolerance=float(selected_config["redshift_tolerance"]),
        )
        for target, output_name, actual in selected_outputs:
            arrays, diagnostics = _load_snapshot_arrays(
                handle["Outputs"][output_name],
                disk_parameters=disk_parameters,
                weight_tolerance=float(
                    selected_config["weight_check"]["maximum_fractional_deviation_from_median"]
                ),
            )
            rows, q_rows = summarize_snapshot(
                arrays,
                evaluation_id=evaluation_id,
                output_name=output_name,
                target_redshift=target,
                redshift=actual,
                config=selected_config,
            )
            fraction_rows.extend(rows)
            quiescent_rows.extend(q_rows)
            diagnostics["target_redshift"] = float(target)
            snapshot_metadata.append(diagnostics)

    galacticus_stat = path.stat()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "extractor": "galacticus_emu.blanton_agn_campaign.extract_evaluation",
        "evaluation_id": evaluation_id,
        "galacticus_file": str(path),
        "params_xml": str(parameter_path),
        "source_files": {
            "galacticus": {
                "path": str(path),
                "size_bytes": galacticus_stat.st_size,
                "mtime_ns": galacticus_stat.st_mtime_ns,
            },
            "parameters": {
                "path": str(parameter_path),
                "sha256": sha256(parameter_path.read_bytes()).hexdigest(),
            },
        },
        "config": selected_config,
        "config_hash": config_hash(selected_config),
        "observable_contract_hash": observable_contract_hash(selected_config),
        "snapshots": snapshot_metadata,
        "n_fagn_rows": len(fraction_rows),
        "n_quiescent_rows": len(quiescent_rows),
    }
    return fraction_rows, quiescent_rows, metadata


def default_output_root(campaign_root: str | Path) -> Path:
    """Return the versioned sidecar directory for a campaign."""

    return (
        Path(campaign_root).expanduser().resolve()
        / "additional_observables"
        / "blanton-agn"
        / f"v{SCHEMA_VERSION}"
    )


def evaluation_ids_from_samples(campaign_root: str | Path) -> list[str]:
    """Read evaluation IDs in campaign order from ``samples.csv``."""

    import pandas as pd

    path = Path(campaign_root).expanduser().resolve() / "samples.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["evaluation_id"])
    ids = frame["evaluation_id"].astype(str).tolist()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{path} must contain unique, non-empty evaluation_id values")
    return ids


def evaluation_paths(campaign_root: str | Path, evaluation_id: str) -> tuple[Path, Path]:
    """Resolve the full Galacticus output and parameter XML for one evaluation."""

    evaluation_dir = Path(campaign_root).expanduser().resolve() / "evaluations" / evaluation_id
    return evaluation_dir / "galacticus.hdf5", evaluation_dir / "params.xml"


def _string_array(values: Sequence[str]) -> np.ndarray:
    encoded = [str(value).encode("utf-8") for value in values]
    width = max((len(value) for value in encoded), default=1)
    return np.asarray(encoded, dtype=f"S{width}")


def _create_string_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    group.create_dataset(
        name,
        data=_string_array(values),
        compression="gzip",
        shuffle=True,
    )


def _write_row_group(group: h5py.Group, rows: Sequence[Mapping[str, Any]], kind: str) -> None:
    if kind == "fagn":
        value_name, count_names = "f_agn", ("n_galaxies", "n_agn")
    elif kind == "quiescent":
        value_name, count_names = "f_quiescent", ("n_total", "n_quiescent")
    else:
        raise ValueError(kind)
    _create_string_dataset(group, "observable_id", [str(row["observable_id"]) for row in rows])
    string_fields = ["sfr_definition", "parent_selection"]
    numeric_fields = ["target_redshift", "ssfr_cut", "mass_low", "mass_high"]
    if kind == "fagn":
        string_fields.extend(["agn_definition", "mode", "population"])
        numeric_fields.append("lambda_threshold")
    for name in string_fields:
        _create_string_dataset(group, name, [str(row[name]) for row in rows])
    for name in numeric_fields:
        group.create_dataset(
            name,
            data=np.asarray([row[name] for row in rows], dtype=np.float64),
            compression="gzip",
            shuffle=True,
        )
    group.create_dataset(
        value_name,
        data=np.asarray([row[value_name] for row in rows], dtype=np.float64),
        compression="gzip",
        shuffle=True,
    )
    for name in count_names:
        group.create_dataset(
            name,
            data=np.asarray([row[name] for row in rows], dtype=np.int64),
            compression="gzip",
            shuffle=True,
        )


def write_evaluation_shard(
    output_path: str | Path,
    fraction_rows: Sequence[Mapping[str, Any]],
    quiescent_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one evaluation's rows to a compact HDF5 shard."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    if not fraction_rows or not quiescent_rows:
        raise ValueError("both F_AGN and quiescent rows must be non-empty")
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)
    try:
        with h5py.File(temp_path, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["evaluation_id"] = str(metadata["evaluation_id"])
            handle.attrs["config_hash"] = str(metadata["config_hash"])
            handle.attrs["observable_contract_hash"] = str(
                metadata.get(
                    "observable_contract_hash",
                    observable_contract_hash(metadata["config"]),
                )
            )
            handle.create_dataset(
                "metadata_json",
                data=json.dumps(metadata, sort_keys=True, allow_nan=False),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            _write_row_group(handle.create_group("fagn"), fraction_rows, "fagn")
            _write_row_group(handle.create_group("quiescent"), quiescent_rows, "quiescent")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return destination


def read_evaluation_shard(path: str | Path) -> dict[str, Any]:
    """Read the normalized arrays needed to aggregate one shard."""

    source = Path(path).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        metadata_raw = handle["metadata_json"][()]
        metadata = json.loads(
            metadata_raw.decode() if isinstance(metadata_raw, bytes) else str(metadata_raw)
        )
        result: dict[str, Any] = {"metadata": metadata}
        for kind, value_name, counts in (
            ("fagn", "f_agn", ("n_galaxies", "n_agn")),
            ("quiescent", "f_quiescent", ("n_total", "n_quiescent")),
        ):
            group = handle[kind]
            result[kind] = {
                "observable_id": group["observable_id"].asstr()[:],
                value_name: np.asarray(group[value_name], dtype=float),
                **{name: np.asarray(group[name], dtype=np.int64) for name in counts},
            }
    return result


def extract_campaign_evaluation(
    campaign_root: str | Path,
    evaluation_id: str,
    *,
    output_root: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Extract and write a single campaign evaluation."""

    campaign = Path(campaign_root).expanduser().resolve()
    selected_config = dict(config) if config is not None else extraction_config()
    destination_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else default_output_root(campaign)
    )
    destination = destination_root / "shards" / f"{evaluation_id}.hdf5"
    if destination.is_file() and not overwrite:
        existing = read_evaluation_shard(destination)
        existing_contract = observable_contract_hash(existing["metadata"]["config"])
        if existing_contract != observable_contract_hash(selected_config):
            raise ValueError(
                f"existing shard {destination} used different observable definitions; "
                "pass --overwrite to replace it"
            )
        return destination
    galacticus_path, parameter_path = evaluation_paths(campaign, evaluation_id)
    fraction_rows, quiescent_rows, metadata = extract_evaluation(
        galacticus_path,
        evaluation_id=evaluation_id,
        params_xml=parameter_path,
        config=selected_config,
    )
    metadata["campaign_root"] = str(campaign)
    return write_evaluation_shard(
        destination,
        fraction_rows,
        quiescent_rows,
        metadata,
        overwrite=overwrite,
    )


def _atomic_to_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if path.name.endswith(".csv.gz") else path.suffix
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(temp_fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False, compression="gzip" if path.name.endswith(".gz") else None)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _copy_descriptors(source: h5py.Group, destination: h5py.Group) -> None:
    value_names = {"f_agn", "n_galaxies", "n_agn", "f_quiescent", "n_total", "n_quiescent"}
    for name, dataset in source.items():
        if name in value_names:
            continue
        if h5py.check_string_dtype(dataset.dtype) is not None:
            _create_string_dataset(destination, name, dataset.asstr()[:])
        else:
            destination.create_dataset(name, data=dataset[:], compression="gzip", shuffle=True)


def aggregate_campaign_shards(
    campaign_root: str | Path,
    *,
    output_root: str | Path | None = None,
    require_complete: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Aggregate per-evaluation shards into HDF5 and emulator-friendly CSV tables."""

    import pandas as pd

    campaign = Path(campaign_root).expanduser().resolve()
    root = Path(output_root).expanduser().resolve() if output_root else default_output_root(campaign)
    samples_path = campaign / "samples.csv"
    samples = pd.read_csv(samples_path)
    if "evaluation_id" not in samples:
        raise ValueError(f"{samples_path} has no evaluation_id column")
    expected_ids = samples["evaluation_id"].astype(str).tolist()
    shard_paths = {path.stem: path for path in sorted((root / "shards").glob("*.hdf5"))}
    missing = [evaluation_id for evaluation_id in expected_ids if evaluation_id not in shard_paths]
    if missing and require_complete:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"missing {len(missing)} evaluation shards, beginning with: {preview}")
    evaluation_ids = [evaluation_id for evaluation_id in expected_ids if evaluation_id in shard_paths]
    if not evaluation_ids:
        raise FileNotFoundError(f"no evaluation shards found under {root / 'shards'}")

    first_path = shard_paths[evaluation_ids[0]]
    first = read_evaluation_shard(first_path)
    expected_hash = first["metadata"]["config_hash"]
    expected_contract_hash = observable_contract_hash(first["metadata"]["config"])
    ids_by_kind = {
        kind: np.asarray(first[kind]["observable_id"], dtype=str)
        for kind in ("fagn", "quiescent")
    }
    arrays: dict[str, dict[str, np.ndarray]] = {
        "fagn": {
            "f_agn": np.full((len(evaluation_ids), len(ids_by_kind["fagn"])), np.nan),
            "n_galaxies": np.zeros((len(evaluation_ids), len(ids_by_kind["fagn"])), dtype=np.int64),
            "n_agn": np.zeros((len(evaluation_ids), len(ids_by_kind["fagn"])), dtype=np.int64),
        },
        "quiescent": {
            "f_quiescent": np.full((len(evaluation_ids), len(ids_by_kind["quiescent"])), np.nan),
            "n_total": np.zeros((len(evaluation_ids), len(ids_by_kind["quiescent"])), dtype=np.int64),
            "n_quiescent": np.zeros((len(evaluation_ids), len(ids_by_kind["quiescent"])), dtype=np.int64),
        },
    }
    shard_config_hashes: list[str] = []
    shard_weight_tolerances: list[float] = []
    for row_index, evaluation_id in enumerate(evaluation_ids):
        shard = read_evaluation_shard(shard_paths[evaluation_id])
        shard_contract_hash = observable_contract_hash(shard["metadata"]["config"])
        if shard_contract_hash != expected_contract_hash:
            raise ValueError(f"observable-definition mismatch in {shard_paths[evaluation_id]}")
        shard_config_hashes.append(str(shard["metadata"]["config_hash"]))
        shard_weight_tolerances.append(
            float(
                shard["metadata"]["config"]["weight_check"][
                    "maximum_fractional_deviation_from_median"
                ]
            )
        )
        for kind in ("fagn", "quiescent"):
            if not np.array_equal(np.asarray(shard[kind]["observable_id"], dtype=str), ids_by_kind[kind]):
                raise ValueError(f"observable ordering mismatch in {shard_paths[evaluation_id]}")
            for name in arrays[kind]:
                arrays[kind][name][row_index] = shard[kind][name]

    hdf5_path = root / "blanton_agn_observables.hdf5"
    fagn_csv = root / "blanton_agn_fractions_with_samples.csv.gz"
    quiescent_csv = root / "blanton_quiescent_fractions_with_samples.csv.gz"
    definitions_path = root / "definitions.json"
    for path in (hdf5_path, fagn_csv, quiescent_csv, definitions_path):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    root.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{hdf5_path.name}.", suffix=".tmp", dir=root)
    os.close(temp_fd)
    temp_hdf5 = Path(temp_name)
    try:
        with h5py.File(first_path, "r") as first_handle, h5py.File(temp_hdf5, "w") as output:
            output.attrs["schema_version"] = SCHEMA_VERSION
            output.attrs["config_hash"] = expected_hash
            output.attrs["observable_contract_hash"] = expected_contract_hash
            output.attrs["campaign_root"] = str(campaign)
            _create_string_dataset(output, "evaluation_id", evaluation_ids)
            _create_string_dataset(output, "shard_config_hash", shard_config_hashes)
            output.create_dataset("weight_check_tolerance", data=shard_weight_tolerances)
            output.create_dataset(
                "config_json",
                data=json.dumps(first["metadata"]["config"], sort_keys=True, allow_nan=False),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            for kind in ("fagn", "quiescent"):
                group = output.create_group(kind)
                _copy_descriptors(first_handle[kind], group)
                for name, values in arrays[kind].items():
                    group.create_dataset(name, data=values, compression="gzip", shuffle=True)
        os.replace(temp_hdf5, hdf5_path)
    finally:
        if temp_hdf5.exists():
            temp_hdf5.unlink()

    selected_samples = samples.set_index("evaluation_id").loc[evaluation_ids].reset_index()
    fagn_wide = pd.DataFrame(arrays["fagn"]["f_agn"], columns=ids_by_kind["fagn"])
    quiescent_wide = pd.DataFrame(
        arrays["quiescent"]["f_quiescent"], columns=ids_by_kind["quiescent"]
    )
    _atomic_to_csv(pd.concat([selected_samples, fagn_wide], axis=1), fagn_csv)
    _atomic_to_csv(pd.concat([selected_samples, quiescent_wide], axis=1), quiescent_csv)
    definitions = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": expected_hash,
        "observable_contract_hash": expected_contract_hash,
        "config": first["metadata"]["config"],
        "shard_config_hashes": {
            value: shard_config_hashes.count(value) for value in sorted(set(shard_config_hashes))
        },
        "weight_check_tolerances": sorted(set(shard_weight_tolerances)),
        "source_campaign": {
            "path": str(campaign),
            "samples_csv": str(samples_path),
            "samples_csv_sha256": sha256(samples_path.read_bytes()).hexdigest(),
        },
        "n_expected_evaluations": len(expected_ids),
        "n_aggregated_evaluations": len(evaluation_ids),
        "missing_evaluation_ids": missing,
        "representation": {
            "zero": "literal zero when n_galaxies > 0 and n_agn == 0",
            "missing": "NaN only when the relevant denominator is zero",
            "weights": "integer counts after the per-snapshot uniform-weight check",
            "emulator_transform": "not applied; choose log10(F_AGN + F_floor) downstream",
        },
    }
    temp_json = definitions_path.with_name(f".{definitions_path.name}.tmp")
    temp_json.write_text(json.dumps(definitions, indent=2, sort_keys=True) + "\n")
    os.replace(temp_json, definitions_path)
    return {
        "hdf5": hdf5_path,
        "fagn_csv": fagn_csv,
        "quiescent_csv": quiescent_csv,
        "definitions": definitions_path,
    }
