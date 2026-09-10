"""AGN-demographic measurements from Galacticus node catalogues.

This module contains the numerical definitions needed to compare a Galacticus
snapshot with Suresh & Blanton (2026, arXiv:2607.19603).  The array-level
functions are deliberately independent of plotting and file output so that a
saved catalogue can be reclassified without rerunning Galacticus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.stats import beta


SPEED_OF_LIGHT_CGS = 2.99792458e10
SPEED_OF_LIGHT_SI = 2.99792458e8
SOLAR_MASS_G = 1.98847e33
GIGAYEAR_S = 365.25 * 86400.0 * 1.0e9
REST_MASS_POWER_PER_MSUN_GYR = SOLAR_MASS_G * SPEED_OF_LIGHT_CGS**2 / GIGAYEAR_S
JET_POWER_CGS_PER_GALACTICUS_UNIT = SOLAR_MASS_G * 1.0e10 / GIGAYEAR_S
DEFAULT_MDOT_UNITS_IN_SI = 6.302397367723697e13

# Black_Hole_Eddington_Accretion_Rate in Galacticus is L_Edd/c^2 (i.e. it
# contains no radiative-efficiency factor). This coefficient is reconstructed
# from the native GSL 2.6/Galacticus constants recorded by the example build.
GALACTICUS_MDOT_EDD_PER_MBH_MSUN_GYR = 2.2206176144141274

DEFAULT_LEDD_COEFFICIENT = 1.26e38
PAPER_II_MASS_EDGES = np.asarray([10.0, 10.4, 10.8, 11.2, 11.6, 12.0])
PAPER_II_SSFR_SPLIT = -11.5
PAPER_II_THRESHOLDS = np.asarray([1.0e-1, 1.0e-2, 1.0e-3])
ModeCriterion = Literal["inclusive", "thin_disk_dominated"]


@dataclass(frozen=True)
class SwitchedDiskParameters:
    """Parameters of Galacticus' smooth thin-disk/ADAF switch."""

    thin_disk_minimum: float | None = 0.01
    thin_disk_maximum: float | None = 21.784434725712401
    transition_width: float = 0.1

    def __post_init__(self) -> None:
        if self.thin_disk_minimum is not None and self.thin_disk_minimum <= 0.0:
            raise ValueError("thin_disk_minimum must be positive or None")
        if self.thin_disk_maximum is not None and self.thin_disk_maximum <= 0.0:
            raise ValueError("thin_disk_maximum must be positive or None")
        if self.transition_width <= 0.0:
            raise ValueError("transition_width must be positive")


@dataclass(frozen=True)
class HostProperties:
    """Combined disk+spheroid host-galaxy properties."""

    stellar_mass_msun: np.ndarray
    sfr_raw_msun_per_gyr: np.ndarray
    sfr_msun_per_year: np.ndarray
    log10_ssfr_per_year: np.ndarray
    negative_sfr_count: int
    nonpositive_sfr_count: int


@dataclass(frozen=True)
class AccretionQuantities:
    """Derived black-hole accretion quantities in cgs and Eddington units."""

    mdot_rest_g_per_s: np.ndarray
    rest_mass_power_erg_s: np.ndarray
    eddington_luminosity_erg_s: np.ndarray
    bolometric_luminosity_catalog_erg_s: np.ndarray
    bolometric_luminosity_erg_s: np.ndarray
    radiative_efficiency_used: np.ndarray
    eddington_ratio_catalog: np.ndarray
    eddington_ratio: np.ndarray
    accretion_rate_eddington_paper: np.ndarray
    accretion_rate_eddington_galacticus: np.ndarray
    valid_accreting_black_hole: np.ndarray


@dataclass(frozen=True)
class WeightedFraction:
    """A weighted fraction and an effective-count Jeffreys interval."""

    fraction: float
    lower: float
    upper: float
    effective_n: float
    effective_successes: float
    total_weight: float
    success_weight: float
    raw_n: int
    raw_successes: int
    is_upper_limit: bool


@dataclass(frozen=True)
class FractionBin:
    """One population/mass-bin measurement of an AGN fraction."""

    target_redshift: float
    redshift: float
    output_name: str
    mode_criterion: ModeCriterion
    lambda_threshold: float
    population: Literal["star_forming", "quiescent"]
    mass_low: float
    mass_high: float
    mass_center: float
    estimate: WeightedFraction


@dataclass(frozen=True)
class TrendMetrics:
    """Paper-II qualitative trend metrics in log fraction versus log mass."""

    star_forming_slope: float
    quiescent_slope: float
    separation_at_log10_mass_11: float
    star_forming_bins_used: int
    quiescent_bins_used: int


@dataclass(frozen=True)
class SnapshotCatalog:
    """Processed arrays for one selected Galacticus output."""

    target_redshift: float
    redshift: float
    output_name: str
    columns: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, int | float]


def pool_snapshot_catalogs(
    catalogs: Sequence[SnapshotCatalog], *, name: str = "pooled_snapshots"
) -> SnapshotCatalog:
    """Concatenate selected snapshots for a descriptive pooled comparison.

    This operation does not make snapshots statistically independent.  It is
    provided because Paper II pools nearby snapshots for its primary curves;
    callers should label effective-count intervals on such a pool as
    indicative when merger trees are shared.
    """

    if not catalogs:
        raise ValueError("At least one snapshot catalogue is required")
    keys = tuple(catalogs[0].columns)
    for catalog in catalogs[1:]:
        if set(catalog.columns) != set(keys):
            raise ValueError("All snapshot catalogues must contain identical columns")
    columns = {
        key: np.concatenate([np.asarray(catalog.columns[key]) for catalog in catalogs])
        for key in keys
    }
    return SnapshotCatalog(
        target_redshift=float(np.mean([catalog.target_redshift for catalog in catalogs])),
        redshift=float(np.mean([catalog.redshift for catalog in catalogs])),
        output_name=name,
        columns=columns,
        diagnostics={
            "snapshots_pooled": len(catalogs),
            "nodes_retained": int(sum(len(catalog.columns["weight"]) for catalog in catalogs)),
            "redshift_min": float(min(catalog.redshift for catalog in catalogs)),
            "redshift_max": float(max(catalog.redshift for catalog in catalogs)),
        },
    )


REQUIRED_NODE_DATASETS = (
    "nodeIndex",
    "nodeIsIsolated",
    "diskMassStellar",
    "spheroidMassStellar",
    "diskStarFormationRate",
    "spheroidStarFormationRate",
    "blackHoleMass",
    "massAccretionRateBlackHoles",
    "radiativeEfficiencyBlackHoles",
    "powerJetBlackHoles",
)


def output_redshift(output_group: h5py.Group) -> float:
    """Return the redshift recorded on a Galacticus output group."""

    return 1.0 / float(output_group.attrs["outputExpansionFactor"]) - 1.0


def nearest_output(outputs: h5py.Group, target_redshift: float) -> tuple[str, float]:
    """Return the output name and actual redshift nearest a target redshift."""

    candidates = [(name, output_redshift(group)) for name, group in outputs.items()]
    if not candidates:
        raise ValueError("The HDF5 file contains no groups under Outputs")
    return min(candidates, key=lambda pair: abs(pair[1] - target_redshift))


def select_outputs(
    outputs: h5py.Group,
    target_redshifts: Iterable[float],
    *,
    tolerance: float | None = None,
) -> list[tuple[float, str, float]]:
    """Resolve target redshifts to unique nearest outputs.

    Parameters
    ----------
    tolerance
        Maximum permitted ``abs(z_actual-z_target)``.  ``None`` accepts any
        nearest output.
    """

    selected: list[tuple[float, str, float]] = []
    names: set[str] = set()
    for target in target_redshifts:
        target = float(target)
        name, actual = nearest_output(outputs, target)
        if tolerance is not None and abs(actual - target) > tolerance:
            raise ValueError(
                f"Nearest output to z={target:g} is {name} at z={actual:.8g}, "
                f"outside tolerance {tolerance:g}"
            )
        if name in names:
            raise ValueError(f"Multiple target redshifts resolve to the same output {name}")
        selected.append((target, name, actual))
        names.add(name)
    return selected


def expand_tree_values(
    n_nodes: int,
    starts: Sequence[int],
    counts: Sequence[int],
    values: Sequence[float | int],
    *,
    dtype: np.dtype | type = float,
) -> np.ndarray:
    """Expand one value per merger tree onto its contiguous node range."""

    starts_array = np.asarray(starts, dtype=int)
    counts_array = np.asarray(counts, dtype=int)
    values_array = np.asarray(values, dtype=dtype)
    if not (starts_array.size == counts_array.size == values_array.size):
        raise ValueError("starts, counts, and values must have equal lengths")
    result = np.empty(int(n_nodes), dtype=dtype)
    assigned = np.zeros(int(n_nodes), dtype=bool)
    for start, count, value in zip(starts_array, counts_array, values_array, strict=True):
        stop = int(start + count)
        if start < 0 or count < 0 or stop > n_nodes:
            raise ValueError(f"Merger-tree node range [{start}:{stop}] is invalid for {n_nodes} nodes")
        if np.any(assigned[start:stop]):
            raise ValueError(f"Overlapping merger-tree node range [{start}:{stop}]")
        result[start:stop] = value
        assigned[start:stop] = True
    if not np.all(assigned):
        raise ValueError(f"{np.count_nonzero(~assigned)} nodes have no merger-tree value")
    return result


def node_weights(output_group: h5py.Group) -> np.ndarray:
    """Return ``mergerTreeWeight * nodeSubsamplingWeight`` for every node."""

    node_data = output_group["nodeData"]
    required = ("mergerTreeStartIndex", "mergerTreeCount", "mergerTreeWeight")
    missing = [name for name in required if name not in output_group]
    if missing:
        raise KeyError(f"{output_group.name} is missing weight datasets {missing}")
    n_nodes = len(node_data["nodeIndex"])
    weights = expand_tree_values(
        n_nodes,
        output_group["mergerTreeStartIndex"],
        output_group["mergerTreeCount"],
        output_group["mergerTreeWeight"],
    )
    if "nodeSubsamplingWeight" in node_data:
        weights *= np.asarray(node_data["nodeSubsamplingWeight"], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Node weights must be finite and non-negative")
    return weights


def node_tree_indices(output_group: h5py.Group) -> np.ndarray:
    """Return the saved merger-tree index associated with each node."""

    n_nodes = len(output_group["nodeData"]["nodeIndex"])
    values = (
        output_group["mergerTreeIndex"]
        if "mergerTreeIndex" in output_group
        else np.arange(len(output_group["mergerTreeCount"]), dtype=int)
    )
    return expand_tree_values(
        n_nodes,
        output_group["mergerTreeStartIndex"],
        output_group["mergerTreeCount"],
        values,
        dtype=np.int64,
    )


def first_vlen_element(values: Sequence[object], *, fill_value: float = np.nan) -> np.ndarray:
    """Extract element zero from every scalar, fixed-length, or VLEN row."""

    array = np.asarray(values)
    if array.dtype != object and array.ndim == 2:
        if array.shape[1] == 0:
            return np.full(array.shape[0], fill_value, dtype=float)
        return np.asarray(array[:, 0], dtype=float)
    if array.dtype != object and array.ndim == 1 and np.issubdtype(array.dtype, np.number):
        return np.asarray(array, dtype=float)

    result = np.full(len(values), fill_value, dtype=float)
    for index, value in enumerate(values):
        row = np.asarray(value).reshape(-1)
        if row.size:
            result[index] = float(row[0])
    return result


def vlen_row_sizes(values: Sequence[object]) -> np.ndarray:
    """Return the number of elements in each VLEN/fixed-length row."""

    array = np.asarray(values)
    if array.dtype != object and array.ndim == 2:
        return np.full(array.shape[0], array.shape[1], dtype=int)
    if array.dtype != object and array.ndim == 1:
        return np.ones(array.shape[0], dtype=int)
    return np.asarray([np.asarray(value).size for value in values], dtype=int)


def combine_host_properties(
    disk_stellar_mass_msun: Sequence[float],
    spheroid_stellar_mass_msun: Sequence[float],
    disk_sfr_msun_per_gyr: Sequence[float],
    spheroid_sfr_msun_per_gyr: Sequence[float],
) -> HostProperties:
    """Combine disk+spheroid mass/SFR and safely calculate log sSFR.

    Galacticus saves SFR in Msun/Gyr.  Numerically negative total rates are
    clipped to zero.  Zero or clipped rates have ``log10(sSFR/yr^-1)=-inf``.
    """

    stellar_mass = np.asarray(disk_stellar_mass_msun, dtype=float) + np.asarray(
        spheroid_stellar_mass_msun, dtype=float
    )
    sfr_raw = np.asarray(disk_sfr_msun_per_gyr, dtype=float) + np.asarray(
        spheroid_sfr_msun_per_gyr, dtype=float
    )
    if stellar_mass.shape != sfr_raw.shape:
        raise ValueError("Mass and SFR inputs must have matching shapes")
    negative = np.isfinite(sfr_raw) & (sfr_raw < 0.0)
    nonpositive = np.isfinite(sfr_raw) & (sfr_raw <= 0.0)
    sfr_per_year = np.maximum(sfr_raw, 0.0) / 1.0e9
    log_ssfr = np.full(stellar_mass.shape, np.nan, dtype=float)
    zero_rate = np.isfinite(stellar_mass) & (stellar_mass > 0.0) & (sfr_per_year == 0.0)
    log_ssfr[zero_rate] = -np.inf
    positive = (
        np.isfinite(stellar_mass)
        & (stellar_mass > 0.0)
        & np.isfinite(sfr_per_year)
        & (sfr_per_year > 0.0)
    )
    log_ssfr[positive] = np.log10(sfr_per_year[positive] / stellar_mass[positive])
    return HostProperties(
        stellar_mass_msun=stellar_mass,
        sfr_raw_msun_per_gyr=sfr_raw,
        sfr_msun_per_year=sfr_per_year,
        log10_ssfr_per_year=log_ssfr,
        negative_sfr_count=int(np.count_nonzero(negative)),
        nonpositive_sfr_count=int(np.count_nonzero(nonpositive)),
    )


def accretion_quantities(
    black_hole_mass_msun: Sequence[float],
    mdot_rest_msun_per_gyr: Sequence[float],
    radiative_efficiency_catalog: Sequence[float],
    *,
    ledd_coefficient: float = DEFAULT_LEDD_COEFFICIENT,
    fixed_radiative_efficiency: float | None = None,
    galacticus_mdot_edd_per_mbh: float = GALACTICUS_MDOT_EDD_PER_MBH_MSUN_GYR,
    mdot_units_in_si: float = DEFAULT_MDOT_UNITS_IN_SI,
) -> AccretionQuantities:
    """Calculate luminosity, Eddington ratio, and two accretion-rate ratios.

    ``eddington_ratio`` uses the configurable Paper-II luminosity convention.
    ``accretion_rate_eddington_galacticus`` is kept separate and is the value
    used by Galacticus' switched-disk state model.
    """

    mass = np.asarray(black_hole_mass_msun, dtype=float)
    mdot = np.asarray(mdot_rest_msun_per_gyr, dtype=float)
    eta_catalog = np.asarray(radiative_efficiency_catalog, dtype=float)
    if not (mass.shape == mdot.shape == eta_catalog.shape):
        raise ValueError("Black-hole mass, Mdot, and efficiency must have matching shapes")
    if ledd_coefficient <= 0.0 or galacticus_mdot_edd_per_mbh <= 0.0 or mdot_units_in_si <= 0.0:
        raise ValueError("Eddington coefficients must be positive")
    if fixed_radiative_efficiency is not None and fixed_radiative_efficiency < 0.0:
        raise ValueError("fixed_radiative_efficiency must be non-negative")

    eta_used = (
        eta_catalog.copy()
        if fixed_radiative_efficiency is None
        else np.full(mass.shape, float(fixed_radiative_efficiency))
    )
    # Use the units written on the HDF5 extractor, not a separately chosen
    # modern solar-mass/year conversion: threshold-adjacent objects can
    # otherwise move between lambda bins. unitsInSI is kg/s per stored unit.
    mdot_g_s = mdot * mdot_units_in_si * 1.0e3
    rest_power = mdot * mdot_units_in_si * SPEED_OF_LIGHT_SI**2 * 1.0e7
    ledd = ledd_coefficient * mass
    lbol_catalog = eta_catalog * rest_power
    lbol = eta_used * rest_power

    valid_bh = np.isfinite(mass) & (mass > 0.0)
    valid_accreting = valid_bh & np.isfinite(mdot) & (mdot > 0.0)
    lambda_catalog = np.full(mass.shape, np.nan, dtype=float)
    lambda_used = np.full(mass.shape, np.nan, dtype=float)
    x_paper = np.full(mass.shape, np.nan, dtype=float)
    x_native = np.full(mass.shape, np.nan, dtype=float)
    valid_ratio = valid_bh & np.isfinite(mdot) & (mdot >= 0.0)
    lambda_catalog[valid_ratio] = lbol_catalog[valid_ratio] / ledd[valid_ratio]
    lambda_used[valid_ratio] = lbol[valid_ratio] / ledd[valid_ratio]
    x_paper[valid_ratio] = rest_power[valid_ratio] / ledd[valid_ratio]
    x_native[valid_ratio] = mdot[valid_ratio] / (
        galacticus_mdot_edd_per_mbh * mass[valid_ratio]
    )
    return AccretionQuantities(
        mdot_rest_g_per_s=mdot_g_s,
        rest_mass_power_erg_s=rest_power,
        eddington_luminosity_erg_s=ledd,
        bolometric_luminosity_catalog_erg_s=lbol_catalog,
        bolometric_luminosity_erg_s=lbol,
        radiative_efficiency_used=eta_used,
        eddington_ratio_catalog=lambda_catalog,
        eddington_ratio=lambda_used,
        accretion_rate_eddington_paper=x_paper,
        accretion_rate_eddington_galacticus=x_native,
        valid_accreting_black_hole=valid_accreting,
    )


def switched_disk_adaf_fraction(
    accretion_rate_eddington_galacticus: Sequence[float],
    *,
    thin_disk_minimum: float | None = 0.01,
    thin_disk_maximum: float | None = 21.784434725712401,
    transition_width: float = 0.1,
    valid_accreting_black_hole: Sequence[bool] | None = None,
) -> np.ndarray:
    """Reproduce ``accretion_disks.switched.F90::fractionADAF``.

    Both the low-x and high-x ADAF branches are retained.  Galacticus returns
    zero for a missing/non-accreting black hole; callers must additionally use
    ``valid_accreting_black_hole`` when interpreting a thin-disk mode flag.
    """

    params = SwitchedDiskParameters(thin_disk_minimum, thin_disk_maximum, transition_width)
    x = np.asarray(accretion_rate_eddington_galacticus, dtype=float)
    valid = np.isfinite(x) & (x > 0.0)
    if valid_accreting_black_hole is not None:
        valid_gate = np.asarray(valid_accreting_black_hole, dtype=bool)
        if valid_gate.shape != x.shape:
            raise ValueError("valid_accreting_black_hole must match x")
        valid &= valid_gate
    fraction = np.zeros(x.shape, dtype=float)
    log_x = np.log(x[valid])

    def inverse_one_plus_exp(argument: np.ndarray) -> np.ndarray:
        # Algebraically exact logistic evaluation without overflow.
        output = np.empty(argument.shape, dtype=float)
        positive = argument >= 0.0
        exp_negative = np.exp(-argument[positive])
        output[positive] = exp_negative / (1.0 + exp_negative)
        exp_positive = np.exp(argument[~positive])
        output[~positive] = 1.0 / (1.0 + exp_positive)
        return output

    selected = np.zeros(log_x.shape, dtype=float)
    if params.thin_disk_minimum is not None:
        argument = (log_x - np.log(params.thin_disk_minimum)) / params.transition_width
        selected += inverse_one_plus_exp(np.minimum(argument, 60.0))
    if params.thin_disk_maximum is not None:
        argument = -(log_x - np.log(params.thin_disk_maximum)) / params.transition_width
        selected += inverse_one_plus_exp(np.minimum(argument, 60.0))
    fraction[valid] = selected
    return fraction


def activity_mask(
    eddington_ratio: Sequence[float],
    threshold: float,
    *,
    mode_criterion: ModeCriterion = "inclusive",
    fraction_adaf: Sequence[float] | None = None,
    valid_accreting_black_hole: Sequence[bool] | None = None,
) -> np.ndarray:
    """Select AGN above a lambda threshold, optionally requiring thin mode."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    ratio = np.asarray(eddington_ratio, dtype=float)
    active = np.isfinite(ratio) & (ratio > threshold)
    if mode_criterion == "inclusive":
        return active
    if mode_criterion != "thin_disk_dominated":
        raise ValueError(f"Unknown mode criterion {mode_criterion!r}")
    if fraction_adaf is None or valid_accreting_black_hole is None:
        raise ValueError("thin_disk_dominated requires fraction_adaf and a valid-accreting-BH mask")
    adaf = np.asarray(fraction_adaf, dtype=float)
    valid = np.asarray(valid_accreting_black_hole, dtype=bool)
    if adaf.shape != ratio.shape or valid.shape != ratio.shape:
        raise ValueError("Mode arrays must have the same shape as eddington_ratio")
    return active & valid & np.isfinite(adaf) & (adaf < 0.5)


def weighted_jeffreys_fraction(
    successes: Sequence[bool],
    weights: Sequence[float],
    *,
    confidence: float = 0.682689492137,
) -> WeightedFraction:
    """Return a Kish-effective-weight Jeffreys credible interval.

    A zero weighted numerator is explicitly flagged as an upper limit. Its
    ``upper`` value is the upper edge (84.13th percentile by default) of the
    central Jeffreys interval used by the plotting CLI.
    """

    selected = np.asarray(successes, dtype=bool)
    weight = np.asarray(weights, dtype=float)
    if selected.shape != weight.shape:
        raise ValueError("successes and weights must have matching shapes")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    valid = np.isfinite(weight) & (weight > 0.0)
    selected, weight = selected[valid], weight[valid]
    raw_n = int(weight.size)
    raw_successes = int(np.count_nonzero(selected))
    if raw_n == 0:
        return WeightedFraction(
            fraction=np.nan,
            lower=np.nan,
            upper=np.nan,
            effective_n=np.nan,
            effective_successes=np.nan,
            total_weight=0.0,
            success_weight=0.0,
            raw_n=0,
            raw_successes=0,
            is_upper_limit=False,
        )

    total_weight = float(np.sum(weight))
    success_weight = float(np.sum(weight[selected]))
    fraction = success_weight / total_weight
    effective_n = total_weight**2 / float(np.sum(weight**2))
    effective_successes = fraction * effective_n
    alpha = 0.5 * (1.0 - confidence)
    lower = float(beta.ppf(alpha, effective_successes + 0.5, effective_n - effective_successes + 0.5))
    upper = float(beta.ppf(1.0 - alpha, effective_successes + 0.5, effective_n - effective_successes + 0.5))
    return WeightedFraction(
        fraction=fraction,
        lower=lower,
        upper=upper,
        effective_n=effective_n,
        effective_successes=effective_successes,
        total_weight=total_weight,
        success_weight=success_weight,
        raw_n=raw_n,
        raw_successes=raw_successes,
        is_upper_limit=success_weight == 0.0,
    )


def summarize_fractions(
    catalog: SnapshotCatalog,
    *,
    mass_edges: Sequence[float] = PAPER_II_MASS_EDGES,
    ssfr_split: float = PAPER_II_SSFR_SPLIT,
    thresholds: Sequence[float] = PAPER_II_THRESHOLDS,
    mode_criteria: Sequence[ModeCriterion] = ("inclusive", "thin_disk_dominated"),
    confidence: float = 0.682689492137,
) -> list[FractionBin]:
    """Measure weighted AGN fractions for every requested definition/bin."""

    edges = np.asarray(mass_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("mass_edges must be a strictly increasing 1D array")
    columns = catalog.columns
    log_mass = np.asarray(columns["log10_stellar_mass_msun"], dtype=float)
    log_ssfr = np.asarray(columns["log10_ssfr_per_year"], dtype=float)
    weights = np.asarray(columns["weight"], dtype=float)
    results: list[FractionBin] = []
    populations = {
        "star_forming": np.isfinite(log_ssfr) & (log_ssfr > ssfr_split),
        # -inf (zero/clipped SFR) is deliberately quiescent; NaN is excluded.
        "quiescent": ~np.isnan(log_ssfr) & (log_ssfr <= ssfr_split),
    }
    for mode in mode_criteria:
        for threshold in thresholds:
            active = activity_mask(
                columns["eddington_ratio"],
                float(threshold),
                mode_criterion=mode,
                fraction_adaf=columns["fraction_adaf"],
                valid_accreting_black_hole=columns["valid_accreting_black_hole"],
            )
            for population_name, population_mask in populations.items():
                for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                    upper_test = log_mass <= high if index == edges.size - 2 else log_mass < high
                    denominator = (
                        np.isfinite(log_mass)
                        & (log_mass >= low)
                        & upper_test
                        & population_mask
                        & np.isfinite(weights)
                        & (weights > 0.0)
                    )
                    estimate = weighted_jeffreys_fraction(active[denominator], weights[denominator], confidence=confidence)
                    results.append(
                        FractionBin(
                            target_redshift=catalog.target_redshift,
                            redshift=catalog.redshift,
                            output_name=catalog.output_name,
                            mode_criterion=mode,
                            lambda_threshold=float(threshold),
                            population=population_name,  # type: ignore[arg-type]
                            mass_low=float(low),
                            mass_high=float(high),
                            mass_center=float(0.5 * (low + high)),
                            estimate=estimate,
                        )
                    )
    return results


def fit_trend_metrics(rows: Sequence[FractionBin]) -> TrendMetrics:
    """Fit the Paper-II slopes and log-fraction separation at log Mstar=11."""

    fits: dict[str, tuple[float, float, int]] = {}
    for population in ("star_forming", "quiescent"):
        population_rows = [row for row in rows if row.population == population]
        x = np.asarray([row.mass_center for row in population_rows], dtype=float)
        y = np.asarray([row.estimate.fraction for row in population_rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(valid) < 2:
            fits[population] = (np.nan, np.nan, int(np.count_nonzero(valid)))
        else:
            slope, intercept = np.polyfit(x[valid], np.log10(y[valid]), 1)
            fits[population] = (float(slope), float(intercept), int(np.count_nonzero(valid)))
    sf_slope, sf_intercept, sf_n = fits["star_forming"]
    q_slope, q_intercept, q_n = fits["quiescent"]
    separation = (
        (sf_slope * 11.0 + sf_intercept) - (q_slope * 11.0 + q_intercept)
        if np.all(np.isfinite([sf_slope, sf_intercept, q_slope, q_intercept]))
        else np.nan
    )
    return TrendMetrics(sf_slope, q_slope, float(separation), sf_n, q_n)


def _xml_optional_float(parent: ET.Element, name: str, default: float | None) -> float | None:
    child = parent.find(name)
    if child is None:
        return default
    text = child.attrib.get("value", "").strip()
    if text.lower() == "none":
        return None
    return float(text)


def switched_disk_parameters_from_xml(path: str | Path) -> SwitchedDiskParameters:
    """Read switched-disk state parameters from a Galacticus params.xml."""

    root = ET.parse(path).getroot()
    disk = root.find(".//accretionDisks[@value='switched']")
    if disk is None:
        raise ValueError(f"{path} has no <accretionDisks value='switched'> block")
    return SwitchedDiskParameters(
        thin_disk_minimum=_xml_optional_float(disk, "accretionRateThinDiskMinimum", 0.01),
        thin_disk_maximum=_xml_optional_float(disk, "accretionRateThinDiskMaximum", 0.3),
        transition_width=float(_xml_optional_float(disk, "accretionRateTransitionWidth", 0.1)),
    )


def load_snapshot_catalog(
    output_group: h5py.Group,
    *,
    target_redshift: float,
    switched_disk_parameters: SwitchedDiskParameters,
    ledd_coefficient: float = DEFAULT_LEDD_COEFFICIENT,
    fixed_radiative_efficiency: float | None = None,
    galacticus_mdot_edd_per_mbh: float = GALACTICUS_MDOT_EDD_PER_MBH_MSUN_GYR,
    centrals_only: bool = False,
) -> SnapshotCatalog:
    """Read and derive the comparison catalogue for one output group."""

    node_data = output_group["nodeData"]
    missing = [name for name in REQUIRED_NODE_DATASETS if name not in node_data]
    if missing:
        raise KeyError(f"{output_group.name}/nodeData is missing required datasets {missing}")

    host = combine_host_properties(
        node_data["diskMassStellar"],
        node_data["spheroidMassStellar"],
        node_data["diskStarFormationRate"],
        node_data["spheroidStarFormationRate"],
    )
    vlen_values: dict[str, np.ndarray] = {}
    row_sizes: dict[str, np.ndarray] = {}
    for output_name, dataset_name in (
        ("mdot_rest_msun_per_gyr", "massAccretionRateBlackHoles"),
        ("radiative_efficiency_catalog", "radiativeEfficiencyBlackHoles"),
        ("power_jet_galacticus", "powerJetBlackHoles"),
    ):
        raw = node_data[dataset_name][:]
        vlen_values[output_name] = first_vlen_element(raw)
        row_sizes[dataset_name] = vlen_row_sizes(raw)

    mdot_units_in_si = float(
        node_data["massAccretionRateBlackHoles"].attrs.get("unitsInSI", DEFAULT_MDOT_UNITS_IN_SI)
    )
    jet_power_units_in_si = float(
        node_data["powerJetBlackHoles"].attrs.get(
            "unitsInSI", JET_POWER_CGS_PER_GALACTICUS_UNIT / 1.0e7
        )
    )
    accretion = accretion_quantities(
        node_data["blackHoleMass"],
        vlen_values["mdot_rest_msun_per_gyr"],
        vlen_values["radiative_efficiency_catalog"],
        ledd_coefficient=ledd_coefficient,
        fixed_radiative_efficiency=fixed_radiative_efficiency,
        galacticus_mdot_edd_per_mbh=galacticus_mdot_edd_per_mbh,
        mdot_units_in_si=mdot_units_in_si,
    )
    fraction_adaf = switched_disk_adaf_fraction(
        accretion.accretion_rate_eddington_galacticus,
        thin_disk_minimum=switched_disk_parameters.thin_disk_minimum,
        thin_disk_maximum=switched_disk_parameters.thin_disk_maximum,
        transition_width=switched_disk_parameters.transition_width,
        valid_accreting_black_hole=accretion.valid_accreting_black_hole,
    )
    central = np.asarray(node_data["nodeIsIsolated"], dtype=int) == 1
    keep = central if centrals_only else np.ones(central.shape, dtype=bool)
    stellar_mass = host.stellar_mass_msun
    log_stellar_mass = np.full(stellar_mass.shape, np.nan, dtype=float)
    positive_mass = np.isfinite(stellar_mass) & (stellar_mass > 0.0)
    log_stellar_mass[positive_mass] = np.log10(stellar_mass[positive_mass])
    weights = node_weights(output_group)
    power_jet = vlen_values["power_jet_galacticus"]

    columns = {
        "node_index": np.asarray(node_data["nodeIndex"], dtype=np.int64)[keep],
        "merger_tree_index": node_tree_indices(output_group)[keep],
        "is_central_galaxy": central[keep],
        "weight": weights[keep],
        "stellar_mass_msun": stellar_mass[keep],
        "log10_stellar_mass_msun": log_stellar_mass[keep],
        "sfr_raw_msun_per_gyr": host.sfr_raw_msun_per_gyr[keep],
        "sfr_msun_per_year": host.sfr_msun_per_year[keep],
        "log10_ssfr_per_year": host.log10_ssfr_per_year[keep],
        "black_hole_mass_msun": np.asarray(node_data["blackHoleMass"], dtype=float)[keep],
        "mdot_rest_msun_per_gyr": vlen_values["mdot_rest_msun_per_gyr"][keep],
        "mdot_rest_g_per_s": accretion.mdot_rest_g_per_s[keep],
        "radiative_efficiency_catalog": vlen_values["radiative_efficiency_catalog"][keep],
        "radiative_efficiency_used": accretion.radiative_efficiency_used[keep],
        "rest_mass_power_erg_s": accretion.rest_mass_power_erg_s[keep],
        "eddington_luminosity_erg_s": accretion.eddington_luminosity_erg_s[keep],
        "bolometric_luminosity_catalog_erg_s": accretion.bolometric_luminosity_catalog_erg_s[keep],
        "bolometric_luminosity_erg_s": accretion.bolometric_luminosity_erg_s[keep],
        "eddington_ratio_catalog": accretion.eddington_ratio_catalog[keep],
        "eddington_ratio": accretion.eddington_ratio[keep],
        "accretion_rate_eddington_paper": accretion.accretion_rate_eddington_paper[keep],
        "accretion_rate_eddington_galacticus": accretion.accretion_rate_eddington_galacticus[keep],
        "fraction_adaf": fraction_adaf[keep],
        "valid_accreting_black_hole": accretion.valid_accreting_black_hole[keep],
        "power_jet_galacticus": power_jet[keep],
        "power_jet_erg_s": power_jet[keep] * jet_power_units_in_si * 1.0e7,
    }
    diagnostics = {
        "nodes_in_output": int(central.size),
        "nodes_retained": int(np.count_nonzero(keep)),
        "negative_total_sfr_count_all_nodes": host.negative_sfr_count,
        "nonpositive_total_sfr_count_all_nodes": host.nonpositive_sfr_count,
        "negative_total_sfr_count_retained": int(
            np.count_nonzero(np.isfinite(host.sfr_raw_msun_per_gyr[keep]) & (host.sfr_raw_msun_per_gyr[keep] < 0.0))
        ),
        "nonpositive_total_sfr_count_retained": int(
            np.count_nonzero(np.isfinite(host.sfr_raw_msun_per_gyr[keep]) & (host.sfr_raw_msun_per_gyr[keep] <= 0.0))
        ),
        "massAccretionRateBlackHoles_unitsInSI_kg_s": mdot_units_in_si,
        "powerJetBlackHoles_unitsInSI_watt": jet_power_units_in_si,
    }
    for dataset_name, sizes in row_sizes.items():
        diagnostics[f"{dataset_name}_empty_rows"] = int(np.count_nonzero(sizes == 0))
        diagnostics[f"{dataset_name}_multiple_rows"] = int(np.count_nonzero(sizes > 1))
    return SnapshotCatalog(
        target_redshift=float(target_redshift),
        redshift=output_redshift(output_group),
        output_name=output_group.name.rsplit("/", 1)[-1],
        columns=columns,
        diagnostics=diagnostics,
    )
