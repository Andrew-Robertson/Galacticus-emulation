#!/usr/bin/env python3
"""Compare raw and Blanton-like mock-observed AGN quantities for the MAP run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from galacticus_emu.agn_demographics import (
    BLANTON_LEDD_COEFFICIENT,
    BLANTON_SIGMA_MINIMUM_KM_S,
    PAPER_II_MASS_EDGES,
    PAPER_II_SSFR_SPLIT,
    SnapshotCatalog,
    SwitchedDiskParameters,
    load_snapshot_catalog,
    mock_observed_agn_quantities,
    pool_snapshot_catalogs,
    select_outputs,
    switched_disk_parameters_from_xml,
    weighted_jeffreys_fraction,
)


DEFAULT_HDF5_RELATIVE = Path(
    "runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/"
    "automatedPipeline_transformedParams_finalPaper_definitive/"
    "MCMCs_production_from_exploratory_MAP/standard_observables_plus_emission_line_lfs/"
    "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr/"
    "bestFitModel_GalacticusRun/romanEPS_massFunction.hdf5"
)
DEFAULT_HDF5 = REPO_ROOT / DEFAULT_HDF5_RELATIVE
# A Git worktree normally does not duplicate large untracked campaign files.
# Fall back to the sibling saved-project checkout for the local MAP diagnostic.
SHARED_CHECKOUT_HDF5 = REPO_ROOT.parent.parent / "Galacticus-emulation" / DEFAULT_HDF5_RELATIVE
if not DEFAULT_HDF5.is_file() and SHARED_CHECKOUT_HDF5.is_file():
    DEFAULT_HDF5 = SHARED_CHECKOUT_HDF5
DEFAULT_OBSERVATIONS = (
    REPO_ROOT
    / "data/observations/suresh_blanton_2026/table1_radiative_agn_fraction_bands.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper/agn_demographics_mock_observed"

DEFINITION_ORDER = (
    "raw",
    "hbeta_true_bh",
    "raw_sigma_bh",
    "mock_observed",
)
DEFINITION_LABELS = {
    "raw": r"Raw: $L_{\rm bol}^{\rm Gal}/L_{\rm Edd}(M_{\rm BH}^{\rm Gal})$",
    "hbeta_true_bh": r"H$\beta$ luminosity inference only",
    "raw_sigma_bh": r"$M_{\rm BH}$--$\sigma$ inference only",
    "mock_observed": r"Mock observed: H$\beta$ + $M_{\rm BH}$--$\sigma$",
}
DEFINITION_COLORS = {
    "raw": "#111111",
    "hbeta_true_bh": "#d95f02",
    "raw_sigma_bh": "#7570b3",
    "mock_observed": "#1b9e77",
}
DEFINITION_MARKERS = {"raw": "o", "hbeta_true_bh": "^", "raw_sigma_bh": "s", "mock_observed": "D"}
POPULATION_COLORS = {"star_forming": "#2166ac", "quiescent": "#b2182b"}
POPULATION_LABELS = {"star_forming": "Star-forming", "quiescent": "Quiescent"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--params", type=Path, help="Defaults to params.xml beside --hdf5")
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--redshifts", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--redshift-tolerance", type=float, default=0.02)
    parser.add_argument("--mass-edges", type=float, nargs="+", default=PAPER_II_MASS_EDGES.tolist())
    parser.add_argument("--ssfr-split", type=float, default=PAPER_II_SSFR_SPLIT)
    parser.add_argument("--lambda-threshold", type=float, default=1.0e-3)
    parser.add_argument("--sigma-minimum", type=float, default=BLANTON_SIGMA_MINIMUM_KM_S)
    parser.add_argument("--confidence", type=float, default=0.682689492137)
    parser.add_argument("--centrals-only", action="store_true")
    parser.add_argument("--max-scatter-points", type=int, default=10000)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.hdf5.is_file():
        raise FileNotFoundError(args.hdf5)
    if not args.observations.is_file():
        raise FileNotFoundError(args.observations)
    edges = np.asarray(args.mass_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("--mass-edges must be strictly increasing")
    if args.lambda_threshold <= 0.0 or args.sigma_minimum <= 0.0:
        raise ValueError("lambda and sigma thresholds must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie in (0, 1)")
    if args.max_scatter_points <= 0:
        raise ValueError("--max-scatter-points must be positive")


def subset_catalog(catalog: SnapshotCatalog, mask: np.ndarray, *, name: str | None = None) -> SnapshotCatalog:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != np.asarray(catalog.columns["weight"]).shape:
        raise ValueError("catalog mask has the wrong shape")
    diagnostics = dict(catalog.diagnostics)
    diagnostics["nodes_before_common_sigma_selection"] = int(mask.size)
    diagnostics["nodes_after_common_sigma_selection"] = int(np.count_nonzero(mask))
    return SnapshotCatalog(
        catalog.target_redshift,
        catalog.redshift,
        name or catalog.output_name,
        {key: np.asarray(value)[mask] for key, value in catalog.columns.items()},
        diagnostics,
    )


def add_mock_columns(catalog: SnapshotCatalog, *, sigma_minimum: float) -> tuple[SnapshotCatalog, np.ndarray]:
    required = {"hbeta_agn_intrinsic_erg_s", "velocity_dispersion_km_s"}
    missing = required - set(catalog.columns)
    if missing:
        raise KeyError(f"{catalog.output_name} lacks mock-observation columns {sorted(missing)}")
    columns = {key: np.asarray(value) for key, value in catalog.columns.items()}
    inferred = mock_observed_agn_quantities(
        hbeta_agn_intrinsic_erg_s=columns["hbeta_agn_intrinsic_erg_s"],
        velocity_dispersion_km_s=columns["velocity_dispersion_km_s"],
        black_hole_mass_true_msun=columns["black_hole_mass_msun"],
        bolometric_luminosity_raw_erg_s=columns["bolometric_luminosity_catalog_erg_s"],
        ledd_coefficient=BLANTON_LEDD_COEFFICIENT,
        sigma_minimum_km_s=sigma_minimum,
    )
    columns.update(
        {
            "bolometric_luminosity_hbeta_erg_s": inferred.bolometric_luminosity_hbeta_erg_s,
            "black_hole_mass_sigma_msun": inferred.black_hole_mass_sigma_msun,
            "lambda_raw": inferred.eddington_ratio_raw_true_bh,
            "lambda_hbeta_true_bh": inferred.eddington_ratio_hbeta_true_bh,
            "lambda_raw_sigma_bh": inferred.eddington_ratio_raw_sigma_bh,
            "lambda_mock_observed": inferred.eddington_ratio_hbeta_sigma_bh,
            "valid_sigma_sample": inferred.valid_sigma_sample,
        }
    )
    enriched = SnapshotCatalog(
        catalog.target_redshift,
        catalog.redshift,
        catalog.output_name,
        columns,
        dict(catalog.diagnostics),
    )
    return enriched, inferred.valid_sigma_sample


def definition_ratios(catalog: SnapshotCatalog) -> Mapping[str, np.ndarray]:
    return {
        "raw": np.asarray(catalog.columns["lambda_raw"], dtype=float),
        "hbeta_true_bh": np.asarray(catalog.columns["lambda_hbeta_true_bh"], dtype=float),
        "raw_sigma_bh": np.asarray(catalog.columns["lambda_raw_sigma_bh"], dtype=float),
        "mock_observed": np.asarray(catalog.columns["lambda_mock_observed"], dtype=float),
    }


def load_observations(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected = {
        "mass_low_log10_msun",
        "mass_high_log10_msun",
        "population",
        "log10_fraction_lower",
        "log10_fraction_upper",
    }
    if not expected.issubset(data):
        raise ValueError(f"Observation table lacks {sorted(expected - set(data))}")
    return data


def fraction_frame(
    catalog: SnapshotCatalog,
    *,
    mass_edges: Sequence[float],
    ssfr_split: float,
    threshold: float,
    confidence: float,
) -> pd.DataFrame:
    columns = catalog.columns
    mass = np.asarray(columns["log10_stellar_mass_msun"], dtype=float)
    ssfr = np.asarray(columns["log10_ssfr_per_year"], dtype=float)
    weight = np.asarray(columns["weight"], dtype=float)
    populations = {
        "star_forming": np.isfinite(ssfr) & (ssfr > ssfr_split),
        "quiescent": ~np.isnan(ssfr) & (ssfr <= ssfr_split),
    }
    records: list[dict[str, float | int | bool | str]] = []
    edges = np.asarray(mass_edges, dtype=float)
    for definition, ratio in definition_ratios(catalog).items():
        active = np.isfinite(ratio) & (ratio > threshold)
        for population, population_mask in populations.items():
            for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
                upper = mass <= high if index == edges.size - 2 else mass < high
                selected = (
                    np.isfinite(mass)
                    & (mass >= low)
                    & upper
                    & population_mask
                    & np.isfinite(weight)
                    & (weight > 0.0)
                )
                estimate = weighted_jeffreys_fraction(
                    active[selected], weight[selected], confidence=confidence
                )
                records.append(
                    {
                        "definition": definition,
                        "population": population,
                        "mass_low_log10_msun": low,
                        "mass_high_log10_msun": high,
                        "mass_center_log10_msun": 0.5 * (low + high),
                        **estimate.__dict__,
                    }
                )
    return pd.DataFrame.from_records(records)


def add_observation_band(axis: plt.Axes, observations: pd.DataFrame, population: str) -> None:
    selected = observations[observations["population"] == population]
    for row in selected.itertuples(index=False):
        lower = 10.0**row.log10_fraction_lower
        upper = 10.0**row.log10_fraction_upper
        axis.add_patch(
            Rectangle(
                (row.mass_low_log10_msun, lower),
                row.mass_high_log10_msun - row.mass_low_log10_msun,
                upper - lower,
                facecolor=POPULATION_COLORS[population],
                edgecolor="none",
                alpha=0.17,
                zorder=0,
            )
        )


def plot_fraction_series(
    axis: plt.Axes,
    table: pd.DataFrame,
    definition: str,
    population: str,
    *,
    offset: float,
    label: str | None = None,
) -> None:
    rows = table[(table["definition"] == definition) & (table["population"] == population)]
    detections = rows[np.isfinite(rows["fraction"]) & ~rows["is_upper_limit"]]
    color = DEFINITION_COLORS[definition]
    marker = DEFINITION_MARKERS[definition]
    if len(detections):
        x = detections["mass_center_log10_msun"].to_numpy() + offset
        y = detections["fraction"].to_numpy()
        axis.errorbar(
            x,
            y,
            yerr=np.vstack(
                (
                    np.maximum(y - detections["lower"].to_numpy(), 0.0),
                    np.maximum(detections["upper"].to_numpy() - y, 0.0),
                )
            ),
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.45,
            markersize=5.2,
            capsize=2,
            label=label,
            zorder=3,
        )
    elif label:
        axis.plot([], [], color=color, marker=marker, label=label)
    limits = rows[np.isfinite(rows["fraction"]) & rows["is_upper_limit"]]
    if len(limits):
        axis.errorbar(
            limits["mass_center_log10_msun"].to_numpy() + offset,
            limits["upper"].to_numpy(),
            yerr=0.35 * limits["upper"].to_numpy(),
            uplims=True,
            fmt=marker,
            color=color,
            capsize=2,
            markersize=5.2,
            zorder=4,
        )


def style_fraction_axes(axes: Sequence[plt.Axes], mass_edges: Sequence[float]) -> None:
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
        axis.set_ylim(1.0e-4, 1.0)
        axis.grid(alpha=0.16, linewidth=0.6)
        axis.tick_params(direction="in", top=True, right=True)
        axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")


def save_figure(fig: plt.Figure, stem: Path, *, dpi: int, write_pdf: bool) -> list[Path]:
    outputs = [stem.with_suffix(".png")]
    fig.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    if write_pdf:
        outputs.append(stem.with_suffix(".pdf"))
        fig.savefig(outputs[-1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def plot_raw_vs_mock(
    table: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    mass_edges: Sequence[float],
    threshold: float,
    sigma_minimum: float,
    ssfr_split: float,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), sharex=True, sharey=True)
    for axis, population in zip(axes, ("star_forming", "quiescent"), strict=True):
        add_observation_band(axis, observations, population)
        plot_fraction_series(axis, table, "raw", population, offset=-0.018, label=DEFINITION_LABELS["raw"])
        plot_fraction_series(
            axis, table, "mock_observed", population, offset=0.018, label=DEFINITION_LABELS["mock_observed"]
        )
        axis.set_title(POPULATION_LABELS[population])
    style_fraction_axes(axes, mass_edges)
    axes[0].set_ylabel(r"$F_{\rm AGN}(\lambda>10^{-3})$")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="0.45", alpha=0.17))
    labels.append("Observed 68% band")
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False, fontsize=8.5
    )
    fig.text(
        0.5,
        0.018,
        "Downward arrows are 68% upper limits for bins with zero weighted AGN.",
        ha="center",
        fontsize=8,
    )
    fig.suptitle(
        rf"MAP Galacticus: raw versus Blanton-like mock observation; common $\sigma>{sigma_minimum:g}$ km s$^{{-1}}$ sample"
        "\n"
        rf"pooled $z=0,0.1$; sSFR split $={ssfr_split:g}$; $\lambda_{{\rm cut}}={threshold:g}$",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.79, bottom=0.23, wspace=0.08)
    return fig


def plot_decomposition(
    table: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    mass_edges: Sequence[float],
    threshold: float,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharex=True, sharey=True)
    offsets = dict(zip(DEFINITION_ORDER, np.linspace(-0.045, 0.045, len(DEFINITION_ORDER)), strict=True))
    for axis, population in zip(axes, ("star_forming", "quiescent"), strict=True):
        add_observation_band(axis, observations, population)
        for definition in DEFINITION_ORDER:
            plot_fraction_series(
                axis,
                table,
                definition,
                population,
                offset=float(offsets[definition]),
                label=DEFINITION_LABELS[definition],
            )
        axis.set_title(POPULATION_LABELS[population])
    style_fraction_axes(axes, mass_edges)
    axes[0].set_ylabel(rf"$F_{{\rm AGN}}(\lambda>{threshold:g})$")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="0.45", alpha=0.17))
    labels.append("Observed 68% band")
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=3, frameon=False, fontsize=8.1
    )
    fig.text(
        0.5,
        0.018,
        "Downward arrows are 68% upper limits for bins with zero weighted AGN.",
        ha="center",
        fontsize=8,
    )
    fig.suptitle("Which inference step moves the MAP Galacticus AGN fraction?", fontsize=11)
    fig.subplots_adjust(left=0.078, right=0.99, top=0.88, bottom=0.25, wspace=0.08)
    return fig


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values[valid])
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]
    cumulative = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / np.sum(sorted_weights)
    return np.interp(quantiles, cumulative, sorted_values)


def plot_quantity_diagnostics(
    catalog: SnapshotCatalog,
    *,
    mass_edges: Sequence[float],
    ssfr_split: float,
    max_scatter_points: int,
    lambda_floor: float | None = None,
    lambda_threshold: float = 1.0e-3,
) -> plt.Figure:
    c = catalog.columns
    log_mass = np.asarray(c["log10_stellar_mass_msun"], dtype=float)
    ssfr = np.asarray(c["log10_ssfr_per_year"], dtype=float)
    weight = np.asarray(c["weight"], dtype=float)
    population_masks = {
        "star_forming": np.isfinite(ssfr) & (ssfr > ssfr_split),
        "quiescent": ~np.isnan(ssfr) & (ssfr <= ssfr_split),
    }
    relevance = np.ones(log_mass.shape, dtype=bool)
    if lambda_floor is not None:
        raw_lambda = np.asarray(c["lambda_raw"], dtype=float)
        mock_lambda = np.asarray(c["lambda_mock_observed"], dtype=float)
        relevance = (
            (np.isfinite(raw_lambda) & (raw_lambda > lambda_floor))
            | (np.isfinite(mock_lambda) & (mock_lambda > lambda_floor))
        )
    comparisons = (
        (
            np.asarray(c["bolometric_luminosity_catalog_erg_s"], dtype=float),
            np.asarray(c["bolometric_luminosity_hbeta_erg_s"], dtype=float),
            r"$L_{\rm bol}^{\rm Gal}$ [erg s$^{-1}$]",
            r"$L_{\rm bol}^{\rm inferred}(\mathrm{H}\beta)$ [erg s$^{-1}$]",
            r"$\Delta\log L_{\rm bol}$ (H$\beta$ inferred $-$ raw)",
        ),
        (
            np.asarray(c["black_hole_mass_msun"], dtype=float),
            np.asarray(c["black_hole_mass_sigma_msun"], dtype=float),
            r"$M_{\rm BH}^{\rm Gal}$ [$M_\odot$]",
            r"$M_{\rm BH}^{\rm inferred}(\sigma)$ [$M_\odot$]",
            r"$\Delta\log M_{\rm BH}$ ($\sigma$ inferred $-$ raw)",
        ),
        (
            np.asarray(c["lambda_raw"], dtype=float),
            np.asarray(c["lambda_mock_observed"], dtype=float),
            r"$\lambda_{\rm raw}$",
            r"$\lambda_{\rm mock\ observed}$",
            r"$\Delta\log\lambda$ (mock $-$ raw)",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
    rng = np.random.default_rng(27011983)
    centers = 0.5 * (np.asarray(mass_edges[:-1]) + np.asarray(mass_edges[1:]))
    for column, (raw, inferred, xlabel, ylabel, delta_label) in enumerate(comparisons):
        positive = (
            np.isfinite(raw)
            & (raw > 0.0)
            & np.isfinite(inferred)
            & (inferred > 0.0)
            & np.isfinite(log_mass)
            & (log_mass >= mass_edges[0])
            & (log_mass <= mass_edges[-1])
            & relevance
        )
        candidates = np.flatnonzero(positive)
        if candidates.size > max_scatter_points:
            candidates = rng.choice(candidates, size=max_scatter_points, replace=False)
        top = axes[0, column]
        for population, mask in population_masks.items():
            indices = candidates[mask[candidates]]
            top.scatter(
                raw[indices],
                inferred[indices],
                s=8,
                alpha=0.22,
                linewidths=0,
                color=POPULATION_COLORS[population],
                rasterized=True,
            )
        combined = np.concatenate((raw[positive], inferred[positive]))
        limits = np.nanpercentile(combined, [0.5, 99.5])
        limits = 10.0 ** np.asarray([np.floor(np.log10(limits[0])), np.ceil(np.log10(limits[1]))])
        top.plot(limits, limits, color="0.25", linestyle="--", linewidth=1.1)
        if column == 2:
            top.axvline(lambda_threshold, color="0.45", linestyle=":", linewidth=1.1)
            top.axhline(lambda_threshold, color="0.45", linestyle=":", linewidth=1.1)
        top.set_xscale("log")
        top.set_yscale("log")
        top.set_xlim(limits)
        top.set_ylim(limits)
        top.set_xlabel(xlabel)
        top.set_ylabel(ylabel)
        top.grid(alpha=0.14, linewidth=0.5)
        top.tick_params(direction="in", top=True, right=True)

        delta = np.full(raw.shape, np.nan)
        delta[positive] = np.log10(inferred[positive] / raw[positive])
        bottom = axes[1, column]
        for population, population_mask in population_masks.items():
            medians, lower, upper = [], [], []
            for index, (low, high) in enumerate(zip(mass_edges[:-1], mass_edges[1:], strict=True)):
                upper_test = log_mass <= high if index == len(mass_edges) - 2 else log_mass < high
                selected = positive & population_mask & (log_mass >= low) & upper_test
                q16, q50, q84 = weighted_quantile(delta[selected], weight[selected], [0.16, 0.50, 0.84])
                lower.append(q16)
                medians.append(q50)
                upper.append(q84)
            medians_array = np.asarray(medians)
            bottom.plot(
                centers,
                medians_array,
                color=POPULATION_COLORS[population],
                marker="o",
                linewidth=1.5,
                markersize=4.5,
                label=POPULATION_LABELS[population],
            )
            bottom.fill_between(
                centers,
                lower,
                upper,
                color=POPULATION_COLORS[population],
                alpha=0.14,
                linewidth=0,
            )
        bottom.axhline(0.0, color="0.25", linestyle="--", linewidth=1.1)
        bottom.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
        bottom.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        bottom.set_ylabel(delta_label)
        bottom.grid(alpha=0.16, linewidth=0.6)
        bottom.tick_params(direction="in", top=True, right=True)
    legend = [
        Line2D([], [], color=POPULATION_COLORS[name], marker="o", label=POPULATION_LABELS[name])
        for name in ("star_forming", "quiescent")
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False)
    selection_text = (
        "positive finite quantities only"
        if lambda_floor is None
        else rf"objects with raw or mock $\lambda>{lambda_floor:g}$"
    )
    fig.suptitle(
        "MAP Galacticus: raw quantities versus the Blanton-like inference\n"
        f"Bottom panels show weighted medians and 16th–84th percentiles; {selection_text}",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.10, hspace=0.30, wspace=0.28)
    return fig


def diagnostic_summary(catalog: SnapshotCatalog, *, mass_edges: Sequence[float], ssfr_split: float) -> pd.DataFrame:
    c = catalog.columns
    mass = np.asarray(c["log10_stellar_mass_msun"], dtype=float)
    ssfr = np.asarray(c["log10_ssfr_per_year"], dtype=float)
    weight = np.asarray(c["weight"], dtype=float)
    quantities = {
        "log10_lbol_hbeta_over_raw": (
            np.asarray(c["bolometric_luminosity_hbeta_erg_s"]),
            np.asarray(c["bolometric_luminosity_catalog_erg_s"]),
        ),
        "log10_mbh_sigma_over_true": (
            np.asarray(c["black_hole_mass_sigma_msun"]),
            np.asarray(c["black_hole_mass_msun"]),
        ),
        "log10_lambda_mock_over_raw": (
            np.asarray(c["lambda_mock_observed"]),
            np.asarray(c["lambda_raw"]),
        ),
    }
    records: list[dict[str, float | int | str]] = []
    populations = {
        "all": ~np.isnan(ssfr),
        "star_forming": np.isfinite(ssfr) & (ssfr > ssfr_split),
        "quiescent": ~np.isnan(ssfr) & (ssfr <= ssfr_split),
    }
    mass_sample = np.isfinite(mass) & (mass >= mass_edges[0]) & (mass <= mass_edges[-1])
    for population, population_mask in populations.items():
        for quantity, (numerator, denominator) in quantities.items():
            valid = (
                mass_sample
                & population_mask
                & np.isfinite(numerator)
                & (numerator > 0.0)
                & np.isfinite(denominator)
                & (denominator > 0.0)
            )
            q16, q50, q84 = weighted_quantile(
                np.log10(numerator[valid] / denominator[valid]), weight[valid], [0.16, 0.50, 0.84]
            )
            records.append(
                {
                    "population": population,
                    "quantity": quantity,
                    "raw_n": int(np.count_nonzero(valid)),
                    "weighted_q16_dex": q16,
                    "weighted_median_dex": q50,
                    "weighted_q84_dex": q84,
                }
            )
    return pd.DataFrame.from_records(records)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.hdf5 = args.hdf5.expanduser().resolve()
    args.observations = args.observations.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.params is not None:
        args.params = args.params.expanduser().resolve()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = load_observations(args.observations)
    params_path = args.params or args.hdf5.with_name("params.xml")
    disk_parameters = (
        switched_disk_parameters_from_xml(params_path)
        if params_path.is_file()
        else SwitchedDiskParameters()
    )

    snapshots: list[SnapshotCatalog] = []
    with h5py.File(args.hdf5, "r") as handle:
        selected = select_outputs(
            handle["Outputs"], args.redshifts, tolerance=args.redshift_tolerance
        )
        for target, output_name, _ in selected:
            base = load_snapshot_catalog(
                handle["Outputs"][output_name],
                target_redshift=target,
                switched_disk_parameters=disk_parameters,
                ledd_coefficient=BLANTON_LEDD_COEFFICIENT,
                centrals_only=args.centrals_only,
            )
            enriched, sigma_mask = add_mock_columns(base, sigma_minimum=args.sigma_minimum)
            snapshots.append(subset_catalog(enriched, sigma_mask))

    pooled = pool_snapshot_catalogs(snapshots, name="pooled_z0_z0p1_common_sigma_sample")
    fractions = fraction_frame(
        pooled,
        mass_edges=args.mass_edges,
        ssfr_split=args.ssfr_split,
        threshold=args.lambda_threshold,
        confidence=args.confidence,
    )
    summary = diagnostic_summary(pooled, mass_edges=args.mass_edges, ssfr_split=args.ssfr_split)

    catalog_columns = {
        "target_redshift": np.concatenate(
            [np.full(len(snapshot.columns["weight"]), snapshot.target_redshift) for snapshot in snapshots]
        ),
        "redshift": np.concatenate(
            [np.full(len(snapshot.columns["weight"]), snapshot.redshift) for snapshot in snapshots]
        ),
        "output_name": np.concatenate(
            [np.full(len(snapshot.columns["weight"]), snapshot.output_name) for snapshot in snapshots]
        ),
        **{key: np.asarray(value) for key, value in pooled.columns.items()},
    }
    catalog = pd.DataFrame(catalog_columns)
    catalog["population"] = np.where(
        catalog["log10_ssfr_per_year"] > args.ssfr_split, "star_forming", "quiescent"
    )
    low, high = float(args.mass_edges[0]), float(args.mass_edges[-1])
    catalog = catalog[
        np.isfinite(catalog["log10_stellar_mass_msun"])
        & (catalog["log10_stellar_mass_msun"] >= low)
        & (catalog["log10_stellar_mass_msun"] <= high)
    ].copy()

    files: list[Path] = []
    catalog_path = args.output_dir / "mock_observed_catalog.csv"
    fraction_path = args.output_dir / "fraction_comparison.csv"
    summary_path = args.output_dir / "diagnostic_summary.csv"
    catalog.to_csv(catalog_path, index=False)
    fractions.to_csv(fraction_path, index=False)
    summary.to_csv(summary_path, index=False)
    files.extend([catalog_path, fraction_path, summary_path])
    files.extend(
        save_figure(
            plot_raw_vs_mock(
                fractions,
                observations,
                mass_edges=args.mass_edges,
                threshold=args.lambda_threshold,
                sigma_minimum=args.sigma_minimum,
                ssfr_split=args.ssfr_split,
            ),
            args.output_dir / "fagn_raw_vs_mock_observed",
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )
    files.extend(
        save_figure(
            plot_decomposition(
                fractions,
                observations,
                mass_edges=args.mass_edges,
                threshold=args.lambda_threshold,
            ),
            args.output_dir / "fagn_inference_decomposition",
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )
    files.extend(
        save_figure(
            plot_quantity_diagnostics(
                pooled,
                mass_edges=args.mass_edges,
                ssfr_split=args.ssfr_split,
                max_scatter_points=args.max_scatter_points,
                lambda_threshold=args.lambda_threshold,
            ),
            args.output_dir / "raw_vs_inferred_quantities",
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )
    files.extend(
        save_figure(
            plot_quantity_diagnostics(
                pooled,
                mass_edges=args.mass_edges,
                ssfr_split=args.ssfr_split,
                max_scatter_points=args.max_scatter_points,
                lambda_floor=args.lambda_threshold / 100.0,
                lambda_threshold=args.lambda_threshold,
            ),
            args.output_dir / "raw_vs_inferred_quantities_near_threshold",
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_hdf5": str(args.hdf5),
        "params_xml": str(params_path.resolve()) if params_path.is_file() else None,
        "observations": str(args.observations),
        "definitions": {
            "common_sample": f"all four definitions use velocityDispersion > {args.sigma_minimum:g} km/s",
            "raw": "Galacticus saved radiative efficiency * Mdot*c^2 divided by L_Edd(true Galacticus M_BH)",
            "hbeta_true_bh": "Netzer(2019) Lbol from intrinsic Galacticus AGN Hbeta divided by L_Edd(true Galacticus M_BH)",
            "raw_sigma_bh": "direct Galacticus Lbol divided by L_Edd(Kormendy-Ho-2013 M_BH from Galacticus sigma)",
            "mock_observed": "Netzer(2019) Lbol from intrinsic Galacticus AGN Hbeta divided by L_Edd(Kormendy-Ho-2013 M_BH from Galacticus sigma)",
            "ledd_coefficient_erg_s_per_msun": BLANTON_LEDD_COEFFICIENT,
            "lambda_threshold": args.lambda_threshold,
            "ssfr_split_log10_per_year": args.ssfr_split,
            "mass_edges_log10_msun": list(map(float, args.mass_edges)),
            "weights": "mergerTreeWeight * nodeSubsamplingWeight",
            "mode_selection": "inclusive; no explicit thin-disk/ADAF cut",
            "redshift_pool": "descriptive z=0 plus z=0.1 pool; snapshots share merger trees",
        },
        "selected_outputs": [
            {
                "output_name": snapshot.output_name,
                "redshift": snapshot.redshift,
                "diagnostics": dict(snapshot.diagnostics),
            }
            for snapshot in snapshots
        ],
        "output_files": [path.name for path in files],
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    files.append(metadata_path)

    readme_path = args.output_dir / "README.md"
    readme_path.write_text(
        "# Raw versus mock-observed Galacticus AGN diagnostics\n\n"
        "Generated by `scripts/plot_blanton_mock_observed_diagnostics.py` from the MAP Galacticus run.\n\n"
        "All four Eddington-ratio definitions use the same sigma-selected host sample, the same "
        "Blanton Eddington-luminosity normalization, the same stellar-mass/sSFR bins, and the same "
        "Galacticus weights. This makes the crossed definitions a controlled decomposition of the "
        "H-beta bolometric correction and M_BH-sigma substitution.\n\n"
        "The mock-observed calculation is a latent, idealized measurement: intrinsic Galacticus AGN "
        "H-beta is treated as perfectly dust-corrected, and no MaNGA noise, aperture, or raw Seyfert "
        "detection cut is imposed because the published F_AGN constraint is completeness-corrected.\n\n"
        "The pooled z=0 and z=0.1 output is descriptive; the snapshots share merger trees and are not "
        "independent realizations. See `metadata.json` for exact definitions and provenance.\n"
    )
    files.append(readme_path)
    print(f"Wrote {len(catalog):,} selected catalogue rows to {args.output_dir}")
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
