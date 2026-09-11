#!/usr/bin/env python3
"""Compare Galacticus AGN demographics with Suresh & Blanton (2026)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from galacticus_emu.agn_demographics import (
    DEFAULT_LEDD_COEFFICIENT,
    GALACTICUS_MDOT_EDD_PER_MBH_MSUN_GYR,
    PAPER_II_MASS_EDGES,
    PAPER_II_SSFR_SPLIT,
    PAPER_II_THRESHOLDS,
    FractionBin,
    SnapshotCatalog,
    SwitchedDiskParameters,
    fit_trend_metrics,
    load_snapshot_catalog,
    pool_snapshot_catalogs,
    select_outputs,
    summarize_fractions,
    switched_disk_parameters_from_xml,
    weighted_jeffreys_fraction,
)


DEFAULT_HDF5 = REPO_ROOT / (
    "runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/"
    "automatedPipeline_transformedParams_finalPaper_definitive/"
    "MCMCs_production_from_exploratory_MAP/standard_observables_plus_emission_line_lfs/"
    "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr/"
    "bestFitModel_GalacticusRun/romanEPS_massFunction.hdf5"
)
DEFAULT_OBSERVATIONS = (
    REPO_ROOT
    / "data/observations/suresh_blanton_2026/table1_radiative_agn_fraction_bands.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper/agn_demographics"
POPULATION_COLORS = {"star_forming": "#2166ac", "quiescent": "#b2182b"}
POPULATION_LABELS = {"star_forming": "Star-forming", "quiescent": "Quiescent"}


def _optional_positive_float(text: str) -> float | None:
    if text.lower() == "none":
        return None
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5, help="Galacticus HDF5 output")
    parser.add_argument(
        "--params",
        type=Path,
        help="Galacticus params.xml; defaults to params.xml beside --hdf5",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--observations",
        type=Path,
        default=DEFAULT_OBSERVATIONS,
        help="Tracked transcription of Paper-II Table 1 bounds",
    )
    parser.add_argument(
        "--redshifts",
        type=float,
        nargs="+",
        default=[0.0, 0.1],
        help="Target redshifts (nearest saved output within tolerance is used)",
    )
    parser.add_argument(
        "--redshift-tolerance",
        type=float,
        default=0.02,
        help="Maximum |saved redshift - target redshift|",
    )
    parser.add_argument(
        "--mass-edges",
        type=float,
        nargs="+",
        default=PAPER_II_MASS_EDGES.tolist(),
        metavar="LOGM",
        help="log10 stellar-mass bin edges",
    )
    parser.add_argument(
        "--ssfr-split",
        type=float,
        default=PAPER_II_SSFR_SPLIT,
        help="Star-forming if log10(sSFR/yr^-1) is above this value",
    )
    parser.add_argument(
        "--ssfr-panel-splits",
        type=float,
        nargs="+",
        default=[-11.5, -11.0, -10.5],
        metavar="LOGSSFR",
        help="Three SF/quiescent boundaries shown in the fixed-lambda companion figure",
    )
    parser.add_argument(
        "--ssfr-panel-lambda-threshold",
        type=float,
        default=1.0e-3,
        metavar="LAMBDA",
        help="Fixed Eddington-ratio threshold in the sSFR-split companion figure",
    )
    parser.add_argument(
        "--lambda-thresholds",
        type=float,
        nargs="+",
        default=PAPER_II_THRESHOLDS.tolist(),
        metavar="LAMBDA",
        help="Eddington-ratio thresholds shown in the Fig. 2-style panels",
    )
    parser.add_argument(
        "--ledd-coefficient",
        type=float,
        default=DEFAULT_LEDD_COEFFICIENT,
        help="L_Edd / (M_BH/Msun) in erg/s for the Paper-II lambda definition",
    )
    parser.add_argument(
        "--fixed-radiative-efficiency",
        type=float,
        default=None,
        help="Use this eta for Lbol/lambda; omit to use the saved per-BH eta",
    )
    parser.add_argument(
        "--galacticus-mdot-edd-per-mbh",
        type=float,
        default=GALACTICUS_MDOT_EDD_PER_MBH_MSUN_GYR,
        help="Native Galacticus Mdot_Edd/M_BH in Gyr^-1, used only for disk-state x",
    )
    parser.add_argument(
        "--thin-disk-minimum",
        type=_optional_positive_float,
        default=argparse.SUPPRESS,
        metavar="XMIN|none",
        help="Override params.xml low-x ADAF transition; 'none' disables it",
    )
    parser.add_argument(
        "--thin-disk-maximum",
        type=_optional_positive_float,
        default=argparse.SUPPRESS,
        metavar="XMAX|none",
        help="Override params.xml high-x ADAF transition; 'none' disables it",
    )
    parser.add_argument(
        "--transition-width",
        type=float,
        default=argparse.SUPPRESS,
        help="Override params.xml smooth transition width in natural-log x",
    )
    parser.add_argument(
        "--centrals-only",
        action="store_true",
        help="Retain only central galaxies (nodeIsIsolated == 1); default is all galaxies",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.682689492137,
        help="Central Jeffreys interval probability",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-pdf", action="store_true", help="Write PNG figures only")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.hdf5.is_file():
        raise FileNotFoundError(f"Galacticus file not found: {args.hdf5}")
    if not args.observations.is_file():
        raise FileNotFoundError(f"Observation table not found: {args.observations}")
    edges = np.asarray(args.mass_edges, dtype=float)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("--mass-edges must be finite and strictly increasing")
    thresholds = np.asarray(args.lambda_thresholds, dtype=float)
    if thresholds.size != 3:
        raise ValueError("The Fig. 2-style output requires exactly three --lambda-thresholds")
    if np.any(~np.isfinite(thresholds)) or np.any(thresholds < 0.0):
        raise ValueError("--lambda-thresholds must be finite and non-negative")
    ssfr_panel_splits = np.asarray(args.ssfr_panel_splits, dtype=float)
    if ssfr_panel_splits.size != 3 or np.any(~np.isfinite(ssfr_panel_splits)):
        raise ValueError("--ssfr-panel-splits requires exactly three finite values")
    if not np.isfinite(args.ssfr_panel_lambda_threshold) or args.ssfr_panel_lambda_threshold < 0.0:
        raise ValueError("--ssfr-panel-lambda-threshold must be finite and non-negative")
    if args.redshift_tolerance < 0.0:
        raise ValueError("--redshift-tolerance must be non-negative")
    if args.ledd_coefficient <= 0.0 or args.galacticus_mdot_edd_per_mbh <= 0.0:
        raise ValueError("Eddington coefficients must be positive")
    if args.fixed_radiative_efficiency is not None and not (
        0.0 <= args.fixed_radiative_efficiency <= 1.0
    ):
        raise ValueError("--fixed-radiative-efficiency must lie in [0, 1]")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie in (0, 1)")


def _resolve_disk_parameters(args: argparse.Namespace) -> tuple[SwitchedDiskParameters, Path | None, str]:
    parameter_path = args.params if args.params is not None else args.hdf5.with_name("params.xml")
    if parameter_path.is_file():
        parameters = switched_disk_parameters_from_xml(parameter_path)
        source = "params.xml"
    elif args.params is not None:
        raise FileNotFoundError(f"Parameter file not found: {args.params}")
    else:
        parameters = SwitchedDiskParameters()
        parameter_path = None
        source = "script fallback defaults"
    parameters = SwitchedDiskParameters(
        thin_disk_minimum=getattr(args, "thin_disk_minimum", parameters.thin_disk_minimum),
        thin_disk_maximum=getattr(args, "thin_disk_maximum", parameters.thin_disk_maximum),
        transition_width=getattr(args, "transition_width", parameters.transition_width),
    )
    if any(hasattr(args, name) for name in ("thin_disk_minimum", "thin_disk_maximum", "transition_width")):
        source += " plus CLI override(s)"
    return parameters, parameter_path, source


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_observations(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected = {
        "mass_low_log10_msun",
        "mass_high_log10_msun",
        "population",
        "log10_fraction_lower",
        "log10_fraction_upper",
    }
    if not expected.issubset(data.columns):
        raise ValueError(f"{path} lacks columns {sorted(expected - set(data.columns))}")
    if set(data["population"]) != {"star_forming", "quiescent"}:
        raise ValueError(f"{path} must contain star_forming and quiescent rows")
    if np.any(data["log10_fraction_lower"] > data["log10_fraction_upper"]):
        raise ValueError(f"{path} has lower bounds above upper bounds")
    return data


def _summary_frame(rows: Sequence[FractionBin]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            "target_redshift": row.target_redshift,
            "redshift": row.redshift,
            "output_name": row.output_name,
            "mode_criterion": row.mode_criterion,
            "lambda_threshold": row.lambda_threshold,
            "population": row.population,
            "mass_low_log10_msun": row.mass_low,
            "mass_high_log10_msun": row.mass_high,
            "mass_center_log10_msun": row.mass_center,
        }
        record.update(asdict(row.estimate))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _quiescent_fraction_frame(
    snapshot: SnapshotCatalog,
    *,
    mass_edges: Sequence[float],
    ssfr_splits: Sequence[float],
    confidence: float,
) -> pd.DataFrame:
    """Measure the weighted quiescent share of the parent sample in each bin."""

    edges = np.asarray(mass_edges, dtype=float)
    log_mass = np.asarray(snapshot.columns["log10_stellar_mass_msun"], dtype=float)
    log_ssfr = np.asarray(snapshot.columns["log10_ssfr_per_year"], dtype=float)
    weights = np.asarray(snapshot.columns["weight"], dtype=float)
    records: list[dict[str, Any]] = []
    for ssfr_split in ssfr_splits:
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            upper_test = log_mass <= high if index == edges.size - 2 else log_mass < high
            denominator = (
                np.isfinite(log_mass)
                & (log_mass >= low)
                & upper_test
                & ~np.isnan(log_ssfr)
                & np.isfinite(weights)
                & (weights > 0.0)
            )
            estimate = weighted_jeffreys_fraction(
                log_ssfr[denominator] <= float(ssfr_split),
                weights[denominator],
                confidence=confidence,
            )
            record = {
                "ssfr_split_log10_per_year": float(ssfr_split),
                "mass_low_log10_msun": float(low),
                "mass_high_log10_msun": float(high),
                "mass_center_log10_msun": float(0.5 * (low + high)),
            }
            record.update(asdict(estimate))
            records.append(record)
    return pd.DataFrame.from_records(records)


def _catalog_frame(
    snapshots: Sequence[SnapshotCatalog], mass_edges: Sequence[float], ssfr_split: float
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    low, high = float(mass_edges[0]), float(mass_edges[-1])
    for snapshot in snapshots:
        columns = {name: np.asarray(values) for name, values in snapshot.columns.items()}
        keep = (
            np.isfinite(columns["log10_stellar_mass_msun"])
            & (columns["log10_stellar_mass_msun"] >= low)
            & (columns["log10_stellar_mass_msun"] <= high)
            & ~np.isnan(columns["log10_ssfr_per_year"])
            & np.isfinite(columns["weight"])
            & (columns["weight"] > 0.0)
        )
        frame = pd.DataFrame({name: values[keep] for name, values in columns.items()})
        frame.insert(0, "output_name", snapshot.output_name)
        frame.insert(0, "redshift", snapshot.redshift)
        frame.insert(0, "target_redshift", snapshot.target_redshift)
        frame["population_default"] = np.where(
            frame["log10_ssfr_per_year"] > ssfr_split, "star_forming", "quiescent"
        )
        frame["thin_disk_dominated"] = (
            frame["valid_accreting_black_hole"].astype(bool) & (frame["fraction_adaf"] < 0.5)
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _rows_for(
    rows: Sequence[FractionBin],
    snapshot: SnapshotCatalog,
    mode: str,
    threshold: float,
    population: str,
) -> list[FractionBin]:
    return sorted(
        [
            row
            for row in rows
            if row.output_name == snapshot.output_name
            and row.mode_criterion == mode
            and np.isclose(row.lambda_threshold, threshold, rtol=1.0e-12, atol=0.0)
            and row.population == population
        ],
        key=lambda row: row.mass_center,
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(alpha=0.16, linewidth=0.6)
    axis.tick_params(direction="in", top=True, right=True)


def _add_observation_rectangles(axis: plt.Axes, observations: pd.DataFrame) -> None:
    for record in observations.itertuples(index=False):
        lower = 10.0 ** record.log10_fraction_lower
        upper = 10.0 ** record.log10_fraction_upper
        axis.add_patch(
            Rectangle(
                (record.mass_low_log10_msun, lower),
                record.mass_high_log10_msun - record.mass_low_log10_msun,
                upper - lower,
                facecolor=POPULATION_COLORS[record.population],
                edgecolor="none",
                alpha=0.17,
                zorder=0,
            )
        )


def _plot_fraction_series(
    axis: plt.Axes,
    rows: Sequence[FractionBin],
    *,
    color: str,
    marker: str,
    linestyle: str,
    label: str,
    x_offset: float,
) -> None:
    finite_rows = [row for row in rows if np.isfinite(row.estimate.fraction)]
    nonzero = [row for row in finite_rows if not row.estimate.is_upper_limit]
    if nonzero:
        x = np.asarray([row.mass_center + x_offset for row in nonzero])
        y = np.asarray([row.estimate.fraction for row in nonzero])
        lower = np.asarray([row.estimate.lower for row in nonzero])
        upper = np.asarray([row.estimate.upper for row in nonzero])
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((np.maximum(y - lower, 0.0), np.maximum(upper - y, 0.0))),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.8,
            capsize=2,
            label=label,
            zorder=3,
        )
    else:
        # Retain a legend entry even when every bin is a limit/empty.
        axis.plot([], [], color=color, marker=marker, linestyle=linestyle, label=label)
    limits = [row for row in finite_rows if row.estimate.is_upper_limit]
    if limits:
        x = np.asarray([row.mass_center + x_offset for row in limits])
        upper = np.asarray([row.estimate.upper for row in limits])
        axis.errorbar(
            x,
            upper,
            yerr=0.35 * upper,
            uplims=True,
            fmt=marker,
            color=color,
            markersize=4.8,
            capsize=2,
            linestyle="none",
            zorder=4,
        )


def _save_figure(fig: plt.Figure, stem: Path, *, dpi: int, write_pdf: bool) -> list[Path]:
    paths = [stem.with_suffix(".png")]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    if write_pdf:
        paths.append(stem.with_suffix(".pdf"))
        fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_fig2_style(
    output_stem: Path,
    *,
    snapshots: Sequence[SnapshotCatalog],
    rows: Sequence[FractionBin],
    observations: pd.DataFrame,
    thresholds: Sequence[float],
    mass_edges: Sequence[float],
    mode: str,
    ssfr_split: float,
    fixed_efficiency: float | None,
    dpi: int,
    write_pdf: bool,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), sharex=True, sharey=True)
    marker_cycle = ("o", "s", "D", "^")
    line_cycle = ("-", "--", "-.", ":")
    redshift_offsets = np.linspace(-0.035, 0.035, len(snapshots)) if len(snapshots) > 1 else [0.0]
    for axis, threshold in zip(axes, thresholds, strict=True):
        _add_observation_rectangles(axis, observations)
        for snapshot_index, snapshot in enumerate(snapshots):
            if snapshot.output_name.startswith("pooled_"):
                redshift_min = float(snapshot.diagnostics.get("redshift_min", snapshot.redshift))
                redshift_max = float(snapshot.diagnostics.get("redshift_max", snapshot.redshift))
                snapshot_label = f"pooled z={redshift_min:.3g}–{redshift_max:.3g}"
            else:
                snapshot_label = f"z={snapshot.redshift:.3g}"
            for population, population_offset in (("star_forming", -0.008), ("quiescent", 0.008)):
                series = _rows_for(rows, snapshot, mode, float(threshold), population)
                _plot_fraction_series(
                    axis,
                    series,
                    color=POPULATION_COLORS[population],
                    marker=marker_cycle[snapshot_index % len(marker_cycle)],
                    linestyle=line_cycle[snapshot_index % len(line_cycle)],
                    label=f"{POPULATION_LABELS[population]}, {snapshot_label}",
                    x_offset=float(redshift_offsets[snapshot_index]) + population_offset,
                )
        axis.set_yscale("log")
        axis.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
        axis.set_ylim(1.0e-4, 1.0)
        axis.set_title(rf"Galacticus: $\lambda>{threshold:g}$")
        axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        _style_axis(axis)
    axes[0].set_ylabel(r"$F_{\rm AGN}$")
    eta_text = "catalog eta" if fixed_efficiency is None else f"fixed eta={fixed_efficiency:g}"
    mode_text = "lambda-only (all accretion states)" if mode == "inclusive" else r"lambda + $f_{\rm ADAF}<0.5$"
    fig.suptitle(
        f"Suresh & Blanton (2026) radiative-AGN comparison — {mode_text}; {eta_text}; "
        rf"sSFR split $={ssfr_split:g}$",
        fontsize=11,
    )
    handles, labels = axes[-1].get_legend_handles_labels()
    handles.extend(
        [
            Patch(facecolor=POPULATION_COLORS["star_forming"], alpha=0.17, label="Observed SF 1 sigma bounds"),
            Patch(facecolor=POPULATION_COLORS["quiescent"], alpha=0.17, label="Observed Q 1 sigma bounds"),
        ]
    )
    labels.extend(["Observed SF 1 sigma bounds", "Observed Q 1 sigma bounds"])
    # Each panel has the same entries; one compact figure-level legend is sufficient.
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        unique[label] = handle
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=min(4, len(unique)),
        fontsize=8,
        frameon=False,
    )
    fig.text(
        0.5,
        0.018,
        "Downward arrows: zero weighted numerator, plotted at the upper edge of the central 68% effective-weight Jeffreys interval.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.27, wspace=0.12)
    return _save_figure(fig, output_stem, dpi=dpi, write_pdf=write_pdf)


def _plot_ssfr_split_style(
    output_stem: Path,
    *,
    snapshot: SnapshotCatalog,
    rows_by_split: dict[float, Sequence[FractionBin]],
    quiescent_fractions: pd.DataFrame,
    observations: pd.DataFrame,
    ssfr_splits: Sequence[float],
    lambda_threshold: float,
    mass_edges: Sequence[float],
    fixed_efficiency: float | None,
    dpi: int,
    write_pdf: bool,
) -> list[Path]:
    """Plot a Fig. 2-style comparison at fixed lambda for three sSFR cuts."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.4, 6.4),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [3.0, 1.05]},
    )
    fagn_axes = axes[0]
    fq_axes = axes[1]
    redshift_min = float(snapshot.diagnostics.get("redshift_min", snapshot.redshift))
    redshift_max = float(snapshot.diagnostics.get("redshift_max", snapshot.redshift))
    snapshot_label = f"pooled z={redshift_min:.3g}–{redshift_max:.3g}"
    for fagn_axis, fq_axis, ssfr_split in zip(
        fagn_axes,
        fq_axes,
        ssfr_splits,
        strict=True,
    ):
        _add_observation_rectangles(fagn_axis, observations)
        rows = rows_by_split[float(ssfr_split)]
        for population, population_offset in (("star_forming", -0.008), ("quiescent", 0.008)):
            series = _rows_for(
                rows,
                snapshot,
                "inclusive",
                float(lambda_threshold),
                population,
            )
            _plot_fraction_series(
                fagn_axis,
                series,
                color=POPULATION_COLORS[population],
                marker="o",
                linestyle="-",
                label=f"{POPULATION_LABELS[population]}, {snapshot_label}",
                x_offset=population_offset,
            )
        fagn_axis.set_yscale("log")
        fagn_axis.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
        fagn_axis.set_ylim(1.0e-4, 1.0)
        fagn_axis.set_title(
            rf"SF if $\log_{{10}}({{\rm sSFR}}/{{\rm yr}}^{{-1}})>{ssfr_split:g}$"
        )
        _style_axis(fagn_axis)

        selected_fq = quiescent_fractions[
            np.isclose(
                quiescent_fractions["ssfr_split_log10_per_year"],
                float(ssfr_split),
                rtol=0.0,
                atol=1.0e-12,
            )
        ].sort_values("mass_center_log10_msun")
        x = selected_fq["mass_center_log10_msun"].to_numpy()
        y = selected_fq["fraction"].to_numpy()
        lower = selected_fq["lower"].to_numpy()
        upper = selected_fq["upper"].to_numpy()
        fq_axis.errorbar(
            x,
            y,
            yerr=np.vstack((np.maximum(y - lower, 0.0), np.maximum(upper - y, 0.0))),
            color="#5e3c99",
            marker="o",
            linestyle="-",
            linewidth=1.35,
            markersize=4.8,
            capsize=2,
        )
        fq_axis.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
        fq_axis.set_ylim(0.0, 1.0)
        fq_axis.set_yticks([0.0, 0.5, 1.0])
        fq_axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        _style_axis(fq_axis)
    fagn_axes[0].set_ylabel(r"$F_{\rm AGN}$")
    fq_axes[0].set_ylabel(r"$f_{\rm quiescent}$")
    eta_text = "catalog eta" if fixed_efficiency is None else f"fixed eta={fixed_efficiency:g}"
    fig.suptitle(
        "Suresh & Blanton (2026) radiative-AGN comparison — "
        rf"inclusive; {eta_text}; fixed $\lambda>{lambda_threshold:g}$",
        fontsize=11,
    )
    handles, labels = fagn_axes[-1].get_legend_handles_labels()
    handles.extend(
        [
            Patch(
                facecolor=POPULATION_COLORS["star_forming"],
                alpha=0.17,
                label="Observed SF 1 sigma bounds",
            ),
            Patch(
                facecolor=POPULATION_COLORS["quiescent"],
                alpha=0.17,
                label="Observed Q 1 sigma bounds",
            ),
        ]
    )
    labels.extend(["Observed SF 1 sigma bounds", "Observed Q 1 sigma bounds"])
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        unique[label] = handle
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=min(4, len(unique)),
        fontsize=8,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "Downward arrows: zero weighted numerator, plotted at the upper edge of the central 68% effective-weight Jeffreys interval.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.86,
        bottom=0.19,
        hspace=0.08,
        wspace=0.12,
    )
    return _save_figure(fig, output_stem, dpi=dpi, write_pdf=write_pdf)


def _plot_accretion_states(
    output_stem: Path,
    *,
    snapshots: Sequence[SnapshotCatalog],
    disk_parameters: SwitchedDiskParameters,
    mass_edges: Sequence[float],
    ssfr_split: float,
    galacticus_mdot_edd_per_mbh: float,
    dpi: int,
    write_pdf: bool,
) -> list[Path]:
    fig, axes_array = plt.subplots(
        1, len(snapshots), figsize=(6.2 * len(snapshots), 5.0), sharex=True, sharey=True, squeeze=False
    )
    axes = axes_array[0]
    black_hole_grid = np.logspace(4.0, 11.0, 300)
    for axis, snapshot in zip(axes, snapshots, strict=True):
        columns = snapshot.columns
        log_mass = np.asarray(columns["log10_stellar_mass_msun"])
        mdot = np.asarray(columns["mdot_rest_msun_per_gyr"])
        bh_mass = np.asarray(columns["black_hole_mass_msun"])
        log_ssfr = np.asarray(columns["log10_ssfr_per_year"])
        base = (
            np.asarray(columns["valid_accreting_black_hole"], dtype=bool)
            & (log_mass >= mass_edges[0])
            & (log_mass <= mass_edges[-1])
            & np.isfinite(log_ssfr)
        )
        for population, population_mask in (
            ("star_forming", log_ssfr > ssfr_split),
            ("quiescent", log_ssfr <= ssfr_split),
        ):
            selected = base & population_mask
            axis.scatter(
                bh_mass[selected],
                mdot[selected],
                s=7,
                alpha=0.28,
                linewidths=0,
                color=POPULATION_COLORS[population],
                label=POPULATION_LABELS[population],
                rasterized=True,
            )
        if disk_parameters.thin_disk_minimum is not None:
            lower = disk_parameters.thin_disk_minimum * galacticus_mdot_edd_per_mbh * black_hole_grid
            axis.plot(black_hole_grid, lower, color="black", linestyle="--", linewidth=1.7, label="low-x fADAF=0.5")
        else:
            lower = None
        if disk_parameters.thin_disk_maximum is not None:
            upper = disk_parameters.thin_disk_maximum * galacticus_mdot_edd_per_mbh * black_hole_grid
            axis.plot(black_hole_grid, upper, color="0.35", linestyle="--", linewidth=1.7, label="high-x fADAF=0.5")
        else:
            upper = None
        if lower is not None and upper is not None:
            axis.fill_between(black_hole_grid, lower, upper, color="goldenrod", alpha=0.08, label="thin-disk-dominated")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(2.0e4, 2.0e10)
        axis.set_ylim(1.0e-5, 1.0e12)
        axis.set_title(f"{snapshot.output_name}: z={snapshot.redshift:.3g}")
        axis.set_xlabel(r"$M_{\rm BH}\ [M_\odot]$")
        _style_axis(axis)
    axes[0].set_ylabel(r"$\dot{M}_{\rm rest}\ [M_\odot\,\mathrm{Gyr}^{-1}]$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
    fig.suptitle(
        "Galacticus accretion states (Paper-II Fig. 5-style); dashed lines are smooth-switch midpoints",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.19, wspace=0.06)
    return _save_figure(fig, output_stem, dpi=dpi, write_pdf=write_pdf)


def _plot_mstar_ssfr(
    output_stem: Path,
    *,
    snapshot: SnapshotCatalog,
    mass_edges: Sequence[float],
    ssfr_splits: Sequence[float],
    dpi: int,
    write_pdf: bool,
) -> list[Path]:
    """Plot the pooled stellar-mass–sSFR distribution and classification cuts."""

    log_mass = np.asarray(snapshot.columns["log10_stellar_mass_msun"], dtype=float)
    log_ssfr = np.asarray(snapshot.columns["log10_ssfr_per_year"], dtype=float)
    selected = (
        np.isfinite(log_mass)
        & (log_mass >= float(mass_edges[0]))
        & (log_mass <= float(mass_edges[-1]))
        & ~np.isnan(log_ssfr)
    )
    x = log_mass[selected]
    y = log_ssfr[selected]
    plotting_floor = -15.0
    finite = np.isfinite(y)
    below_floor = finite & (y < plotting_floor)
    visible = finite & ~below_floor

    fig, axis = plt.subplots(figsize=(7.8, 5.7))
    axis.scatter(
        x[visible],
        y[visible],
        s=9,
        color="#343a40",
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        label="Galacticus galaxies",
    )
    if np.any(below_floor):
        axis.scatter(
            x[below_floor],
            np.full(np.count_nonzero(below_floor), plotting_floor),
            s=14,
            marker="v",
            color="#343a40",
            alpha=0.5,
            linewidths=0,
            rasterized=True,
            label=rf"$\log_{{10}}({{\rm sSFR}}/{{\rm yr}}^{{-1}})<{plotting_floor:g}$",
        )
    cut_colors = ("#2166ac", "#d95f02", "#5e3c99")
    for ssfr_split, color in zip(ssfr_splits, cut_colors, strict=True):
        axis.axhline(
            float(ssfr_split),
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=rf"SF/Q cut: ${float(ssfr_split):g}$",
        )
    redshift_min = float(snapshot.diagnostics.get("redshift_min", snapshot.redshift))
    redshift_max = float(snapshot.diagnostics.get("redshift_max", snapshot.redshift))
    axis.set_title(
        rf"Galacticus stellar mass–sSFR distribution; pooled $z={redshift_min:.3g}$–${redshift_max:.3g}$"
    )
    axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axis.set_ylabel(r"$\log_{10}({\rm sSFR}/{\rm yr}^{-1})$")
    axis.set_xlim(float(mass_edges[0]), float(mass_edges[-1]))
    axis.set_ylim(plotting_floor - 0.15, -8.0)
    _style_axis(axis)
    axis.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    return _save_figure(fig, output_stem, dpi=dpi, write_pdf=write_pdf)


def _metrics_records(
    rows: Sequence[FractionBin], snapshots: Sequence[SnapshotCatalog], thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        for mode in ("inclusive", "thin_disk_dominated"):
            for threshold in thresholds:
                selected = [
                    row
                    for row in rows
                    if row.output_name == snapshot.output_name
                    and row.mode_criterion == mode
                    and np.isclose(row.lambda_threshold, threshold, rtol=1.0e-12, atol=0.0)
                ]
                metrics = fit_trend_metrics(selected)
                records.append(
                    {
                        "target_redshift": snapshot.target_redshift,
                        "redshift": snapshot.redshift,
                        "output_name": snapshot.output_name,
                        "mode_criterion": mode,
                        "lambda_threshold": float(threshold),
                        **asdict(metrics),
                    }
                )
    return records


def _write_readme(
    path: Path,
    *,
    args: argparse.Namespace,
    disk_parameters: SwitchedDiskParameters,
    disk_parameter_source: str,
    snapshots: Sequence[SnapshotCatalog],
    pooled_snapshot: SnapshotCatalog,
    metrics: Sequence[dict[str, Any]],
    files: Sequence[Path],
) -> None:
    command_parts = [
        "python",
        "scripts/plot_suresh_blanton_agn_demographics.py",
        "--hdf5",
        shlex.quote(str(args.hdf5)),
        "--output-dir",
        shlex.quote(str(args.output_dir)),
    ]
    if args.fixed_radiative_efficiency is not None:
        command_parts.extend(
            ["--fixed-radiative-efficiency", f"{args.fixed_radiative_efficiency:g}"]
        )
    command = " ".join(command_parts)
    lines = [
        "# Galacticus AGN-demographics first pass",
        "",
        "This directory is generated by `scripts/plot_suresh_blanton_agn_demographics.py`.",
        "It compares the existing MAP Galacticus run to the radiative-AGN constraint in",
        "Suresh & Blanton (2026), arXiv:2607.19603v1.",
        "",
        "## Outputs",
        "",
    ]
    descriptions = {
        "catalog.csv": "Per-galaxy derived catalogue within the requested stellar-mass range.",
        "fraction_summary.csv": "All weighted fractions, effective counts, intervals, and upper-limit flags.",
        "fraction_summary_ssfr_splits.csv": "Fixed-lambda fractions for each sSFR boundary in the companion figure.",
        "quiescent_fraction_ssfr_splits.csv": "Weighted quiescent share of the parent sample shown in the lower companion panels.",
        "metrics_and_provenance.json": "Run definitions, diagnostics, provenance, and Paper-II trend metrics.",
        "fig2_inclusive": "Three lambda panels with no accretion-mode cut.",
        "fig2_thin_disk_dominated": "Same panels additionally requiring a valid accreting BH and f_ADAF < 0.5.",
        "fig2_ssfr_splits_inclusive": "Three sSFR boundaries at fixed lambda with no accretion-mode cut.",
        "fig5_accretion_states": "Mdot-MBH view with both Galacticus ADAF transition branches.",
        "mstar_ssfr_ssfr_cuts": "Pooled stellar-mass–sSFR distribution with the three classification boundaries.",
    }
    for file_path in sorted(set(files)):
        key = file_path.stem if file_path.suffix in {".png", ".pdf"} else file_path.name
        lines.append(f"- `{file_path.name}` — {descriptions.get(key, 'Generated analysis product.')}")
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            f"- Outputs: {', '.join(f'{item.output_name} (z={item.redshift:.8g})' for item in snapshots)}.",
            f"- Primary Fig. 2-style curves pool those outputs as `{pooled_snapshot.output_name}`; the pool is descriptive because the snapshots share merger trees.",
            f"- Stellar-mass edges: {', '.join(f'{value:g}' for value in args.mass_edges)}.",
            f"- Star-forming split: log10(sSFR/yr^-1) > {args.ssfr_split:g}; finite values at/below the cut plus zero/clipped rates are quiescent.",
            f"- sSFR companion panels: {', '.join(f'{value:g}' for value in args.ssfr_panel_splits)}, all at lambda > {args.ssfr_panel_lambda_threshold:g}.",
            "- Total stellar mass and SFR are disk+spheroid. Saved SFR is converted from Msun/Gyr to Msun/yr; negative totals are clipped to zero and counted in the JSON.",
            "- The element at index 0 is used from each VLEN Mdot, radiative-efficiency, and jet-power row.",
            f"- L_Edd = {args.ledd_coefficient:.8g} (M_BH/Msun) erg/s.",
            "- Lbol = eta Mdot_rest c^2. "
            + ("The saved per-BH eta is used." if args.fixed_radiative_efficiency is None else f"A fixed eta={args.fixed_radiative_efficiency:g} is used; saved eta remains in catalog.csv."),
            f"- Galacticus disk-state x uses Mdot_Edd/M_BH={args.galacticus_mdot_edd_per_mbh:.8g} Gyr^-1, independently of the configurable Paper-II L_Edd coefficient.",
            f"- Switched disk ({disk_parameter_source}): xmin={disk_parameters.thin_disk_minimum}, xmax={disk_parameters.thin_disk_maximum}, Delta_ln_x={disk_parameters.transition_width}.",
            "- Fractions use mergerTreeWeight * nodeSubsamplingWeight. Indicative intervals are central Jeffreys beta intervals using Kish effective counts. A zero numerator is plotted at the interval's upper edge (the 84.13th percentile for the default 68.27% interval).",
            f"- Galaxy selection: {'centrals only (nodeIsIsolated=1)' if args.centrals_only else 'all galaxies (central and satellite)'}.",
            "",
            "## Paper-II qualitative metrics",
            "",
            "Observed reference values are SF slope = 0.14, Q slope = -1.11, and SF-Q separation at log10(Mstar)=11 of 1.27 dex.",
            "",
            "| sample | z | mode | lambda cut | SF slope | Q slope | separation (dex) |",
            "|:---|---:|:---|---:|---:|---:|---:|",
        ]
    )
    for record in metrics:
        values = [record["star_forming_slope"], record["quiescent_slope"], record["separation_at_log10_mass_11"]]
        formatted = [("n/a" if not np.isfinite(value) else f"{value:.3f}") for value in values]
        redshift_label = "pooled" if record["output_name"].startswith("pooled_") else f"{record['redshift']:.3g}"
        lines.append(
            f"| {record['output_name']} | {redshift_label} | {record['mode_criterion']} | {record['lambda_threshold']:.0e} | "
            f"{formatted[0]} | {formatted[1]} | {formatted[2]} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "This is an intrinsic, instantaneous SAM catalogue comparison, not a mock narrow-line selection. It does not include MaNGA selection/completeness, aperture effects, Eddington-ratio variability, or systematic stellar/BH-mass errors. The observational rectangles are the published Table-1 lower/upper bounds, not midpoint measurements. The model intervals are only effective-sampling diagnostics and do not include merger-tree cosmic variance. The selected snapshots reuse merger trees and are correlated; the Paper-II-style pool is descriptive and must not be treated as a set of independent realizations for inference.",
            "",
            "The two mode definitions can be changed downstream using `eddington_ratio`, `fraction_adaf`, and `valid_accreting_black_hole` in `catalog.csv`.",
            "",
            "## Re-run",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    args.hdf5 = args.hdf5.expanduser().resolve()
    args.observations = args.observations.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.params is not None:
        args.params = args.params.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    disk_parameters, params_path, disk_parameter_source = _resolve_disk_parameters(args)
    observations = _load_observations(args.observations)

    snapshots: list[SnapshotCatalog] = []
    with h5py.File(args.hdf5, "r") as handle:
        if "Outputs" not in handle:
            raise KeyError(f"{args.hdf5} has no Outputs group")
        output_selection = select_outputs(
            handle["Outputs"], args.redshifts, tolerance=args.redshift_tolerance
        )
        hdf5_attributes = {key: _json_value(value) for key, value in handle.attrs.items()}
        for target, output_name, _actual in output_selection:
            snapshots.append(
                load_snapshot_catalog(
                    handle["Outputs"][output_name],
                    target_redshift=target,
                    switched_disk_parameters=disk_parameters,
                    ledd_coefficient=args.ledd_coefficient,
                    fixed_radiative_efficiency=args.fixed_radiative_efficiency,
                    galacticus_mdot_edd_per_mbh=args.galacticus_mdot_edd_per_mbh,
                    centrals_only=args.centrals_only,
                )
            )

    fraction_rows: list[FractionBin] = []
    for snapshot in snapshots:
        fraction_rows.extend(
            summarize_fractions(
                snapshot,
                mass_edges=args.mass_edges,
                ssfr_split=args.ssfr_split,
                thresholds=args.lambda_thresholds,
                confidence=args.confidence,
            )
        )
    pooled_snapshot = pool_snapshot_catalogs(snapshots, name="pooled_selected_outputs")
    fraction_rows.extend(
        summarize_fractions(
            pooled_snapshot,
            mass_edges=args.mass_edges,
            ssfr_split=args.ssfr_split,
            thresholds=args.lambda_thresholds,
            confidence=args.confidence,
        )
    )
    catalog = _catalog_frame(snapshots, args.mass_edges, args.ssfr_split)
    summary = _summary_frame(fraction_rows)
    catalog_path = args.output_dir / "catalog.csv"
    summary_path = args.output_dir / "fraction_summary.csv"
    catalog.to_csv(catalog_path, index=False, float_format="%.12g")
    summary.to_csv(summary_path, index=False, float_format="%.12g")

    ssfr_panel_rows: dict[float, list[FractionBin]] = {}
    ssfr_summary_frames: list[pd.DataFrame] = []
    for ssfr_split in args.ssfr_panel_splits:
        split = float(ssfr_split)
        rows = summarize_fractions(
            pooled_snapshot,
            mass_edges=args.mass_edges,
            ssfr_split=split,
            thresholds=[args.ssfr_panel_lambda_threshold],
            confidence=args.confidence,
        )
        ssfr_panel_rows[split] = rows
        frame = _summary_frame(rows)
        frame.insert(0, "ssfr_split_log10_per_year", split)
        ssfr_summary_frames.append(frame)
    ssfr_summary_path = args.output_dir / "fraction_summary_ssfr_splits.csv"
    pd.concat(ssfr_summary_frames, ignore_index=True).to_csv(
        ssfr_summary_path,
        index=False,
        float_format="%.12g",
    )
    quiescent_fraction_summary = _quiescent_fraction_frame(
        pooled_snapshot,
        mass_edges=args.mass_edges,
        ssfr_splits=args.ssfr_panel_splits,
        confidence=args.confidence,
    )
    quiescent_fraction_path = args.output_dir / "quiescent_fraction_ssfr_splits.csv"
    quiescent_fraction_summary.to_csv(
        quiescent_fraction_path,
        index=False,
        float_format="%.12g",
    )

    figure_paths: list[Path] = []
    figure_paths.extend(
        _plot_fig2_style(
            args.output_dir / "fig2_inclusive",
            snapshots=[pooled_snapshot],
            rows=fraction_rows,
            observations=observations,
            thresholds=args.lambda_thresholds,
            mass_edges=args.mass_edges,
            mode="inclusive",
            ssfr_split=args.ssfr_split,
            fixed_efficiency=args.fixed_radiative_efficiency,
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )
    figure_paths.extend(
        _plot_ssfr_split_style(
            args.output_dir / "fig2_ssfr_splits_inclusive",
            snapshot=pooled_snapshot,
            rows_by_split=ssfr_panel_rows,
            quiescent_fractions=quiescent_fraction_summary,
            observations=observations,
            ssfr_splits=args.ssfr_panel_splits,
            lambda_threshold=args.ssfr_panel_lambda_threshold,
            mass_edges=args.mass_edges,
            fixed_efficiency=args.fixed_radiative_efficiency,
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )
    figure_paths.extend(
        _plot_fig2_style(
            args.output_dir / "fig2_thin_disk_dominated",
            snapshots=[pooled_snapshot],
            rows=fraction_rows,
            observations=observations,
            thresholds=args.lambda_thresholds,
            mass_edges=args.mass_edges,
            mode="thin_disk_dominated",
            ssfr_split=args.ssfr_split,
            fixed_efficiency=args.fixed_radiative_efficiency,
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )
    figure_paths.extend(
        _plot_accretion_states(
            args.output_dir / "fig5_accretion_states",
            snapshots=snapshots,
            disk_parameters=disk_parameters,
            mass_edges=args.mass_edges,
            ssfr_split=args.ssfr_split,
            galacticus_mdot_edd_per_mbh=args.galacticus_mdot_edd_per_mbh,
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )
    figure_paths.extend(
        _plot_mstar_ssfr(
            args.output_dir / "mstar_ssfr_ssfr_cuts",
            snapshot=pooled_snapshot,
            mass_edges=args.mass_edges,
            ssfr_splits=args.ssfr_panel_splits,
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
    )

    metrics = _metrics_records(fraction_rows, [*snapshots, pooled_snapshot], args.lambda_thresholds)
    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "hdf5": str(args.hdf5),
            "hdf5_size_bytes": args.hdf5.stat().st_size,
            "hdf5_attributes": hdf5_attributes,
            "params_xml": str(params_path.resolve()) if params_path is not None else None,
            "observational_bounds_csv": str(args.observations),
            "observational_source": "Suresh & Blanton 2026, arXiv:2607.19603v1, Table 1",
        },
        "definitions": {
            "target_redshifts": list(map(float, args.redshifts)),
            "redshift_tolerance": args.redshift_tolerance,
            "mass_edges_log10_msun": list(map(float, args.mass_edges)),
            "ssfr_split_log10_per_year": args.ssfr_split,
            "ssfr_panel_splits_log10_per_year": list(map(float, args.ssfr_panel_splits)),
            "ssfr_panel_lambda_threshold": args.ssfr_panel_lambda_threshold,
            "lambda_thresholds": list(map(float, args.lambda_thresholds)),
            "mode_criteria": ["inclusive", "thin_disk_dominated: valid accreting BH and f_ADAF < 0.5"],
            "centrals_only": args.centrals_only,
            "ledd_coefficient_erg_s_per_msun": args.ledd_coefficient,
            "fixed_radiative_efficiency": args.fixed_radiative_efficiency,
            "galacticus_mdot_edd_per_mbh_gyr_inverse": args.galacticus_mdot_edd_per_mbh,
            "switched_disk": asdict(disk_parameters),
            "switched_disk_parameter_source": disk_parameter_source,
            "confidence": args.confidence,
            "interval": "indicative central Jeffreys beta interval with Kish effective n; zero numerator rendered at its upper edge",
            "weights": "mergerTreeWeight * nodeSubsamplingWeight",
            "vlen_selection": "element 0 (central black hole) from Mdot/eta/jet-power rows",
            "sfr": "disk+spheroid Msun/Gyr, negative totals clipped to zero, then divided by 1e9",
            "formulas": {
                "lbol": "eta_used * Mdot_rest * c^2",
                "lambda": "Lbol / (ledd_coefficient * M_BH/Msun)",
                "x_paper": "Mdot_rest*c^2 / (ledd_coefficient * M_BH/Msun)",
                "x_galacticus": "Mdot_rest / (galacticus_mdot_edd_per_mbh * M_BH)",
                "fraction_adaf": "[1+exp(ln(x/xmin)/width)]^-1 + [1+exp(-ln(x/xmax)/width)]^-1",
            },
        },
        "selected_outputs": [
            {
                "target_redshift": snapshot.target_redshift,
                "output_name": snapshot.output_name,
                "redshift": snapshot.redshift,
                "diagnostics": dict(snapshot.diagnostics),
            }
            for snapshot in snapshots
        ],
        "pooled_comparison": {
            "output_name": pooled_snapshot.output_name,
            "constituent_outputs": [snapshot.output_name for snapshot in snapshots],
            "redshift_range": [
                pooled_snapshot.diagnostics["redshift_min"],
                pooled_snapshot.diagnostics["redshift_max"],
            ],
            "statistical_note": "descriptive Paper-II-style pool; snapshots share merger trees and are correlated",
        },
        "catalog_rows": int(len(catalog)),
        "observational_reference_metrics": {
            "star_forming_slope": 0.14,
            "quiescent_slope": -1.11,
            "separation_at_log10_mass_11": 1.27,
        },
        "metrics": [{key: _json_value(value) for key, value in record.items()} for record in metrics],
    }
    metrics_path = args.output_dir / "metrics_and_provenance.json"
    metrics_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    readme_path = args.output_dir / "README.md"
    output_files = [
        catalog_path,
        summary_path,
        ssfr_summary_path,
        quiescent_fraction_path,
        metrics_path,
        readme_path,
        *figure_paths,
    ]
    _write_readme(
        readme_path,
        args=args,
        disk_parameters=disk_parameters,
        disk_parameter_source=disk_parameter_source,
        snapshots=snapshots,
        pooled_snapshot=pooled_snapshot,
        metrics=metrics,
        files=output_files,
    )
    print(f"Wrote {len(catalog):,} catalogue rows and {len(summary):,} fraction rows to {args.output_dir}")
    for path in output_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
