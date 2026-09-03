from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from galacticus_emu.observable_plot_metadata import (
    DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET,
    HALPHA_LF_X_AXIS_LABEL,
    HALPHA_LF_Y_AXIS_LABEL,
    STANDARD_OBSERVABLE_PLOT_METADATA,
    standard_observable_y_display_offset,
)


RUN_DIR = Path(
    "runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced/"
    "automatedPipeline_transformedParams_finalPaper_definitive/"
    "MCMCs_production_from_exploratory_MAP/"
    "standard_observables_plus_emission_line_lfs/"
    "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"
)

RUN_PREFIX = "mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"


@dataclass(frozen=True)
class PanelSpec:
    observable_key: str
    analysis: str
    title: str
    panel_tag: str | None
    panel_tag_location: str
    xlabel: str
    ylabel: str


STANDARD_PANELS = [
    PanelSpec(
        "smf_z0",
        "massFunctionStellarTomczak2014ZFOURGEz0",
        "Stellar mass function",
        r"$0.20<z<0.50$",
        "upper right",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$\log_{10}[(\mathrm{d}n/\mathrm{d}\log_{10}M_\star)/\mathrm{Mpc}^{-3}]$",
    ),
    PanelSpec(
        "smf_z3",
        "massFunctionStellarTomczak2014ZFOURGEz3",
        "Stellar mass function",
        r"$1.00<z<1.25$",
        "upper right",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$\log_{10}[(\mathrm{d}n/\mathrm{d}\log_{10}M_\star)/\mathrm{Mpc}^{-3}]$",
    ),
    PanelSpec(
        "size_mass_vdw2014_star_forming_z0",
        "stellarSizeMassRelationvanDerWel2014Sample1",
        "Star-forming galaxy sizes",
        r"$0<z<0.5$",
        "upper left",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$\log_{10}(R_{\mathrm{eff}}/\mathrm{kpc})$",
    ),
    PanelSpec(
        "size_mass_vdw2014_quiescent_z0",
        "stellarSizeMassRelationvanDerWel2014Sample7",
        "Quiescent galaxy sizes",
        r"$0<z<0.5$",
        "upper left",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$\log_{10}(R_{\mathrm{eff}}/\mathrm{kpc})$",
    ),
    PanelSpec(
        "sfr_function_robotham2011",
        "starFormationRateFunctionRobotham2011",
        "Star formation rate function",
        r"$0.013<z<0.1$",
        "upper right",
        r"$\log_{10}(\dot{M}_\star/(M_\odot\,\mathrm{yr}^{-1}))$",
        r"$\log_{10}[(\mathrm{d}n/\mathrm{d}\log_{10}\dot{M}_\star)/\mathrm{Mpc}^{-3}]$",
    ),
    PanelSpec(
        "bh_velocity_dispersion",
        "blackHoleVelocityDispersionRelation",
        "Black hole mass-velocity dispersion",
        r"$z\approx0$",
        "upper left",
        r"$\log_{10}(\sigma_{\star,\mathrm{spheroid}}/\mathrm{km\,s}^{-1})$",
        r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$",
    ),
    PanelSpec(
        "mzr_blanc2019",
        "massMetallicityBlanc2019",
        "Mass-metallicity relation",
        r"$z\approx0$",
        "upper left",
        r"$\log_{10}(M_\star/M_\odot)$",
        r"$12+\log_{10}(\mathrm{O/H})$",
    ),
]

HALPHA_TITLES = {
    "halpha_sobral_z1": r"H$\alpha$ luminosity function",
    "halpha_sobral_z2": r"H$\alpha$ luminosity function",
    "halpha_sobral_z3": r"H$\alpha$ luminosity function",
    "halpha_sobral_z4": r"H$\alpha$ luminosity function",
}

HALPHA_PANEL_TAGS = {
    "halpha_sobral_z1": r"$z\approx0.40$",
    "halpha_sobral_z2": r"$z\approx0.84$",
    "halpha_sobral_z3": r"$z\approx1.47$",
    "halpha_sobral_z4": r"$z\approx2.23$",
}

HALPHA_SAMPLE_LABELS = {
    "halpha_sobral_z1": "Z1",
    "halpha_sobral_z2": "Z2",
    "halpha_sobral_z3": "Z3",
    "halpha_sobral_z4": "Z4",
}

STANDARD_X_DISPLAY_OFFSETS = {
    "sfr_function_robotham2011": -9.0,
}

STANDARD_X_TICKS = {
    "size_mass_vdw2014_star_forming_z0": [9.5, 10.0, 10.5],
}

HALPHA_Y_TICKS = {
    "halpha_sobral_z1": [-4, -3, -2],
    "halpha_sobral_z2": [-5, -4, -3, -2],
    "halpha_sobral_z3": [-5, -4, -3, -2],
    "halpha_sobral_z4": [-5, -4, -3, -2],
}

HALPHA_Y_LIMITS = {
    "halpha_sobral_z2": (-5.3, None),
}

HALPHA_X_TICKS = {
    "halpha_sobral_z4": [41.5, 42.0, 42.5, 43.0],
}

STANDARD_TARGET_LEGEND_LOCS = {
    "smf_z0": "lower left",
    "smf_z3": "lower left",
    "size_mass_vdw2014_star_forming_z0": "lower right",
    "size_mass_vdw2014_quiescent_z0": "lower right",
    "sfr_function_robotham2011": "lower left",
    "bh_velocity_dispersion": "lower right",
    "mzr_blanc2019": "lower right",
}

DATA_COLOR = "0.12"
EMULATOR_COLOR = "#1f77b4"
ACTUAL_COLOR = "0.02"
UNIT_COLOR = "#ff7f0e"
UNIT_LINESTYLE = ":"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make paper-style final MAP validation figures.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / RUN_DIR,
        help="Final joint calibration run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to RUN_DIR/paperFigures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.15,
        help="Multiply all paper-figure font sizes by this factor.",
    )
    parser.add_argument(
        "--standard-figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(7.3, 10.5),
        help="Standard-observable figure size in inches.",
    )
    parser.add_argument(
        "--halpha-figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(7.3, 5.45),
        help="Halpha figure size in inches.",
    )
    parser.add_argument(
        "--halpha-layout",
        choices=("2x2", "4x1"),
        default="2x2",
        help="Panel layout for the Halpha luminosity-function figure.",
    )
    parser.add_argument(
        "--combined-figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(10.95, 10.5),
        help="Combined standard-plus-Halpha figure size in inches.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended before file extensions, e.g. _columnDraft.",
    )
    parser.add_argument(
        "--unit-run-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional UNIT direct-run directory to include in the N-body overlay. "
            "Can be repeated. Defaults to all completed unit_* runs under "
            "RUN_DIR/bestFitModels_UNIT10_array."
        ),
    )
    parser.add_argument(
        "--unit-label",
        default="Galacticus MAP (N-body)",
        help="Legend label for the UNIT direct-run overlay.",
    )
    return parser.parse_args()


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _configure_matplotlib(font_scale: float) -> None:
    scale = float(font_scale)
    label_title_scale = 1.15
    mpl.rcParams.update(
        {
            "font.size": 8.5 * scale,
            "axes.titlesize": 9.5 * scale * label_title_scale,
            "axes.labelsize": 8.5 * scale * label_title_scale,
            "xtick.labelsize": 7.5 * scale,
            "ytick.labelsize": 7.5 * scale,
            "legend.fontsize": 8.0 * scale,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "mathtext.default": "regular",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _actual_standard_panel_data(actual_hdf5: Path, panel: PanelSpec) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(actual_hdf5, "r") as handle:
        group = handle[f"/analyses/{panel.analysis}"]
        attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
        x = np.asarray(group[attrs["xDataset"]][...], dtype=float)
        y = np.asarray(group[attrs["yDataset"]][...], dtype=float)
    return x, y, attrs


def _display_standard_panel(
    x: np.ndarray,
    y: np.ndarray,
    attrs: dict,
    panel: PanelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    x_display = np.asarray(x, dtype=float)
    y_display = np.asarray(y, dtype=float)
    if bool(attrs.get("xAxisIsLog", False)):
        x_display = np.log10(x_display)
    if bool(attrs.get("yAxisIsLog", False)):
        y_display = np.log10(np.maximum(y_display, 1.0e-300))
        y_display = y_display + DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET
    else:
        y_display = y_display + standard_observable_y_display_offset(panel.observable_key)
    return x_display, y_display


def _actual_standard_panel(actual_hdf5: Path, panel: PanelSpec) -> tuple[np.ndarray, np.ndarray]:
    x, y, attrs = _actual_standard_panel_data(actual_hdf5, panel)
    return _display_standard_panel(x, y, attrs, panel)


def _combined_standard_panel(actual_hdf5s: list[Path], panel: PanelSpec) -> tuple[np.ndarray, np.ndarray] | None:
    if not actual_hdf5s:
        return None
    x_reference: np.ndarray | None = None
    attrs_reference: dict | None = None
    y_values = []
    for actual_hdf5 in actual_hdf5s:
        x, y, attrs = _actual_standard_panel_data(actual_hdf5, panel)
        if x_reference is None:
            x_reference = x
            attrs_reference = attrs
        elif x.shape != x_reference.shape or not np.allclose(x, x_reference, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{actual_hdf5} has incompatible x bins for {panel.analysis}")
        y_values.append(y)
    assert x_reference is not None
    assert attrs_reference is not None
    y_mean = np.nanmean(np.stack(y_values, axis=0), axis=0)
    return _display_standard_panel(x_reference, y_mean, attrs_reference, panel)


def _finite_limits(*arrays: np.ndarray, fractional_margin: float = 0.08) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    lower = float(np.min(values))
    upper = float(np.max(values))
    margin = fractional_margin * (upper - lower if upper > lower else 1.0)
    return lower - margin, upper + margin


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    axis.tick_params(length=3.0, width=0.8)
    for spine in axis.spines.values():
        spine.set_color("0.2")


def _add_panel_tag(axis: plt.Axes, text: str | None, location: str) -> None:
    if not text:
        return
    if location == "upper right":
        x, ha = 0.935, "right"
    elif location == "upper left":
        x, ha = 0.065, "left"
    else:
        raise ValueError(f"Unknown panel-tag location: {location}")
    axis.text(
        x,
        0.885,
        text,
        transform=axis.transAxes,
        ha=ha,
        va="top",
        fontsize=mpl.rcParams["font.size"],
        color="0.1",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
        zorder=6,
    )


def _legend_handles(*, include_unit: bool, unit_label: str) -> list[object]:
    handles: list[object] = [
        Line2D([], [], marker="o", linestyle="None", color=DATA_COLOR, markersize=4.5, label="observations"),
        Line2D([], [], color=EMULATOR_COLOR, lw=1.7, label="emulator MAP"),
        Patch(facecolor=EMULATOR_COLOR, alpha=0.18, edgecolor="none", label=r"emulator $1\sigma$"),
        Line2D([], [], color=ACTUAL_COLOR, lw=1.7, ls="--", label="Galacticus MAP (EPS)"),
    ]
    if include_unit:
        handles.append(Line2D([], [], color=UNIT_COLOR, lw=1.9, ls=UNIT_LINESTYLE, label=unit_label))
    return handles


def _default_unit_run_dirs(run_dir: Path) -> list[Path]:
    unit_root = run_dir / "bestFitModels_UNIT10_array"
    if not unit_root.exists():
        return []
    return [
        path
        for path in sorted(unit_root.glob("unit_[0-9][0-9][0-9]_[0-9][0-9][0-9]"))
        if path.is_dir() and (_unit_standard_hdf5(path) is not None or _unit_halpha_csv(path) is not None)
    ]


def _unit_standard_hdf5(unit_run_dir: Path | None) -> Path | None:
    if unit_run_dir is None:
        return None
    path = unit_run_dir / "romanUNIT_reduced.hdf5"
    return path if path.exists() else None


def _unit_halpha_csv(unit_run_dir: Path | None) -> Path | None:
    if unit_run_dir is None:
        return None
    path = unit_run_dir / "emission_line_dust" / "emission_line_dust_lf_long.csv"
    return path if path.exists() else None


def _unit_standard_hdf5s(unit_run_dirs: list[Path]) -> list[Path]:
    return [path for run_dir in unit_run_dirs if (path := _unit_standard_hdf5(run_dir)) is not None]


def _unit_halpha_csvs(unit_run_dirs: list[Path]) -> list[Path]:
    return [path for run_dir in unit_run_dirs if (path := _unit_halpha_csv(run_dir)) is not None]


def _plot_standard_axis(
    axis: plt.Axes,
    table: pd.DataFrame,
    actual_hdf5: Path,
    unit_hdf5s: list[Path],
    panel: PanelSpec,
) -> None:
    sub = table[
        (table["observable_key"] == panel.observable_key) & (table["analysis"] == panel.analysis)
    ].sort_values("bin_index_after_mask")
    x_offset = STANDARD_X_DISPLAY_OFFSETS.get(panel.observable_key, 0.0)
    x = sub["x_plot"].to_numpy(dtype=float) + x_offset
    target = sub["target_plot"].to_numpy(dtype=float)
    target_sigma = sub["target_sigma_plot"].to_numpy(dtype=float)
    prediction = sub["prediction_plot"].to_numpy(dtype=float)
    prediction_sigma = sub["prediction_sigma_plot"].to_numpy(dtype=float)
    actual_x, actual_y = _actual_standard_panel(actual_hdf5, panel)
    actual_x = actual_x + x_offset
    unit_x: np.ndarray | None = None
    unit_y: np.ndarray | None = None
    unit_panel = _combined_standard_panel(unit_hdf5s, panel)
    if unit_panel is not None:
        unit_x, unit_y = unit_panel
        unit_x = unit_x + x_offset

    target_handle = axis.errorbar(
        x,
        target,
        yerr=target_sigma,
        fmt="o",
        color=DATA_COLOR,
        ecolor=DATA_COLOR,
        elinewidth=0.8,
        capsize=0,
        ms=3.2,
        zorder=4,
    )
    axis.fill_between(
        x,
        prediction - prediction_sigma,
        prediction + prediction_sigma,
        color=EMULATOR_COLOR,
        alpha=0.18,
        lw=0,
        zorder=1,
    )
    axis.plot(x, prediction, color=EMULATOR_COLOR, lw=1.7, zorder=3)
    axis.plot(actual_x, actual_y, color=ACTUAL_COLOR, lw=1.7, ls="--", zorder=2)
    if unit_x is not None and unit_y is not None:
        axis.plot(unit_x, unit_y, color=UNIT_COLOR, lw=1.9, ls=UNIT_LINESTYLE, zorder=2.5)

    x_limit_arrays = [x, actual_x]
    y_limit_arrays = [
        target - target_sigma,
        target + target_sigma,
        prediction - prediction_sigma,
        prediction + prediction_sigma,
        actual_y,
    ]
    if unit_x is not None and unit_y is not None:
        x_limit_arrays.append(unit_x)
        y_limit_arrays.append(unit_y)
    x_lower, x_upper = _finite_limits(*x_limit_arrays, fractional_margin=0.06)
    y_lower, y_upper = _finite_limits(*y_limit_arrays, fractional_margin=0.10)
    axis.set_xlim(x_lower, x_upper)
    axis.set_ylim(y_lower, y_upper)
    if panel.observable_key in STANDARD_X_TICKS:
        axis.set_xticks(STANDARD_X_TICKS[panel.observable_key])
    axis.set_title(panel.title, pad=3)
    axis.set_xlabel(panel.xlabel)
    axis.set_ylabel(panel.ylabel)
    _add_panel_tag(axis, panel.panel_tag, panel.panel_tag_location)
    target_label = STANDARD_OBSERVABLE_PLOT_METADATA.get(panel.observable_key, {}).get("target_label")
    if target_label:
        scale = mpl.rcParams["font.size"] / 8.5
        axis.legend(
            handles=[target_handle],
            labels=[target_label],
            loc=STANDARD_TARGET_LEGEND_LOCS[panel.observable_key],
            frameon=False,
            fontsize=8.06 * scale,
            handlelength=1.0,
            borderpad=0.1,
            labelspacing=0.2,
            handletextpad=0.35,
        )
    _style_axis(axis)


def _plot_halpha_axis(
    axis: plt.Axes,
    table: pd.DataFrame,
    actual_csv: Path,
    unit_actual_csvs: list[Path],
    observable_key: str,
    title: str | None = None,
    *,
    target_label: str | None = None,
    reserve_title_space: bool = False,
) -> None:
    sub = table[table["observable_key"] == observable_key].sort_values("bin_index")
    x = sub["x_plot"].to_numpy(dtype=float)
    target = sub["target_log10_phi"].to_numpy(dtype=float)
    target_sigma = sub["target_log10_phi_std"].to_numpy(dtype=float)
    prediction = sub["prediction_log10_phi"].to_numpy(dtype=float)
    prediction_sigma = sub["prediction_log10_phi_std"].to_numpy(dtype=float)
    actual_x, actual_y = _actual_halpha(actual_csv, HALPHA_SAMPLE_LABELS[observable_key])
    unit_x: np.ndarray | None = None
    unit_y: np.ndarray | None = None
    unit_panel = _combined_halpha(unit_actual_csvs, HALPHA_SAMPLE_LABELS[observable_key])
    if unit_panel is not None:
        unit_x, unit_y = unit_panel

    target_handle = axis.errorbar(
        x,
        target,
        yerr=target_sigma,
        fmt="o",
        color=DATA_COLOR,
        ecolor=DATA_COLOR,
        elinewidth=0.8,
        capsize=0,
        ms=3.2,
        zorder=4,
    )
    axis.fill_between(
        x,
        prediction - prediction_sigma,
        prediction + prediction_sigma,
        color=EMULATOR_COLOR,
        alpha=0.18,
        lw=0,
        zorder=1,
    )
    axis.plot(x, prediction, color=EMULATOR_COLOR, lw=1.7, zorder=3)
    axis.plot(actual_x, actual_y, color=ACTUAL_COLOR, lw=1.7, ls="--", zorder=2)
    if unit_x is not None and unit_y is not None:
        axis.plot(unit_x, unit_y, color=UNIT_COLOR, lw=1.9, ls=UNIT_LINESTYLE, zorder=2.5)

    x_limit_arrays = [x, actual_x]
    y_limit_arrays = [
        target - target_sigma,
        target + target_sigma,
        prediction - prediction_sigma,
        prediction + prediction_sigma,
        actual_y,
    ]
    if unit_x is not None and unit_y is not None:
        x_limit_arrays.append(unit_x)
        y_limit_arrays.append(unit_y)
    x_lower, x_upper = _finite_limits(*x_limit_arrays, fractional_margin=0.06)
    y_lower, y_upper = _finite_limits(*y_limit_arrays, fractional_margin=0.10)
    axis.set_xlim(x_lower, x_upper)
    axis.set_ylim(y_lower, y_upper)
    if observable_key in HALPHA_Y_LIMITS:
        requested_lower, requested_upper = HALPHA_Y_LIMITS[observable_key]
        current_lower, current_upper = axis.get_ylim()
        axis.set_ylim(
            current_lower if requested_lower is None else requested_lower,
            current_upper if requested_upper is None else requested_upper,
        )
    if observable_key in HALPHA_Y_TICKS:
        axis.set_yticks(HALPHA_Y_TICKS[observable_key])
        axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%d"))
    if observable_key in HALPHA_X_TICKS:
        axis.set_xticks(HALPHA_X_TICKS[observable_key])
        axis.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    if title:
        axis.set_title(title, pad=3)
    elif reserve_title_space:
        axis.set_title(" ", pad=3)
    axis.set_xlabel(HALPHA_LF_X_AXIS_LABEL)
    axis.set_ylabel(HALPHA_LF_Y_AXIS_LABEL)
    _add_panel_tag(axis, HALPHA_PANEL_TAGS[observable_key], "upper right")
    if target_label:
        scale = mpl.rcParams["font.size"] / 8.5
        axis.legend(
            handles=[target_handle],
            labels=[target_label],
            loc="lower left",
            frameon=False,
            fontsize=8.06 * scale,
            handlelength=1.0,
            borderpad=0.1,
            labelspacing=0.2,
            handletextpad=0.35,
        )
    _style_axis(axis)


def plot_standard(
    run_dir: Path,
    output_dir: Path,
    dpi: int,
    figsize: tuple[float, float],
    output_suffix: str,
    unit_run_dirs: list[Path],
    unit_label: str,
) -> list[Path]:
    csv_path = run_dir / f"{RUN_PREFIX}_best_fit_standard_observables.csv"
    actual_hdf5 = run_dir / "bestFitModel_GalacticusRun" / "romanEPS_massFunction.hdf5"
    unit_hdf5s = _unit_standard_hdf5s(unit_run_dirs)
    table = pd.read_csv(csv_path)

    fig, axes = plt.subplots(4, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, panel in zip(axes_flat, STANDARD_PANELS, strict=False):
        _plot_standard_axis(axis, table, actual_hdf5, unit_hdf5s, panel)

    legend_axis = axes_flat[-1]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=_legend_handles(include_unit=bool(unit_hdf5s), unit_label=unit_label),
        loc="upper left",
        frameon=False,
        borderpad=0.2,
        handlelength=2.2,
        labelspacing=0.9,
    )

    stem = f"final_map_validation_standard_observables_4x2{output_suffix}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _actual_halpha(actual_csv: Path, sample_label: str) -> tuple[np.ndarray, np.ndarray]:
    x, phi = _actual_halpha_data(actual_csv, sample_label)
    return x, np.log10(np.maximum(phi, 1.0e-300))


def _actual_halpha_data(actual_csv: Path, sample_label: str) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(actual_csv)
    sub = table[
        (table["observable"] == "halpha_sobral") & (table["sample_label"].str.upper() == sample_label)
    ].sort_values("bin_index")
    x = sub["log10_luminosity_center"].to_numpy(dtype=float)
    phi = sub["phi_mpc3_dex"].to_numpy(dtype=float)
    return x, phi


def _combined_halpha(actual_csvs: list[Path], sample_label: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not actual_csvs:
        return None
    x_reference: np.ndarray | None = None
    phi_values = []
    for actual_csv in actual_csvs:
        x, phi = _actual_halpha_data(actual_csv, sample_label)
        if x_reference is None:
            x_reference = x
        elif x.shape != x_reference.shape or not np.allclose(x, x_reference, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{actual_csv} has incompatible Halpha luminosity bins for {sample_label}")
        phi_values.append(phi)
    assert x_reference is not None
    phi_mean = np.nanmean(np.stack(phi_values, axis=0), axis=0)
    return x_reference, np.log10(np.maximum(phi_mean, 1.0e-300))


def plot_halpha(
    run_dir: Path,
    output_dir: Path,
    dpi: int,
    figsize: tuple[float, float],
    layout: str,
    output_suffix: str,
    unit_run_dirs: list[Path],
    unit_label: str,
) -> list[Path]:
    csv_path = run_dir / f"{RUN_PREFIX}_best_fit_sidecar_lfs.csv"
    actual_csv = run_dir / "bestFitModel_GalacticusRun" / "emission_line_dust" / "emission_line_dust_lf_long.csv"
    unit_actual_csvs = _unit_halpha_csvs(unit_run_dirs)
    table = pd.read_csv(csv_path)

    if layout == "2x2":
        nrows, ncols = 2, 2
    elif layout == "4x1":
        nrows, ncols = 4, 1
    else:
        raise ValueError(f"Unknown Halpha layout: {layout}")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False)
    fig.suptitle(r"H$\alpha$ luminosity functions", fontsize=mpl.rcParams["axes.titlesize"] * 1.08)
    axes_by_key: dict[str, plt.Axes] = {}
    for axis, observable_key in zip(axes.ravel(), HALPHA_TITLES, strict=True):
        axes_by_key[observable_key] = axis
        _plot_halpha_axis(axis, table, actual_csv, unit_actual_csvs, observable_key)

    scale = mpl.rcParams["font.size"] / 8.5
    axes_by_key["halpha_sobral_z1"].legend(
        handles=[
            Line2D([], [], color=EMULATOR_COLOR, lw=1.7, label="emulator MAP"),
            Patch(facecolor=EMULATOR_COLOR, alpha=0.18, edgecolor="none", label=r"emulator $1\sigma$"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8.97 * scale,
        handlelength=1.6,
        borderpad=0.1,
        labelspacing=0.25,
        handletextpad=0.45,
    )
    axes_by_key["halpha_sobral_z2"].legend(
        handles=[
            Line2D([], [], marker="o", linestyle="None", color=DATA_COLOR, markersize=4.5, label="Sobral et al. (2013)"),
            Line2D([], [], color=ACTUAL_COLOR, lw=1.7, ls="--", label="Galacticus MAP (EPS)"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8.97 * scale,
        handlelength=1.6,
        borderpad=0.1,
        labelspacing=0.25,
        handletextpad=0.45,
    )
    if unit_actual_csvs:
        axes_by_key["halpha_sobral_z3"].legend(
            handles=[
                Line2D([], [], color=UNIT_COLOR, lw=1.9, ls=UNIT_LINESTYLE, label=unit_label),
            ],
            loc="lower left",
            frameon=False,
            fontsize=8.97 * scale,
            handlelength=1.6,
            borderpad=0.1,
            labelspacing=0.25,
            handletextpad=0.45,
        )

    stem = f"final_map_validation_halpha_lfs_{layout}{output_suffix}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def plot_combined(
    run_dir: Path,
    output_dir: Path,
    dpi: int,
    figsize: tuple[float, float],
    output_suffix: str,
    unit_run_dirs: list[Path],
    unit_label: str,
) -> list[Path]:
    standard_csv_path = run_dir / f"{RUN_PREFIX}_best_fit_standard_observables.csv"
    standard_actual_hdf5 = run_dir / "bestFitModel_GalacticusRun" / "romanEPS_massFunction.hdf5"
    unit_hdf5s = _unit_standard_hdf5s(unit_run_dirs)
    standard_table = pd.read_csv(standard_csv_path)

    halpha_csv_path = run_dir / f"{RUN_PREFIX}_best_fit_sidecar_lfs.csv"
    halpha_actual_csv = run_dir / "bestFitModel_GalacticusRun" / "emission_line_dust" / "emission_line_dust_lf_long.csv"
    unit_actual_csvs = _unit_halpha_csvs(unit_run_dirs)
    halpha_table = pd.read_csv(halpha_csv_path)

    fig, axes = plt.subplots(4, 3, figsize=figsize, constrained_layout=True)
    standard_axes = axes[:, :2].ravel()
    for axis, panel in zip(standard_axes, STANDARD_PANELS, strict=False):
        _plot_standard_axis(axis, standard_table, standard_actual_hdf5, unit_hdf5s, panel)

    legend_axis = standard_axes[-1]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=_legend_handles(include_unit=bool(unit_hdf5s or unit_actual_csvs), unit_label=unit_label),
        loc="center",
        bbox_to_anchor=(0.47, 0.5),
        frameon=False,
        borderpad=0.2,
        handlelength=2.2,
        labelspacing=0.9,
    )

    for row, observable_key in enumerate(HALPHA_TITLES):
        _plot_halpha_axis(
            axes[row, 2],
            halpha_table,
            halpha_actual_csv,
            unit_actual_csvs,
            observable_key,
            title=HALPHA_TITLES[observable_key],
            target_label="Sobral et al. (2013)",
        )

    stem = f"final_map_validation_all_observables_4x3{output_suffix}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "paperFigures")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib(args.font_scale)
    if args.unit_run_dir is None:
        unit_run_dirs = _default_unit_run_dirs(run_dir)
    else:
        unit_run_dirs = [path.expanduser().resolve() for path in args.unit_run_dir]
        unit_run_dirs = [
            path
            for path in unit_run_dirs
            if _unit_standard_hdf5(path) is not None or _unit_halpha_csv(path) is not None
        ]
    if unit_run_dirs:
        print(
            "Using UNIT N-body overlays from: "
            + ", ".join(path.name for path in unit_run_dirs),
            file=sys.stderr,
        )

    outputs = []
    outputs.extend(
        plot_standard(
            run_dir,
            output_dir,
            args.dpi,
            tuple(args.standard_figsize),
            args.output_suffix,
            unit_run_dirs,
            args.unit_label,
        )
    )
    outputs.extend(
        plot_halpha(
            run_dir,
            output_dir,
            args.dpi,
            tuple(args.halpha_figsize),
            args.halpha_layout,
            args.output_suffix,
            unit_run_dirs,
            args.unit_label,
        )
    )
    outputs.extend(
        plot_combined(
            run_dir,
            output_dir,
            args.dpi,
            tuple(args.combined_figsize),
            args.output_suffix,
            unit_run_dirs,
            args.unit_label,
        )
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
