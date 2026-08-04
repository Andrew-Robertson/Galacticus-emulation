from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from galacticus_emu.lhs import (
    NormalPrior,
    TruncatedLogNormalPrior,
    TruncatedNormalPrior,
    UniformPrior,
)
from galacticus_emu.mcmc_results import load_posterior_frame
from galacticus_emu.specs import trinity_parameter_specs


DEFAULT_PARAMETERS = [
    "diskVelocityCharacteristic",
    "diskExponent",
    "coolingMultiplier",
    "diskSFRFrequencyNorm",
    "stellarPopulationMetalYield",
]

DEFAULT_CASES = [
    (
        "SMF z0+z3",
        [
            "smf_z0_z3/mcmc_smf_z0_z3_mcmc_results.hdf5",
            "smf_z0_z3/mcmc_smf_z0_z3_posterior_samples.csv",
        ],
        "#1f77b4",
    ),
    (
        "SFRF",
        [
            "sfr_function_robotham2011/mcmc_sfr_function_robotham2011_mcmc_results.hdf5",
            "sfr_function_robotham2011/mcmc_sfr_function_robotham2011_posterior_samples.csv",
        ],
        "#ff7f0e",
    ),
    (
        "Sizes",
        [
            "size_mass_vdw2014_sf_q/mcmc_size_mass_vdw2014_sf_q_mcmc_results.hdf5",
            "size_mass_vdw2014_sf_q/mcmc_size_mass_vdw2014_sf_q_posterior_samples.csv",
        ],
        "#2ca02c",
    ),
    (
        "BH-halo",
        [
            "bh_velocity_dispersion/mcmc_bh_velocity_dispersion_mcmc_results.hdf5",
            "bh_halo_mass_trinity_z1_mhalo12_13p8/mcmc_bh_halo_mass_trinity_z1_mhalo12_13p8_posterior_samples.csv",
        ],
        "#d62728",
    ),
    (
        "MZR",
        [
            "mzr_blanc2019/mcmc_mzr_blanc2019_mcmc_results.hdf5",
            "mzr_blanc2019/mcmc_mzr_blanc2019_posterior_samples.csv",
        ],
        "#9467bd",
    ),
    (
        "Combined",
        [
            "all_standard_observables/mcmc_all_standard_observables_mcmc_results.hdf5",
            "all_standard_observables/mcmc_all_standard_observables_posterior_samples.csv",
        ],
        "#000000",
    ),
]


LABELS = {
    "diskVelocityCharacteristic": r"\log_{10}(V_{\rm out,d}/\mathrm{km\,s^{-1}})",
    "diskExponent": r"\alpha_{\rm out,d}",
    "coolingMultiplier": r"\log_{10} f_{\rm cool}",
    "diskSFRFrequencyNorm": r"\log_{10}(\nu_{\rm SF,d}/\mathrm{Gyr}^{-1})",
    "spheroidSFREfficiency": r"\log_{10}\epsilon_{\star,{\rm sph}}",
    "stellarPopulationMetalYield": r"\log_{10}p_Z",
}


@dataclass(frozen=True)
class PlotParameter:
    name: str
    label: str
    scale: float
    log10: bool


@dataclass(frozen=True)
class Case:
    label: str
    path: Path
    color: str
    samples: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make a reduced corner plot showing how standard-observable MCMCs "
            "constrain a selected subset of Galacticus parameters."
        )
    )
    parser.add_argument(
        "mcmc_root",
        type=Path,
        help="Path to a campaign MCMCs/standard_observables directory.",
    )
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        help=(
            "Parameter short name to include. Repeat to set the full plotting order. "
            "Defaults to the five-parameter disk-feedback/cooling/yield subset."
        ),
    )
    parser.add_argument(
        "--linear-parameter",
        action="append",
        default=[],
        help="Truncated-lognormal parameter to plot linearly rather than in log10 space.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output prefix. If relative, it is interpreted under mcmc_root. "
            "PNG, PDF, and provenance JSON files are written."
        ),
    )
    parser.add_argument(
        "--combined-posterior",
        type=Path,
        default=None,
        help=(
            "Optional explicit all-standard posterior path. This is useful when "
            "single-observable chains are written to a new output root but the "
            "joint all-standard chain should be reused from an existing pipeline run."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=7000)
    parser.add_argument("--grid-size", type=int, default=72)
    parser.add_argument(
        "--kde-bandwidth-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to scipy's default Gaussian KDE bandwidth.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--show-priors", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _plot_parameters(parameter_names: list[str], linear_parameters: set[str]) -> list[PlotParameter]:
    spec_by_name = {spec.short_name: spec for spec in trinity_parameter_specs()}
    missing = sorted(set(parameter_names).difference(spec_by_name))
    if missing:
        raise ValueError(f"Unknown parameter(s): {missing}")
    result = []
    for name in parameter_names:
        prior = spec_by_name[name].prior
        log10 = isinstance(prior, TruncatedLogNormalPrior) and name not in linear_parameters
        scale = 1.0e9 if name == "diskSFRFrequencyNorm" else 1.0
        result.append(
            PlotParameter(
                name=name,
                label=LABELS.get(name, rf"\log_{{10}} {name}" if log10 else name),
                scale=scale,
                log10=log10,
            )
        )
    return result


def _transform_values(values: np.ndarray, parameter: PlotParameter) -> np.ndarray:
    transformed = np.asarray(values, dtype=float) * parameter.scale
    if parameter.log10:
        transformed = np.where(transformed > 0.0, np.log10(transformed), np.nan)
    return transformed


def _resolve_existing_path(mcmc_root: Path, relative_paths: list[str]) -> Path:
    candidates = [mcmc_root / relative_path for relative_path in relative_paths]
    for path in candidates:
        if path.exists():
            return path
    rendered = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of the candidate posterior paths exist:\n  {rendered}")


def _load_case(
    label: str,
    path: Path,
    color: str,
    parameters: list[PlotParameter],
    rng: np.random.Generator,
    max_samples: int,
) -> Case:
    frame = load_posterior_frame(path)
    missing = [parameter.name for parameter in parameters if parameter.name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing parameter column(s): {missing}")
    columns = [_transform_values(frame[parameter.name].to_numpy(dtype=float), parameter) for parameter in parameters]
    samples = np.column_stack(columns)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if max_samples > 0 and samples.shape[0] > max_samples:
        indices = rng.choice(samples.shape[0], size=max_samples, replace=False)
        samples = samples[np.sort(indices)]
    return Case(label=label, path=path, color=color, samples=samples)


def _default_output_prefix(mcmc_root: Path, n_parameters: int) -> Path:
    return mcmc_root / f"standard_observables_degeneracy_zoom_{n_parameters}param"


def _parameter_ranges(cases: list[Case], n_parameters: int) -> list[tuple[float, float]]:
    stacked = np.vstack([case.samples for case in cases])
    ranges = []
    for index in range(n_parameters):
        lower, upper = np.percentile(stacked[:, index], [0.5, 99.5])
        width = upper - lower
        if not np.isfinite(width) or width <= 0.0:
            width = 1.0
        ranges.append((float(lower - 0.06 * width), float(upper + 0.06 * width)))
    return ranges


def _credible_levels(density: np.ndarray, probabilities: tuple[float, ...] = (0.68, 0.95)) -> list[float]:
    values = np.asarray(density, dtype=float).ravel()
    values = values[np.isfinite(values)]
    values = values[values > 0.0]
    if values.size == 0:
        return []
    ordered = np.sort(values)[::-1]
    cdf = np.cumsum(ordered)
    cdf /= cdf[-1]
    levels = []
    for probability in probabilities:
        index = min(int(np.searchsorted(cdf, probability)), ordered.size - 1)
        levels.append(float(ordered[index]))
    return sorted(set(levels))


def _kde(values: np.ndarray, bandwidth_scale: float) -> gaussian_kde:
    kde = gaussian_kde(values)
    if bandwidth_scale != 1.0:
        kde.set_bandwidth(kde.factor * bandwidth_scale)
    return kde


def _plot_1d_density(
    axis: plt.Axes,
    samples: np.ndarray,
    plot_range: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    bandwidth_scale: float,
) -> None:
    x = np.linspace(plot_range[0], plot_range[1], 256)
    try:
        density = _kde(samples, bandwidth_scale)(x)
    except Exception:
        axis.hist(samples, bins=50, range=plot_range, density=True, histtype="step", color=color, lw=linewidth)
        return
    axis.plot(x, density, color=color, lw=linewidth)


def _plot_2d_contours(
    axis: plt.Axes,
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    grid_size: int,
    bandwidth_scale: float,
) -> None:
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    values = np.vstack([samples_x, samples_y])
    try:
        density = _kde(values, bandwidth_scale)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except Exception:
        hist, x_edges, y_edges = np.histogram2d(samples_x, samples_y, bins=grid_size, range=[x_range, y_range])
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        xx, yy = np.meshgrid(x, y)
        density = hist.T
    levels = _credible_levels(density)
    if levels:
        axis.contour(xx, yy, density, levels=levels, colors=[color], linewidths=linewidth)


def _prior_density(parameter: PlotParameter, x: np.ndarray) -> np.ndarray:
    spec = {spec.short_name: spec for spec in trinity_parameter_specs()}[parameter.name]
    prior = spec.prior
    if parameter.log10:
        raw = 10.0**x / parameter.scale
        log_density = prior.logpdf(raw) + math.log(math.log(10.0)) + np.log(raw)
        density = np.exp(log_density)
    else:
        raw = x / parameter.scale
        density = np.exp(prior.logpdf(raw)) / parameter.scale
    density[~np.isfinite(density)] = 0.0
    return density


def _draw_prior(axis: plt.Axes, parameter: PlotParameter, plot_range: tuple[float, float]) -> None:
    x = np.linspace(plot_range[0], plot_range[1], 256)
    density = _prior_density(parameter, x)
    peak = float(np.max(density))
    if peak <= 0.0:
        return
    y_top = axis.get_ylim()[1]
    axis.plot(x, density / peak * 0.35 * y_top, color="0.55", lw=1.1, ls="--", zorder=0)


def make_plot(
    cases: list[Case],
    parameters: list[PlotParameter],
    output_prefix: Path,
    *,
    grid_size: int,
    dpi: int,
    show_priors: bool,
    bandwidth_scale: float,
) -> None:
    n_parameters = len(parameters)
    ranges = _parameter_ranges(cases, n_parameters)
    fig, axes = plt.subplots(
        n_parameters,
        n_parameters,
        figsize=(1.85 * n_parameters, 1.85 * n_parameters),
        constrained_layout=False,
    )

    for row in range(n_parameters):
        for column in range(n_parameters):
            axis = axes[row, column]
            if row < column:
                axis.axis("off")
                continue
            if row == column:
                for case in cases:
                    linewidth = 2.2 if case.label == "Combined" else 1.45
                    _plot_1d_density(
                        axis,
                        case.samples[:, column],
                        ranges[column],
                        color=case.color,
                        linewidth=linewidth,
                        bandwidth_scale=bandwidth_scale,
                    )
                if show_priors:
                    _draw_prior(axis, parameters[column], ranges[column])
            else:
                for case in cases:
                    linewidth = 2.0 if case.label == "Combined" else 1.15
                    _plot_2d_contours(
                        axis,
                        case.samples[:, column],
                        case.samples[:, row],
                        ranges[column],
                        ranges[row],
                        color=case.color,
                        linewidth=linewidth,
                        grid_size=grid_size,
                        bandwidth_scale=bandwidth_scale,
                    )

            axis.set_xlim(*ranges[column])
            if row != column:
                axis.set_ylim(*ranges[row])
            axis.tick_params(labelsize=7, length=2.5, pad=1.5)
            if row < n_parameters - 1:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(f"${parameters[column].label}$", fontsize=10)
            if column > 0:
                axis.set_yticklabels([])
            elif row > 0:
                axis.set_ylabel(f"${parameters[row].label}$", fontsize=10)

    handles = [
        Line2D(
            [0],
            [0],
            color=case.color,
            lw=2.4 if case.label == "Combined" else 1.7,
            label=case.label,
        )
        for case in cases
    ]
    if show_priors:
        handles.append(Line2D([0], [0], color="0.55", lw=1.1, ls="--", label="Prior"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.105, right=0.88, bottom=0.09, top=0.965, wspace=0.08, hspace=0.08)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def write_provenance(output_prefix: Path, cases: list[Case], parameters: list[PlotParameter], args: argparse.Namespace) -> None:
    provenance = {
        "mcmc_root": str(args.mcmc_root.expanduser().resolve()),
        "parameters": [
            {
                "name": parameter.name,
                "label": parameter.label,
                "scale": parameter.scale,
                "log10": parameter.log10,
            }
            for parameter in parameters
        ],
        "cases": [
            {
                "label": case.label,
                "path": str(case.path),
                "color": case.color,
                "n_plot_samples": int(case.samples.shape[0]),
            }
            for case in cases
        ],
        "max_samples": int(args.max_samples),
        "grid_size": int(args.grid_size),
        "seed": int(args.seed),
        "show_priors": bool(args.show_priors),
        "kde_bandwidth_scale": float(args.kde_bandwidth_scale),
    }
    output_prefix.with_name(output_prefix.name + "_provenance.json").write_text(json.dumps(provenance, indent=2))


def main() -> None:
    args = parse_args()
    mcmc_root = args.mcmc_root.expanduser().resolve()
    parameter_names = args.parameter or DEFAULT_PARAMETERS
    parameters = _plot_parameters(parameter_names, set(args.linear_parameter))
    rng = np.random.default_rng(args.seed)
    cases = []
    for label, relative_paths, color in DEFAULT_CASES:
        if label == "Combined" and args.combined_posterior is not None:
            path = args.combined_posterior.expanduser().resolve()
        else:
            path = _resolve_existing_path(mcmc_root, relative_paths)
        cases.append(_load_case(label, path, color, parameters, rng, args.max_samples))
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = _default_output_prefix(mcmc_root, len(parameters))
    elif not output_prefix.is_absolute():
        output_prefix = mcmc_root / output_prefix
    output_prefix = output_prefix.expanduser().resolve()
    make_plot(
        cases,
        parameters,
        output_prefix,
        grid_size=args.grid_size,
        dpi=args.dpi,
        show_priors=args.show_priors,
        bandwidth_scale=args.kde_bandwidth_scale,
    )
    write_provenance(output_prefix, cases, parameters, args)
    print(output_prefix.with_suffix(".png"))
    print(output_prefix.with_suffix(".pdf"))
    print(output_prefix.with_name(output_prefix.name + "_provenance.json"))


if __name__ == "__main__":
    main()
