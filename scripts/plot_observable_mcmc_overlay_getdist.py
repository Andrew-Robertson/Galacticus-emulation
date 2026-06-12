from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

try:
    from getdist import MCSamples, plots as gd_plots
except ModuleNotFoundError:  # pragma: no cover
    MCSamples = None
    gd_plots = None

from galacticus_emu.lhs import (
    NormalPrior,
    TruncatedLogNormalPrior,
    TruncatedNormalPrior,
    UniformPrior,
)
from galacticus_emu.mcmc_results import load_posterior_frame, load_run_summary
from galacticus_emu.specs import trinity_parameter_specs
from run_mass_function_observable_emulator_mcmc import INPUT_COLUMNS, INPUT_PARAMETER_SPECS
from fit_halpha_sobol_pca_gp_cv import DUST_INPUT_PARAMETER_SPECS


DEFAULT_STYLE = {
    "SMF pair": {"color": "#1b9e77", "label": "SMF pair"},
    "SFRF": {"color": "#d95f02", "label": "SFR function"},
    "Size pair": {"color": "#7570b3", "label": "Size pair"},
    "SMF z0+z3": {"color": "#1f77b4", "label": "SMF z0+z3"},
    "Sizes": {"color": "#2ca02c", "label": "Sizes"},
    "BH-halo": {"color": "#d62728", "label": "BH-halo"},
    "MZR": {"color": "#9467bd", "label": "MZR"},
    "Combined": {"color": "#000000", "label": "Combined", "linewidth": 2.2},
}

FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]


JOINT_INPUT_PARAMETER_SPECS = list(INPUT_PARAMETER_SPECS) + list(DUST_INPUT_PARAMETER_SPECS)
JOINT_INPUT_COLUMNS = [spec.short_name for spec in JOINT_INPUT_PARAMETER_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay observable-emulator MCMC posterior samples with GetDist."
    )
    parser.add_argument(
        "--posterior",
        action="append",
        required=True,
        help="Case in the form label=/path/to/*_mcmc_results.hdf5 or label=/path/to/posterior_samples.csv.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--filled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width-inch", type=float, default=18.0)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument(
        "--parameter-set",
        choices=["auto", "observable19", "joint24"],
        default="auto",
        help="Which parameter specification set to use when interpreting posterior columns.",
    )
    parser.add_argument(
        "--marker-summary",
        type=Path,
        default=None,
        help=(
            "Optional *_mcmc_results.hdf5 or run_summary.json file containing a best_theta dict to mark on the corner plot. "
            "This legacy form draws one neutral marker set."
        ),
    )
    parser.add_argument(
        "--posterior-marker-summary",
        action="append",
        default=[],
        help=(
            "Marker tied to a posterior label in the form label=/path/to/*_mcmc_results.hdf5 or label=/path/to/run_summary.json. "
            "Repeat to mark multiple MAP points using the matching posterior colour."
        ),
    )
    return parser.parse_args()


def _resolve_parameter_specs(parameter_set: str, posterior_columns: list[str]):
    if parameter_set == "observable19":
        return list(INPUT_PARAMETER_SPECS)
    if parameter_set == "joint24":
        return list(JOINT_INPUT_PARAMETER_SPECS)
    filtered = [column for column in posterior_columns if column != "log_probability"]
    if filtered == INPUT_COLUMNS:
        return list(INPUT_PARAMETER_SPECS)
    if filtered == JOINT_INPUT_COLUMNS:
        return list(JOINT_INPUT_PARAMETER_SPECS)
    if len(filtered) == len(JOINT_INPUT_COLUMNS) and all(column in JOINT_INPUT_COLUMNS for column in filtered):
        return [spec for spec in JOINT_INPUT_PARAMETER_SPECS if spec.short_name in filtered]
    if len(filtered) == len(INPUT_COLUMNS) and all(column in INPUT_COLUMNS for column in filtered):
        return [spec for spec in INPUT_PARAMETER_SPECS if spec.short_name in filtered]
    trinity_specs_by_name = {spec.short_name: spec for spec in trinity_parameter_specs()}
    if all(column in trinity_specs_by_name for column in filtered):
        return [trinity_specs_by_name[column] for column in filtered]
    recognized_specs_by_name = {
        spec.short_name: spec
        for spec in [*INPUT_PARAMETER_SPECS, *trinity_parameter_specs(), *DUST_INPUT_PARAMETER_SPECS]
    }
    if all(column in recognized_specs_by_name for column in filtered):
        return [recognized_specs_by_name[column] for column in filtered]
    raise ValueError(
        "Could not infer parameter set from posterior columns. "
        "Pass --parameter-set observable19 or --parameter-set joint24 explicitly if this is an older chain."
    )


def _parse_case(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected label=path format, got: {text}")
    label, path_text = text.split("=", 1)
    return label.strip(), Path(path_text.strip()).expanduser().resolve()


def _transform_samples_for_plotting(samples: np.ndarray, input_columns: list[str], log10_plotted_columns: set[str]) -> np.ndarray:
    transformed = np.asarray(samples, dtype=float).copy()
    for index, name in enumerate(input_columns):
        if name not in log10_plotted_columns:
            continue
        if np.any(transformed[:, index] <= 0.0):
            raise ValueError(f"Cannot log10-transform non-positive samples for parameter '{name}'")
        transformed[:, index] = np.log10(transformed[:, index])
    return transformed


def _transform_parameter_dict_for_plotting(values: dict[str, float], input_columns: list[str], log10_plotted_columns: set[str]) -> dict[str, float]:
    transformed = {}
    for name in input_columns:
        if name not in values:
            continue
        value = float(values[name])
        if name in log10_plotted_columns:
            if value <= 0.0:
                raise ValueError(f"Cannot log10-transform non-positive marker value for parameter '{name}'")
            transformed[f"log10_{name}"] = float(np.log10(value))
        else:
            transformed[name] = value
    return transformed


def _marker_array_from_summary(
    path: Path,
    input_columns: list[str],
    plot_parameter_names: list[str],
    log10_plotted_columns: set[str],
) -> np.ndarray:
    summary = load_run_summary(path)
    if "best_theta" not in summary:
        raise KeyError(f"{path} does not contain 'best_theta'")
    marker_values = _transform_parameter_dict_for_plotting(
        summary["best_theta"],
        input_columns,
        log10_plotted_columns,
    )
    return np.asarray([float(marker_values.get(name, np.nan)) for name in plot_parameter_names], dtype=float)


def _load_labeled_marker_summaries(
    values: list[str],
    input_columns: list[str],
    plot_parameter_names: list[str],
    log10_plotted_columns: set[str],
) -> dict[str, np.ndarray]:
    markers = {}
    for value in values:
        label, path = _parse_case(value)
        markers[label] = _marker_array_from_summary(
            path,
            input_columns,
            plot_parameter_names,
            log10_plotted_columns,
        )
    return markers


def _ranges(
    samples_by_case: dict[str, np.ndarray],
    input_columns_by_case: dict[str, list[str]],
    input_columns: list[str],
    plot_parameter_names: list[str],
    spec_by_name: dict[str, object],
) -> dict[str, list[float]]:
    ranges = {}
    for index, plot_name in enumerate(plot_parameter_names):
        parameter_name = input_columns[index]
        prior = spec_by_name[parameter_name].prior
        parameter_samples = []
        for case_name, samples in samples_by_case.items():
            case_columns = input_columns_by_case[case_name]
            if parameter_name in case_columns:
                parameter_samples.append(samples[:, case_columns.index(parameter_name)])
        if not parameter_samples:
            raise ValueError(f"No posterior contains parameter {parameter_name!r}")
        stacked = np.concatenate(parameter_samples)

        lower = None
        upper = None
        if isinstance(prior, UniformPrior):
            lower = float(prior.lower)
            upper = float(prior.upper)
        elif isinstance(prior, TruncatedNormalPrior):
            lower = float(prior.lower)
            if np.isfinite(prior.upper):
                upper = float(prior.upper)
        elif isinstance(prior, TruncatedLogNormalPrior):
            lower = float(np.log10(prior.lower))
            upper = float(np.log10(prior.upper))

        if plot_name == parameter_name:
            sample_lower = float(np.min(stacked))
            sample_upper = float(np.max(stacked))
        else:
            sample_lower = float(np.min(stacked))
            sample_upper = float(np.max(stacked))

        if lower is None:
            lower = sample_lower
        if upper is None:
            upper = sample_upper

        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        if isinstance(prior, NormalPrior):
            plot_lower = lower - margin
            plot_upper = upper + margin
        elif isinstance(prior, TruncatedNormalPrior) and not np.isfinite(prior.upper):
            plot_lower = lower
            plot_upper = upper + margin
        else:
            plot_lower = lower
            plot_upper = upper

        ranges[plot_name] = [float(plot_lower), float(plot_upper)]
    return ranges


def _hard_plot_bounds(parameter_name: str, spec_by_name: dict[str, object]) -> tuple[float | None, float | None]:
    prior = spec_by_name[parameter_name].prior
    if isinstance(prior, UniformPrior):
        return float(prior.lower), float(prior.upper)
    if isinstance(prior, TruncatedNormalPrior):
        upper = float(prior.upper) if np.isfinite(prior.upper) else None
        return float(prior.lower), upper
    if isinstance(prior, TruncatedLogNormalPrior):
        return float(np.log10(prior.lower)), float(np.log10(prior.upper))
    return None, None


def _prior_density_in_plot_space(parameter_name: str, grid: np.ndarray, spec_by_name: dict[str, object]) -> np.ndarray:
    spec = spec_by_name[parameter_name]
    prior = spec.prior
    density = np.zeros_like(grid, dtype=float)

    if isinstance(prior, UniformPrior):
        mask = np.logical_and(grid >= prior.lower, grid <= prior.upper)
        density[mask] = 1.0 / (prior.upper - prior.lower)
        return density

    if isinstance(prior, NormalPrior):
        z = (grid - prior.mean) / prior.sigma
        return np.exp(-0.5 * z**2) / (prior.sigma * math.sqrt(2.0 * math.pi))

    if isinstance(prior, TruncatedNormalPrior):
        lower_cdf = 0.5 * (1.0 + math.erf((prior.lower - prior.mean) / (prior.sigma * math.sqrt(2.0))))
        upper_cdf = 0.5 * (1.0 + math.erf((prior.upper - prior.mean) / (prior.sigma * math.sqrt(2.0))))
        norm = upper_cdf - lower_cdf
        mask = np.logical_and(grid >= prior.lower, grid <= prior.upper)
        z = (grid[mask] - prior.mean) / prior.sigma
        density[mask] = np.exp(-0.5 * z**2) / (prior.sigma * math.sqrt(2.0 * math.pi) * norm)
        return density

    if isinstance(prior, TruncatedLogNormalPrior):
        x = 10.0 ** grid
        logpdf_x = prior.logpdf(x)
        valid = np.isfinite(logpdf_x)
        if np.any(valid):
            density[valid] = np.exp(logpdf_x[valid] + math.log(math.log(10.0)) + np.log(x[valid]))
        return density

    raise TypeError(f"Unsupported prior type for {parameter_name}: {type(prior)!r}")


def _overlay_priors_and_rotate_labels(
    plotter,
    ranges: dict[str, list[float]],
    input_columns: list[str],
    plot_parameter_names: list[str],
    spec_by_name: dict[str, object],
) -> None:
    subplots = getattr(plotter, "subplots", None)
    if subplots is None:
        return
    n_param = len(input_columns)
    for index, parameter_name in enumerate(input_columns):
        axis = subplots[index, index]
        if axis is None:
            continue
        lower, upper = ranges[plot_parameter_names[index]]
        grid = np.linspace(lower, upper, 512)
        prior_density = _prior_density_in_plot_space(parameter_name, grid, spec_by_name)
        prior_peak = np.max(prior_density)
        if prior_peak > 0.0:
            prior = spec_by_name[parameter_name].prior
            y_top = axis.get_ylim()[1]
            peak_fraction = 0.25 if isinstance(prior, UniformPrior) else 1.0
            scaled_prior = prior_density / prior_peak * (peak_fraction * y_top)
            axis.plot(grid, scaled_prior, color="0.5", linestyle="--", linewidth=1.3, zorder=1)

    for row in range(n_param):
        for col in range(n_param):
            axis = subplots[row, col]
            if axis is None:
                continue
            xlabel = axis.xaxis.label
            ylabel = axis.yaxis.label
            if xlabel.get_text():
                xlabel.set_rotation(18)
                xlabel.set_ha("right")
                xlabel.set_va("top")
            if ylabel.get_text():
                ylabel.set_rotation(72)
                ylabel.set_ha("right")
                ylabel.set_va("center")


def _overlay_labeled_markers(
    plotter,
    markers_by_case: dict[str, np.ndarray],
    case_order: list[str],
) -> None:
    if not markers_by_case:
        return
    subplots = getattr(plotter, "subplots", None)
    if subplots is None:
        return
    for case_index, case_name in enumerate(case_order):
        if case_name not in markers_by_case:
            continue
        style = _style_for_case(case_name, case_index)
        color = style.get("color", FALLBACK_COLORS[case_index % len(FALLBACK_COLORS)])
        marker = np.asarray(markers_by_case[case_name], dtype=float)
        for row in range(marker.size):
            for col in range(row + 1):
                axis = subplots[row, col]
                if axis is None:
                    continue
                if np.isfinite(marker[col]):
                    axis.axvline(marker[col], color=color, lw=1.25, ls="--", alpha=0.95, zorder=20)
                if row != col and np.isfinite(marker[row]):
                    axis.axhline(marker[row], color=color, lw=1.25, ls="--", alpha=0.95, zorder=20)


def _style_for_case(case_name: str, fallback_index: int) -> dict[str, object]:
    if case_name in DEFAULT_STYLE:
        return dict(DEFAULT_STYLE[case_name])
    return {
        "color": FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)],
        "label": case_name,
        "linewidth": 1.8,
    }


def main() -> None:
    args = parse_args()
    if MCSamples is None or gd_plots is None:
        raise ModuleNotFoundError(
            "The 'getdist' package is required for plot_observable_mcmc_overlay_getdist.py."
        )

    case_order = []
    posterior_frames = {}
    union_columns = []
    for text in args.posterior:
        label, path = _parse_case(text)
        df = load_posterior_frame(path)
        posterior_frames[label] = df
        for column in df.columns:
            if column != "log_probability" and column not in union_columns:
                union_columns.append(column)
        case_order.append(label)

    parameter_specs = _resolve_parameter_specs(args.parameter_set, union_columns)
    input_columns = [spec.short_name for spec in parameter_specs]
    log10_plotted_columns = {
        spec.short_name for spec in parameter_specs if isinstance(spec.prior, TruncatedLogNormalPrior)
    }

    samples_by_case = {}
    input_columns_by_case = {}
    for label in case_order:
        df = posterior_frames[label]
        case_columns = [column for column in input_columns if column in df.columns]
        if not case_columns:
            raise ValueError(f"Posterior {label!r} contains none of the requested plotting parameters")
        log10_plotted_columns = {
            spec.short_name for spec in parameter_specs if isinstance(spec.prior, TruncatedLogNormalPrior)
        }
        samples_by_case[label] = _transform_samples_for_plotting(
            df[case_columns].to_numpy(dtype=float),
            case_columns,
            log10_plotted_columns,
        )
        input_columns_by_case[label] = case_columns

    spec_by_name = {spec.short_name: spec for spec in parameter_specs}
    log10_plotted_columns = {
        spec.short_name for spec in parameter_specs if isinstance(spec.prior, TruncatedLogNormalPrior)
    }
    plot_parameter_names = [
        f"log10_{name}" if name in log10_plotted_columns else name for name in input_columns
    ]
    plot_parameter_labels = [
        rf"\log_{{10}}({name})" if name in log10_plotted_columns else name for name in input_columns
    ]

    marker_values = None
    if args.marker_summary is not None:
        marker_dict = _transform_parameter_dict_for_plotting(
            load_run_summary(args.marker_summary)["best_theta"],
            input_columns,
            log10_plotted_columns,
        )
        marker_values = {name: float(marker_dict[name]) for name in plot_parameter_names}
    markers_by_case = _load_labeled_marker_summaries(
        args.posterior_marker_summary,
        input_columns,
        plot_parameter_names,
        log10_plotted_columns,
    )

    ranges = _ranges(samples_by_case, input_columns_by_case, input_columns, plot_parameter_names, spec_by_name)
    roots = []
    contour_colors = []
    line_args = []
    legend_labels = []
    for case_index, case_name in enumerate(case_order):
        style = _style_for_case(case_name, case_index)
        case_columns = input_columns_by_case[case_name]
        case_plot_parameter_names = [
            f"log10_{name}" if name in log10_plotted_columns else name for name in case_columns
        ]
        case_plot_parameter_labels = [
            rf"\log_{{10}}({name})" if name in log10_plotted_columns else name for name in case_columns
        ]
        roots.append(
            MCSamples(
                samples=samples_by_case[case_name],
                names=case_plot_parameter_names,
                labels=case_plot_parameter_labels,
                label=style.get("label", case_name),
                ranges={name: ranges[name] for name in case_plot_parameter_names},
                settings={"smooth_scale_1D": 0.7, "smooth_scale_2D": 0.7},
            )
        )
        contour_colors.append(style.get("color"))
        line_args.append({"lw": style.get("linewidth", 1.8), "color": style.get("color")})
        legend_labels.append(style.get("label", case_name))

    plotter = gd_plots.get_subplot_plotter(width_inch=args.width_inch)
    plotter.settings.figure_legend_frame = False
    plotter.settings.alpha_factor_contour_lines = 1.0
    plotter.settings.linewidth = 1.5
    plotter.settings.linewidth_contour = 1.6
    plotter.settings.num_plot_contours = 2
    plotter.settings.legend_fontsize = 13
    plotter.settings.axes_labelsize = 10
    plotter.settings.lab_fontsize = 10
    if args.filled:
        plotter.settings.alpha_filled_add = 0.4
        plotter.settings.solid_colors = contour_colors
    plotter.triangle_plot(
        roots,
        params=plot_parameter_names,
        filled=args.filled,
        contour_colors=contour_colors,
        legend_labels=legend_labels,
        line_args=line_args,
        markers=marker_values,
        marker_args={"color": "0.25", "lw": 1.2, "ls": "--"},
    )
    _overlay_priors_and_rotate_labels(plotter, ranges, input_columns, plot_parameter_names, spec_by_name)
    _overlay_labeled_markers(plotter, markers_by_case, case_order)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {}
    if args.dpi is not None:
        export_kwargs["dpi"] = args.dpi
    plotter.export(str(args.output_path), **export_kwargs)
    print(args.output_path)


if __name__ == "__main__":
    main()
