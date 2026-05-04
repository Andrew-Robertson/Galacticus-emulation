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
from run_mass_function_observable_emulator_mcmc import INPUT_COLUMNS, INPUT_PARAMETER_SPECS


DEFAULT_STYLE = {
    "SMF pair": {"color": "#1b9e77", "label": "SMF pair"},
    "SFRF": {"color": "#d95f02", "label": "SFR function"},
    "Size pair": {"color": "#7570b3", "label": "Size pair"},
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


LOG10_PLOTTED_COLUMNS = {
    spec.short_name for spec in INPUT_PARAMETER_SPECS if isinstance(spec.prior, TruncatedLogNormalPrior)
}
PLOT_PARAMETER_NAMES = [
    f"log10_{name}" if name in LOG10_PLOTTED_COLUMNS else name for name in INPUT_COLUMNS
]
PLOT_PARAMETER_LABELS = [
    rf"\log_{{10}}({name})" if name in LOG10_PLOTTED_COLUMNS else name for name in INPUT_COLUMNS
]
SPEC_BY_NAME = {spec.short_name: spec for spec in INPUT_PARAMETER_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay observable-emulator MCMC posterior samples with GetDist."
    )
    parser.add_argument(
        "--posterior",
        action="append",
        required=True,
        help="Case in the form label=/path/to/posterior_samples.csv. Can be supplied multiple times.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--filled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width-inch", type=float, default=18.0)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument(
        "--marker-summary",
        type=Path,
        default=None,
        help="Optional run_summary.json file containing a best_theta dict to mark on the corner plot.",
    )
    return parser.parse_args()


def _parse_case(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected label=path format, got: {text}")
    label, path_text = text.split("=", 1)
    return label.strip(), Path(path_text.strip()).expanduser().resolve()


def _transform_samples_for_plotting(samples: np.ndarray) -> np.ndarray:
    transformed = np.asarray(samples, dtype=float).copy()
    for index, name in enumerate(INPUT_COLUMNS):
        if name not in LOG10_PLOTTED_COLUMNS:
            continue
        if np.any(transformed[:, index] <= 0.0):
            raise ValueError(f"Cannot log10-transform non-positive samples for parameter '{name}'")
        transformed[:, index] = np.log10(transformed[:, index])
    return transformed


def _transform_parameter_dict_for_plotting(values: dict[str, float]) -> dict[str, float]:
    transformed = {}
    for name in INPUT_COLUMNS:
        value = float(values[name])
        if name in LOG10_PLOTTED_COLUMNS:
            if value <= 0.0:
                raise ValueError(f"Cannot log10-transform non-positive marker value for parameter '{name}'")
            transformed[f"log10_{name}"] = float(np.log10(value))
        else:
            transformed[name] = value
    return transformed


def _ranges(samples_by_case: dict[str, np.ndarray]) -> dict[str, list[float]]:
    stacked = np.vstack(list(samples_by_case.values()))
    ranges = {}
    for index, plot_name in enumerate(PLOT_PARAMETER_NAMES):
        parameter_name = INPUT_COLUMNS[index]
        prior = SPEC_BY_NAME[parameter_name].prior

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

        if parameter_name not in LOG10_PLOTTED_COLUMNS:
            sample_lower = float(np.min(stacked[:, index]))
            sample_upper = float(np.max(stacked[:, index]))
        else:
            sample_lower = float(np.min(stacked[:, index]))
            sample_upper = float(np.max(stacked[:, index]))

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


def _hard_plot_bounds(parameter_name: str) -> tuple[float | None, float | None]:
    prior = SPEC_BY_NAME[parameter_name].prior
    if isinstance(prior, UniformPrior):
        return float(prior.lower), float(prior.upper)
    if isinstance(prior, TruncatedNormalPrior):
        upper = float(prior.upper) if np.isfinite(prior.upper) else None
        return float(prior.lower), upper
    if isinstance(prior, TruncatedLogNormalPrior):
        return float(np.log10(prior.lower)), float(np.log10(prior.upper))
    return None, None


def _prior_density_in_plot_space(parameter_name: str, grid: np.ndarray) -> np.ndarray:
    spec = SPEC_BY_NAME[parameter_name]
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


def _overlay_priors_and_rotate_labels(plotter, ranges: dict[str, list[float]]) -> None:
    subplots = getattr(plotter, "subplots", None)
    if subplots is None:
        return
    n_param = len(INPUT_COLUMNS)
    for index, parameter_name in enumerate(INPUT_COLUMNS):
        axis = subplots[index, index]
        if axis is None:
            continue
        lower, upper = ranges[PLOT_PARAMETER_NAMES[index]]
        grid = np.linspace(lower, upper, 512)
        prior_density = _prior_density_in_plot_space(parameter_name, grid)
        prior_peak = np.max(prior_density)
        if prior_peak > 0.0:
            prior = SPEC_BY_NAME[parameter_name].prior
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
    samples_by_case = {}
    for text in args.posterior:
        label, path = _parse_case(text)
        df = pd.read_csv(path)
        samples_by_case[label] = _transform_samples_for_plotting(df[INPUT_COLUMNS].to_numpy(dtype=float))
        case_order.append(label)

    marker_values = None
    if args.marker_summary is not None:
        summary = json.loads(args.marker_summary.expanduser().resolve().read_text())
        if "best_theta" not in summary:
            raise KeyError(f"{args.marker_summary} does not contain 'best_theta'")
        marker_values = _transform_parameter_dict_for_plotting(summary["best_theta"])

    ranges = _ranges(samples_by_case)
    roots = []
    contour_colors = []
    line_args = []
    legend_labels = []
    for case_index, case_name in enumerate(case_order):
        style = _style_for_case(case_name, case_index)
        roots.append(
            MCSamples(
                samples=samples_by_case[case_name],
                names=PLOT_PARAMETER_NAMES,
                labels=PLOT_PARAMETER_LABELS,
                label=style.get("label", case_name),
                ranges=ranges,
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
        params=PLOT_PARAMETER_NAMES,
        filled=args.filled,
        contour_colors=contour_colors,
        legend_labels=legend_labels,
        line_args=line_args,
        markers=marker_values,
        marker_args={"color": "0.25", "lw": 1.2, "ls": "--"},
    )
    _overlay_priors_and_rotate_labels(plotter, ranges)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {}
    if args.dpi is not None:
        export_kwargs["dpi"] = args.dpi
    plotter.export(str(args.output_path), **export_kwargs)
    print(args.output_path)


if __name__ == "__main__":
    main()
