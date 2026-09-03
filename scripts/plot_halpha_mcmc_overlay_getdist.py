from __future__ import annotations

import argparse
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

from fit_halpha_allz_gp_cv import INPUT_COLUMNS
from galacticus_emu.mcmc_results import load_posterior_frame


DEFAULT_STYLE = {
    "Z1": {"color": "#1b9e77", "label": "z=0.40"},
    "Z2": {"color": "#d95f02", "label": "z=0.84"},
    "Z3": {"color": "#7570b3", "label": "z=1.47"},
    "Z4": {"color": "#e7298a", "label": "z=2.23"},
    "Combined": {"color": "#000000", "label": "Combined", "linewidth": 2.2},
}

DISPLAY_LABELS = {
    "diskVelocityCharacteristic": r"diskVelocityCharacteristic",
    "delta_0": r"delta_0",
    "delta_z": r"delta_z",
    "delta_M": r"delta_M",
    "delta_Mz": r"delta_Mz",
    "attenuation_scatter": r"attenuationScatter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Halpha MCMC posterior samples with GetDist."
    )
    parser.add_argument(
        "--posterior",
        action="append",
        required=True,
        help="Case in the form label=/path/to/*_mcmc_results.hdf5 or label=/path/to/posterior_samples.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to the output PNG.",
    )
    parser.add_argument(
        "--filled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use filled GetDist contours.",
    )
    parser.add_argument(
        "--width-inch",
        type=float,
        default=11.0,
    )
    return parser.parse_args()


def _parse_case(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected label=path format, got: {text}")
    label, path_text = text.split("=", 1)
    label = label.strip()
    path = Path(path_text.strip()).expanduser().resolve()
    if not label:
        raise ValueError(f"Missing label in: {text}")
    return label, path


def _ranges(samples_by_case: dict[str, np.ndarray], parameter_columns: list[str]) -> dict[str, list[float]]:
    stacked = np.vstack(list(samples_by_case.values()))
    ranges: dict[str, list[float]] = {}
    for index, name in enumerate(parameter_columns):
        lower = float(np.min(stacked[:, index]))
        upper = float(np.max(stacked[:, index]))
        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        ranges[name] = [lower - margin, upper + margin]
    return ranges


def main() -> None:
    args = parse_args()
    if MCSamples is None or gd_plots is None:
        raise ModuleNotFoundError(
            "The 'getdist' package is required for plot_halpha_mcmc_overlay_getdist.py."
        )

    case_order: list[str] = []
    samples_by_case: dict[str, np.ndarray] = {}
    for text in args.posterior:
        label, path = _parse_case(text)
        df = load_posterior_frame(path)
        samples = df[INPUT_COLUMNS].to_numpy(dtype=float)
        samples_by_case[label] = samples
        case_order.append(label)

    labels = [DISPLAY_LABELS.get(name, name) for name in INPUT_COLUMNS]
    ranges = _ranges(samples_by_case, INPUT_COLUMNS)

    roots = []
    contour_colors = []
    line_args = []
    legend_labels = []
    for case_name in case_order:
        style = DEFAULT_STYLE.get(case_name, {"color": None, "label": case_name})
        roots.append(
            MCSamples(
                samples=samples_by_case[case_name],
                names=INPUT_COLUMNS,
                labels=labels,
                label=style.get("label", case_name),
                ranges=ranges,
                settings={"smooth_scale_1D": 0.7, "smooth_scale_2D": 0.7},
            )
        )
        contour_colors.append(style.get("color"))
        line_args.append(
            {
                "lw": style.get("linewidth", 1.8),
                "color": style.get("color"),
            }
        )
        legend_labels.append(style.get("label", case_name))

    plotter = gd_plots.get_subplot_plotter(width_inch=args.width_inch)
    plotter.settings.figure_legend_frame = False
    plotter.settings.alpha_factor_contour_lines = 1.0
    plotter.settings.linewidth = 1.5
    plotter.settings.linewidth_contour = 1.6
    plotter.settings.num_plot_contours = 2
    plotter.settings.legend_fontsize = 13
    plotter.settings.axes_labelsize = 11
    plotter.settings.lab_fontsize = 11
    if args.filled:
        plotter.settings.alpha_filled_add = 0.4
        plotter.settings.solid_colors = contour_colors
    plotter.triangle_plot(
        roots,
        params=INPUT_COLUMNS,
        filled=args.filled,
        contour_colors=contour_colors,
        legend_labels=legend_labels,
        line_args=line_args,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.export(str(args.output_path))
    print(args.output_path)


if __name__ == "__main__":
    main()
