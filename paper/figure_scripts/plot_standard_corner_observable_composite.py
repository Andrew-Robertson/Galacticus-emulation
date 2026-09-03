from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from galacticus_emu.interactive_observables import load_observables_bundle, refresh_observables_training_preview
from galacticus_emu.plotting import set_ylim_from_values
from plot_observable_mcmc_overlay_getdist import _style_for_case
from plot_standard_observable_posterior_band_overlay import (
    DEFAULT_OBSERVABLES,
    _parse_labeled_path,
    _parse_observable_ymins,
    _posterior_summaries,
    _sample_posterior,
    _summary_rows,
)
from run_interactive_observables_mcmc import _parse_x_ranges, _selected_targets


DEFAULT_PANEL_RECTS = {
    "smf_z0": [0.470, 0.720, 0.220, 0.225],
    "smf_z3": [0.740, 0.720, 0.220, 0.225],
    "size_mass_vdw2014_star_forming_z0": [0.470, 0.420, 0.220, 0.225],
    "size_mass_vdw2014_quiescent_z0": [0.740, 0.420, 0.220, 0.225],
    "sfr_function_robotham2011": [0.740, 0.120, 0.220, 0.225],
}

DEFAULT_TARGET_LEGEND_LOCS = {
    "smf_z0": "lower left",
    "smf_z3": "lower left",
    "size_mass_vdw2014_star_forming_z0": "lower right",
    "size_mass_vdw2014_quiescent_z0": "lower right",
    "sfr_function_robotham2011": "lower left",
}

PANEL_TITLES = {
    "smf_z0": "Stellar mass function",
    "smf_z3": "Stellar mass function",
    "size_mass_vdw2014_star_forming_z0": "Star-forming galaxy sizes",
    "size_mass_vdw2014_quiescent_z0": "Quiescent galaxy sizes",
    "sfr_function_robotham2011": "Star formation rate function",
}

PANEL_TAGS = {
    "smf_z0": (r"$0.20<z<0.50$", "upper right"),
    "smf_z3": (r"$1.00<z<1.25$", "upper right"),
    "size_mass_vdw2014_star_forming_z0": (r"$0<z<0.5$", "upper left"),
    "size_mass_vdw2014_quiescent_z0": (r"$0<z<0.5$", "upper left"),
    "sfr_function_robotham2011": (r"$0.013<z<0.1$", "upper right"),
}

PANEL_TAG_COORDS = {
    "smf_z0": (0.965, 0.965),
    "smf_z3": (0.965, 0.965),
    "size_mass_vdw2014_star_forming_z0": (0.035, 0.965),
    "size_mass_vdw2014_quiescent_z0": (0.035, 0.965),
    "sfr_function_robotham2011": (0.965, 0.965),
}

PANEL_X_DISPLAY_OFFSETS = {
    "sfr_function_robotham2011": -9.0,
}

PANEL_X_AXIS_LABELS = {
    "sfr_function_robotham2011": r"$\log_{10}(\dot{M}_\star/(M_\odot\,\mathrm{yr}^{-1}))$",
}

PANEL_X_TICKS = {
    "size_mass_vdw2014_star_forming_z0": [9.5, 10.0, 10.5],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Composite a GetDist corner image with posterior-band standard-observable panels. "
            "The corner panel is embedded as an image; observable panels are drawn with Matplotlib."
        )
    )
    parser.add_argument(
        "--corner-image",
        type=Path,
        default=None,
        help="Existing corner image to embed. Omit with --generate-corner to create it first.",
    )
    parser.add_argument(
        "--generate-corner",
        action="store_true",
        help="Generate the GetDist corner PNG first, then embed it into the composite figure.",
    )
    parser.add_argument(
        "--corner-output-path",
        type=Path,
        default=None,
        help="Path for the generated corner PNG. Defaults to <output stem>_corner.png.",
    )
    parser.add_argument(
        "--corner-parameter",
        action="append",
        default=None,
        help="Parameter short name to include in the generated corner plot. Repeat to select and order a subset.",
    )
    parser.add_argument(
        "--corner-posterior-filter",
        action="append",
        default=[],
        help="Filter passed through to plot_observable_mcmc_overlay_getdist.py as --posterior-filter.",
    )
    parser.add_argument(
        "--corner-label-style",
        choices=["short-name", "paper"],
        default="paper",
        help="Label style for a generated corner plot.",
    )
    parser.add_argument(
        "--corner-parameter-set",
        choices=["auto", "observable19", "joint24"],
        default="auto",
        help="Parameter set passed through when generating the corner plot.",
    )
    parser.add_argument("--corner-width-inch", type=float, default=None)
    parser.add_argument("--corner-axes-labelsize", type=float, default=None)
    parser.add_argument("--corner-tick-labelsize", type=float, default=None)
    parser.add_argument("--corner-tick-length-scale", type=float, default=1.0)
    parser.add_argument("--corner-linewidth-scale", type=float, default=1.0)
    parser.add_argument("--corner-x-labelpad", type=float, default=None)
    parser.add_argument("--corner-y-labelpad", type=float, default=None)
    parser.add_argument("--corner-filled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--corner-dpi",
        type=int,
        default=None,
        help="DPI for a generated corner PNG. Defaults to --dpi.",
    )
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument(
        "--posterior",
        action="append",
        required=True,
        help="Posterior in label=/path/to/*_mcmc_results.hdf5 or label=/path/to/posterior_samples.csv form.",
    )
    parser.add_argument(
        "--observable",
        action="append",
        default=None,
        help="Observable key to plot. Repeat to override the default SMF/SFRF/size set.",
    )
    parser.add_argument(
        "--observable-x-range",
        action="append",
        default=[],
        help="Restrict one plotted observable in x units as key=xmin:xmax.",
    )
    parser.add_argument(
        "--observable-ymin",
        action="append",
        default=[],
        help="Set the lower y-axis limit for one observable as key=ymin. Repeat for multiple panels.",
    )
    parser.add_argument(
        "--training-campaign-root",
        type=Path,
        default=None,
        help="Campaign root used to refresh and plot training curves behind the posterior bands.",
    )
    parser.add_argument(
        "--training-preview-rows",
        default=None,
        help="Number of training curves to plot, or 'all'. Requires --training-campaign-root.",
    )
    parser.add_argument("--training-alpha", type=float, default=0.08)
    parser.add_argument("--training-linewidth", type=float, default=0.28)
    parser.add_argument("--include-training-in-ylim", action="store_true")
    parser.add_argument("--n-draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--credible-interval", type=float, default=0.68)
    parser.add_argument("--target-sigma-floor", type=float, default=1.0e-3)
    parser.add_argument("--band-alpha", type=float, default=0.16)
    parser.add_argument("--line-width", type=float, default=1.65)
    parser.add_argument("--combined-line-width", type=float, default=2.2)
    parser.add_argument("--fig-width", type=float, default=15.4)
    parser.add_argument("--fig-height", type=float, default=9.7)
    parser.add_argument(
        "--corner-rect",
        nargs=4,
        type=float,
        default=[0.000, 0.020, 0.555, 0.960],
        metavar=("LEFT", "BOTTOM", "WIDTH", "HEIGHT"),
        help="Figure-relative rectangle used for the corner image.",
    )
    parser.add_argument(
        "--corner-auto-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically crop white margins from the embedded corner image.",
    )
    parser.add_argument(
        "--corner-crop-padding",
        type=int,
        default=8,
        help="Pixel padding retained around the auto-cropped corner image.",
    )
    parser.add_argument(
        "--corner-crop-threshold",
        type=float,
        default=0.985,
        help="RGB threshold used by --corner-auto-crop to identify non-white pixels.",
    )
    parser.add_argument(
        "--legend-rect",
        nargs=4,
        type=float,
        default=[0.285, 0.805, 0.160, 0.155],
        metavar=("LEFT", "BOTTOM", "WIDTH", "HEIGHT"),
        help="Figure-relative rectangle used for the shared legend.",
    )
    parser.add_argument(
        "--panel-rect",
        action="append",
        default=[],
        help="Override one panel rectangle as observable=left,bottom,width,height.",
    )
    parser.add_argument(
        "--panel-legends",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw full per-panel legends including posterior cases and target data.",
    )
    parser.add_argument(
        "--target-legends",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a compact target-dataset legend in each observable panel.",
    )
    parser.add_argument("--target-legend-fontsize", type=float, default=6.7)
    parser.add_argument(
        "--panel-font-scale",
        type=float,
        default=1.0,
        help=(
            "Scale observable-panel axis labels, ticks, and target legends. "
            "The shared SMFs/SFRF/Sizes/Combined legend is unchanged."
        ),
    )
    parser.add_argument(
        "--panel-title-font-scale",
        type=float,
        default=1.0,
        help="Scale observable-panel titles independently of --panel-font-scale.",
    )
    parser.add_argument(
        "--panel-axis-label-scale",
        type=float,
        default=1.15,
        help="Additional scale applied only to observable-panel x/y axis labels.",
    )
    parser.add_argument(
        "--panel-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw grid lines in the observable panels.",
    )
    parser.add_argument(
        "--shared-legend-fontsize",
        type=float,
        default=16.0,
        help="Font size for the shared posterior-case legend.",
    )
    parser.add_argument(
        "--target-legend-loc",
        action="append",
        default=[],
        help="Set one target legend location as observable='matplotlib loc', e.g. smf_z0='lower left'.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--extra-output-path",
        action="append",
        type=Path,
        default=[],
        help="Additional path to save the same composite figure, e.g. a PNG next to the PDF.",
    )
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument(
        "--save-provenance",
        action="store_true",
        help="Write a provenance sidecar next to --output-path as <output-path>.command.sh.",
    )
    parser.add_argument(
        "--provenance-command-path",
        type=Path,
        default=None,
        help=(
            "Optional sidecar path where the exact plotting command and git context will be written. "
            "Overrides the default path used by --save-provenance."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _parse_panel_rects(values: list[str]) -> dict[str, list[float]]:
    rects = {key: list(value) for key, value in DEFAULT_PANEL_RECTS.items()}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected panel rect as observable=left,bottom,width,height, got: {value}")
        key, raw_rect = value.split("=", 1)
        numbers = [float(part.strip()) for part in raw_rect.split(",")]
        if len(numbers) != 4:
            raise ValueError(f"Expected four comma-separated numbers for {key}, got: {raw_rect}")
        rects[key.strip()] = numbers
    return rects


def _parse_target_legend_locs(values: list[str]) -> dict[str, str]:
    locs = dict(DEFAULT_TARGET_LEGEND_LOCS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected target legend location as observable=loc, got: {value}")
        key, loc = value.split("=", 1)
        key = key.strip()
        loc = loc.strip().strip("'\"")
        if not key or not loc:
            raise ValueError(f"Invalid target legend location: {value}")
        locs[key] = loc
    return locs


def _load_bundle(args: argparse.Namespace) -> dict:
    bundle = load_observables_bundle(args.bundle_path)
    if args.training_campaign_root is not None:
        bundle["campaign_root"] = str(args.training_campaign_root)
        bundle = refresh_observables_training_preview(
            bundle,
            training_preview_rows=args.training_preview_rows or "all",
        )
    return bundle


def _generated_corner_default_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_corner.png")


def _append_optional_float(command: list[str], flag: str, value: float | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _generate_corner_image(args: argparse.Namespace) -> Path:
    output_path = args.corner_output_path or _generated_corner_default_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plot_observable_mcmc_overlay_getdist.py"),
    ]
    for posterior in args.posterior:
        command.extend(["--posterior", posterior])
    for parameter in args.corner_parameter or []:
        command.extend(["--parameter", parameter])
    for posterior_filter in args.corner_posterior_filter:
        command.extend(["--posterior-filter", posterior_filter])
    command.extend(
        [
            "--output-path",
            str(output_path),
            "--parameter-set",
            args.corner_parameter_set,
            "--label-style",
            args.corner_label_style,
            "--legend-loc",
            "none",
            "--linewidth-scale",
            str(args.corner_linewidth_scale),
            "--tick-length-scale",
            str(args.corner_tick_length_scale),
            "--dpi",
            str(args.corner_dpi if args.corner_dpi is not None else args.dpi),
        ]
    )
    command.append("--filled" if args.corner_filled else "--no-filled")
    _append_optional_float(command, "--width-inch", args.corner_width_inch)
    _append_optional_float(command, "--axes-labelsize", args.corner_axes_labelsize)
    _append_optional_float(command, "--tick-labelsize", args.corner_tick_labelsize)
    _append_optional_float(command, "--x-labelpad", args.corner_x_labelpad)
    _append_optional_float(command, "--y-labelpad", args.corner_y_labelpad)
    print(shlex.join(command))
    subprocess.run(command, cwd=Path.cwd(), check=True)
    return output_path


def _evaluate_posteriors(
    args: argparse.Namespace,
    bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    slices: dict[str, slice],
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, object]]]:
    summaries_by_case = {}
    rows = []
    input_columns = list(bundle["input_columns"])
    for case_index, posterior_text in enumerate(args.posterior):
        label, path = _parse_labeled_path(posterior_text)
        theta = _sample_posterior(path, input_columns, n_draws=args.n_draws, seed=args.seed + case_index)
        summaries = _posterior_summaries(
            bundle=bundle,
            observable_keys=observable_keys,
            masks=masks,
            slices=slices,
            theta=theta,
            interval=args.credible_interval,
        )
        summaries_by_case[label] = summaries
        rows.extend(_summary_rows(label, bundle, observable_keys, masks, summaries))
        print(f"{label}: evaluated {theta.shape[0]} posterior samples")
    return summaries_by_case, rows


def _auto_crop_white_margin(image: np.ndarray, *, threshold: float, padding: int) -> np.ndarray:
    rgb = np.asarray(image[..., :3], dtype=float)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    non_white = np.any(rgb < float(threshold), axis=2)
    if not np.any(non_white):
        return image
    rows = np.where(np.any(non_white, axis=1))[0]
    cols = np.where(np.any(non_white, axis=0))[0]
    top = max(int(rows[0]) - int(padding), 0)
    bottom = min(int(rows[-1]) + int(padding) + 1, image.shape[0])
    left = max(int(cols[0]) - int(padding), 0)
    right = min(int(cols[-1]) + int(padding) + 1, image.shape[1])
    return image[top:bottom, left:right]


def _display_x_values(observable_key: str, values: np.ndarray) -> np.ndarray:
    offset = float(PANEL_X_DISPLAY_OFFSETS.get(observable_key, 0.0))
    return np.asarray(values, dtype=float) + offset


def _panel_x_axis_label(observable_key: str, observable: dict) -> str:
    return PANEL_X_AXIS_LABELS.get(observable_key, observable["x_axis_label"])


def _add_panel_tag(axis: plt.Axes, observable_key: str, *, font_scale: float) -> None:
    tag = PANEL_TAGS.get(observable_key)
    if tag is None:
        return
    text, location = tag
    if location == "upper right":
        x, ha = 0.935, "right"
    elif location == "upper left":
        x, ha = 0.065, "left"
    else:
        raise ValueError(f"Unknown panel-tag location: {location}")
    x, y = PANEL_TAG_COORDS.get(observable_key, (x, 0.885))
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha=ha,
        va="top",
        fontsize=8.5 * font_scale,
        color="0.1",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
        zorder=6,
    )


def _plot_corner(
    fig: plt.Figure,
    corner_image: Path,
    rect: list[float],
    *,
    auto_crop: bool,
    crop_threshold: float,
    crop_padding: int,
) -> None:
    axis = fig.add_axes(rect)
    image = mpimg.imread(corner_image.expanduser().resolve())
    if auto_crop:
        image = _auto_crop_white_margin(
            image,
            threshold=crop_threshold,
            padding=crop_padding,
        )
    axis.imshow(image)
    axis.set_axis_off()


def _legend_patch(label: str, case_index: int) -> Patch:
    style = _style_for_case(label, case_index)
    color = str(style.get("color"))
    display_label = str(style.get("label", label))
    if label == "Combined":
        return Patch(facecolor="0.35", edgecolor="0.15", alpha=0.55, label=display_label)
    return Patch(facecolor=color, edgecolor=color, alpha=0.55, label=display_label)


def _plot_shared_legend(
    fig: plt.Figure,
    rect: list[float],
    case_order: list[str],
    *,
    fontsize: float,
) -> None:
    axis = fig.add_axes(rect)
    axis.set_axis_off()
    handles = [_legend_patch(label, index) for index, label in enumerate(case_order)]
    axis.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=fontsize,
        handlelength=1.6,
        handleheight=0.9,
        borderaxespad=0.0,
        labelspacing=0.55,
    )


def _plot_observable_panel(
    axis: plt.Axes,
    *,
    bundle: dict,
    observable_key: str,
    mask: np.ndarray,
    summaries_by_case: dict[str, dict[str, dict[str, np.ndarray]]],
    observable_ymins: dict[str, float],
    training_alpha: float,
    training_linewidth: float,
    include_training_in_ylim: bool,
    band_alpha: float,
    line_width: float,
    combined_line_width: float,
    panel_legends: bool,
    target_legends: bool,
    target_legend_fontsize: float,
    target_legend_locs: dict[str, str],
    panel_font_scale: float,
    panel_title_font_scale: float,
    panel_axis_label_scale: float,
    panel_grid: bool,
) -> None:
    observable = bundle["observables"][observable_key]
    x = _display_x_values(observable_key, np.asarray(observable["x_plot"], dtype=float)[mask])
    target = np.asarray(observable["target_plot"], dtype=float)[mask]
    target_sigma = observable.get("target_sigma_plot")
    if target_sigma is not None:
        target_sigma = np.asarray(target_sigma, dtype=float)[mask]
    y_train = observable.get("y_train_preview")
    if y_train is not None:
        y_train = np.asarray(y_train, dtype=float)[:, mask]
        axis.plot(
            x,
            y_train.T,
            color="0.55",
            alpha=training_alpha,
            linewidth=training_linewidth,
            zorder=1,
        )
    target_handle = axis.errorbar(
        x,
        target,
        yerr=target_sigma,
        fmt="o",
        color="0.15",
        ms=2.6,
        elinewidth=0.8,
        label=observable.get("target_label") or "target",
        zorder=5,
    )

    y_limits_values = [target]
    for case_index, (case_label, summaries) in enumerate(summaries_by_case.items()):
        style = _style_for_case(case_label, case_index)
        color = str(style.get("color"))
        linewidth = combined_line_width if case_label == "Combined" else line_width
        label = str(style.get("label", case_label))
        summary = summaries[observable_key]
        axis.plot(x, summary["median"], color=color, lw=linewidth, label=label, zorder=4)
        axis.fill_between(
            x,
            summary["lower"],
            summary["upper"],
            color=color,
            alpha=band_alpha,
            zorder=2,
        )
        y_limits_values.extend([summary["lower"], summary["median"], summary["upper"]])
    if include_training_in_ylim and y_train is not None:
        y_limits_values.append(y_train)

    set_ylim_from_values(axis, *y_limits_values)
    if observable_key in observable_ymins:
        _, ymax = axis.get_ylim()
        ymin = observable_ymins[observable_key]
        axis.set_ylim(ymin, max(ymax, ymin + 1.0))

    font_scale = float(panel_font_scale)
    title_font_scale = float(panel_title_font_scale)
    label_scale = float(panel_axis_label_scale)
    axis.set_title(
        PANEL_TITLES.get(observable_key, observable["label"]),
        fontsize=10.5 * title_font_scale,
        pad=2.5 * title_font_scale,
    )
    axis.set_xlabel(
        _panel_x_axis_label(observable_key, observable),
        fontsize=8.5 * font_scale * label_scale,
        labelpad=1.5 * font_scale,
    )
    axis.set_ylabel(
        observable["y_axis_label"],
        fontsize=8.5 * font_scale * label_scale,
        labelpad=1.5 * font_scale,
    )
    axis.tick_params(labelsize=7.5 * font_scale, length=2.5, pad=1.5 * font_scale)
    if observable_key in PANEL_X_TICKS:
        axis.set_xticks(PANEL_X_TICKS[observable_key])
    axis.grid(alpha=0.22 if panel_grid else 0.0)
    if not panel_grid:
        axis.grid(False)
    _add_panel_tag(axis, observable_key, font_scale=font_scale)
    if panel_legends:
        axis.legend(frameon=False, fontsize=6.5 * font_scale)
    elif target_legends:
        axis.legend(
            [target_handle],
            [observable.get("target_label") or "target"],
            frameon=False,
            fontsize=target_legend_fontsize * font_scale,
            loc=target_legend_locs.get(observable_key, "best"),
        )


def _git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return f"<git unavailable: {error}>"
    output = completed.stdout.strip()
    error_output = completed.stderr.strip()
    if completed.returncode != 0:
        return error_output or f"<git {' '.join(args)} failed with code {completed.returncode}>"
    return output


def _write_provenance(path: Path, *, output_paths: list[Path]) -> None:
    command = shlex.join([sys.executable, *sys.argv])
    git_head = _git_output(["rev-parse", "HEAD"])
    git_status = _git_output(["status", "--short"])
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "# Provenance for plot_standard_corner_observable_composite.py",
        f"# created_utc: {datetime.now(timezone.utc).isoformat()}",
        f"# cwd: {Path.cwd()}",
        f"# git_head: {git_head}",
        "# git_status_short:",
        *[f"#   {line}" for line in (git_status.splitlines() or ["<clean>"])],
        f"# output_path: {output_paths[0]}",
        "# extra_output_paths:",
        *[f"#   {output_path}" for output_path in output_paths[1:]],
        "",
        command,
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.credible_interval < 1.0:
        raise ValueError("--credible-interval must be between 0 and 1")
    if args.generate_corner:
        corner_image = _generate_corner_image(args)
    elif args.corner_image is not None:
        corner_image = args.corner_image
    else:
        raise ValueError("Pass --corner-image, or pass --generate-corner to create one first.")
    bundle = _load_bundle(args)
    observable_keys = args.observable or DEFAULT_OBSERVABLES
    missing = [key for key in observable_keys if key not in bundle["observables"]]
    if missing:
        raise ValueError(f"Observable(s) absent from bundle: {', '.join(missing)}")

    x_ranges = _parse_x_ranges(args.observable_x_range)
    observable_ymins = _parse_observable_ymins(args.observable_ymin)
    invalid_ymins = sorted(set(observable_ymins).difference(observable_keys))
    if invalid_ymins:
        raise ValueError(f"--observable-ymin given for absent observable(s): {', '.join(invalid_ymins)}")
    _, _, masks, slices = _selected_targets(bundle, observable_keys, x_ranges, args.target_sigma_floor)
    summaries_by_case, rows = _evaluate_posteriors(args, bundle, observable_keys, masks, slices)

    if args.csv_path is not None:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv_path, index=False)
        print(args.csv_path)

    panel_rects = _parse_panel_rects(args.panel_rect)
    target_legend_locs = _parse_target_legend_locs(args.target_legend_loc)
    fig = plt.figure(figsize=(args.fig_width, args.fig_height), facecolor="white")
    _plot_corner(
        fig,
        corner_image,
        list(args.corner_rect),
        auto_crop=args.corner_auto_crop,
        crop_threshold=args.corner_crop_threshold,
        crop_padding=args.corner_crop_padding,
    )
    _plot_shared_legend(
        fig,
        list(args.legend_rect),
        list(summaries_by_case),
        fontsize=args.shared_legend_fontsize,
    )
    for key in observable_keys:
        rect = panel_rects.get(key)
        if rect is None:
            continue
        axis = fig.add_axes(rect)
        _plot_observable_panel(
            axis,
            bundle=bundle,
            observable_key=key,
            mask=masks[key],
            summaries_by_case=summaries_by_case,
            observable_ymins=observable_ymins,
            training_alpha=args.training_alpha,
            training_linewidth=args.training_linewidth,
            include_training_in_ylim=args.include_training_in_ylim,
            band_alpha=args.band_alpha,
            line_width=args.line_width,
            combined_line_width=args.combined_line_width,
            panel_legends=args.panel_legends,
            target_legends=args.target_legends,
            target_legend_fontsize=args.target_legend_fontsize,
            target_legend_locs=target_legend_locs,
            panel_font_scale=args.panel_font_scale,
            panel_title_font_scale=args.panel_title_font_scale,
            panel_axis_label_scale=args.panel_axis_label_scale,
            panel_grid=args.panel_grid,
        )
    output_paths = [args.output_path, *args.extra_output_path]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=args.dpi)
        print(output_path)
    plt.close(fig)
    provenance_path = args.provenance_command_path
    if provenance_path is None and args.save_provenance:
        provenance_path = Path(f"{args.output_path}.command.sh")
    if provenance_path is not None:
        _write_provenance(provenance_path, output_paths=output_paths)
        print(provenance_path)


if __name__ == "__main__":
    main()
