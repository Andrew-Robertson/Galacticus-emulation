from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu.interactive_observables import load_observables_bundle, refresh_observables_training_preview
from galacticus_emu.mcmc_results import load_posterior_frame
from galacticus_emu.plotting import set_ylim_from_values
from plot_observable_mcmc_overlay_getdist import _style_for_case
from run_interactive_observables_mcmc import (
    BundlePosterior,
    _parameter_specs_for_columns,
    _parse_x_ranges,
    _selected_targets,
)


DEFAULT_OBSERVABLES = [
    "smf_z0",
    "smf_z3",
    "sfr_function_robotham2011",
    "size_mass_vdw2014_star_forming_z0",
    "size_mass_vdw2014_quiescent_z0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay posterior bands for standard-observable emulator mean relations. "
            "The shaded regions show posterior spread in the emulator mean, not GP predictive uncertainty."
        )
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
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional CSV path for the plotted percentile summaries.",
    )
    parser.add_argument("--n-draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--credible-interval", type=float, default=0.68)
    parser.add_argument("--target-sigma-floor", type=float, default=1.0e-3)
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--band-alpha", type=float, default=0.16)
    parser.add_argument("--line-width", type=float, default=1.9)
    parser.add_argument("--combined-line-width", type=float, default=2.5)
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
    parser.add_argument("--training-alpha", type=float, default=0.035)
    parser.add_argument("--training-linewidth", type=float, default=0.35)
    parser.add_argument(
        "--include-training-in-ylim",
        action="store_true",
        help="Include the plotted training curves when setting y-axis limits.",
    )
    return parser.parse_args()


def _parse_labeled_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected label=path format, got: {text}")
    label, path_text = text.split("=", 1)
    return label.strip(), Path(path_text.strip()).expanduser().resolve()


def _parse_observable_ymins(values: list[str]) -> dict[str, float]:
    ymins = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected observable-ymin in key=ymin format, got: {value}")
        key, raw_ymin = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty observable key in observable-ymin value: {value}")
        ymins[key] = float(raw_ymin)
    return ymins


def _sample_posterior(path: Path, columns: list[str], *, n_draws: int, seed: int) -> np.ndarray:
    frame = load_posterior_frame(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required posterior column(s): {', '.join(missing)}")
    if n_draws > 0 and n_draws < len(frame):
        frame = frame.sample(n=n_draws, random_state=seed)
    return frame[columns].to_numpy(dtype=float)


def _posterior_summaries(
    *,
    bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    slices: dict[str, slice],
    theta: np.ndarray,
    interval: float,
) -> dict[str, dict[str, np.ndarray]]:
    parameter_specs = _parameter_specs_for_columns(list(bundle["input_columns"]))
    target_size = sum(np.count_nonzero(masks[key]) for key in observable_keys)
    posterior = BundlePosterior(
        bundle=bundle,
        observable_keys=observable_keys,
        masks=masks,
        target_vector=np.zeros(target_size, dtype=float),
        target_variance=np.ones(target_size, dtype=float),
        parameter_specs=parameter_specs,
        include_emulator_variance=False,
    )
    predictions, _ = posterior.predict_batch(theta)
    tail = 0.5 * (1.0 - interval)
    percentiles = [100.0 * tail, 50.0, 100.0 * (1.0 - tail)]
    summaries = {}
    for key in observable_keys:
        lower, median, upper = np.percentile(predictions[:, slices[key]], percentiles, axis=0)
        summaries[key] = {"lower": lower, "median": median, "upper": upper}
    return summaries


def _summary_rows(
    label: str,
    bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    summaries: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows = []
    for key in observable_keys:
        observable = bundle["observables"][key]
        mask = masks[key]
        x = np.asarray(observable["x_plot"], dtype=float)[mask]
        target = np.asarray(observable["target_plot"], dtype=float)[mask]
        target_sigma = observable.get("target_sigma_plot")
        if target_sigma is None:
            target_sigma = np.full_like(target, np.nan, dtype=float)
        else:
            target_sigma = np.asarray(target_sigma, dtype=float)[mask]
        summary = summaries[key]
        for index, values in enumerate(
            zip(
                x,
                target,
                target_sigma,
                summary["lower"],
                summary["median"],
                summary["upper"],
                strict=True,
            )
        ):
            x_value, target_value, sigma_value, lower, median, upper = values
            rows.append(
                {
                    "posterior_label": label,
                    "observable_key": key,
                    "analysis": observable["analysis"],
                    "bin_index_after_mask": int(index),
                    "x_plot": float(x_value),
                    "target_plot": float(target_value),
                    "target_sigma_plot": float(sigma_value),
                    "posterior_prediction_lower": float(lower),
                    "posterior_prediction_median": float(median),
                    "posterior_prediction_upper": float(upper),
                }
            )
    return rows


def _plot(
    *,
    bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    summaries_by_case: dict[str, dict[str, dict[str, np.ndarray]]],
    output_path: Path,
    interval: float,
    ncols: int,
    dpi: int,
    band_alpha: float,
    line_width: float,
    combined_line_width: float,
    training_alpha: float,
    training_linewidth: float,
    include_training_in_ylim: bool,
    observable_ymins: dict[str, float],
) -> None:
    ncols = max(1, int(ncols))
    n_panels = len(observable_keys)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 4.0 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, key in zip(axes_flat, observable_keys, strict=False):
        observable = bundle["observables"][key]
        mask = masks[key]
        x = np.asarray(observable["x_plot"], dtype=float)[mask]
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
        axis.errorbar(
            x,
            target,
            yerr=target_sigma,
            fmt="o",
            color="0.15",
            ms=4.0,
            label=observable.get("target_label") or "target",
            zorder=5,
        )
        y_limits_values = [target]
        for case_index, (case_label, summaries) in enumerate(summaries_by_case.items()):
            style = _style_for_case(case_label, case_index)
            color = str(style.get("color"))
            label = str(style.get("label", case_label))
            linewidth = combined_line_width if case_label == "Combined" else line_width
            summary = summaries[key]
            axis.plot(x, summary["median"], color=color, lw=linewidth, label=label, zorder=4)
            axis.fill_between(
                x,
                summary["lower"],
                summary["upper"],
                color=color,
                alpha=band_alpha,
                label="_nolegend_",
                zorder=2,
            )
            y_limits_values.extend([summary["lower"], summary["median"], summary["upper"]])
        if include_training_in_ylim and y_train is not None:
            y_limits_values.append(y_train)
        set_ylim_from_values(axis, *y_limits_values)
        if key in observable_ymins:
            _, ymax = axis.get_ylim()
            ymin = observable_ymins[key]
            if ymax <= ymin:
                ymax = ymin + 1.0
            axis.set_ylim(ymin, ymax)
        axis.set_title(observable["label"])
        axis.set_xlabel(observable["x_axis_label"])
        axis.set_ylabel(observable["y_axis_label"])
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.credible_interval < 1.0:
        raise ValueError("--credible-interval must be between 0 and 1")
    bundle = load_observables_bundle(args.bundle_path)
    if args.training_campaign_root is not None:
        bundle["campaign_root"] = str(args.training_campaign_root)
        bundle = refresh_observables_training_preview(
            bundle,
            training_preview_rows=args.training_preview_rows or "all",
        )
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

    summaries_by_case = {}
    rows = []
    input_columns = list(bundle["input_columns"])
    for case_index, posterior_text in enumerate(args.posterior):
        label, path = _parse_labeled_path(posterior_text)
        theta = _sample_posterior(
            path,
            input_columns,
            n_draws=args.n_draws,
            seed=args.seed + case_index,
        )
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

    if args.csv_path is not None:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv_path, index=False)
        print(args.csv_path)
    _plot(
        bundle=bundle,
        observable_keys=observable_keys,
        masks=masks,
        summaries_by_case=summaries_by_case,
        output_path=args.output_path,
        interval=args.credible_interval,
        ncols=args.ncols,
        dpi=args.dpi,
        band_alpha=args.band_alpha,
        line_width=args.line_width,
        combined_line_width=args.combined_line_width,
        training_alpha=args.training_alpha,
        training_linewidth=args.training_linewidth,
        include_training_in_ylim=args.include_training_in_ylim,
        observable_ymins=observable_ymins,
    )
    print(args.output_path)


if __name__ == "__main__":
    main()
