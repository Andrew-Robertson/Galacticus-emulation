from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu.interactive_observables import load_observables_bundle, refresh_observables_training_preview
from galacticus_emu.mcmc_results import load_posterior_frame
from galacticus_emu.plotting import set_ylim_from_values
from run_interactive_observables_mcmc import BundlePosterior, _parameter_specs_for_columns, _parse_x_ranges, _selected_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot standard-observable emulator-mean predictions across posterior draws. "
            "The shaded band is posterior spread in the emulator mean, not GP predictive uncertainty."
        )
    )
    parser.add_argument("--mcmc-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=None,
        help="Override the bundle path stored in the MCMC inputs file.",
    )
    parser.add_argument(
        "--posterior-samples",
        type=Path,
        default=None,
        help=(
            "Posterior samples CSV or HDF5 results file. Defaults to the posterior CSV when present, "
            "otherwise <mcmc-dir>/<prefix>_mcmc_results.hdf5."
        ),
    )
    parser.add_argument("--n-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--target-sigma-floor", type=float, default=None)
    parser.add_argument(
        "--observable",
        action="append",
        default=None,
        help=(
            "Observable key to plot. Repeat for multiple. Defaults to the keys fitted by the MCMC."
        ),
    )
    parser.add_argument(
        "--observable-x-range",
        action="append",
        default=[],
        help="Restrict one plotted observable in x units as key=xmin:xmax.",
    )
    parser.add_argument(
        "--credible-interval",
        type=float,
        default=0.68,
        help="Central posterior interval to shade. Default: 0.68.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="PNG path. Defaults to figures/<prefix>_posterior_mean_standard_observables.png.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="CSV path. Defaults to <mcmc-dir>/<prefix>_posterior_mean_standard_observables.csv.",
    )
    parser.add_argument(
        "--training-campaign-root",
        type=Path,
        default=None,
        help="Campaign root used to refresh and plot training curves behind the posterior band.",
    )
    parser.add_argument(
        "--training-preview-rows",
        default=None,
        help="Number of training curves to plot, or 'all'. Requires --training-campaign-root.",
    )
    parser.add_argument("--training-alpha", type=float, default=0.035)
    parser.add_argument("--training-linewidth", type=float, default=0.35)
    parser.add_argument(
        "--legend-mode",
        choices=["full", "full-first-target-rest", "target-only"],
        default="full",
        help=(
            "Legend content for each panel. 'full' repeats all legend entries in every panel; "
            "'full-first-target-rest' uses a full legend in the first panel and only the observational "
            "data-set label thereafter; 'target-only' shows only the data-set label in every panel."
        ),
    )
    parser.add_argument(
        "--include-training-in-ylim",
        action="store_true",
        help="Include the plotted training curves when setting y-axis limits.",
    )
    return parser.parse_args()


def _posterior_prediction_rows(
    bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    predictions: np.ndarray,
    slices: dict[str, slice],
    interval: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    tail = 0.5 * (1.0 - interval)
    percentiles = [100.0 * tail, 50.0, 100.0 * (1.0 - tail)]
    rows = []
    summaries = {}
    for key in observable_keys:
        observable = bundle["observables"][key]
        mask = masks[key]
        block = predictions[:, slices[key]]
        lower, median, upper = np.percentile(block, percentiles, axis=0)
        mean = np.mean(block, axis=0)
        std = np.std(block, axis=0, ddof=1) if block.shape[0] > 1 else np.zeros(block.shape[1])
        summaries[key] = {
            "lower": lower,
            "median": median,
            "upper": upper,
            "mean": mean,
            "std": std,
        }
        target_sigma = observable.get("target_sigma_plot")
        if target_sigma is None:
            target_sigma = np.full(np.count_nonzero(mask), np.nan, dtype=float)
        else:
            target_sigma = np.asarray(target_sigma, dtype=float)[mask]
        for bin_index, values in enumerate(
            zip(
                np.asarray(observable["x_plot"], dtype=float)[mask],
                np.asarray(observable["target_plot"], dtype=float)[mask],
                target_sigma,
                lower,
                median,
                upper,
                mean,
                std,
                strict=True,
            )
        ):
            x_value, target_value, target_std, lo, med, hi, avg, sigma = values
            rows.append(
                {
                    "observable_key": key,
                    "analysis": observable["analysis"],
                    "bin_index_after_mask": int(bin_index),
                    "x_plot": float(x_value),
                    "target_plot": float(target_value),
                    "target_sigma_plot": float(target_std),
                    "posterior_prediction_lower": float(lo),
                    "posterior_prediction_median": float(med),
                    "posterior_prediction_upper": float(hi),
                    "posterior_prediction_mean": float(avg),
                    "posterior_prediction_std": float(sigma),
                }
            )
    return pd.DataFrame(rows), summaries


def _plot(
    bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    summaries: dict[str, dict[str, np.ndarray]],
    path: Path,
    interval: float,
    *,
    training_alpha: float,
    training_linewidth: float,
    include_training_in_ylim: bool,
    legend_mode: str,
) -> None:
    n_panels = len(observable_keys)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    band_label = f"{100.0 * interval:.0f}% posterior band"
    for panel_index, (axis, key) in enumerate(zip(axes_flat, observable_keys, strict=False)):
        observable = bundle["observables"][key]
        use_full_legend = legend_mode == "full" or (
            legend_mode == "full-first-target-rest" and panel_index == 0
        )
        mask = masks[key]
        x = np.asarray(observable["x_plot"], dtype=float)[mask]
        target = np.asarray(observable["target_plot"], dtype=float)[mask]
        target_sigma = observable.get("target_sigma_plot")
        if target_sigma is not None:
            target_sigma = np.asarray(target_sigma, dtype=float)[mask]
        summary = summaries[key]
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
            axis.plot(
                [],
                [],
                color="0.55",
                alpha=0.6,
                linewidth=max(training_linewidth, 0.8),
                label=f"{y_train.shape[0]} training runs" if use_full_legend else "_nolegend_",
            )
        target_label = str(observable.get("target_label") or "target")
        axis.errorbar(x, target, yerr=target_sigma, fmt="o", color="0.15", label=target_label)
        axis.plot(
            x,
            summary["median"],
            color="tab:blue",
            lw=1.8,
            label="posterior median" if use_full_legend else "_nolegend_",
        )
        axis.fill_between(
            x,
            summary["lower"],
            summary["upper"],
            color="tab:blue",
            alpha=0.2,
            label=band_label if use_full_legend else "_nolegend_",
        )
        ylim_training = y_train if include_training_in_ylim else None
        set_ylim_from_values(axis, target, summary["lower"], summary["upper"], summary["median"], ylim_training)
        axis.set_title(observable["label"])
        axis.set_xlabel(observable["x_axis_label"])
        axis.set_ylabel(observable["y_axis_label"])
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.credible_interval < 1.0:
        raise ValueError("--credible-interval must be between 0 and 1")

    inputs_path = args.mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib"
    summary_path = args.mcmc_dir / f"{args.output_prefix}_run_summary.json"
    posterior_path = args.mcmc_dir / f"{args.output_prefix}_posterior_samples.csv"
    inputs = joblib.load(inputs_path)
    summary = json.loads(summary_path.read_text())
    bundle_path = args.bundle_path or inputs.get("standard_bundle_path") or inputs.get("bundle_path")
    if bundle_path is None:
        raise KeyError(f"{inputs_path} does not contain standard_bundle_path or bundle_path")
    bundle = load_observables_bundle(Path(bundle_path))
    if args.training_campaign_root is not None:
        bundle["campaign_root"] = str(args.training_campaign_root)
        bundle = refresh_observables_training_preview(
            bundle,
            training_preview_rows=args.training_preview_rows or "all",
        )

    observable_keys = args.observable or list(
        inputs.get("standard_observable_keys") or inputs.get("observable_keys") or []
    )
    if not observable_keys:
        raise ValueError("No observables requested and none found in MCMC inputs.")
    x_ranges = {
        key: tuple(value)
        for key, value in (
            summary.get("standard_observable_x_ranges") or summary.get("observable_x_ranges") or {}
        ).items()
    }
    x_ranges.update(_parse_x_ranges(args.observable_x_range))
    target_sigma_floor = (
        float(summary.get("target_sigma_floor", 1.0e-3))
        if args.target_sigma_floor is None
        else float(args.target_sigma_floor)
    )
    _, _, masks, slices = _selected_targets(bundle, observable_keys, x_ranges, target_sigma_floor)

    posterior_source = args.posterior_samples
    if posterior_source is None:
        posterior_source = posterior_path if posterior_path.exists() else args.mcmc_dir / f"{args.output_prefix}_mcmc_results.hdf5"
    posterior_df = load_posterior_frame(posterior_source)
    parameter_names = list(inputs.get("parameter_names") or inputs.get("input_columns") or bundle["input_columns"])
    slow_count = int(inputs.get("slow_count", len(parameter_names)))
    parameter_names = parameter_names[:slow_count]
    if args.n_draws > 0 and args.n_draws < len(posterior_df):
        posterior_df = posterior_df.sample(n=args.n_draws, random_state=args.seed)
    theta = posterior_df[parameter_names].to_numpy(dtype=float)

    parameter_specs = _parameter_specs_for_columns(list(bundle["input_columns"]))
    posterior = BundlePosterior(
        bundle=bundle,
        observable_keys=observable_keys,
        masks=masks,
        target_vector=np.zeros(sum(np.count_nonzero(masks[key]) for key in observable_keys), dtype=float),
        target_variance=np.ones(sum(np.count_nonzero(masks[key]) for key in observable_keys), dtype=float),
        parameter_specs=parameter_specs,
        include_emulator_variance=False,
    )
    predictions, _ = posterior.predict_batch(theta)
    rows, summaries = _posterior_prediction_rows(
        bundle,
        observable_keys,
        masks,
        predictions,
        slices,
        args.credible_interval,
    )

    output_path = args.output_path or (
        args.mcmc_dir
        / "figures"
        / f"{args.output_prefix}_posterior_mean_standard_observables.png"
    )
    csv_path = args.csv_path or args.mcmc_dir / f"{args.output_prefix}_posterior_mean_standard_observables.csv"
    rows.to_csv(csv_path, index=False)
    _plot(
        bundle,
        observable_keys,
        masks,
        summaries,
        output_path,
        args.credible_interval,
        training_alpha=args.training_alpha,
        training_linewidth=args.training_linewidth,
        include_training_in_ylim=args.include_training_in_ylim,
        legend_mode=args.legend_mode,
    )
    print(csv_path)
    print(output_path)


if __name__ == "__main__":
    main()
