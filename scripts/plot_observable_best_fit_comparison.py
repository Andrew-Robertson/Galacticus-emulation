from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE_BY_CASE = {
    "smf_pair": {"color": "#1b9e77", "label": "SMF pair best fit"},
    "sfr": {"color": "#d95f02", "label": "SFR function best fit"},
    "smf_pair_sfr": {"color": "#000000", "label": "SMF+SFR combined best fit"},
    "size_pair": {"color": "#7570b3", "label": "Size pair best fit"},
    "combined": {"color": "#000000", "label": "Combined best fit"},
}
ACTUAL_STYLE = {"color": "#c23b22", "label": "Actual Galacticus best fit", "linestyle": "--"}

ANALYSIS_ORDER = ["smf_z0", "smf_z3", "size_sf", "size_q", None, "sfr"]
INDIVIDUAL_CASE_BY_ANALYSIS_KEY = {
    "smf_z0": "smf_pair",
    "smf_z3": "smf_pair",
    "sfr": "sfr",
    "size_sf": "size_pair",
    "size_q": "size_pair",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target data with best-fit GP predictions from individual-observable and combined MCMC fits."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write figures into. Defaults to campaign_root/figures_observable_mcmc_overlay.",
    )
    parser.add_argument(
        "--output-prefix",
        default="observable_best_fit_comparison",
        help="Filename prefix for the output figures.",
    )
    parser.add_argument("--save-pdf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--actual-hdf5",
        type=Path,
        default=None,
        help="Optional reduced HDF5 file for the actual slow Galacticus best-fit model to overlay.",
    )
    parser.add_argument(
        "--individual-case",
        action="append",
        default=None,
        help="Case name to include as an individual-fit curve source. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--combined-case",
        default="combined",
        help="Case name to use as the black combined fit curve source.",
    )
    parser.add_argument(
        "--analysis-order",
        nargs="+",
        default=None,
        help="Ordered analysis-key layout. Use 'none' for a blank panel.",
    )
    return parser.parse_args()


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_reduced_hdf5(campaign_root: Path) -> Path:
    for path in sorted((campaign_root / "evaluations").glob("*/galacticus_reduced.hdf5")):
        return path
    raise FileNotFoundError(f"No galacticus_reduced.hdf5 files found under {campaign_root}")


def _analysis_attrs(campaign_root: Path) -> dict[str, dict[str, object]]:
    first_path = _first_reduced_hdf5(campaign_root)
    attrs_by_analysis = {}
    with h5py.File(first_path, "r") as handle:
        analyses = handle["/analyses"]
        for analysis_name in analyses:
            group = analyses[analysis_name]
            attrs_by_analysis[analysis_name] = {key: _decode_attr(value) for key, value in group.attrs.items()}
    return attrs_by_analysis


def _transform_x(values: np.ndarray, *, is_log: bool) -> np.ndarray:
    if is_log:
        if np.any(values <= 0.0):
            raise ValueError("Cannot log10-transform non-positive x values")
        return np.log10(values)
    return values


def _load_actual_predictions(actual_hdf5: Path, analysis_meta: dict[str, dict[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(actual_hdf5, "r") as handle:
        for analysis_key, meta in analysis_meta.items():
            group = handle[f"/analyses/{meta['analysis']}"]
            attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
            x_values = np.asarray(group[attrs["xDataset"]][...], dtype=float)
            y_values = np.asarray(group[attrs["yDataset"]][...], dtype=float)
            results[analysis_key] = {
                "x_plot": _transform_x(x_values, is_log=bool(meta["x_is_log"])),
                "y_value": y_values,
            }
    return results


def _load_case_predictions(campaign_root: Path, case_name: str) -> pd.DataFrame:
    case_to_prefix = {
        "smf_pair": "observable_smf_pair",
        "sfr": "observable_sfr",
        "smf_pair_sfr": "observable_smf_pair_sfr",
        "size_pair": "observable_size_pair",
        "combined": "observable_combined",
    }
    path = (
        campaign_root
        / f"emulator_observable_mcmc_{case_name}"
        / f"{case_to_prefix[case_name]}_best_fit_observables.csv"
    )
    return pd.read_csv(path)


def _plot_target_and_prediction(
    axis: plt.Axes,
    x: np.ndarray,
    target: np.ndarray,
    target_std: np.ndarray,
    pred: np.ndarray,
    pred_std: np.ndarray,
    *,
    color: str,
    label: str,
    y_is_log: bool,
) -> None:
    if y_is_log:
        lower_target = np.maximum(target - target_std, 1.0e-30)
        upper_target = target + target_std
        target_plot = np.log10(np.maximum(target, 1.0e-30))
        yerr_lower = target_plot - np.log10(lower_target)
        yerr_upper = np.log10(upper_target) - target_plot

        pred_plot = np.log10(np.maximum(pred, 1.0e-30))
        lower_pred = np.maximum(pred - pred_std, 1.0e-30)
        upper_pred = pred + pred_std
        pred_lower_plot = np.log10(lower_pred)
        pred_upper_plot = np.log10(upper_pred)

        axis.plot(x, pred_plot, color=color, lw=2.0, label=label, zorder=2)
        axis.fill_between(x, pred_lower_plot, pred_upper_plot, color=color, alpha=0.18, zorder=1)
        axis.errorbar(
            x,
            target_plot,
            yerr=np.vstack([yerr_lower, yerr_upper]),
            fmt="o",
            color="0.1",
            mfc="0.1",
            mec="0.1",
            ms=4.0,
            capsize=0.0,
            zorder=4,
            label="Target" if label == STYLE_BY_CASE["combined"]["label"] else None,
        )
        axis.plot(x, target_plot, color="0.1", lw=1.6, zorder=3)
        return pred_plot, pred_lower_plot, pred_upper_plot
    else:
        axis.plot(x, pred, color=color, lw=2.0, label=label, zorder=2)
        axis.errorbar(
            x,
            target,
            yerr=target_std,
            fmt="o",
            color="0.1",
            mfc="0.1",
            mec="0.1",
            ms=4.0,
            capsize=0.0,
            zorder=4,
            label="Target" if label == STYLE_BY_CASE["combined"]["label"] else None,
        )
        axis.plot(x, target, color="0.1", lw=1.6, zorder=3)
        return pred, pred - pred_std, pred + pred_std


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else campaign_root / "figures_observable_mcmc_overlay"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    attrs_by_analysis = _analysis_attrs(campaign_root)
    individual_cases = args.individual_case or ["smf_pair", "sfr", "size_pair"]
    combined_case = args.combined_case
    requested_cases = list(dict.fromkeys([*individual_cases, combined_case]))
    predictions_by_case = {
        case_name: _load_case_predictions(campaign_root, case_name)
        for case_name in requested_cases
    }

    analysis_order = ANALYSIS_ORDER if args.analysis_order is None else [
        None if value.lower() == "none" else value for value in args.analysis_order
    ]

    analysis_meta = {}
    for case_df in predictions_by_case.values():
        for row in case_df[["analysis_key", "analysis", "label"]].drop_duplicates().itertuples(index=False):
            if row.analysis_key not in analysis_meta:
                attrs = attrs_by_analysis[row.analysis]
                analysis_meta[row.analysis_key] = {
                    "analysis": row.analysis,
                    "label": row.label,
                    "attrs": attrs,
                    "x_is_log": bool(attrs.get("xAxisIsLog", False)),
                    "y_is_log": bool(attrs.get("yAxisIsLog", False)),
                }

    actual_predictions = None
    if args.actual_hdf5 is not None:
        actual_predictions = _load_actual_predictions(args.actual_hdf5.expanduser().resolve(), analysis_meta)

    n_panels = len(analysis_order)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 4.35 * nrows), constrained_layout=True)
    axes_flat = axes.ravel()
    legend_handles = None
    legend_labels = None

    for axis, analysis_key in zip(axes_flat, analysis_order, strict=False):
        if analysis_key is None:
            axis.axis("off")
            continue
        meta = analysis_meta[analysis_key]
        matching_individual_cases = [
            case_name
            for case_name in individual_cases
            if analysis_key in set(predictions_by_case[case_name]["analysis_key"].unique())
        ]
        if len(matching_individual_cases) != 1:
            raise ValueError(
                f"Expected exactly one individual case for analysis_key={analysis_key!r}, "
                f"found {matching_individual_cases}"
            )
        individual_case = matching_individual_cases[0]
        individual_df = predictions_by_case[individual_case].query("analysis_key == @analysis_key").sort_values("bin_index")
        combined_df = predictions_by_case[combined_case].query("analysis_key == @analysis_key").sort_values("bin_index")

        x = individual_df["x_plot"].to_numpy(dtype=float)
        target = individual_df["target_value"].to_numpy(dtype=float)
        target_std = individual_df["target_std"].to_numpy(dtype=float)

        individual_line, individual_lower, individual_upper = _plot_target_and_prediction(
            axis,
            x,
            target,
            target_std,
            individual_df["prediction_value"].to_numpy(dtype=float),
            individual_df["prediction_std"].to_numpy(dtype=float),
            color=STYLE_BY_CASE[individual_case]["color"],
            label=STYLE_BY_CASE[individual_case]["label"],
            y_is_log=bool(meta["y_is_log"]),
        )
        combined_line, combined_lower, combined_upper = _plot_target_and_prediction(
            axis,
            x,
            target,
            target_std,
            combined_df["prediction_value"].to_numpy(dtype=float),
            combined_df["prediction_std"].to_numpy(dtype=float),
            color=STYLE_BY_CASE[combined_case]["color"],
            label=STYLE_BY_CASE[combined_case]["label"],
            y_is_log=bool(meta["y_is_log"]),
        )

        if actual_predictions is not None:
            actual = actual_predictions[analysis_key]
            actual_x = actual["x_plot"]
            actual_y = actual["y_value"]
            if bool(meta["y_is_log"]):
                actual_plot = np.log10(np.maximum(actual_y, 1.0e-30))
            else:
                actual_plot = actual_y
            axis.plot(
                actual_x,
                actual_plot,
                color=ACTUAL_STYLE["color"],
                lw=2.0,
                ls=ACTUAL_STYLE["linestyle"],
                zorder=2.5,
                label=ACTUAL_STYLE["label"],
            )
        axis.relim()
        axis.autoscale_view()
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()

        axis.fill_between(
            x,
            individual_lower,
            individual_upper,
            color=STYLE_BY_CASE[individual_case]["color"],
            alpha=0.18,
            zorder=1,
        )
        axis.fill_between(
            x,
            combined_lower,
            combined_upper,
            color=STYLE_BY_CASE[combined_case]["color"],
            alpha=0.18,
            zorder=1,
        )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)

        axis.set_title(str(meta["label"]))
        axis.set_xlabel(str(meta["attrs"].get("xAxisLabel", "x")))
        if meta["y_is_log"]:
            axis.set_ylabel(r"$\log_{10}(\Phi)$")
        else:
            axis.set_ylabel(str(meta["attrs"].get("yAxisLabel", "Observable")))
        axis.grid(alpha=0.22)

        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    for axis in axes_flat[n_panels:]:
        axis.axis("off")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(4, len(legend_labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    png_path = output_dir / f"{args.output_prefix}.png"
    fig.savefig(png_path, dpi=220)
    if args.save_pdf:
        pdf_path = output_dir / f"{args.output_prefix}.pdf"
        fig.savefig(pdf_path)
    plt.close(fig)

    print(png_path)
    if args.save_pdf:
        print(pdf_path)


if __name__ == "__main__":
    main()
