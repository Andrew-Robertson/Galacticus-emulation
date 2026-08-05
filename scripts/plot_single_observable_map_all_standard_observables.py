from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu.interactive_observables import load_observables_bundle, predict_observables_bundle
from galacticus_emu.mcmc_results import load_run_summary
from galacticus_emu.plotting import set_ylim_from_values


DEFAULT_ALL_STANDARD_OBSERVABLES = [
    "smf_z0",
    "smf_z3",
    "sfr_function_robotham2011",
    "size_mass_vdw2014_star_forming_z0",
    "size_mass_vdw2014_quiescent_z0",
    "bh_velocity_dispersion",
    "mzr_blanc2019",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each single-observable MCMC result, evaluate its MAP point with a standard-observables "
            "emulator bundle and plot the prediction for all standard calibration observables."
        )
    )
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--mcmc-root", type=Path, required=True)
    parser.add_argument(
        "--observable",
        action="append",
        default=None,
        help=(
            "Observable key to include in the all-standard prediction plot. Repeat to override the default "
            "seven-observable production set."
        ),
    )
    parser.add_argument(
        "--summary-glob",
        default="*/*_run_summary.json",
        help="Glob, relative to --mcmc-root, used to find MCMC run summaries.",
    )
    parser.add_argument(
        "--output-suffix",
        default="best_fit_all_standard_observables",
        help="Suffix used in output filenames after the MCMC prefix.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV path. Defaults to <mcmc-root>/all_standard_prediction_figures.csv.",
    )
    return parser.parse_args()


def _masked_observable_arrays(observable: dict) -> dict[str, np.ndarray | None]:
    x = np.asarray(observable["x_plot"], dtype=float)
    mask = np.ones(x.shape, dtype=bool)
    target = np.asarray(observable["target_plot"], dtype=float)[mask]
    sigma = observable.get("target_sigma_plot")
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)[mask]
    return {
        "mask": mask,
        "x": x[mask],
        "target": target,
        "target_sigma": sigma,
    }


def _write_prediction_csv(
    *,
    bundle: dict,
    observable_keys: list[str],
    predictions: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    source_summary: dict,
) -> None:
    rows = []
    source_observables = ",".join(source_summary.get("observable_keys", []))
    for key in observable_keys:
        observable = bundle["observables"][key]
        arrays = _masked_observable_arrays(observable)
        prediction = predictions[key]
        pred = np.asarray(prediction["y_pred_plot"], dtype=float)[arrays["mask"]]
        pred_std = np.asarray(prediction["y_std_plot"], dtype=float)[arrays["mask"]]
        target_sigma = arrays["target_sigma"]
        if target_sigma is None:
            target_sigma = np.full_like(arrays["target"], np.nan, dtype=float)
        for local_bin, (x_value, target_value, sigma_value, pred_value, pred_std_value) in enumerate(
            zip(
                arrays["x"],
                arrays["target"],
                target_sigma,
                pred,
                pred_std,
                strict=True,
            )
        ):
            rows.append(
                {
                    "source_mcmc_observable_keys": source_observables,
                    "observable_key": key,
                    "analysis": observable["analysis"],
                    "bin_index_after_mask": int(local_bin),
                    "x_plot": float(x_value),
                    "target_plot": float(target_value),
                    "target_sigma_plot": float(sigma_value),
                    "prediction_plot": float(pred_value),
                    "prediction_sigma_plot": float(pred_std_value),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _plot_predictions(
    *,
    bundle: dict,
    observable_keys: list[str],
    predictions: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    dpi: int,
) -> None:
    n_panels = len(observable_keys)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, key in zip(axes_flat, observable_keys, strict=False):
        observable = bundle["observables"][key]
        arrays = _masked_observable_arrays(observable)
        prediction = predictions[key]
        pred = np.asarray(prediction["y_pred_plot"], dtype=float)[arrays["mask"]]
        pred_std = np.asarray(prediction["y_std_plot"], dtype=float)[arrays["mask"]]
        axis.errorbar(
            arrays["x"],
            arrays["target"],
            yerr=arrays["target_sigma"],
            fmt="o",
            color="0.15",
            label=observable.get("target_label") or "target",
        )
        axis.plot(arrays["x"], pred, color="tab:blue", lw=1.8, label="emulator MAP")
        axis.fill_between(arrays["x"], pred - pred_std, pred + pred_std, color="tab:blue", alpha=0.2)
        set_ylim_from_values(axis, arrays["target"], pred)
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


def _summary_prefix(summary_path: Path) -> str:
    suffix = "_run_summary"
    stem = summary_path.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Cannot infer MCMC output prefix from {summary_path}")
    return stem[: -len(suffix)]


def main() -> None:
    args = parse_args()
    observable_keys = args.observable or DEFAULT_ALL_STANDARD_OBSERVABLES
    bundle = load_observables_bundle(args.bundle_path)
    missing = [key for key in observable_keys if key not in bundle["observables"]]
    if missing:
        raise ValueError(f"Observable(s) absent from bundle: {', '.join(missing)}")

    summary_paths = sorted(args.mcmc_root.glob(args.summary_glob))
    if not summary_paths:
        raise FileNotFoundError(f"No run summaries matched {args.mcmc_root / args.summary_glob}")

    manifest_path = args.manifest or args.mcmc_root / "all_standard_prediction_figures.csv"
    manifest_rows = []
    for summary_path in summary_paths:
        summary = load_run_summary(summary_path)
        best_theta = summary.get("best_theta")
        if not isinstance(best_theta, dict):
            raise ValueError(f"{summary_path} does not contain a best_theta dictionary")
        prefix = _summary_prefix(summary_path)
        predictions = predict_observables_bundle(bundle, best_theta)
        mcmc_dir = summary_path.parent
        csv_path = mcmc_dir / f"{prefix}_{args.output_suffix}.csv"
        figure_path = mcmc_dir / "figures" / f"{prefix}_{args.output_suffix}.png"
        _write_prediction_csv(
            bundle=bundle,
            observable_keys=observable_keys,
            predictions=predictions,
            output_path=csv_path,
            source_summary=summary,
        )
        _plot_predictions(
            bundle=bundle,
            observable_keys=observable_keys,
            predictions=predictions,
            output_path=figure_path,
            dpi=args.dpi,
        )
        manifest_rows.append(
            {
                "source_mcmc_dir": str(mcmc_dir),
                "source_mcmc_observable_keys": ",".join(summary.get("observable_keys", [])),
                "run_summary": str(summary_path),
                "prediction_csv": str(csv_path),
                "figure": str(figure_path),
            }
        )
        print(f"Wrote {figure_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_mcmc_dir", "source_mcmc_observable_keys", "run_summary", "prediction_csv", "figure"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
