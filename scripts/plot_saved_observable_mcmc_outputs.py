from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import joblib
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import corner
except ModuleNotFoundError:  # pragma: no cover
    corner = None

from galacticus_emu.interactive_observables import load_observables_bundle
from galacticus_emu.mcmc_corner import corner_with_log10_priors
from galacticus_emu.mcmc_results import load_posterior_frame, load_run_summary
from run_interactive_observables_mcmc import _best_fit_plot as _standard_best_fit_plot
from run_interactive_observables_mcmc import _selected_targets
from run_interactive_observables_mcmc import _trace_plot
from run_sidecar_lf_pca_emulator_mcmc import _plot_best_fit as _sidecar_best_fit_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate figures from saved observable-MCMC outputs without rerunning emcee."
    )
    parser.add_argument("mcmc_dir", type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--kind",
        choices=["standard", "sidecar", "combined"],
        required=True,
        help="Which MCMC script produced the saved outputs.",
    )
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--skip-trace-plot", action="store_true")
    parser.add_argument("--skip-corner-plot", action="store_true")
    parser.add_argument("--skip-best-fit-plot", action="store_true")
    return parser.parse_args()


def _prediction_by_key(
    frame: pd.DataFrame,
    *,
    key_column: str = "observable_key",
    prediction_column: str = "prediction_plot",
    sigma_column: str = "prediction_sigma_plot",
    order_column: str = "bin_index_after_mask",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for key, rows in frame.groupby(key_column):
        if order_column in rows.columns:
            rows = rows.sort_values(order_column)
        else:
            rows = rows.sort_values("x_plot")
        result[str(key)] = (
            rows[prediction_column].to_numpy(dtype=float),
            rows[sigma_column].to_numpy(dtype=float),
        )
    return result


def _truths(summary: dict, names: list[str]) -> np.ndarray | None:
    best_theta = summary.get("best_theta")
    if not isinstance(best_theta, dict):
        return None
    missing = [name for name in names if name not in best_theta]
    if missing:
        return None
    return np.asarray([best_theta[name] for name in names], dtype=float)


def _plot_corner(
    posterior_path: Path,
    summary: dict,
    parameter_names: list[str],
    parameter_specs: list,
    output_path: Path,
) -> bool:
    if corner is None:
        print("Skipping corner plot: corner is not installed")
        return False
    posterior_df = load_posterior_frame(posterior_path)
    fig = corner_with_log10_priors(
        corner,
        posterior_df,
        parameter_names,
        parameter_specs,
        truths=_truths(summary, parameter_names),
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _plot_trace(
    mcmc_dir: Path,
    output_prefix: str,
    summary: dict,
    parameter_names: list[str],
    parameter_specs: list,
    output_path: Path,
) -> bool:
    chain_path = Path(summary.get("chain_path", mcmc_dir / f"{output_prefix}_chain_thinned.npy"))
    if not chain_path.is_absolute():
        chain_path = mcmc_dir / chain_path
    if chain_path.exists() and not chain_path.name.endswith("_chain.npy"):
        chain = np.load(chain_path)
        burn_in = int(summary.get("trace_autocorrelation_time_burn_in_thinned", 0))
        if burn_in <= 0 and "burn_in" in summary and "thin" in summary:
            burn_in = int(np.ceil(float(summary["burn_in"]) / float(summary["thin"])))
    else:
        results_path = Path(summary.get("results_hdf5_path", mcmc_dir / f"{output_prefix}_mcmc_results.hdf5"))
        if not results_path.is_absolute():
            results_path = mcmc_dir / results_path
        if not results_path.exists():
            print(f"Skipping trace plot: no thinned chain found at {chain_path} and no HDF5 at {results_path}")
            return False
        with h5py.File(results_path, "r") as handle:
            burn_chain = handle["burn_in/chain"][...]
            posterior_chain = handle["posterior/chain"][...]
        chain = np.concatenate([burn_chain, posterior_chain], axis=0)
        burn_in = burn_chain.shape[0]
    _trace_plot(
        chain,
        parameter_names,
        output_path,
        parameter_specs=parameter_specs,
        burn_in=burn_in,
    )
    return True


def _plot_standard(args: argparse.Namespace, figures_dir: Path, summary: dict) -> list[Path]:
    mcmc_dir = args.mcmc_dir
    inputs = joblib.load(mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib")
    bundle = load_observables_bundle(Path(inputs["bundle_path"]))
    parameter_names = list(inputs["input_columns"])
    parameter_specs = list(inputs["parameter_specs"])
    created: list[Path] = []

    if not args.skip_trace_plot:
        path = figures_dir / f"{args.output_prefix}_trace.png"
        if _plot_trace(mcmc_dir, args.output_prefix, summary, parameter_names, parameter_specs, path):
            created.append(path)
    if not args.skip_corner_plot:
        path = figures_dir / f"{args.output_prefix}_corner.png"
        if _plot_corner(
            mcmc_dir / f"{args.output_prefix}_mcmc_results.hdf5",
            summary,
            parameter_names,
            parameter_specs,
            path,
        ):
            created.append(path)
    if not args.skip_best_fit_plot:
        predictions = pd.read_csv(mcmc_dir / f"{args.output_prefix}_best_fit_observables.csv")
        path = figures_dir / f"{args.output_prefix}_best_fit_observables.png"
        masks = inputs.get("masks")
        if masks is None:
            x_ranges = {key: tuple(value) for key, value in summary.get("observable_x_ranges", {}).items()}
            _, _, masks, _ = _selected_targets(
                bundle,
                list(inputs["observable_keys"]),
                x_ranges,
                float(summary.get("target_sigma_floor", 1.0e-3)),
            )
        _standard_best_fit_plot(
            bundle,
            list(inputs["observable_keys"]),
            masks,
            _prediction_by_key(predictions),
            path,
        )
        created.append(path)
    return created


def _plot_sidecar(args: argparse.Namespace, figures_dir: Path, summary: dict) -> list[Path]:
    mcmc_dir = args.mcmc_dir
    inputs = joblib.load(mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib")
    bundle = joblib.load(Path(inputs["bundle_path"]))
    parameter_names = list(inputs["parameter_names"])
    parameter_specs = list(inputs["parameter_specs"])
    selected = {key: bundle["observables"][key] for key in inputs["observable_keys"]}
    created: list[Path] = []

    if not args.skip_trace_plot:
        path = figures_dir / f"{args.output_prefix}_trace.png"
        if _plot_trace(mcmc_dir, args.output_prefix, summary, parameter_names, parameter_specs, path):
            created.append(path)
    if not args.skip_corner_plot:
        path = figures_dir / f"{args.output_prefix}_corner.png"
        if _plot_corner(
            mcmc_dir / f"{args.output_prefix}_mcmc_results.hdf5",
            summary,
            parameter_names,
            parameter_specs,
            path,
        ):
            created.append(path)
    if not args.skip_best_fit_plot:
        predictions = pd.read_csv(mcmc_dir / f"{args.output_prefix}_best_fit_observables.csv")
        path = figures_dir / f"{args.output_prefix}_best_fit_observables.png"
        _sidecar_best_fit_plot(
            selected,
            _prediction_by_key(
                predictions,
                prediction_column="prediction_log10_phi",
                sigma_column="prediction_log10_phi_std",
                order_column="bin_index",
            ),
            path,
        )
        created.append(path)
    return created


def _plot_combined(args: argparse.Namespace, figures_dir: Path, summary: dict) -> list[Path]:
    mcmc_dir = args.mcmc_dir
    inputs = joblib.load(mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib")
    standard_bundle = load_observables_bundle(Path(inputs["standard_bundle_path"]))
    sidecar_bundle = joblib.load(Path(inputs["sidecar_bundle_path"]))
    parameter_names = list(inputs["parameter_names"])
    parameter_specs = list(inputs["parameter_specs"])
    selected_sidecars = {key: sidecar_bundle["observables"][key] for key in inputs["sidecar_observable_keys"]}
    created: list[Path] = []

    if not args.skip_trace_plot:
        path = figures_dir / f"{args.output_prefix}_trace.png"
        if _plot_trace(mcmc_dir, args.output_prefix, summary, parameter_names, parameter_specs, path):
            created.append(path)
    if not args.skip_corner_plot:
        path = figures_dir / f"{args.output_prefix}_corner.png"
        if _plot_corner(
            mcmc_dir / f"{args.output_prefix}_mcmc_results.hdf5",
            summary,
            parameter_names,
            parameter_specs,
            path,
        ):
            created.append(path)
    if not args.skip_best_fit_plot:
        standard_predictions = pd.read_csv(mcmc_dir / f"{args.output_prefix}_best_fit_standard_observables.csv")
        standard_path = figures_dir / f"{args.output_prefix}_best_fit_standard_observables.png"
        masks = inputs.get("masks")
        if masks is None:
            x_ranges = {key: tuple(value) for key, value in summary.get("standard_observable_x_ranges", {}).items()}
            _, _, masks, _ = _selected_targets(
                standard_bundle,
                list(inputs["standard_observable_keys"]),
                x_ranges,
                float(summary.get("target_sigma_floor", 1.0e-3)),
            )
        _standard_best_fit_plot(
            standard_bundle,
            list(inputs["standard_observable_keys"]),
            masks,
            _prediction_by_key(standard_predictions),
            standard_path,
        )
        created.append(standard_path)

        sidecar_predictions = pd.read_csv(mcmc_dir / f"{args.output_prefix}_best_fit_sidecar_lfs.csv")
        sidecar_path = figures_dir / f"{args.output_prefix}_best_fit_sidecar_lfs.png"
        _sidecar_best_fit_plot(
            selected_sidecars,
            _prediction_by_key(
                sidecar_predictions,
                prediction_column="prediction_log10_phi",
                sigma_column="prediction_log10_phi_std",
                order_column="bin_index",
            ),
            sidecar_path,
        )
        created.append(sidecar_path)
    return created


def main() -> None:
    args = parse_args()
    args.mcmc_dir = args.mcmc_dir.expanduser().resolve()
    figures_dir = args.figures_dir.expanduser().resolve() if args.figures_dir else args.mcmc_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.mcmc_dir / f"{args.output_prefix}_mcmc_results.hdf5"
    summary_path = results_path if results_path.exists() else args.mcmc_dir / f"{args.output_prefix}_run_summary.json"
    summary = load_run_summary(summary_path)

    if args.kind == "standard":
        created = _plot_standard(args, figures_dir, summary)
    elif args.kind == "sidecar":
        created = _plot_sidecar(args, figures_dir, summary)
    else:
        created = _plot_combined(args, figures_dir, summary)

    for path in created:
        print(path)


if __name__ == "__main__":
    main()
