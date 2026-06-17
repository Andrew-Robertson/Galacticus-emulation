from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import emcee
except ModuleNotFoundError:  # pragma: no cover
    emcee = None

try:
    import corner
except ModuleNotFoundError:  # pragma: no cover
    corner = None

from galacticus_emu.lhs import (
    transform_from_prior_quantiles,
    transform_to_prior_quantiles,
)
from galacticus_emu.mcmc_coordinates import (
    SamplerCoordinatePosterior,
    add_sampler_coordinate_arguments,
    best_physical_sample_from_sampler_history,
    physical_log_prob_from_sampler_log_prob,
    physical_to_sampler_coordinates,
    sampler_coordinate_summary,
    sampler_to_physical_coordinates,
)
from galacticus_emu.mcmc_backend import (
    add_emcee_backend_arguments,
    emcee_backend_run_settings,
    original_steps_for_saved_chain,
    prepare_emcee_backend,
    resolve_emcee_backend_path,
    run_mcmc_with_backend_settings,
    trace_chain_thin,
)
from galacticus_emu.mcmc_corner import corner_with_log10_priors
from galacticus_emu.mcmc_moves import add_emcee_move_arguments, build_emcee_moves, describe_emcee_moves
from galacticus_emu.mcmc_results import split_thinned_chain, write_mcmc_results_hdf5
from run_sidecar_lf_pca_emulator_mcmc import (
    SidecarLFPosterior,
    _plot_best_fit,
    _trace_plot,
    _write_model_changes_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emcee MCMC against a saved sidecar emission-line LF emulator bundle."
    )
    parser.add_argument("bundle_path", type=Path)
    parser.add_argument(
        "--observable",
        action="append",
        required=True,
        help="Observable key or group to include. For example: halpha_sobral or halpha_sobral_z1.",
    )
    parser.add_argument("--mcmc-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-walkers", type=int, default=96)
    parser.add_argument("--n-steps", type=int, default=6000)
    parser.add_argument("--burn-in", type=int, default=1200)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--init-quantile-sigma", type=float, default=0.04)
    parser.add_argument("--target-sigma-floor", type=float, default=1.0e-3)
    parser.add_argument("--include-emulator-variance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--save-full-chain",
        action="store_true",
        help=(
            "Save the full unthinned physical chain as *_chain.npy. "
            "By default chains are stored only in the canonical HDF5 results file."
        ),
    )
    parser.add_argument(
        "--save-thinned-chain",
        action="store_true",
        help="Also save *_chain_thinned.npy and *_log_prob_thinned.npy compatibility files.",
    )
    parser.add_argument(
        "--save-sampler-chain",
        action="store_true",
        help="Also save sampler-coordinate chains. Full sampler chains are only saved with --save-full-chain.",
    )
    parser.add_argument("--trace-thin", type=int, default=0, help="Trace-plot thinning. Default 0 means use --thin.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip all plot generation.")
    parser.add_argument("--skip-trace-plot", action="store_true", help="Skip only the trace plot.")
    parser.add_argument("--skip-corner-plot", action="store_true", help="Skip only the corner plot.")
    parser.add_argument("--skip-best-fit-plot", action="store_true", help="Skip only the best-fit observable plot.")
    parser.add_argument("--save-posterior-csv", action="store_true", help="Also export posterior samples to CSV.")
    parser.add_argument(
        "--results-hdf5",
        type=Path,
        default=None,
        help="Canonical HDF5 results path. Defaults to mcmc-dir/output-prefix_mcmc_results.hdf5.",
    )
    add_emcee_backend_arguments(parser)
    add_emcee_move_arguments(parser)
    add_sampler_coordinate_arguments(parser)
    return parser.parse_args()


def _expand_observable_keys(bundle: dict, requested: list[str]) -> list[str]:
    result = []
    groups = bundle.get("observable_groups", {})
    observables = bundle.get("observables", {})
    for key in requested:
        if key in groups:
            result.extend(groups[key])
        elif key in observables:
            result.append(key)
        else:
            raise ValueError(
                f"Unknown observable {key!r}. Available groups: {sorted(groups)}; "
                f"available observables: {sorted(observables)}"
            )
    deduped = []
    seen = set()
    for key in result:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def _write_model_changes_with_sidecar_summary(
    parameter_specs,
    slow_count: int,
    parameter_names: list[str],
    theta: np.ndarray,
    output_path: Path,
) -> Path:
    path = _write_model_changes_file(parameter_specs[:slow_count], theta[:slow_count], output_path)
    root = ET.parse(path).getroot()
    sidecar_comment = ET.Comment(
        " Sidecar LF nuisance parameters at MAP: "
        + ", ".join(
            f"{name}={float(value):.6g}"
            for name, value in zip(parameter_names[slow_count:], theta[slow_count:], strict=True)
        )
        + " "
    )
    root.append(sidecar_comment)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
    return path


def main() -> None:
    args = parse_args()
    if emcee is None:
        raise ModuleNotFoundError("The 'emcee' package is required. Activate galacticus-workspace or install emcee.")
    if args.thin < 1:
        raise ValueError("--thin must be >= 1")
    trace_thin = int(args.trace_thin) if int(args.trace_thin) > 0 else int(args.thin)
    if trace_thin < 1:
        raise ValueError("--trace-thin must be >= 1 when supplied")

    bundle = joblib.load(args.bundle_path)
    if bundle.get("bundle_type") != "sidecar_lf_pca_gp":
        raise ValueError(f"{args.bundle_path} is not a sidecar_lf_pca_gp bundle")
    observable_keys = _expand_observable_keys(bundle, args.observable)
    selected_bundles = {key: bundle["observables"][key] for key in observable_keys}
    parameter_specs = bundle["parameter_specs"]
    parameter_names = bundle["parameter_names"]
    slow_count = int(bundle["slow_count"])
    if args.n_walkers < 2 * len(parameter_specs):
        raise ValueError(f"--n-walkers should be at least {2 * len(parameter_specs)} for {len(parameter_specs)} dimensions")

    posterior = SidecarLFPosterior(
        parameter_specs=parameter_specs,
        slow_count=slow_count,
        bundles=selected_bundles,
        include_emulator_variance=args.include_emulator_variance,
        target_sigma_floor=args.target_sigma_floor,
    )

    rng = np.random.default_rng(args.seed)
    center_quantiles = np.full(len(parameter_specs), 0.5, dtype=float)
    initial_quantiles = np.clip(
        center_quantiles + args.init_quantile_sigma * rng.normal(size=(args.n_walkers, len(parameter_specs))),
        1.0e-4,
        1.0 - 1.0e-4,
    )
    initial_positions_physical = transform_from_prior_quantiles(parameter_specs, initial_quantiles)
    initial_positions = physical_to_sampler_coordinates(
        parameter_specs,
        initial_positions_physical,
        enabled=args.sample_transformed_parameters,
    )

    args.mcmc_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.figures_dir or args.mcmc_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    emcee_backend_path = resolve_emcee_backend_path(
        args.emcee_backend_hdf5,
        mcmc_dir=args.mcmc_dir,
        output_prefix=args.output_prefix,
    )
    backend_run = emcee_backend_run_settings(
        backend_path=emcee_backend_path,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        thin=args.thin,
    )
    if backend_run.enabled and args.save_full_chain:
        raise ValueError(
            "--save-full-chain is incompatible with the default thinned emcee backend; "
            "pass --emcee-backend-hdf5 none to save a full unthinned chain."
        )

    moves = build_emcee_moves(emcee, args.move)
    sampler_kwargs = {"nwalkers": args.n_walkers, "ndim": len(parameter_specs)}
    if moves is not None:
        sampler_kwargs["moves"] = moves
    backend = prepare_emcee_backend(
        emcee,
        emcee_backend_path,
        n_walkers=args.n_walkers,
        n_dim=len(parameter_specs),
    )
    if backend is not None:
        sampler_kwargs["backend"] = backend
    sampler_posterior = SamplerCoordinatePosterior(
        posterior,
        parameter_specs,
        enabled=args.sample_transformed_parameters,
    )
    if args.n_processes > 1:
        from multiprocessing import Pool

        with Pool(processes=args.n_processes) as pool:
            sampler = emcee.EnsembleSampler(
                log_prob_fn=sampler_posterior.log_probability,
                pool=pool,
                vectorize=False,
                **sampler_kwargs,
            )
            run_mcmc_with_backend_settings(
                sampler,
                initial_positions,
                backend_run=backend_run,
                n_steps=args.n_steps,
                progress=args.progress,
                skip_initial_state_check=True,
            )
    else:
        sampler = emcee.EnsembleSampler(
            log_prob_fn=sampler_posterior.log_probability_batch,
            vectorize=True,
            **sampler_kwargs,
        )
        run_mcmc_with_backend_settings(
            sampler,
            initial_positions,
            backend_run=backend_run,
            n_steps=args.n_steps,
            progress=args.progress,
            skip_initial_state_check=True,
        )

    thinned_sampler_chain = sampler.get_chain(thin=backend_run.chain_thin)
    thinned_sampler_log_prob = sampler.get_log_prob(thin=backend_run.chain_thin)
    saved_original_steps = original_steps_for_saved_chain(
        thinned_sampler_chain.shape[0],
        store_thin=backend_run.store_thin,
    )
    saved_burn_in_count = int(np.count_nonzero(saved_original_steps < int(args.burn_in)))
    thinned_chain = sampler_to_physical_coordinates(
        parameter_specs,
        thinned_sampler_chain,
        enabled=args.sample_transformed_parameters,
    )
    thinned_log_prob = physical_log_prob_from_sampler_log_prob(
        parameter_specs,
        thinned_sampler_chain,
        thinned_sampler_log_prob,
        enabled=args.sample_transformed_parameters,
    )
    flat_sampler_samples = sampler.get_chain(discard=backend_run.burn_in_discard, thin=backend_run.chain_thin, flat=True)
    flat_sampler_log_prob = sampler.get_log_prob(discard=backend_run.burn_in_discard, thin=backend_run.chain_thin, flat=True)
    flat_samples = sampler_to_physical_coordinates(
        parameter_specs,
        flat_sampler_samples,
        enabled=args.sample_transformed_parameters,
    )
    flat_log_prob = physical_log_prob_from_sampler_log_prob(
        parameter_specs,
        flat_sampler_samples,
        flat_sampler_log_prob,
        enabled=args.sample_transformed_parameters,
    )
    chain_path = None
    log_prob_path = None
    if args.save_thinned_chain:
        chain_path = args.mcmc_dir / f"{args.output_prefix}_chain_thinned.npy"
        log_prob_path = args.mcmc_dir / f"{args.output_prefix}_log_prob_thinned.npy"
        np.save(chain_path, thinned_chain)
        np.save(log_prob_path, thinned_log_prob)
        if args.sample_transformed_parameters and args.save_sampler_chain:
            np.save(args.mcmc_dir / f"{args.output_prefix}_sampler_chain_thinned.npy", thinned_sampler_chain)
            np.save(args.mcmc_dir / f"{args.output_prefix}_sampler_log_prob_thinned.npy", thinned_sampler_log_prob)

    full_chain_path = None
    full_log_prob_path = None
    if args.save_full_chain:
        sampler_chain = sampler.get_chain()
        sampler_log_prob = sampler.get_log_prob()
        full_chain = sampler_to_physical_coordinates(
            parameter_specs,
            sampler_chain,
            enabled=args.sample_transformed_parameters,
        )
        full_log_prob = physical_log_prob_from_sampler_log_prob(
            parameter_specs,
            sampler_chain,
            sampler_log_prob,
            enabled=args.sample_transformed_parameters,
        )
        full_chain_path = args.mcmc_dir / f"{args.output_prefix}_chain.npy"
        full_log_prob_path = args.mcmc_dir / f"{args.output_prefix}_log_prob.npy"
        np.save(full_chain_path, full_chain)
        np.save(full_log_prob_path, full_log_prob)
        if args.sample_transformed_parameters and args.save_sampler_chain:
            np.save(args.mcmc_dir / f"{args.output_prefix}_sampler_chain.npy", sampler_chain)
            np.save(args.mcmc_dir / f"{args.output_prefix}_sampler_log_prob.npy", sampler_log_prob)

    posterior_df = pd.DataFrame(flat_samples, columns=parameter_names)
    posterior_df["log_probability"] = flat_log_prob
    posterior_path = None
    if args.save_posterior_csv:
        posterior_path = args.mcmc_dir / f"{args.output_prefix}_posterior_samples.csv"
        posterior_df.to_csv(posterior_path, index=False)

    best_theta, best_log_probability, best_chain_index = best_physical_sample_from_sampler_history(
        parameter_specs,
        thinned_sampler_chain[saved_burn_in_count:],
        thinned_sampler_log_prob[saved_burn_in_count:],
        enabled=args.sample_transformed_parameters,
    )
    best_mean, best_var = posterior.predict(best_theta)
    best_std = np.sqrt(np.maximum(best_var, 0.0))

    prediction_rows = []
    prediction_by_key = {}
    start = 0
    for key, observable in selected_bundles.items():
        n_bins = len(observable["x_plot"])
        pred = best_mean[start : start + n_bins]
        pred_std = best_std[start : start + n_bins]
        start += n_bins
        prediction_by_key[key] = (pred, pred_std)
        target_sigma = observable["target_sigma_plot"]
        if target_sigma is None:
            target_sigma = np.full(n_bins, np.nan)
        for bin_index, values in enumerate(
            zip(observable["x_plot"], observable["target_plot"], target_sigma, pred, pred_std, strict=True)
        ):
            x_value, target_value, target_std, pred_value, pred_std_value = values
            prediction_rows.append(
                {
                    "observable_key": key,
                    "observable": observable["observable"],
                    "sample_label": observable["sample_label"],
                    "bin_index": int(bin_index),
                    "x_plot": float(x_value),
                    "target_log10_phi": float(target_value),
                    "target_log10_phi_std": float(target_std),
                    "prediction_log10_phi": float(pred_value),
                    "prediction_log10_phi_std": float(pred_std_value),
                }
            )
    predictions_path = args.mcmc_dir / f"{args.output_prefix}_best_fit_observables.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    changes_path = _write_model_changes_with_sidecar_summary(
        parameter_specs,
        slow_count,
        parameter_names,
        best_theta,
        args.mcmc_dir / "maximum_a_posteriori_model_changes.xml",
    )
    best_quantiles = transform_to_prior_quantiles(parameter_specs, best_theta[None, :])[0]
    summary = {
        "bundle_path": str(args.bundle_path.resolve()),
        "observable_keys": observable_keys,
        "input_columns": bundle["input_columns"],
        "parameter_names": parameter_names,
        "n_dimensions": len(parameter_specs),
        "n_data": int(posterior.target_vector.size),
        "include_emulator_variance": bool(args.include_emulator_variance),
        "target_sigma_floor": float(args.target_sigma_floor),
        "n_walkers": int(args.n_walkers),
        "n_steps": int(args.n_steps),
        "burn_in": int(args.burn_in),
        "thin": int(args.thin),
        "trace_thin": int(trace_thin),
        "seed": int(args.seed),
        "emcee_moves": describe_emcee_moves(args.move),
        "sampler_coordinates": sampler_coordinate_summary(
            parameter_specs,
            parameter_names,
            enabled=args.sample_transformed_parameters,
        ),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "best_log_probability": float(best_log_probability),
        "best_chain_index_after_burn_in": list(best_chain_index),
        "best_theta": {name: float(value) for name, value in zip(parameter_names, best_theta, strict=True)},
        "best_quantiles": {name: float(value) for name, value in zip(parameter_names, best_quantiles, strict=True)},
        "maximum_a_posteriori_model_changes": str(changes_path),
        "chain_is_thinned": True,
        "save_thinned_chain": bool(args.save_thinned_chain),
        "save_full_chain": bool(args.save_full_chain),
        "save_posterior_csv": bool(args.save_posterior_csv),
        "emcee_backend_hdf5": str(emcee_backend_path) if emcee_backend_path is not None else None,
        "emcee_backend_enabled": bool(backend_run.enabled),
        "emcee_backend_thinned_on_write": bool(backend_run.enabled),
        "emcee_backend_store_thin": int(backend_run.store_thin),
        "emcee_backend_saved_steps": int(thinned_sampler_chain.shape[0]),
        "emcee_backend_proposal_steps_completed": int(thinned_sampler_chain.shape[0] * backend_run.store_thin),
        "delete_emcee_backend_on_success": bool(args.delete_emcee_backend_on_success),
    }
    if chain_path is not None and log_prob_path is not None:
        summary["chain_path"] = str(chain_path)
        summary["log_prob_path"] = str(log_prob_path)
    if full_chain_path is not None and full_log_prob_path is not None:
        summary["full_chain_path"] = str(full_chain_path)
        summary["full_log_prob_path"] = str(full_log_prob_path)
    summary_path = args.mcmc_dir / f"{args.output_prefix}_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    fitted_bundle_path = args.mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib"
    joblib.dump(
        {
            "bundle_path": str(args.bundle_path.resolve()),
            "observable_keys": observable_keys,
            "parameter_names": parameter_names,
            "parameter_specs": parameter_specs,
            "slow_count": slow_count,
            "target_vector": posterior.target_vector,
            "target_variance": posterior.target_variance,
            "best_theta": summary["best_theta"],
            "sampler_coordinates": summary["sampler_coordinates"],
        },
        fitted_bundle_path,
    )
    split_chain = split_thinned_chain(
        thinned_chain,
        thinned_log_prob,
        burn_in=args.burn_in,
        thin=args.thin,
    )
    results_hdf5_path = args.results_hdf5 or args.mcmc_dir / f"{args.output_prefix}_mcmc_results.hdf5"
    write_mcmc_results_hdf5(
        results_hdf5_path,
        parameter_names=parameter_names,
        parameter_specs=parameter_specs,
        summary=summary,
        map_changes_path=changes_path,
        prediction_paths={"best_fit_observables": predictions_path},
        posterior_samples_path=posterior_path,
        mcmc_inputs_path=fitted_bundle_path,
        source_chain_path=chain_path,
        source_log_prob_path=log_prob_path,
        **split_chain,
    )
    summary["results_hdf5_path"] = str(results_hdf5_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    trace_path = figures_dir / f"{args.output_prefix}_trace.png"
    trace_created = False
    if not args.skip_plots and not args.skip_trace_plot:
        effective_trace_chain_thin = trace_chain_thin(
            requested_trace_thin=trace_thin,
            stored_thin=backend_run.store_thin,
            backend_enabled=backend_run.enabled,
        )
        effective_trace_thin = backend_run.store_thin * effective_trace_chain_thin if backend_run.enabled else trace_thin
        if effective_trace_chain_thin == backend_run.chain_thin:
            trace_chain = thinned_chain
        else:
            trace_sampler_chain = sampler.get_chain(thin=effective_trace_chain_thin)
            trace_chain = sampler_to_physical_coordinates(
                parameter_specs,
                trace_sampler_chain,
                enabled=args.sample_transformed_parameters,
            )
        trace_burn_in = int(np.ceil(args.burn_in / effective_trace_thin))
        trace_tau = _trace_plot(
            trace_chain,
            parameter_names,
            trace_path,
            parameter_specs=parameter_specs,
            burn_in=trace_burn_in,
        )
        summary["trace_autocorrelation_time"] = trace_tau
        summary["trace_autocorrelation_time_burn_in"] = int(args.burn_in)
        summary["trace_autocorrelation_time_burn_in_thinned"] = trace_burn_in
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        trace_created = True
    corner_path = figures_dir / f"{args.output_prefix}_corner.png"
    corner_created = False
    if not args.skip_plots and not args.skip_corner_plot and corner is not None:
        corner_fig = corner_with_log10_priors(
            corner,
            posterior_df,
            parameter_names,
            parameter_specs,
            truths=best_theta,
        )
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)
        corner_created = True
    best_fit_path = figures_dir / f"{args.output_prefix}_best_fit_observables.png"
    best_fit_created = False
    if not args.skip_plots and not args.skip_best_fit_plot:
        _plot_best_fit(selected_bundles, prediction_by_key, best_fit_path)
        best_fit_created = True

    if backend_run.enabled and args.delete_emcee_backend_on_success and emcee_backend_path is not None:
        emcee_backend_path.unlink(missing_ok=True)
        summary["emcee_backend_deleted_on_success"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(summary_path)
    print(changes_path)
    print(predictions_path)
    print(fitted_bundle_path)
    print(results_hdf5_path)
    if posterior_path is not None:
        print(posterior_path)
    if chain_path is not None:
        print(chain_path)
    if trace_created:
        print(trace_path)
    if corner_created:
        print(corner_path)
    if best_fit_created:
        print(best_fit_path)


if __name__ == "__main__":
    main()
