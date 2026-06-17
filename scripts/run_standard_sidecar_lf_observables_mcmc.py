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

try:
    import corner
except ModuleNotFoundError:  # pragma: no cover
    corner = None

try:
    import emcee
except ModuleNotFoundError:  # pragma: no cover
    emcee = None

from galacticus_emu.interactive_observables import load_observables_bundle, predict_observables_bundle
from galacticus_emu.lhs import log_prior_density, transform_from_prior_quantiles, transform_to_prior_quantiles
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
from run_interactive_observables_mcmc import (
    BundlePosterior,
    _best_fit_plot as _standard_best_fit_plot,
    _initial_center_from_results,
    _parameter_specs_for_columns,
    _parse_x_ranges,
    _selected_targets,
    _trace_plot,
)
from run_sidecar_lf_observables_mcmc import (
    _expand_observable_keys,
    _write_model_changes_with_sidecar_summary,
)
from run_sidecar_lf_pca_emulator_mcmc import SidecarLFPosterior, _plot_best_fit as _sidecar_best_fit_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emcee MCMC against saved standard-observable and sidecar LF emulator bundles."
    )
    parser.add_argument("standard_bundle_path", type=Path)
    parser.add_argument("sidecar_bundle_path", type=Path)
    parser.add_argument(
        "--standard-observable",
        action="append",
        required=True,
        help="Standard observable key to include. Repeat for multiple.",
    )
    parser.add_argument(
        "--standard-observable-x-range",
        action="append",
        default=[],
        help="Restrict a standard observable in plotted x units as key=xmin:xmax.",
    )
    parser.add_argument(
        "--sidecar-observable",
        action="append",
        required=True,
        help="Sidecar LF observable key or group to include, e.g. halpha_sobral.",
    )
    parser.add_argument("--mcmc-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-walkers", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--init-quantile-sigma", type=float, default=0.04)
    parser.add_argument(
        "--init-from-results",
        type=Path,
        default=None,
        help=(
            "Initialize walkers around a point saved in an existing MCMC HDF5 results file. "
            "The point is read in physical parameter space, converted to prior quantiles, "
            "then transformed into sampler coordinates when --sample-transformed-parameters is used."
        ),
    )
    parser.add_argument(
        "--init-source",
        choices=["map"],
        default="map",
        help="Which point to read from --init-from-results. Currently only 'map' is supported.",
    )
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
    parser.add_argument("--skip-best-fit-plot", action="store_true", help="Skip only the best-fit observable plots.")
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


def _log_likelihood(
    target: np.ndarray,
    target_variance: np.ndarray,
    means: np.ndarray,
    emulator_variances: np.ndarray,
    *,
    include_emulator_variance: bool,
) -> np.ndarray:
    variance = target_variance[None, :]
    if include_emulator_variance:
        variance = variance + emulator_variances
    variance = np.maximum(variance, 1.0e-12)
    residuals = target[None, :] - means
    return -0.5 * np.sum((residuals**2) / variance + np.log(2.0 * np.pi * variance), axis=1)


class JointPosterior:
    def __init__(
        self,
        *,
        standard_posterior: BundlePosterior,
        sidecar_posterior: SidecarLFPosterior,
        parameter_specs,
        slow_count: int,
        include_emulator_variance: bool,
    ):
        self.standard_posterior = standard_posterior
        self.sidecar_posterior = sidecar_posterior
        self.parameter_specs = parameter_specs
        self.slow_count = slow_count
        self.include_emulator_variance = include_emulator_variance

    def log_probability(self, theta: np.ndarray) -> float:
        return float(self.log_probability_batch(theta[None, :])[0])

    def log_probability_batch(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        log_prior = np.asarray(log_prior_density(self.parameter_specs, theta), dtype=float)
        result = np.full(theta.shape[0], -np.inf, dtype=float)
        valid = np.isfinite(log_prior)
        if not np.any(valid):
            return result

        valid_theta = theta[valid]
        standard_mean, standard_variance = self.standard_posterior.predict_batch(valid_theta[:, : self.slow_count])
        sidecar_mean, sidecar_variance = self.sidecar_posterior.predict_batch(valid_theta)
        standard_like = _log_likelihood(
            self.standard_posterior.target_vector,
            self.standard_posterior.target_variance,
            standard_mean,
            standard_variance,
            include_emulator_variance=self.include_emulator_variance,
        )
        sidecar_like = _log_likelihood(
            self.sidecar_posterior.target_vector,
            self.sidecar_posterior.target_variance,
            sidecar_mean,
            sidecar_variance,
            include_emulator_variance=self.include_emulator_variance,
        )
        log_like = standard_like + sidecar_like
        valid_indices = np.where(valid)[0]
        finite = np.isfinite(log_like)
        result[valid_indices[finite]] = log_prior[valid_indices[finite]] + log_like[finite]
        return result


def _validate_parameter_order(standard_input_columns: list[str], sidecar_bundle: dict) -> tuple[list, list[str], int]:
    parameter_specs = list(sidecar_bundle["parameter_specs"])
    parameter_names = list(sidecar_bundle["parameter_names"])
    slow_count = int(sidecar_bundle["slow_count"])
    sidecar_slow_names = parameter_names[:slow_count]
    if standard_input_columns != sidecar_slow_names:
        raise ValueError(
            "Standard and sidecar bundles have different slow-parameter orders: "
            f"standard={standard_input_columns}, sidecar={sidecar_slow_names}"
        )
    return parameter_specs, parameter_names, slow_count


def _write_standard_prediction_rows(
    bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    best_predictions: dict,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows = []
    prediction_by_key = {}
    for key in observable_keys:
        observable = bundle["observables"][key]
        mask = masks[key]
        prediction = best_predictions[key]
        pred = np.asarray(prediction["y_pred_plot"], dtype=float)[mask]
        pred_std = np.asarray(prediction["y_std_plot"], dtype=float)[mask]
        prediction_by_key[key] = (pred, pred_std)
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
                pred,
                pred_std,
                strict=True,
            )
        ):
            x_value, target_value, sigma_value, pred_value, pred_std_value = values
            rows.append(
                {
                    "observable_key": key,
                    "analysis": observable["analysis"],
                    "bin_index_after_mask": int(bin_index),
                    "x_plot": float(x_value),
                    "target_plot": float(target_value),
                    "target_sigma_plot": float(sigma_value),
                    "prediction_plot": float(pred_value),
                    "prediction_sigma_plot": float(pred_std_value),
                }
            )
    return pd.DataFrame(rows), prediction_by_key


def _write_sidecar_prediction_rows(
    selected_bundles: dict[str, dict],
    mean: np.ndarray,
    variance: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows = []
    prediction_by_key = {}
    start = 0
    std = np.sqrt(np.maximum(variance, 0.0))
    for key, observable in selected_bundles.items():
        n_bins = len(observable["x_plot"])
        pred = mean[start : start + n_bins]
        pred_std = std[start : start + n_bins]
        start += n_bins
        prediction_by_key[key] = (pred, pred_std)
        target_sigma = observable["target_sigma_plot"]
        if target_sigma is None:
            target_sigma = np.full(n_bins, np.nan)
        for bin_index, values in enumerate(
            zip(observable["x_plot"], observable["target_plot"], target_sigma, pred, pred_std, strict=True)
        ):
            x_value, target_value, target_std, pred_value, pred_std_value = values
            rows.append(
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
    return pd.DataFrame(rows), prediction_by_key


def main() -> None:
    args = parse_args()
    if emcee is None:
        raise ModuleNotFoundError("The 'emcee' package is required. Activate galacticus-workspace or install emcee.")
    if args.thin < 1:
        raise ValueError("--thin must be >= 1")
    trace_thin = int(args.trace_thin) if int(args.trace_thin) > 0 else int(args.thin)
    if trace_thin < 1:
        raise ValueError("--trace-thin must be >= 1 when supplied")

    standard_bundle = load_observables_bundle(args.standard_bundle_path)
    sidecar_bundle = joblib.load(args.sidecar_bundle_path)
    if sidecar_bundle.get("bundle_type") != "sidecar_lf_pca_gp":
        raise ValueError(f"{args.sidecar_bundle_path} is not a sidecar_lf_pca_gp bundle")

    standard_input_columns = list(standard_bundle["input_columns"])
    slow_parameter_specs = _parameter_specs_for_columns(standard_input_columns)
    parameter_specs, parameter_names, slow_count = _validate_parameter_order(standard_input_columns, sidecar_bundle)
    if args.n_walkers < 2 * len(parameter_specs):
        raise ValueError(f"--n-walkers should be at least {2 * len(parameter_specs)} for {len(parameter_specs)} dimensions")

    x_ranges = _parse_x_ranges(args.standard_observable_x_range)
    target_vector, target_variance, masks, standard_slices = _selected_targets(
        standard_bundle,
        args.standard_observable,
        x_ranges,
        args.target_sigma_floor,
    )
    standard_posterior = BundlePosterior(
        bundle=standard_bundle,
        observable_keys=args.standard_observable,
        masks=masks,
        target_vector=target_vector,
        target_variance=target_variance,
        parameter_specs=slow_parameter_specs,
        include_emulator_variance=bool(args.include_emulator_variance),
    )
    sidecar_observable_keys = _expand_observable_keys(sidecar_bundle, args.sidecar_observable)
    selected_sidecars = {key: sidecar_bundle["observables"][key] for key in sidecar_observable_keys}
    sidecar_posterior = SidecarLFPosterior(
        parameter_specs=parameter_specs,
        slow_count=slow_count,
        bundles=selected_sidecars,
        include_emulator_variance=bool(args.include_emulator_variance),
        target_sigma_floor=args.target_sigma_floor,
    )
    posterior = JointPosterior(
        standard_posterior=standard_posterior,
        sidecar_posterior=sidecar_posterior,
        parameter_specs=parameter_specs,
        slow_count=slow_count,
        include_emulator_variance=bool(args.include_emulator_variance),
    )

    rng = np.random.default_rng(args.seed)
    center_theta = None
    center_quantiles = np.full(len(parameter_specs), 0.5, dtype=float)
    initialization_summary = {
        "mode": "prior_quantile_center",
        "source": None,
        "path": None,
        "init_quantile_sigma": float(args.init_quantile_sigma),
    }
    if args.init_from_results is not None:
        center_theta = _initial_center_from_results(
            args.init_from_results,
            parameter_names,
            source=args.init_source,
        )
        center_quantiles = np.asarray(transform_to_prior_quantiles(parameter_specs, center_theta[None, :])[0], dtype=float)
        initialization_summary = {
            "mode": "results_file",
            "source": args.init_source,
            "path": str(args.init_from_results.expanduser().resolve()),
            "center_theta": {name: float(value) for name, value in zip(parameter_names, center_theta, strict=True)},
            "center_quantiles": {name: float(value) for name, value in zip(parameter_names, center_quantiles, strict=True)},
            "init_quantile_sigma": float(args.init_quantile_sigma),
        }
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
        sampler = emcee.EnsembleSampler(log_prob_fn=sampler_posterior.log_probability_batch, vectorize=True, **sampler_kwargs)
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
    best_standard_predictions = predict_observables_bundle(
        standard_bundle,
        {name: float(value) for name, value in zip(parameter_names[:slow_count], best_theta[:slow_count], strict=True)},
    )
    standard_df, standard_prediction_by_key = _write_standard_prediction_rows(
        standard_bundle,
        args.standard_observable,
        masks,
        best_standard_predictions,
    )
    standard_predictions_path = args.mcmc_dir / f"{args.output_prefix}_best_fit_standard_observables.csv"
    standard_df.to_csv(standard_predictions_path, index=False)

    best_sidecar_mean, best_sidecar_variance = sidecar_posterior.predict(best_theta)
    sidecar_df, sidecar_prediction_by_key = _write_sidecar_prediction_rows(
        selected_sidecars,
        best_sidecar_mean,
        best_sidecar_variance,
    )
    sidecar_predictions_path = args.mcmc_dir / f"{args.output_prefix}_best_fit_sidecar_lfs.csv"
    sidecar_df.to_csv(sidecar_predictions_path, index=False)

    changes_path = _write_model_changes_with_sidecar_summary(
        parameter_specs,
        slow_count,
        parameter_names,
        best_theta,
        args.mcmc_dir / "maximum_a_posteriori_model_changes.xml",
    )
    best_quantiles = transform_to_prior_quantiles(parameter_specs, best_theta[None, :])[0]
    summary = {
        "standard_bundle_path": str(args.standard_bundle_path.resolve()),
        "sidecar_bundle_path": str(args.sidecar_bundle_path.resolve()),
        "standard_observable_keys": args.standard_observable,
        "standard_observable_x_ranges": {key: list(value) for key, value in x_ranges.items()},
        "sidecar_observable_keys": sidecar_observable_keys,
        "parameter_names": parameter_names,
        "slow_count": slow_count,
        "n_dimensions": len(parameter_specs),
        "n_standard_data": int(standard_posterior.target_vector.size),
        "n_sidecar_data": int(sidecar_posterior.target_vector.size),
        "include_emulator_variance": bool(args.include_emulator_variance),
        "target_sigma_floor": float(args.target_sigma_floor),
        "n_walkers": int(args.n_walkers),
        "n_steps": int(args.n_steps),
        "burn_in": int(args.burn_in),
        "thin": int(args.thin),
        "trace_thin": int(trace_thin),
        "seed": int(args.seed),
        "emcee_moves": describe_emcee_moves(args.move),
        "initialization": initialization_summary,
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
        "standard_output_slices": {key: [slc.start, slc.stop] for key, slc in standard_slices.items()},
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

    joblib.dump(
        {
            "standard_bundle_path": str(args.standard_bundle_path.resolve()),
            "sidecar_bundle_path": str(args.sidecar_bundle_path.resolve()),
            "standard_observable_keys": args.standard_observable,
            "sidecar_observable_keys": sidecar_observable_keys,
            "parameter_names": parameter_names,
            "parameter_specs": parameter_specs,
            "slow_count": slow_count,
            "masks": masks,
            "best_theta": summary["best_theta"],
            "sampler_coordinates": summary["sampler_coordinates"],
        },
        args.mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib",
    )
    fitted_bundle_path = args.mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib"
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
        prediction_paths={
            "best_fit_standard_observables": standard_predictions_path,
            "best_fit_sidecar_lfs": sidecar_predictions_path,
        },
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
    corner_created = False
    if not args.skip_plots and not args.skip_corner_plot and corner is not None:
        corner_path = figures_dir / f"{args.output_prefix}_corner.png"
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

    standard_best_fit_path = figures_dir / f"{args.output_prefix}_best_fit_standard_observables.png"
    standard_best_fit_created = False
    if not args.skip_plots and not args.skip_best_fit_plot:
        _standard_best_fit_plot(
            standard_bundle,
            args.standard_observable,
            masks,
            standard_prediction_by_key,
            standard_best_fit_path,
        )
        standard_best_fit_created = True
    sidecar_best_fit_path = figures_dir / f"{args.output_prefix}_best_fit_sidecar_lfs.png"
    sidecar_best_fit_created = False
    if not args.skip_plots and not args.skip_best_fit_plot:
        _sidecar_best_fit_plot(selected_sidecars, sidecar_prediction_by_key, sidecar_best_fit_path)
        sidecar_best_fit_created = True

    if backend_run.enabled and args.delete_emcee_backend_on_success and emcee_backend_path is not None:
        emcee_backend_path.unlink(missing_ok=True)
        summary["emcee_backend_deleted_on_success"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(summary_path)
    print(changes_path)
    print(standard_predictions_path)
    print(sidecar_predictions_path)
    print(results_hdf5_path)
    if posterior_path is not None:
        print(posterior_path)
    if chain_path is not None:
        print(chain_path)
    if trace_created:
        print(trace_path)
    if corner_created:
        print(corner_path)
    if standard_best_fit_created:
        print(standard_best_fit_path)
    if sidecar_best_fit_created:
        print(sidecar_best_fit_path)


if __name__ == "__main__":
    main()
