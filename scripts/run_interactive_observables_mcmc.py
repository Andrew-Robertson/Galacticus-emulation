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

import joblib
import h5py
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

from galacticus_emu.interactive_observables import (
    load_observables_bundle,
    predict_observables_bundle,
)
from galacticus_emu.gp import predict_scaled_gp
from galacticus_emu.lhs import (
    log_prior_density,
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
from galacticus_emu.mcmc_trace import make_trace_plot
from galacticus_emu.plotting import set_ylim_from_values
from galacticus_emu.specs import trinity_parameter_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emcee MCMC against a saved interactive-observables emulator bundle."
    )
    parser.add_argument("bundle_path", type=Path)
    parser.add_argument("--observable", action="append", required=True, help="Observable key to include. Repeat for multiple.")
    parser.add_argument(
        "--observable-x-range",
        action="append",
        default=[],
        help="Restrict one observable in plotted x units as key=xmin:xmax, e.g. bh_halo_mass_trinity_z1=12:13.8.",
    )
    parser.add_argument("--mcmc-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-walkers", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--burn-in", type=int, default=800)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--init-quantile-sigma", type=float, default=0.04)
    parser.add_argument(
        "--init-from-results",
        type=Path,
        default=None,
        help=(
            "Initialize walkers around a point stored in an existing *_mcmc_results.hdf5 file. "
            "The stored physical parameters are aligned by name and then converted into this run's "
            "sampler coordinates, so this is compatible with --sample-transformed-parameters."
        ),
    )
    parser.add_argument(
        "--init-source",
        choices=["map"],
        default="map",
        help="Which point to read from --init-from-results. Currently only 'map' is supported.",
    )
    parser.add_argument("--target-sigma-floor", type=float, default=1.0e-3)
    parser.add_argument(
        "--include-emulator-variance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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


def _parameter_specs_for_columns(input_columns: list[str]):
    specs_by_name = {spec.short_name: spec for spec in trinity_parameter_specs()}
    missing = [column for column in input_columns if column not in specs_by_name]
    if missing:
        raise ValueError(f"No parameter specification found for input column(s): {missing}")
    return [specs_by_name[column] for column in input_columns]


def _decode_hdf5_strings(values: np.ndarray) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def _initial_center_from_results(path: Path, input_columns: list[str], *, source: str) -> np.ndarray:
    if source != "map":
        raise ValueError(f"Unsupported initialization source: {source!r}")
    path = path.expanduser().resolve()
    with h5py.File(path, "r") as handle:
        if "parameters/names" not in handle:
            raise KeyError(f"{path} does not contain /parameters/names")
        if "map/theta" not in handle:
            raise KeyError(f"{path} does not contain /map/theta")
        source_names = _decode_hdf5_strings(handle["parameters/names"][...])
        source_theta = np.asarray(handle["map/theta"][...], dtype=float)
    if source_theta.shape != (len(source_names),):
        raise ValueError(
            f"{path} /map/theta has shape {source_theta.shape}, expected ({len(source_names)},)"
        )
    by_name = dict(zip(source_names, source_theta, strict=True))
    missing = [name for name in input_columns if name not in by_name]
    if missing:
        raise ValueError(f"{path} is missing initialization parameter(s): {missing}")
    theta = np.asarray([by_name[name] for name in input_columns], dtype=float)
    if not np.all(np.isfinite(theta)):
        bad = [name for name, value in zip(input_columns, theta, strict=True) if not np.isfinite(value)]
        raise ValueError(f"{path} contains non-finite initialization value(s): {bad}")
    return theta


def _write_model_changes_file(
    parameter_specs,
    theta: np.ndarray,
    output_path: Path,
) -> Path:
    root = ET.Element("changes")
    for spec, value in zip(parameter_specs, theta, strict=True):
        ET.SubElement(
            root,
            "change",
            type="update",
            path=spec.path,
            value=f"{float(value):.17g}",
        )
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    return output_path


def _parse_x_ranges(values: list[str]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for value in values:
        if "=" not in value or ":" not in value:
            raise ValueError(f"Invalid --observable-x-range {value!r}; expected key=xmin:xmax")
        key, raw_range = value.split("=", 1)
        raw_min, raw_max = raw_range.split(":", 1)
        x_min = float(raw_min)
        x_max = float(raw_max)
        if x_min > x_max:
            raise ValueError(f"Invalid x range for {key!r}: xmin > xmax")
        ranges[key] = (x_min, x_max)
    return ranges


def _observable_mask(observable: dict, x_range: tuple[float, float] | None) -> np.ndarray:
    x = np.asarray(observable["x_plot"], dtype=float)
    mask = np.ones(x.shape, dtype=bool)
    if x_range is not None:
        mask &= (x >= x_range[0]) & (x <= x_range[1])
    if not np.any(mask):
        raise ValueError(f"No bins remain for {observable['observable_key']} after applying x range {x_range}")
    return mask


def _selected_targets(
    bundle: dict,
    observable_keys: list[str],
    x_ranges: dict[str, tuple[float, float]],
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, slice]]:
    target_blocks = []
    variance_blocks = []
    masks = {}
    slices = {}
    start = 0
    for key in observable_keys:
        if key not in bundle["observables"]:
            raise ValueError(f"Observable {key!r} is not present in bundle")
        observable = bundle["observables"][key]
        mask = _observable_mask(observable, x_ranges.get(key))
        target = np.asarray(observable["target_plot"], dtype=float)[mask]
        sigma = observable.get("target_sigma_plot")
        if sigma is None:
            sigma_values = np.full(target.shape, sigma_floor, dtype=float)
        else:
            sigma_values = np.asarray(sigma, dtype=float)[mask]
            sigma_values = np.where(np.isfinite(sigma_values) & (sigma_values > 0.0), sigma_values, sigma_floor)
            sigma_values = np.maximum(sigma_values, sigma_floor)
        target_blocks.append(target)
        variance_blocks.append(sigma_values**2)
        masks[key] = mask
        slices[key] = slice(start, start + target.size)
        start += target.size
    return np.concatenate(target_blocks), np.concatenate(variance_blocks), masks, slices


class BundlePosterior:
    def __init__(
        self,
        *,
        bundle: dict,
        observable_keys: list[str],
        masks: dict[str, np.ndarray],
        target_vector: np.ndarray,
        target_variance: np.ndarray,
        parameter_specs,
        include_emulator_variance: bool,
    ):
        self.bundle = bundle
        self.observable_keys = observable_keys
        self.masks = masks
        self.target_vector = target_vector
        self.target_variance = target_variance
        self.parameter_specs = parameter_specs
        self.include_emulator_variance = include_emulator_variance
        self.input_columns = list(bundle["input_columns"])

    def _predict_observable_batch(
        self,
        observable: dict,
        x_quantiles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mode = observable.get("emulator_mode", self.bundle.get("emulator_mode", "bin_by_bin"))
        if mode == "bin_by_bin":
            y_pred = np.zeros((x_quantiles.shape[0], observable["n_bins"]), dtype=float)
            y_std = np.zeros_like(y_pred)
            for index, model in enumerate(observable["models"]):
                pred, pred_std = predict_scaled_gp(
                    model,
                    float(observable["y_means"][index]),
                    float(observable["y_stds"][index]),
                    x_quantiles,
                )
                y_pred[:, index] = pred
                y_std[:, index] = pred_std
            return y_pred, y_std

        n_components = len(observable["models"])
        coefficient_predictions = np.zeros((x_quantiles.shape[0], n_components), dtype=float)
        coefficient_stds = np.zeros_like(coefficient_predictions)
        for index, model in enumerate(observable["models"]):
            pred, pred_std = predict_scaled_gp(
                model,
                float(observable["y_means"][index]),
                float(observable["y_stds"][index]),
                x_quantiles,
            )
            coefficient_predictions[:, index] = pred
            coefficient_stds[:, index] = pred_std
        components = np.asarray(observable["pca_components_matrix"], dtype=float)
        pca_mean = np.asarray(observable["pca_mean"], dtype=float)
        preprocessor_mean = np.asarray(observable["preprocessor_mean"], dtype=float)
        preprocessor_scale = np.asarray(observable["preprocessor_scale"], dtype=float)
        y_pred_scaled = coefficient_predictions @ components + pca_mean[None, :]
        y_pred = y_pred_scaled * preprocessor_scale[None, :] + preprocessor_mean[None, :]
        component_variance = coefficient_stds**2
        y_var_scaled = component_variance @ (components**2)
        y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * preprocessor_scale[None, :]
        return y_pred, y_std

    def predict_batch(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        x_quantiles = transform_to_prior_quantiles(self.parameter_specs, theta)
        means = []
        variances = []
        for key in self.observable_keys:
            mask = self.masks[key]
            observable = self.bundle["observables"][key]
            y_pred, y_std = self._predict_observable_batch(observable, x_quantiles)
            means.append(y_pred[:, mask])
            variances.append(y_std[:, mask] ** 2)
        return np.concatenate(means, axis=1), np.concatenate(variances, axis=1)

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_batch(theta[None, :])
        return means[0], variances[0]

    def log_probability(self, theta: np.ndarray) -> float:
        log_prior = float(log_prior_density(self.parameter_specs, theta[None, :])[0])
        if not np.isfinite(log_prior):
            return -np.inf
        mean, emulator_variance = self.predict(theta)
        variance = np.asarray(self.target_variance, dtype=float).copy()
        if self.include_emulator_variance:
            variance = variance + emulator_variance
        variance = np.maximum(variance, 1.0e-12)
        residual = self.target_vector - mean
        log_like = -0.5 * np.sum((residual**2) / variance + np.log(2.0 * np.pi * variance))
        return float(log_prior + log_like)

    def log_probability_batch(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        log_prior = np.asarray(log_prior_density(self.parameter_specs, theta), dtype=float)
        result = np.full(theta.shape[0], -np.inf, dtype=float)
        valid = np.isfinite(log_prior)
        if not np.any(valid):
            return result

        means, emulator_variances = self.predict_batch(theta[valid])
        variance = self.target_variance[None, :]
        if self.include_emulator_variance:
            variance = variance + emulator_variances
        variance = np.maximum(variance, 1.0e-12)
        residuals = self.target_vector[None, :] - means
        log_likelihood = -0.5 * np.sum(
            (residuals**2) / variance + np.log(2.0 * np.pi * variance),
            axis=1,
        )
        valid_indices = np.where(valid)[0]
        finite_likelihood = np.isfinite(log_likelihood)
        result[valid_indices[finite_likelihood]] = (
            log_prior[valid_indices[finite_likelihood]] + log_likelihood[finite_likelihood]
        )
        return result


def _trace_plot(
    chain: np.ndarray,
    labels: list[str],
    path: Path,
    *,
    parameter_specs=None,
    burn_in: int = 0,
) -> dict[str, float]:
    return make_trace_plot(
        chain,
        labels,
        path,
        parameter_specs=parameter_specs,
        burn_in=burn_in,
        row_height=1.65,
        color="tab:blue",
        alpha=0.2,
        linewidth=0.5,
    )


def _best_fit_plot(
    bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    prediction_by_key: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    n_panels = len(observable_keys)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, key in zip(axes_flat, observable_keys, strict=False):
        observable = bundle["observables"][key]
        mask = masks[key]
        x = np.asarray(observable["x_plot"], dtype=float)[mask]
        target = np.asarray(observable["target_plot"], dtype=float)[mask]
        target_sigma = observable.get("target_sigma_plot")
        if target_sigma is not None:
            target_sigma = np.asarray(target_sigma, dtype=float)[mask]
        pred, pred_std = prediction_by_key[key]
        axis.errorbar(x, target, yerr=target_sigma, fmt="o", color="0.15", label="target")
        axis.plot(x, pred, color="tab:blue", lw=1.8, label="best fit")
        axis.fill_between(x, pred - pred_std, pred + pred_std, color="tab:blue", alpha=0.2)
        set_ylim_from_values(axis, target, pred)
        axis.set_title(observable["label"])
        axis.set_xlabel(observable["x_axis_label"])
        axis.set_ylabel(observable["y_axis_label"])
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if emcee is None:
        raise ModuleNotFoundError("The 'emcee' package is required. Activate galacticus-workspace or install emcee.")
    if args.thin < 1:
        raise ValueError("--thin must be >= 1")
    trace_thin = int(args.trace_thin) if int(args.trace_thin) > 0 else int(args.thin)
    if trace_thin < 1:
        raise ValueError("--trace-thin must be >= 1 when supplied")

    bundle = load_observables_bundle(args.bundle_path)
    input_columns = list(bundle["input_columns"])
    parameter_specs = _parameter_specs_for_columns(input_columns)
    if args.n_walkers < 2 * len(input_columns):
        raise ValueError(f"--n-walkers should be at least {2 * len(input_columns)} for {len(input_columns)} dimensions")

    x_ranges = _parse_x_ranges(args.observable_x_range)
    target_vector, target_variance, masks, slices = _selected_targets(
        bundle,
        args.observable,
        x_ranges,
        args.target_sigma_floor,
    )
    posterior = BundlePosterior(
        bundle=bundle,
        observable_keys=args.observable,
        masks=masks,
        target_vector=target_vector,
        target_variance=target_variance,
        parameter_specs=parameter_specs,
        include_emulator_variance=bool(args.include_emulator_variance),
    )

    rng = np.random.default_rng(args.seed)
    initialization_summary = {
        "mode": "prior_quantile_center",
        "init_quantile_sigma": float(args.init_quantile_sigma),
    }
    if args.init_from_results is None:
        center_quantiles = np.full(len(input_columns), 0.5, dtype=float)
    else:
        center_theta = _initial_center_from_results(
            args.init_from_results,
            input_columns,
            source=args.init_source,
        )
        center_quantiles = transform_to_prior_quantiles(parameter_specs, center_theta[None, :])[0]
        initialization_summary = {
            "mode": "results",
            "source": str(args.init_source),
            "results_hdf5": str(args.init_from_results.expanduser().resolve()),
            "center_theta": {name: float(value) for name, value in zip(input_columns, center_theta, strict=True)},
            "center_quantiles": {
                name: float(value) for name, value in zip(input_columns, center_quantiles, strict=True)
            },
            "init_quantile_sigma": float(args.init_quantile_sigma),
        }
    center_quantiles = np.clip(center_quantiles, 1.0e-4, 1.0 - 1.0e-4)
    initial_quantiles = np.clip(
        center_quantiles + args.init_quantile_sigma * rng.normal(size=(args.n_walkers, len(input_columns))),
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
        raise ValueError("--save-full-chain is incompatible with the default thinned emcee backend; pass --emcee-backend-hdf5 none to save a full unthinned chain.")

    moves = build_emcee_moves(emcee, args.move)
    sampler_kwargs = {"nwalkers": args.n_walkers, "ndim": len(input_columns)}
    if moves is not None:
        sampler_kwargs["moves"] = moves
    backend = prepare_emcee_backend(
        emcee,
        emcee_backend_path,
        n_walkers=args.n_walkers,
        n_dim=len(input_columns),
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

    posterior_df = pd.DataFrame(flat_samples, columns=input_columns)
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
    best_params = {name: float(value) for name, value in zip(input_columns, best_theta, strict=True)}
    best_predictions = predict_observables_bundle(bundle, best_params)
    map_changes_path = _write_model_changes_file(
        parameter_specs,
        best_theta,
        args.mcmc_dir / "maximum_a_posteriori_model_changes.xml",
    )

    prediction_rows = []
    prediction_by_key = {}
    for key in args.observable:
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
        for local_bin, (x_value, target_value, sigma_value, pred_value, pred_std_value) in enumerate(
            zip(
                np.asarray(observable["x_plot"], dtype=float)[mask],
                np.asarray(observable["target_plot"], dtype=float)[mask],
                target_sigma,
                pred,
                pred_std,
                strict=True,
            )
        ):
            prediction_rows.append(
                {
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
    predictions_path = args.mcmc_dir / f"{args.output_prefix}_best_fit_observables.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    best_quantiles = transform_to_prior_quantiles(parameter_specs, best_theta[None, :])[0]
    summary = {
        "bundle_path": str(args.bundle_path.resolve()),
        "observable_keys": args.observable,
        "observable_x_ranges": {key: list(value) for key, value in x_ranges.items()},
        "input_columns": input_columns,
        "n_dimensions": len(input_columns),
        "n_data": int(target_vector.size),
        "include_emulator_variance": bool(args.include_emulator_variance),
        "target_sigma_floor": float(args.target_sigma_floor),
        "n_walkers": int(args.n_walkers),
        "n_steps": int(args.n_steps),
        "burn_in": int(args.burn_in),
        "thin": int(args.thin),
        "trace_thin": int(trace_thin),
        "seed": int(args.seed),
        "init_quantile_sigma": float(args.init_quantile_sigma),
        "initialization": initialization_summary,
        "emcee_moves": describe_emcee_moves(args.move),
        "sampler_coordinates": sampler_coordinate_summary(
            parameter_specs,
            input_columns,
            enabled=args.sample_transformed_parameters,
        ),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "best_log_probability": float(best_log_probability),
        "best_chain_index_after_burn_in": list(best_chain_index),
        "best_theta": best_params,
        "best_quantiles": {name: float(value) for name, value in zip(input_columns, best_quantiles, strict=True)},
        "maximum_a_posteriori_model_changes": str(map_changes_path),
        "output_slices": {key: [slc.start, slc.stop] for key, slc in slices.items()},
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
            "observable_keys": args.observable,
            "observable_x_ranges": x_ranges,
            "input_columns": input_columns,
            "parameter_specs": parameter_specs,
            "masks": masks,
            "target_vector": target_vector,
            "target_variance": target_variance,
            "best_theta": best_params,
            "sampler_coordinates": sampler_coordinate_summary(
                parameter_specs,
                input_columns,
                enabled=args.sample_transformed_parameters,
            ),
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
        parameter_names=input_columns,
        parameter_specs=parameter_specs,
        summary=summary,
        map_changes_path=map_changes_path,
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
            input_columns,
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
            input_columns,
            parameter_specs,
            truths=best_theta,
        )
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)
        corner_created = True

    best_fit_path = figures_dir / f"{args.output_prefix}_best_fit_observables.png"
    best_fit_created = False
    if not args.skip_plots and not args.skip_best_fit_plot:
        _best_fit_plot(bundle, args.observable, masks, prediction_by_key, best_fit_path)
        best_fit_created = True

    if backend_run.enabled and args.delete_emcee_backend_on_success and emcee_backend_path is not None:
        emcee_backend_path.unlink(missing_ok=True)
        summary["emcee_backend_deleted_on_success"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(summary_path)
    print(map_changes_path)
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
