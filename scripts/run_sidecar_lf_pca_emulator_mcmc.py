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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fit_sidecar_lf_pca_holdout_demo import (
    _alpha_log10,
    _bin_metadata,
    _input_columns,
    _log10_with_floor,
    _output_columns,
    _read_first_sidecar_table,
    _read_sidecar_tables,
    _sample_labels,
    _target_curve,
)
from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp
from galacticus_emu.lhs import (
    NormalPrior,
    ParameterSpec,
    TruncatedNormalPrior,
    log_prior_density,
    transform_from_prior_quantiles,
    transform_to_prior_quantiles,
)
from galacticus_emu.mcmc_coordinates import (
    SamplerCoordinatePosterior,
    add_sampler_coordinate_arguments,
    physical_log_prob_from_sampler_log_prob,
    physical_to_sampler_coordinates,
    sampler_coordinate_summary,
    sampler_to_physical_coordinates,
)
from galacticus_emu.mcmc_corner import corner_with_log10_priors
from galacticus_emu.mcmc_moves import add_emcee_move_arguments, build_emcee_moves, describe_emcee_moves
from galacticus_emu.mcmc_trace import make_trace_plot
from galacticus_emu.observable_plot_metadata import (
    HALPHA_LF_X_AXIS_LABEL,
    apply_sidecar_lf_y_display_offset,
    sidecar_lf_axis_label,
    sidecar_lf_target_label,
    sidecar_lf_y_axis_label,
)
from galacticus_emu.plotting import set_ylim_from_values
from galacticus_emu.specs import trinity_parameter_specs


DUST_PARAMETER_SPECS = [
    ParameterSpec(path="delta_0", short_name="delta_0", prior=NormalPrior(mean=0.0, sigma=0.5)),
    ParameterSpec(path="delta_z", short_name="delta_z", prior=NormalPrior(mean=0.0, sigma=0.5)),
    ParameterSpec(path="delta_M", short_name="delta_M", prior=NormalPrior(mean=0.0, sigma=0.5)),
    ParameterSpec(path="delta_Mz", short_name="delta_Mz", prior=NormalPrior(mean=0.0, sigma=0.5)),
    ParameterSpec(
        path="attenuation_scatter",
        short_name="attenuation_scatter",
        prior=TruncatedNormalPrior(mean=0.25, sigma=0.1, lower=0.0),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PCA GP emulators for sidecar emission-line LFs and run an MCMC fit."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--sidecar-dir", default="emission_line_dust")
    parser.add_argument("--table-filename", default="emission_line_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="emission_line_dust_lf_long.csv")
    parser.add_argument("--observable", required=True)
    parser.add_argument("--sample-label", action="append", required=True)
    parser.add_argument("--dust-draw-index", type=int, action="append", default=None)
    parser.add_argument(
        "--max-dust-draws-per-eval",
        type=int,
        default=None,
        help="After other filters, keep the first N dust rows per Galacticus evaluation.",
    )
    parser.add_argument("--min-log10-lf", type=float, default=-8.0)
    parser.add_argument("--use-training-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-training-log10-sigma", type=float, default=1.0e-4)
    parser.add_argument("--max-training-log10-sigma", type=float, default=2.0)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--optimize-hyperparameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--pca-variance-threshold", type=float, default=None)
    parser.add_argument("--n-walkers", type=int, default=96)
    parser.add_argument("--n-steps", type=int, default=6000)
    parser.add_argument("--burn-in", type=int, default=1200)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--init-center", choices=["prior_center", "best_training"], default="best_training")
    parser.add_argument("--init-quantile-sigma", type=float, default=0.04)
    parser.add_argument("--target-sigma-floor", type=float, default=1.0e-3)
    parser.add_argument("--include-emulator-variance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--mcmc-dir-name", required=True)
    parser.add_argument("--figures-dir-name", required=True)
    parser.add_argument("--output-prefix", required=True)
    add_emcee_move_arguments(parser)
    add_sampler_coordinate_arguments(parser)
    return parser.parse_args()


def _parameter_specs_for_table(input_columns: list[str]) -> tuple[list[ParameterSpec], list[str], list[str]]:
    slow_quantile_columns = [column for column in input_columns if column.endswith("_quantile")]
    slow_names = [column.removesuffix("_quantile") for column in slow_quantile_columns]
    dust_names = [column for column in input_columns if not column.endswith("_quantile")]
    slow_specs_by_name = {spec.short_name: spec for spec in trinity_parameter_specs()}
    dust_specs_by_name = {spec.short_name: spec for spec in DUST_PARAMETER_SPECS}
    specs = []
    parameter_names = []
    for name in slow_names:
        if name not in slow_specs_by_name:
            raise ValueError(f"No Galacticus parameter spec for {name!r}")
        specs.append(slow_specs_by_name[name])
        parameter_names.append(name)
    for name in dust_names:
        if name not in dust_specs_by_name:
            raise ValueError(f"No dust parameter spec for {name!r}")
        specs.append(dust_specs_by_name[name])
        parameter_names.append(name)
    return specs, parameter_names, dust_names


def _feature_matrix_from_table(table: pd.DataFrame, input_columns: list[str], dust_names: list[str]) -> np.ndarray:
    slow_quantile_columns = [column for column in input_columns if column.endswith("_quantile")]
    blocks = [table[slow_quantile_columns].to_numpy(dtype=float)]
    if dust_names:
        blocks.append(table[dust_names].to_numpy(dtype=float))
    return np.column_stack(blocks)


def _feature_matrix_from_theta(theta: np.ndarray, parameter_specs: list[ParameterSpec], slow_count: int) -> np.ndarray:
    theta = np.atleast_2d(theta)
    slow_quantiles = transform_to_prior_quantiles(parameter_specs[:slow_count], theta[:, :slow_count])
    if slow_count == theta.shape[1]:
        return slow_quantiles
    return np.column_stack([slow_quantiles, theta[:, slow_count:]])


def _filter_table(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = table.copy()
    if args.dust_draw_index is not None:
        filtered = filtered.loc[filtered["dust_draw_index"].isin(args.dust_draw_index)].reset_index(drop=True)
    if args.max_dust_draws_per_eval is not None:
        filtered = (
            filtered.sort_values(["evaluation_id", "dust_draw_index"])
            .groupby("evaluation_id", group_keys=False)
            .head(args.max_dust_draws_per_eval)
            .reset_index(drop=True)
        )
    if filtered.empty:
        raise ValueError("No rows remain after sidecar LF table filtering.")
    return filtered


def _choose_pca_components(y_scaled: np.ndarray, requested: int, threshold: float | None) -> int:
    max_components = min(y_scaled.shape[0], y_scaled.shape[1])
    if threshold is None:
        return min(int(requested), max_components)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("--pca-variance-threshold must be in (0, 1]")
    probe = PCA(n_components=max_components)
    probe.fit(y_scaled)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, threshold) + 1)


def _fit_pca_bundle(
    *,
    x: np.ndarray,
    y: np.ndarray,
    alpha: np.ndarray | None,
    observable_key: str,
    sample_label: str,
    output_columns: list[str],
    x_plot: np.ndarray,
    target_plot: np.ndarray,
    target_std_plot: np.ndarray | None,
    args: argparse.Namespace,
) -> dict:
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)
    n_components = _choose_pca_components(y_scaled, args.pca_components, args.pca_variance_threshold)
    pca = PCA(n_components=n_components)
    coefficients = pca.fit_transform(y_scaled)

    coefficient_alpha = None
    if alpha is not None:
        alpha_scaled = np.asarray(alpha, dtype=float) / (scaler.scale_[None, :] ** 2)
        coefficient_alpha = alpha_scaled @ (pca.components_.T ** 2)
        coefficient_alpha = np.maximum(coefficient_alpha, 1.0e-12)

    models = []
    y_means = []
    y_stds = []
    kernels = []
    for component_index in range(n_components):
        print(f"{observable_key} {sample_label}: fitting PCA component {component_index + 1}/{n_components}", flush=True)
        model, y_mean, y_std = fit_scaled_gp(
            x,
            coefficients[:, component_index],
            n_restarts_optimizer=args.n_restarts_optimizer,
            optimize_hyperparameters=args.optimize_hyperparameters,
            alpha=coefficient_alpha[:, component_index] if coefficient_alpha is not None else None,
        )
        models.append(model)
        y_means.append(y_mean)
        y_stds.append(y_std)
        kernels.append(str(model.kernel_))

    return {
        "observable_key": observable_key,
        "sample_label": sample_label,
        "output_columns": output_columns,
        "x_plot": np.asarray(x_plot, dtype=float),
        "target_plot": np.asarray(target_plot, dtype=float),
        "target_sigma_plot": None if target_std_plot is None else np.asarray(target_std_plot, dtype=float),
        "models": models,
        "y_means": np.asarray(y_means, dtype=float),
        "y_stds": np.asarray(y_stds, dtype=float),
        "pca_components_actual": int(n_components),
        "pca_variance_threshold": args.pca_variance_threshold,
        "pca_explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=float),
        "pca_mean": np.asarray(pca.mean_, dtype=float),
        "pca_components_matrix": np.asarray(pca.components_, dtype=float),
        "preprocessor_mean": np.asarray(scaler.mean_, dtype=float),
        "preprocessor_scale": np.asarray(scaler.scale_, dtype=float),
        "kernels": kernels,
    }


def _predict_bundle(bundle: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_components = len(bundle["models"])
    coefficients = np.zeros((x.shape[0], n_components), dtype=float)
    coefficient_stds = np.zeros_like(coefficients)
    for index, model in enumerate(bundle["models"]):
        pred, pred_std = predict_scaled_gp(
            model,
            float(bundle["y_means"][index]),
            float(bundle["y_stds"][index]),
            x,
        )
        coefficients[:, index] = pred
        coefficient_stds[:, index] = pred_std
    components = np.asarray(bundle["pca_components_matrix"], dtype=float)
    pca_mean = np.asarray(bundle["pca_mean"], dtype=float)
    preprocessor_mean = np.asarray(bundle["preprocessor_mean"], dtype=float)
    preprocessor_scale = np.asarray(bundle["preprocessor_scale"], dtype=float)
    y_scaled = coefficients @ components + pca_mean[None, :]
    y_pred = y_scaled * preprocessor_scale[None, :] + preprocessor_mean[None, :]
    y_var_scaled = (coefficient_stds**2) @ (components**2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * preprocessor_scale[None, :]
    return y_pred, y_std


class SidecarLFPosterior:
    def __init__(
        self,
        *,
        parameter_specs: list[ParameterSpec],
        slow_count: int,
        bundles: dict[str, dict],
        include_emulator_variance: bool,
        target_sigma_floor: float,
    ):
        self.parameter_specs = parameter_specs
        self.slow_count = slow_count
        self.bundles = bundles
        self.include_emulator_variance = include_emulator_variance
        self.target_sigma_floor = target_sigma_floor
        self.target_vector = np.concatenate([bundle["target_plot"] for bundle in bundles.values()])
        target_sigmas = []
        for bundle in bundles.values():
            sigma = bundle["target_sigma_plot"]
            if sigma is None:
                sigma = np.full_like(bundle["target_plot"], target_sigma_floor, dtype=float)
            else:
                sigma = np.asarray(sigma, dtype=float)
                sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, target_sigma_floor)
                sigma = np.maximum(sigma, target_sigma_floor)
            target_sigmas.append(sigma)
        self.target_variance = np.concatenate(target_sigmas) ** 2

    def predict_batch(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = _feature_matrix_from_theta(theta, self.parameter_specs, self.slow_count)
        means = []
        variances = []
        for bundle in self.bundles.values():
            pred, pred_std = _predict_bundle(bundle, x)
            means.append(pred)
            variances.append(pred_std**2)
        return np.concatenate(means, axis=1), np.concatenate(variances, axis=1)

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_batch(theta[None, :])
        return means[0], variances[0]

    def log_probability(self, theta: np.ndarray) -> float:
        return float(self.log_probability_batch(theta[None, :])[0])

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
        residual = self.target_vector[None, :] - means
        log_like = -0.5 * np.sum((residual**2) / variance + np.log(2.0 * np.pi * variance), axis=1)
        valid_indices = np.where(valid)[0]
        finite = np.isfinite(log_like)
        result[valid_indices[finite]] = log_prior[valid_indices[finite]] + log_like[finite]
        return result


def _best_training_theta(table: pd.DataFrame, parameter_names: list[str], output_by_label: dict[str, list[str]], target_by_label: dict[str, np.ndarray], min_log10_lf: float) -> np.ndarray:
    scores = np.zeros(len(table), dtype=float)
    for label, columns in output_by_label.items():
        y, _ = _log10_with_floor(table[columns].to_numpy(dtype=float), min_log10_lf)
        residual = y - target_by_label[label][None, :]
        scores += np.mean(residual**2, axis=1)
    best = table.iloc[int(np.argmin(scores))]
    return best[parameter_names].to_numpy(dtype=float)


def _write_model_changes_file(parameter_specs: list[ParameterSpec], theta: np.ndarray, output_path: Path) -> Path:
    root = ET.Element("changes")
    for spec, value in zip(parameter_specs, theta, strict=True):
        if spec.path.startswith("delta_") or spec.path == "attenuation_scatter":
            continue
        ET.SubElement(root, "change", type="update", path=spec.path, value=f"{float(value):.17g}")
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    return output_path


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
        row_height=1.55,
        alpha=0.22,
        linewidth=0.5,
    )


def _plot_best_fit(bundles: dict[str, dict], prediction_by_label: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    n_panels = len(bundles)
    ncols = 1 if n_panels == 1 else 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, (label, bundle) in zip(axes_flat, bundles.items(), strict=False):
        x = np.asarray(bundle["x_plot"], dtype=float)
        target = apply_sidecar_lf_y_display_offset(bundle, bundle["target_plot"])
        target_sigma = bundle["target_sigma_plot"]
        pred, pred_std = prediction_by_label[label]
        pred = apply_sidecar_lf_y_display_offset(bundle, pred)
        axis.errorbar(
            x,
            target,
            yerr=target_sigma,
            fmt="o",
            color="0.15",
            label=sidecar_lf_target_label(bundle, bundle.get("sample_label")),
        )
        axis.plot(x, pred, color="tab:blue", lw=1.8, label="best fit")
        axis.fill_between(x, pred - pred_std, pred + pred_std, color="tab:blue", alpha=0.2)
        set_ylim_from_values(axis, target, pred)
        axis.set_title(sidecar_lf_axis_label(bundle, bundle.get("sample_label"), label))
        axis.set_xlabel(HALPHA_LF_X_AXIS_LABEL if bundle.get("observable") == "halpha_sobral" else r"$\log_{10}(L/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(sidecar_lf_y_axis_label(bundle))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if emcee is None:
        raise ModuleNotFoundError("The 'emcee' package is required. Activate galacticus-workspace or install emcee.")

    campaign_root = args.campaign_root.resolve()
    mcmc_root = campaign_root / args.mcmc_dir_name
    figures_root = campaign_root / args.figures_dir_name
    mcmc_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    table = _filter_table(_read_sidecar_tables(campaign_root, args.sidecar_dir, args.table_filename), args)
    long_table = _read_first_sidecar_table(campaign_root, args.sidecar_dir, args.long_filename)
    input_columns = _input_columns(table)
    parameter_specs, parameter_names, dust_names = _parameter_specs_for_table(input_columns)
    slow_count = len([column for column in input_columns if column.endswith("_quantile")])
    if args.n_walkers < 2 * len(parameter_specs):
        raise ValueError(f"--n-walkers should be at least {2 * len(parameter_specs)} for {len(parameter_specs)} dimensions")
    x = _feature_matrix_from_table(table, input_columns, dust_names)

    bundles = {}
    output_by_label = {}
    target_by_label = {}
    for sample_label in args.sample_label:
        output_columns = _output_columns(table, args.observable, [sample_label])
        sample_labels = _sample_labels(output_columns)
        if not sample_labels or any(label != sample_labels[0] for label in sample_labels):
            raise ValueError(f"Could not infer a single sample label for {sample_label!r}")
        bin_metadata = _bin_metadata(long_table, args.observable, sample_labels[0], output_columns)
        x_plot = bin_metadata["log10_luminosity_center"].to_numpy(dtype=float)
        target_plot, target_std_plot = _target_curve(campaign_root, args.observable, sample_labels[0], x_plot, args.min_log10_lf)
        y, _floor = _log10_with_floor(table[output_columns].to_numpy(dtype=float), args.min_log10_lf)
        alpha = None
        if args.use_training_alpha:
            alpha_sigma = _alpha_log10(
                table,
                output_columns,
                min_sigma=args.min_training_log10_sigma,
                max_sigma=args.max_training_log10_sigma,
            )
            alpha = alpha_sigma**2
        bundles[sample_labels[0]] = _fit_pca_bundle(
            x=x,
            y=y,
            alpha=alpha,
            observable_key=args.observable,
            sample_label=sample_labels[0],
            output_columns=output_columns,
            x_plot=x_plot,
            target_plot=target_plot,
            target_std_plot=target_std_plot,
            args=args,
        )
        output_by_label[sample_labels[0]] = output_columns
        target_by_label[sample_labels[0]] = target_plot

    posterior = SidecarLFPosterior(
        parameter_specs=parameter_specs,
        slow_count=slow_count,
        bundles=bundles,
        include_emulator_variance=args.include_emulator_variance,
        target_sigma_floor=args.target_sigma_floor,
    )

    if args.init_center == "prior_center":
        center_quantiles = np.full(len(parameter_specs), 0.5, dtype=float)
        center_theta = transform_from_prior_quantiles(parameter_specs, center_quantiles[None, :])[0]
    else:
        center_theta = _best_training_theta(table, parameter_names, output_by_label, target_by_label, args.min_log10_lf)
        center_quantiles = transform_to_prior_quantiles(parameter_specs, center_theta[None, :])[0]

    rng = np.random.default_rng(args.seed)
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

    moves = build_emcee_moves(emcee, args.move)
    sampler_kwargs = {"nwalkers": args.n_walkers, "ndim": len(parameter_specs)}
    if moves is not None:
        sampler_kwargs["moves"] = moves
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
            sampler.run_mcmc(initial_positions, args.n_steps, progress=args.progress, skip_initial_state_check=True)
    else:
        sampler = emcee.EnsembleSampler(
            log_prob_fn=sampler_posterior.log_probability_batch,
            vectorize=True,
            **sampler_kwargs,
        )
        sampler.run_mcmc(initial_positions, args.n_steps, progress=args.progress, skip_initial_state_check=True)

    sampler_chain = sampler.get_chain()
    sampler_log_prob = sampler.get_log_prob()
    chain = sampler_to_physical_coordinates(
        parameter_specs,
        sampler_chain,
        enabled=args.sample_transformed_parameters,
    )
    log_prob = physical_log_prob_from_sampler_log_prob(
        parameter_specs,
        sampler_chain,
        sampler_log_prob,
        enabled=args.sample_transformed_parameters,
    )
    flat_sampler_samples = sampler.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
    flat_sampler_log_prob = sampler.get_log_prob(discard=args.burn_in, thin=args.thin, flat=True)
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
    np.save(mcmc_root / f"{args.output_prefix}_chain.npy", chain)
    np.save(mcmc_root / f"{args.output_prefix}_log_prob.npy", log_prob)
    if args.sample_transformed_parameters:
        np.save(mcmc_root / f"{args.output_prefix}_sampler_chain.npy", sampler_chain)
        np.save(mcmc_root / f"{args.output_prefix}_sampler_log_prob.npy", sampler_log_prob)

    posterior_df = pd.DataFrame(flat_samples, columns=parameter_names)
    posterior_df["log_probability"] = flat_log_prob
    posterior_path = mcmc_root / f"{args.output_prefix}_posterior_samples.csv"
    posterior_df.to_csv(posterior_path, index=False)

    best_index = int(np.argmax(flat_log_prob))
    best_theta = flat_samples[best_index]
    best_mean, best_var = posterior.predict(best_theta)
    best_std = np.sqrt(np.maximum(best_var, 0.0))

    rows = []
    prediction_by_label = {}
    start = 0
    for sample_label, bundle in bundles.items():
        n_bins = len(bundle["x_plot"])
        pred = best_mean[start : start + n_bins]
        pred_std = best_std[start : start + n_bins]
        start += n_bins
        prediction_by_label[sample_label] = (pred, pred_std)
        target_sigma = bundle["target_sigma_plot"]
        if target_sigma is None:
            target_sigma = np.full(n_bins, np.nan)
        for bin_index, values in enumerate(zip(bundle["x_plot"], bundle["target_plot"], target_sigma, pred, pred_std, strict=True)):
            x_value, target_value, target_std, pred_value, pred_std_value = values
            rows.append(
                {
                    "observable": args.observable,
                    "sample_label": sample_label,
                    "bin_index": int(bin_index),
                    "x_plot": float(x_value),
                    "target_log10_phi": float(target_value),
                    "target_log10_phi_std": float(target_std),
                    "prediction_log10_phi": float(pred_value),
                    "prediction_log10_phi_std": float(pred_std_value),
                }
            )
    best_fit_path = mcmc_root / f"{args.output_prefix}_best_fit_observables.csv"
    pd.DataFrame(rows).to_csv(best_fit_path, index=False)

    changes_path = _write_model_changes_file(parameter_specs[:slow_count], best_theta[:slow_count], mcmc_root / "maximum_a_posteriori_model_changes.xml")
    emulator_bundle_path = mcmc_root / f"{args.output_prefix}_emulator_bundle.joblib"
    joblib.dump(
        {
            "bundle_type": "sidecar_lf_pca_gp",
            "campaign_root": str(campaign_root),
            "sidecar_dir": args.sidecar_dir,
            "table_filename": args.table_filename,
            "long_filename": args.long_filename,
            "observable": args.observable,
            "sample_labels": list(bundles),
            "input_columns": input_columns,
            "parameter_names": parameter_names,
            "parameter_specs": parameter_specs,
            "slow_count": slow_count,
            "bundles": bundles,
            "best_theta": {name: float(value) for name, value in zip(parameter_names, best_theta, strict=True)},
            "sampler_coordinates": sampler_coordinate_summary(
                parameter_specs,
                parameter_names,
                enabled=args.sample_transformed_parameters,
            ),
        },
        emulator_bundle_path,
    )

    summary = {
        "campaign_root": str(campaign_root),
        "observable": args.observable,
        "sample_labels": list(bundles),
        "input_columns": input_columns,
        "parameter_names": parameter_names,
        "n_dimensions": len(parameter_specs),
        "n_training_rows": int(len(table)),
        "include_emulator_variance": bool(args.include_emulator_variance),
        "n_walkers": int(args.n_walkers),
        "n_steps": int(args.n_steps),
        "burn_in": int(args.burn_in),
        "thin": int(args.thin),
        "emcee_moves": describe_emcee_moves(args.move),
        "sampler_coordinates": sampler_coordinate_summary(
            parameter_specs,
            parameter_names,
            enabled=args.sample_transformed_parameters,
        ),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "best_log_probability": float(flat_log_prob[best_index]),
        "best_theta": {name: float(value) for name, value in zip(parameter_names, best_theta, strict=True)},
        "maximum_a_posteriori_model_changes": str(changes_path),
        "emulator_bundle": str(emulator_bundle_path),
        "pca_models": {
            label: {
                "pca_components_actual": int(bundle["pca_components_actual"]),
                "pca_variance_threshold": bundle["pca_variance_threshold"],
                "pca_explained_variance_ratio_sum": float(np.sum(bundle["pca_explained_variance_ratio"])),
                "kernels": bundle["kernels"],
            }
            for label, bundle in bundles.items()
        },
    }
    summary_path = mcmc_root / f"{args.output_prefix}_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    trace_path = figures_root / f"{args.output_prefix}_trace.png"
    trace_tau = _trace_plot(
        chain,
        parameter_names,
        trace_path,
        parameter_specs=parameter_specs,
        burn_in=args.burn_in,
    )
    summary["trace_autocorrelation_time"] = trace_tau
    summary["trace_autocorrelation_time_burn_in"] = int(args.burn_in)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    corner_path = figures_root / f"{args.output_prefix}_corner.png"
    if corner is not None:
        corner_fig = corner_with_log10_priors(
            corner,
            posterior_df,
            parameter_names,
            parameter_specs,
            truths=best_theta,
        )
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)
    best_fit_figure_path = figures_root / f"{args.output_prefix}_best_fit_observables.png"
    _plot_best_fit(bundles, prediction_by_label, best_fit_figure_path)

    print(posterior_path)
    print(best_fit_path)
    print(summary_path)
    print(emulator_bundle_path)
    print(changes_path)
    print(trace_path)
    if corner is not None:
        print(corner_path)
    print(best_fit_figure_path)


if __name__ == "__main__":
    main()
