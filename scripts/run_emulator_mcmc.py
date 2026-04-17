from __future__ import annotations

import argparse
import json
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

try:
    import emcee
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    emcee = None

try:
    import corner
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    corner = None

from galacticus_emu import (
    default_parameter_specs,
    fit_scaled_gp,
    log_prior_density,
    predict_scaled_gp,
    transform_from_prior_quantiles,
    transform_to_prior_quantiles,
)


MEAN_OUTPUT_COLUMNS = [
    "mass_stellar_log10_0",
    "mass_stellar_log10_1",
    "mass_stellar_log10_2",
]

SCATTER_OUTPUT_COLUMNS = [
    "mass_stellar_log10_scatter_0",
    "mass_stellar_log10_scatter_1",
    "mass_stellar_log10_scatter_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emcee MCMC using the GP emulator in place of direct Galacticus runs."
    )
    parser.add_argument("campaign_root", help="Path to a completed campaign directory.")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--n-restarts-optimizer", type=int, default=1)
    parser.add_argument(
        "--include-scatter",
        action="store_true",
        help="Include the SHMR scatter outputs in the emulator likelihood.",
    )
    return parser.parse_args()


def _load_emulator_table(campaign_root: Path) -> pd.DataFrame:
    emulator_table_path = campaign_root / "emulator_table.csv"
    if emulator_table_path.exists():
        return pd.read_csv(emulator_table_path)
    samples = pd.read_csv(campaign_root / "samples.csv")
    summary = pd.read_csv(campaign_root / "summary.csv")
    merged = samples.merge(summary, on="evaluation_id", how="inner", validate="one_to_one")
    merged.to_csv(emulator_table_path, index=False)
    return merged


def _load_nested_order(campaign_root: Path, x: np.ndarray) -> np.ndarray:
    order_path = campaign_root / "emulator" / "nested_subset_order.csv"
    if order_path.exists():
        order_df = pd.read_csv(order_path)
        return order_df["ordered_index"].to_numpy(dtype=int)

    remaining = np.ones(x.shape[0], dtype=bool)
    center = np.full(x.shape[1], 0.5)
    first_index = int(np.argmin(np.sum((x - center) ** 2, axis=1)))
    order = [first_index]
    remaining[first_index] = False
    min_dist_sq = np.sum((x - x[first_index]) ** 2, axis=1)
    for _ in range(1, x.shape[0]):
        candidate_indices = np.where(remaining)[0]
        next_index = int(candidate_indices[np.argmax(min_dist_sq[candidate_indices])])
        order.append(next_index)
        remaining[next_index] = False
        min_dist_sq = np.minimum(min_dist_sq, np.sum((x - x[next_index]) ** 2, axis=1))
    return np.asarray(order, dtype=int)


def _load_target_data(campaign_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sample_file = next((campaign_root / "evaluations").glob("*/galacticus.hdf5"))
    with h5py.File(sample_file, "r") as handle:
        mean_group = handle["analyses"]["stellarHaloMassRelationUniverseMachinez6"]
        scatter_group = handle["analyses"]["stellarHaloMassRelationScatterUniverseMachinez6"]
        mask = np.logical_or(
            mean_group["massStellarLog10"][:] != 0.0,
            scatter_group["massStellarLog10Scatter"][:] != 0.0,
        )

        mean_target = mean_group["massStellarLog10Target"][:][mask]
        mean_cov = mean_group["massStellarLog10CovarianceTarget"][:][np.ix_(mask, mask)]
        scatter_target = scatter_group["massStellarLog10ScatterTarget"][:][mask]
        scatter_cov = scatter_group["massStellarLog10ScatterCovarianceTarget"][:][np.ix_(mask, mask)]

    target_vector = np.concatenate([mean_target, scatter_target])
    target_cov = np.block(
        [
            [mean_cov, np.zeros((len(mean_target), len(scatter_target)))],
            [np.zeros((len(scatter_target), len(mean_target))), scatter_cov],
        ]
    )
    output_names = [*MEAN_OUTPUT_COLUMNS, *SCATTER_OUTPUT_COLUMNS]
    return target_vector, target_cov, output_names


class EmulatorPosterior:
    def __init__(
        self,
        parameter_specs,
        gp_models,
        target_vector: np.ndarray,
        target_cov: np.ndarray,
        include_scatter: bool,
    ):
        self.parameter_specs = parameter_specs
        self.gp_models = gp_models
        self.target_vector = target_vector if include_scatter else target_vector[:3]
        self.target_cov = target_cov if include_scatter else target_cov[:3, :3]
        self.output_names = [*MEAN_OUTPUT_COLUMNS, *SCATTER_OUTPUT_COLUMNS] if include_scatter else [*MEAN_OUTPUT_COLUMNS]

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = transform_to_prior_quantiles(self.parameter_specs, theta[None, :])
        means = []
        variances = []
        for output_name in self.output_names:
            model, y_mean, y_std = self.gp_models[output_name]
            pred, pred_std = predict_scaled_gp(model, y_mean, y_std, x)
            means.append(float(pred[0]))
            variances.append(float(pred_std[0] ** 2))
        return np.asarray(means), np.asarray(variances)

    def log_probability(self, theta: np.ndarray) -> float:
        log_prior = float(log_prior_density(self.parameter_specs, theta[None, :])[0])
        if not np.isfinite(log_prior):
            return -np.inf

        prediction_mean, prediction_var = self.predict(theta)
        covariance = self.target_cov + np.diag(prediction_var + 1.0e-8)
        residual = self.target_vector - prediction_mean
        try:
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                return -np.inf
            quadratic = float(residual @ np.linalg.solve(covariance, residual))
        except np.linalg.LinAlgError:
            return -np.inf
        log_likelihood = -0.5 * (
            quadratic
            + logdet
            + len(self.target_vector) * np.log(2.0 * np.pi)
        )
        return log_prior + log_likelihood


def _fit_emulator_models(table: pd.DataFrame, train_indices: np.ndarray, parameter_specs, n_restarts_optimizer: int):
    input_columns = [spec.short_name for spec in parameter_specs]
    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)
    x_train = x[train_indices]
    models = {}
    for output_name in [*MEAN_OUTPUT_COLUMNS, *SCATTER_OUTPUT_COLUMNS]:
        y_train = table.iloc[train_indices][output_name].to_numpy(dtype=float)
        models[output_name] = fit_scaled_gp(
            x=x_train,
            y=y_train,
            n_restarts_optimizer=n_restarts_optimizer,
        )
    return models


def _make_trace_plot(samples: np.ndarray, labels: list[str], output_path: Path) -> None:
    n_steps, n_walkers, n_dim = samples.shape
    fig, axes = plt.subplots(n_dim, 1, figsize=(9, 2.0 * n_dim), sharex=True, constrained_layout=True)
    if n_dim == 1:
        axes = [axes]
    for axis, label, dim_index in zip(axes, labels, range(n_dim), strict=True):
        axis.plot(samples[:, :, dim_index], alpha=0.25, linewidth=0.6)
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Step")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    if emcee is None:
        raise ModuleNotFoundError(
            "The 'emcee' package is required for run_emulator_mcmc.py. "
            "Activate the galacticus-workspace environment or install emcee."
        )
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    emulator_root = campaign_root / "emulator_mcmc"
    figures_root = campaign_root / "figures"
    emulator_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    table = _load_emulator_table(campaign_root)
    parameter_specs = default_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)
    order = _load_nested_order(campaign_root, x)

    train_size = min(args.train_size, len(table))
    train_indices = order[:train_size]
    gp_models = _fit_emulator_models(
        table=table,
        train_indices=train_indices,
        parameter_specs=parameter_specs,
        n_restarts_optimizer=args.n_restarts_optimizer,
    )
    target_vector, target_cov, output_names = _load_target_data(campaign_root)
    posterior = EmulatorPosterior(
        parameter_specs=parameter_specs,
        gp_models=gp_models,
        target_vector=target_vector,
        target_cov=target_cov,
        include_scatter=args.include_scatter,
    )

    best_train_row = table.iloc[train_indices].sort_values("log_likelihood_total", ascending=False).iloc[0]
    center_theta = best_train_row[input_columns].to_numpy(dtype=float)
    center_quantiles = transform_to_prior_quantiles(parameter_specs, center_theta[None, :])[0]
    initial_quantiles = np.clip(
        center_quantiles + 0.03 * np.random.default_rng(42).normal(size=(args.n_walkers, len(parameter_specs))),
        1.0e-4,
        1.0 - 1.0e-4,
    )
    initial_positions = transform_from_prior_quantiles(parameter_specs, initial_quantiles)

    sampler = emcee.EnsembleSampler(
        args.n_walkers,
        len(parameter_specs),
        posterior.log_probability,
    )
    sampler.run_mcmc(initial_positions, args.n_steps, progress=False)

    samples = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    flat_samples = sampler.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=args.burn_in, thin=args.thin, flat=True)

    np.save(emulator_root / "chain.npy", samples)
    np.save(emulator_root / "log_prob.npy", log_prob)

    chain_df = pd.DataFrame(flat_samples, columns=input_columns)
    chain_df["log_probability"] = flat_log_prob
    chain_df.to_csv(emulator_root / "posterior_samples.csv", index=False)

    best_index = int(np.argmax(flat_log_prob))
    best_theta = flat_samples[best_index]
    best_pred_mean, best_pred_var = posterior.predict(best_theta)
    summary = {
        "campaign_root": str(campaign_root),
        "train_size": train_size,
        "include_scatter": args.include_scatter,
        "n_walkers": args.n_walkers,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "input_columns": input_columns,
        "output_names": posterior.output_names,
        "best_log_probability": float(flat_log_prob[best_index]),
        "best_theta": {name: float(value) for name, value in zip(input_columns, best_theta, strict=True)},
        "best_prediction_mean": {name: float(value) for name, value in zip(posterior.output_names, best_pred_mean, strict=True)},
        "best_prediction_std": {name: float(np.sqrt(value)) for name, value in zip(posterior.output_names, best_pred_var, strict=True)},
    }
    (emulator_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    trace_path = figures_root / "emulator_mcmc_trace.png"
    _make_trace_plot(samples, input_columns, trace_path)
    corner_path = figures_root / "emulator_mcmc_corner.png"
    if corner is not None:
        corner_fig = corner.corner(flat_samples, labels=input_columns, truths=best_theta)
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)

    print(emulator_root / "posterior_samples.csv")
    print(emulator_root / "run_summary.json")
    print(trace_path)
    if corner is not None:
        print(corner_path)


if __name__ == "__main__":
    main()
