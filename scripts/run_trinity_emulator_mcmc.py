from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pool
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT.parent / "Galacticus-dust-modelling"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import corner
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    corner = None

try:
    import emcee
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    emcee = None

from dust_model.parameter_changes import write_changes_file
from galacticus_emu import (
    TRINITY_RUN_CONFIGS,
    TRINITY_TRAINABLE_OUTPUT_COLUMNS,
    fit_scaled_gp,
    load_emulator_bundle,
    load_or_build_emulator_table,
    log_prior_density,
    predict_scaled_gp,
    sanitize_output_name,
    transform_from_prior_quantiles,
    transform_to_prior_quantiles,
    trinity_parameter_specs,
)

LOG_PARAMETER_ALIASES = {
    "blackHoleSeedMass",
    "diskSFRFrequencyNorm",
    "spheroidSFREfficiency",
    "bondiEnhancementSpheroid",
    "bondiEnhancementHotHalo",
    "bondiTemperatureSpheroid",
    "BHefficiencyWind",
    "thinDiskMaximum",
}

DISPLAY_LABELS = {
    "diskVelocityCharacteristic": "diskVelocityCharacteristic",
    "diskExponent": "diskExponent",
    "spheroidVelocityCharacteristic": "spheroidVelocityCharacteristic",
    "spheroidExponent": "spheroidExponent",
    "BHefficiencyWind": "BHefficiencyWind",
    "BHefficiencyRadioMode": "BHefficiencyRadioMode",
    "henriquesGamma": "henriquesGamma",
    "henriquesDelta1": "henriquesDelta1",
    "henriquesDelta2": "henriquesDelta2",
    "coreRadiusOverVirialRadius": "hotHaloCoreRadius",
    "diskSFRFrequencyNorm": "normalisationBlitzSFR",
    "barFractionAngularMomentumRetainedSpheroid": "barInstabilityFracAngMomRetSpheroid",
    "coolingMultiplier": "coolingRateMultiplier",
    "spheroidSFREfficiency": "spheroidStarFormationEfficiency",
    "spheroidSFRExponentVelocity": "spheroidStarFormationExponent",
    "thinDiskMaximum": "thinDiskMaxAccretion",
    "bondiEnhancementSpheroid": "bondiHoyleSpheroidEnhancement",
    "bondiEnhancementHotHalo": "bondiHoyleHotHaloEnhancement",
    "bondiTemperatureSpheroid": "bondiHoyleSpheroidTemperature",
    "blackHoleSeedMass": "blackHoleSeedMass",
    "massRatioMajorMerger": "majorMergerMassRatio",
    "energyOrbital": "mergerRemnantOrbitalEnergyFactor",
    "barStabilityThresholdGaseous": "barInstabilityThresholdGaseous",
    "barStabilityThresholdStellar": "barInstabilityThresholdStellar",
}

LIKELIHOOD_CASES = {
    "all36": list(TRINITY_TRAINABLE_OUTPUT_COLUMNS),
    "mstar_z0_z2": [
        *[f"z0_mass_stellar_log10_{index}" for index in range(3)],
        *[f"z2_mass_stellar_log10_{index}" for index in range(3)],
    ],
    "mbh_z0_z2": [
        *[f"z0_mass_black_hole_log10_{index}" for index in range(3)],
        *[f"z2_mass_black_hole_log10_{index}" for index in range(3)],
    ],
    "mbh_z0_z2_no1e11": [
        *[f"z0_mass_black_hole_log10_{index}" for index in (1, 2)],
        *[f"z2_mass_black_hole_log10_{index}" for index in (1, 2)],
    ],
    "four_mean_families": [
        *[f"z0_mass_stellar_log10_{index}" for index in range(3)],
        *[f"z2_mass_stellar_log10_{index}" for index in range(3)],
        *[f"z0_mass_black_hole_log10_{index}" for index in range(3)],
        *[f"z2_mass_black_hole_log10_{index}" for index in range(3)],
    ],
    "four_mean_families_no1e11_mbh": [
        *[f"z0_mass_stellar_log10_{index}" for index in range(3)],
        *[f"z2_mass_stellar_log10_{index}" for index in range(3)],
        *[f"z0_mass_black_hole_log10_{index}" for index in (1, 2)],
        *[f"z2_mass_black_hole_log10_{index}" for index in (1, 2)],
    ],
    "mzr_z0_upper": [],
    "mstar_z0_z2_plus_mzr_z0_upper": [
        *[f"z0_mass_stellar_log10_{index}" for index in range(3)],
        *[f"z2_mass_stellar_log10_{index}" for index in range(3)],
    ],
    "four_mean_families_no1e11_mbh_plus_mzr_z0_upper": [
        *[f"z0_mass_stellar_log10_{index}" for index in range(3)],
        *[f"z2_mass_stellar_log10_{index}" for index in range(3)],
        *[f"z0_mass_black_hole_log10_{index}" for index in (1, 2)],
        *[f"z2_mass_black_hole_log10_{index}" for index in (1, 2)],
    ],
}

MZR_RESIDUAL_OUTPUT_COLUMNS = [
    "z0_mass_metallicity_abundance_oxygen_12logoh_residual_1",
    "z0_mass_metallicity_abundance_oxygen_12logoh_residual_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emcee MCMC using serialized Trinity GP emulators."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--models-summary",
        default=None,
        help="Path to the JSON summary produced by train_trinity_all_outputs.py. "
        "Defaults to emulator/gp_fit_summary_all_outputs_all36_opt.json inside the campaign.",
    )
    parser.add_argument(
        "--likelihood-case",
        choices=sorted(LIKELIHOOD_CASES.keys()),
        default="all36",
        help="Which subset of emulator outputs to include in the likelihood.",
    )
    parser.add_argument("--n-walkers", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--burn-in", type=int, default=400)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument(
        "--mcmc-dir-name",
        default=None,
        help="Optional custom name for the MCMC output directory inside the campaign root.",
    )
    parser.add_argument(
        "--figures-dir-name",
        default="figures",
        help="Name of the figures directory inside the campaign root.",
    )
    parser.add_argument(
        "--init-center",
        choices=["prior_center", "best_training"],
        default="prior_center",
        help="Center of the initial walker cloud in prior-quantile space.",
    )
    parser.add_argument(
        "--init-quantile-sigma",
        type=float,
        default=1.0e-4,
        help="Gaussian width of the initial walker cloud in prior-quantile space.",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help="Number of worker processes to use for parallel likelihood evaluation.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show emcee's progress bar during sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for walker initialization.",
    )
    parser.add_argument(
        "--mzr-sigma-floor",
        type=float,
        default=0.15,
        help="Exploratory Gaussian sigma floor in dex for the approximate MZR residual likelihood.",
    )
    parser.add_argument(
        "--mzr-weight",
        type=float,
        default=1.0,
        help="Multiplicative weight applied to the approximate MZR log-likelihood contribution.",
    )
    return parser.parse_args()


def _covariance_diagonal_errors(covariance: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diag(covariance))


def _transform_samples_for_plotting(df: pd.DataFrame, input_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    transformed_columns = []
    labels = []
    for name in input_columns:
        values = df[name].to_numpy(dtype=float)
        if name in LOG_PARAMETER_ALIASES:
            values = np.log10(values)
            labels.append(f"log10({DISPLAY_LABELS.get(name, name)})")
        else:
            labels.append(DISPLAY_LABELS.get(name, name))
        transformed_columns.append(values)
    return np.column_stack(transformed_columns), labels


def _first_complete_evaluation_dir(campaign_root: Path) -> Path:
    evaluations_root = campaign_root / "evaluations"
    for evaluation_dir in sorted(evaluations_root.iterdir()):
        if not evaluation_dir.is_dir():
            continue
        if all((evaluation_dir / config["filename"]).exists() for config in TRINITY_RUN_CONFIGS.values()):
            return evaluation_dir
    raise FileNotFoundError(f"No completed Trinity evaluations found under {evaluations_root}")


def _load_target_vector_and_covariance(campaign_root: Path) -> tuple[np.ndarray, np.ndarray]:
    evaluation_dir = _first_complete_evaluation_dir(campaign_root)
    target_chunks: list[np.ndarray] = []
    covariance_blocks: list[np.ndarray] = []

    for run_name, config in TRINITY_RUN_CONFIGS.items():
        with h5py.File(evaluation_dir / config["filename"], "r") as handle:
            analyses = handle["analyses"]

            stellar_mean = analyses[config["stellar_mean_key"]]
            stellar_mask = stellar_mean["massStellarLog10"][:] != 0.0
            stellar_mean_target = stellar_mean["massStellarLog10Target"][:][stellar_mask]
            stellar_mean_cov = stellar_mean["massStellarLog10CovarianceTarget"][:][np.ix_(stellar_mask, stellar_mask)]

            black_hole_mean = analyses[config["black_hole_mean_key"]]
            black_hole_mask = black_hole_mean["massBlackHoleLog10"][:] != 0.0
            black_hole_mean_target = black_hole_mean["massBlackHoleLog10Target"][:][black_hole_mask]
            black_hole_mean_cov = black_hole_mean["massBlackHoleLog10CovarianceTarget"][:][
                np.ix_(black_hole_mask, black_hole_mask)
            ]

            stellar_scatter = analyses[config["stellar_scatter_key"]]
            stellar_scatter_target = stellar_scatter["massStellarLog10ScatterTarget"][:][stellar_mask]
            stellar_scatter_cov = stellar_scatter["massStellarLog10ScatterCovarianceTarget"][:][
                np.ix_(stellar_mask, stellar_mask)
            ]

            black_hole_scatter = analyses[config["black_hole_scatter_key"]]
            black_hole_scatter_target = black_hole_scatter["massBlackHoleLog10ScatterTarget"][:][black_hole_mask]
            black_hole_scatter_cov = black_hole_scatter["massBlackHoleLog10ScatterCovarianceTarget"][:][
                np.ix_(black_hole_mask, black_hole_mask)
            ]

        target_chunks.extend(
            [
                stellar_mean_target,
                black_hole_mean_target,
                stellar_scatter_target,
                black_hole_scatter_target,
            ]
        )
        covariance_blocks.extend(
            [
                stellar_mean_cov,
                black_hole_mean_cov,
                stellar_scatter_cov,
                black_hole_scatter_cov,
            ]
        )

    target_vector = np.concatenate(
        [
            *target_chunks[0::4],  # stellar means for z0/z1/z2
            *target_chunks[1::4],  # black-hole means for z0/z1/z2
            *target_chunks[2::4],  # stellar scatters for z0/z1/z2
            *target_chunks[3::4],  # black-hole scatters for z0/z1/z2
        ]
    )

    ordered_blocks = [
        *covariance_blocks[0::4],
        *covariance_blocks[1::4],
        *covariance_blocks[2::4],
        *covariance_blocks[3::4],
    ]
    size = sum(block.shape[0] for block in ordered_blocks)
    covariance = np.zeros((size, size))
    index = 0
    for block in ordered_blocks:
        block_size = block.shape[0]
        covariance[index : index + block_size, index : index + block_size] = block
        index += block_size
    return target_vector, covariance


def _load_models(summary_path: Path):
    summary = json.loads(summary_path.read_text())
    models = {}
    for model_item in summary["models"]:
        bundle = load_emulator_bundle(model_item["model_path"])
        models[bundle["output_column"]] = bundle
    missing = [name for name in TRINITY_TRAINABLE_OUTPUT_COLUMNS if name not in models]
    if missing:
        raise FileNotFoundError(f"Missing serialized emulator bundles for: {missing}")
    return summary, models


def _fit_mass_metallicity_models(
    campaign_root: Path,
    output_columns: list[str],
) -> dict[str, dict]:
    table_path = campaign_root / "mass_metallicity_emulator_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Expected {table_path}; run extract_trinity_mass_metallicity_summary.py before using the MZR likelihood."
        )
    table = pd.read_csv(table_path)
    parameter_specs = trinity_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(parameter_specs, x_raw)
    bundles: dict[str, dict] = {}
    for output_name in output_columns:
        y = table[output_name].to_numpy(dtype=float)
        model, y_mean, y_std = fit_scaled_gp(
            x=x,
            y=y,
            n_restarts_optimizer=1,
            optimize_hyperparameters=True,
        )
        bundles[output_name] = {
            "model": model,
            "y_mean": y_mean,
            "y_std": y_std,
            "output_column": output_name,
        }
    return bundles


def _output_index_map() -> dict[str, int]:
    return {name: index for index, name in enumerate(TRINITY_TRAINABLE_OUTPUT_COLUMNS)}


def _total_log_likelihood_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column.endswith("_log_likelihood_total")]


def _best_training_theta(table: pd.DataFrame, input_columns: list[str]) -> np.ndarray:
    likelihood_columns = _total_log_likelihood_columns(table)
    center_row = table.assign(
        total_log_likelihood=table[likelihood_columns].sum(axis=1)
    ).sort_values("total_log_likelihood", ascending=False).iloc[0]
    return center_row[input_columns].to_numpy(dtype=float)


def _make_trace_plot(samples: np.ndarray, input_columns: list[str], output_path: Path) -> None:
    n_steps, _, n_dim = samples.shape
    fig, axes = plt.subplots(
        n_dim,
        1,
        figsize=(9, 2.0 * n_dim),
        sharex=True,
        constrained_layout=True,
    )
    if n_dim == 1:
        axes = [axes]
    for axis, name, dim_index in zip(axes, input_columns, range(n_dim), strict=True):
        values = samples[:, :, dim_index]
        if name in LOG_PARAMETER_ALIASES:
            values = np.log10(values)
            axis.set_ylabel(f"log10({DISPLAY_LABELS.get(name, name)})")
        else:
            axis.set_ylabel(DISPLAY_LABELS.get(name, name))
        axis.plot(values, alpha=0.2, linewidth=0.6)
    axes[-1].set_xlabel("Step")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


class TrinityEmulatorPosterior:
    def __init__(
        self,
        parameter_specs,
        input_columns: list[str],
        model_bundles: dict[str, dict],
        target_vector: np.ndarray,
        target_covariance: np.ndarray,
        mzr_output_columns: list[str] | None = None,
        mzr_weight: float = 1.0,
    ):
        self.parameter_specs = parameter_specs
        self.input_columns = input_columns
        self.model_bundles = model_bundles
        self.target_vector = target_vector
        self.target_covariance = target_covariance
        self.output_columns = list(TRINITY_TRAINABLE_OUTPUT_COLUMNS)
        self.mzr_output_columns = list(mzr_output_columns or [])
        self.mzr_weight = mzr_weight

    @staticmethod
    def _gaussian_log_likelihood(residual: np.ndarray, covariance: np.ndarray) -> float:
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return -np.inf
        quadratic = float(residual @ np.linalg.solve(covariance, residual))
        return -0.5 * (quadratic + logdet + len(residual) * np.log(2.0 * np.pi))

    def predict_batch(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        x = transform_to_prior_quantiles(self.parameter_specs, theta)
        mean_columns: list[np.ndarray] = []
        variance_columns: list[np.ndarray] = []
        for output_name in self.output_columns:
            bundle = self.model_bundles[output_name]
            pred, pred_std = predict_scaled_gp(
                bundle["model"],
                bundle["y_mean"],
                bundle["y_std"],
                x,
            )
            mean_columns.append(np.asarray(pred, dtype=float))
            variance_columns.append(np.asarray(pred_std, dtype=float) ** 2)
        means = np.column_stack(mean_columns)
        variances = np.column_stack(variance_columns)
        return means, variances

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_batch(theta[None, :])
        return means[0], variances[0]

    def log_probability(self, theta: np.ndarray) -> float:
        log_prior = float(log_prior_density(self.parameter_specs, theta[None, :])[0])
        if not np.isfinite(log_prior):
            return -np.inf

        prediction_mean, prediction_var = self.predict(theta)
        covariance = self.target_covariance + np.diag(prediction_var + 1.0e-8)
        residual = self.target_vector - prediction_mean
        try:
            if self.mzr_output_columns:
                base_count = len(self.output_columns) - len(self.mzr_output_columns)
                log_likelihood = 0.0
                if base_count > 0:
                    base_covariance = covariance[:base_count, :base_count]
                    base_residual = residual[:base_count]
                    log_likelihood += self._gaussian_log_likelihood(base_residual, base_covariance)
                mzr_covariance = covariance[base_count:, base_count:]
                mzr_residual = residual[base_count:]
                log_likelihood_mzr = self._gaussian_log_likelihood(mzr_residual, mzr_covariance)
                if not np.isfinite(log_likelihood_mzr):
                    return -np.inf
                log_likelihood += self.mzr_weight * log_likelihood_mzr
            else:
                log_likelihood = self._gaussian_log_likelihood(residual, covariance)
        except np.linalg.LinAlgError:
            return -np.inf
        if not np.isfinite(log_likelihood):
            return -np.inf
        return log_prior + log_likelihood

    def log_probability_batch(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        log_prior = np.asarray(log_prior_density(self.parameter_specs, theta), dtype=float)
        result = np.full(theta.shape[0], -np.inf, dtype=float)
        valid = np.isfinite(log_prior)
        if not np.any(valid):
            return result

        means, variances = self.predict_batch(theta[valid])
        residuals = self.target_vector[None, :] - means
        target_covariance = self.target_covariance

        valid_indices = np.where(valid)[0]
        for local_index, sample_index in enumerate(valid_indices):
            covariance = target_covariance + np.diag(variances[local_index] + 1.0e-8)
            try:
                if self.mzr_output_columns:
                    base_count = len(self.output_columns) - len(self.mzr_output_columns)
                    log_likelihood = 0.0
                    if base_count > 0:
                        base_covariance = covariance[:base_count, :base_count]
                        base_residual = residuals[local_index][:base_count]
                        log_likelihood += self._gaussian_log_likelihood(base_residual, base_covariance)
                    mzr_covariance = covariance[base_count:, base_count:]
                    mzr_residual = residuals[local_index][base_count:]
                    log_likelihood_mzr = self._gaussian_log_likelihood(mzr_residual, mzr_covariance)
                    if not np.isfinite(log_likelihood_mzr):
                        continue
                    log_likelihood += self.mzr_weight * log_likelihood_mzr
                else:
                    log_likelihood = self._gaussian_log_likelihood(residuals[local_index], covariance)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(log_likelihood):
                continue
            result[sample_index] = log_prior[sample_index] + log_likelihood
        return result


def main() -> None:
    if emcee is None:
        raise ModuleNotFoundError(
            "The 'emcee' package is required for run_trinity_emulator_mcmc.py. "
            "Activate the galacticus-workspace environment or install emcee."
        )

    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    mcmc_dir_name = args.mcmc_dir_name or f"emulator_mcmc_{args.likelihood_case}"
    mcmc_root = campaign_root / mcmc_dir_name
    figures_root = campaign_root / args.figures_dir_name
    mcmc_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    summary_path = (
        Path(args.models_summary).resolve()
        if args.models_summary is not None
        else campaign_root / "emulator" / "gp_fit_summary_all_outputs_all36_opt.json"
    )

    table = load_or_build_emulator_table(campaign_root)
    parameter_specs = trinity_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    _, model_bundles = _load_models(summary_path)
    full_target_vector, full_target_covariance = _load_target_vector_and_covariance(campaign_root)
    selected_output_columns = list(LIKELIHOOD_CASES[args.likelihood_case])
    output_indices = [_output_index_map()[name] for name in selected_output_columns]
    target_vector = full_target_vector[output_indices]
    target_covariance = full_target_covariance[np.ix_(output_indices, output_indices)]

    mzr_output_columns: list[str] = []
    if args.likelihood_case in {
        "mzr_z0_upper",
        "mstar_z0_z2_plus_mzr_z0_upper",
        "four_mean_families_no1e11_mbh_plus_mzr_z0_upper",
    }:
        mzr_output_columns = list(MZR_RESIDUAL_OUTPUT_COLUMNS)
        model_bundles.update(_fit_mass_metallicity_models(campaign_root, mzr_output_columns))
        mzr_target_vector = np.zeros(len(mzr_output_columns), dtype=float)
        mzr_target_covariance = np.eye(len(mzr_output_columns), dtype=float) * args.mzr_sigma_floor**2
        if len(selected_output_columns) > 0:
            combined_size = len(selected_output_columns) + len(mzr_output_columns)
            combined_covariance = np.zeros((combined_size, combined_size), dtype=float)
            n_base = len(selected_output_columns)
            combined_covariance[:n_base, :n_base] = target_covariance
            combined_covariance[n_base:, n_base:] = mzr_target_covariance
            target_covariance = combined_covariance
            target_vector = np.concatenate([target_vector, mzr_target_vector])
        else:
            target_covariance = mzr_target_covariance
            target_vector = mzr_target_vector
        selected_output_columns = [*selected_output_columns, *mzr_output_columns]

    posterior = TrinityEmulatorPosterior(
        parameter_specs=parameter_specs,
        input_columns=input_columns,
        model_bundles=model_bundles,
        target_vector=target_vector,
        target_covariance=target_covariance,
        mzr_output_columns=mzr_output_columns,
        mzr_weight=args.mzr_weight,
    )
    posterior.output_columns = selected_output_columns

    if args.init_center == "prior_center":
        center_quantiles = np.full(len(parameter_specs), 0.5, dtype=float)
    else:
        center_theta = _best_training_theta(table, input_columns)
        center_quantiles = transform_to_prior_quantiles(parameter_specs, center_theta[None, :])[0]
    rng = np.random.default_rng(args.seed)
    initial_quantiles = np.clip(
        center_quantiles + args.init_quantile_sigma * rng.normal(size=(args.n_walkers, len(parameter_specs))),
        1.0e-4,
        1.0 - 1.0e-4,
    )
    initial_positions = transform_from_prior_quantiles(parameter_specs, initial_quantiles)

    sampler_kwargs = {
        "nwalkers": args.n_walkers,
        "ndim": len(parameter_specs),
    }

    if args.n_processes > 1:
        with Pool(processes=args.n_processes) as pool:
            sampler = emcee.EnsembleSampler(
                log_prob_fn=posterior.log_probability,
                pool=pool,
                vectorize=False,
                **sampler_kwargs,
            )
            sampler.run_mcmc(
                initial_positions,
                args.n_steps,
                progress=args.progress,
                skip_initial_state_check=True,
            )
            chain = sampler.get_chain()
            log_prob = sampler.get_log_prob()
            flat_samples = sampler.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
            flat_log_prob = sampler.get_log_prob(discard=args.burn_in, thin=args.thin, flat=True)
            acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=float)
    else:
        sampler = emcee.EnsembleSampler(
            log_prob_fn=posterior.log_probability_batch,
            vectorize=True,
            **sampler_kwargs,
        )
        sampler.run_mcmc(
            initial_positions,
            args.n_steps,
            progress=args.progress,
            skip_initial_state_check=True,
        )
        chain = sampler.get_chain()
        log_prob = sampler.get_log_prob()
        flat_samples = sampler.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
        flat_log_prob = sampler.get_log_prob(discard=args.burn_in, thin=args.thin, flat=True)
        acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=float)

    np.save(mcmc_root / "chain.npy", chain)
    np.save(mcmc_root / "log_prob.npy", log_prob)

    posterior_df = pd.DataFrame(flat_samples, columns=input_columns)
    posterior_df["log_probability"] = flat_log_prob
    posterior_path = mcmc_root / "posterior_samples.csv"
    posterior_df.to_csv(posterior_path, index=False)

    best_index = int(np.argmax(flat_log_prob))
    best_theta = flat_samples[best_index]
    best_pred_mean, best_pred_var = posterior.predict(best_theta)

    map_changes_path = mcmc_root / "maximum_a_posteriori_model_changes.xml"
    write_changes_file(
        [(spec.path, value) for spec, value in zip(parameter_specs, best_theta, strict=True)],
        map_changes_path,
    )

    run_summary = {
        "campaign_root": str(campaign_root),
        "models_summary": str(summary_path),
        "likelihood_case": args.likelihood_case,
        "n_walkers": args.n_walkers,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "init_center": args.init_center,
        "init_quantile_sigma": args.init_quantile_sigma,
        "n_processes": args.n_processes,
        "progress": args.progress,
        "seed": args.seed,
        "mzr_sigma_floor": args.mzr_sigma_floor,
        "mzr_weight": args.mzr_weight,
        "input_columns": input_columns,
        "output_columns": selected_output_columns,
        "best_log_probability": float(flat_log_prob[best_index]),
        "mean_acceptance_fraction": float(np.mean(acceptance_fraction)),
        "acceptance_fraction_by_walker": [float(value) for value in acceptance_fraction],
        "best_theta": {name: float(value) for name, value in zip(input_columns, best_theta, strict=True)},
        "best_prediction_mean": {
            name: float(value) for name, value in zip(selected_output_columns, best_pred_mean, strict=True)
        },
        "best_prediction_std": {
            name: float(np.sqrt(value))
            for name, value in zip(selected_output_columns, best_pred_var, strict=True)
        },
        "target_vector": {
            name: float(value) for name, value in zip(selected_output_columns, target_vector, strict=True)
        },
        "target_std": {
            name: float(value)
            for name, value in zip(
                selected_output_columns,
                _covariance_diagonal_errors(target_covariance),
                strict=True,
            )
        },
        "map_changes_file": str(map_changes_path),
    }
    summary_path_out = mcmc_root / "run_summary.json"
    summary_path_out.write_text(json.dumps(run_summary, indent=2) + "\n")

    trace_path = figures_root / f"emulator_mcmc_{args.likelihood_case}_trace.png"
    _make_trace_plot(chain, input_columns, trace_path)

    corner_path = figures_root / f"emulator_mcmc_{args.likelihood_case}_corner.png"
    if corner is not None:
        plotting_df = posterior_df[input_columns].copy()
        transformed_samples, plotting_labels = _transform_samples_for_plotting(plotting_df, input_columns)
        truths_index = int(np.argmax(flat_log_prob))
        truths = transformed_samples[truths_index]
        corner_fig = corner.corner(transformed_samples, labels=plotting_labels, truths=truths)
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)

    print(posterior_path)
    print(summary_path_out)
    print(map_changes_path)
    print(trace_path)
    if corner is not None:
        print(corner_path)


if __name__ == "__main__":
    main()
