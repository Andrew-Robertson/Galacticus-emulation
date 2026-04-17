from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
except ModuleNotFoundError:  # pragma: no cover
    corner = None

try:
    import emcee
except ModuleNotFoundError:  # pragma: no cover
    emcee = None

from dust_model.parameter_changes import write_changes_file
from galacticus_emu import (
    TRINITY_RUN_CONFIGS,
    TRINITY_TRAINABLE_OUTPUT_COLUMNS,
    load_emulator_bundle,
    load_or_build_emulator_table,
    log_prior_density,
    predict_scaled_gp,
    transform_from_prior_quantiles,
    transform_to_prior_quantiles,
    trinity_parameter_specs,
)


CASE_DEFINITIONS = {
    "z0_mstar_mean": {
        "outputs": [f"z0_mass_stellar_log10_{index}" for index in range(3)],
        "color": "#1b9e77",
        "label": "z=0 Mstar",
    },
    "z2_mstar_mean": {
        "outputs": [f"z2_mass_stellar_log10_{index}" for index in range(3)],
        "color": "#d95f02",
        "label": "z=2 Mstar",
    },
    "z0_mbh_mean": {
        "outputs": [f"z0_mass_black_hole_log10_{index}" for index in range(3)],
        "color": "#7570b3",
        "label": "z=0 Mbh",
    },
    "z2_mbh_mean": {
        "outputs": [f"z2_mass_black_hole_log10_{index}" for index in range(3)],
        "color": "#e7298a",
        "label": "z=2 Mbh",
    },
}

CASE_DEFINITIONS_NO1E11_MBH = {
    "z0_mstar_mean": CASE_DEFINITIONS["z0_mstar_mean"],
    "z2_mstar_mean": CASE_DEFINITIONS["z2_mstar_mean"],
    "z0_mbh_mean": {
        "outputs": [f"z0_mass_black_hole_log10_{index}" for index in (1, 2)],
        "color": "#7570b3",
        "label": "z=0 Mbh",
    },
    "z2_mbh_mean": {
        "outputs": [f"z2_mass_black_hole_log10_{index}" for index in (1, 2)],
        "color": "#e7298a",
        "label": "z=2 Mbh",
    },
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a suite of mean-only Trinity emulator MCMCs and make an overlaid corner plot."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--models-summary",
        default=None,
        help="Path to the serialized model summary JSON. Defaults to the galacticus-workspace-trained bundles.",
    )
    parser.add_argument("--n-walkers", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--thin", type=int, default=4)
    parser.add_argument(
        "--exclude-low-mbh-bin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude the lowest-halo-mass black-hole bin from the z0/z2 Mbh likelihood families.",
    )
    parser.add_argument(
        "--ablation-dir-name",
        default="emulator_mcmc_ablations",
        help="Name of the ablation output directory inside the campaign root.",
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
    parser.add_argument("--top-n-params", type=int, default=8)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show emcee progress bars.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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

    for _, config in TRINITY_RUN_CONFIGS.items():
        with h5py.File(evaluation_dir / config["filename"], "r") as handle:
            analyses = handle["analyses"]

            stellar_mean = analyses[config["stellar_mean_key"]]
            stellar_mask = stellar_mean["massStellarLog10"][:] != 0.0
            stellar_mean_target = stellar_mean["massStellarLog10Target"][:][stellar_mask]
            stellar_mean_cov = stellar_mean["massStellarLog10CovarianceTarget"][:][np.ix_(stellar_mask, stellar_mask)]

            black_hole_mean = analyses[config["black_hole_mean_key"]]
            black_hole_mask = black_hole_mean["massBlackHoleLog10"][:] != 0.0
            black_hole_mean_target = black_hole_mean["massBlackHoleLog10Target"][:][black_hole_mask]
            black_hole_mean_cov = black_hole_mean["massBlackHoleLog10CovarianceTarget"][:][np.ix_(black_hole_mask, black_hole_mask)]

            stellar_scatter = analyses[config["stellar_scatter_key"]]
            stellar_scatter_target = stellar_scatter["massStellarLog10ScatterTarget"][:][stellar_mask]
            stellar_scatter_cov = stellar_scatter["massStellarLog10ScatterCovarianceTarget"][:][np.ix_(stellar_mask, stellar_mask)]

            black_hole_scatter = analyses[config["black_hole_scatter_key"]]
            black_hole_scatter_target = black_hole_scatter["massBlackHoleLog10ScatterTarget"][:][black_hole_mask]
            black_hole_scatter_cov = black_hole_scatter["massBlackHoleLog10ScatterCovarianceTarget"][:][np.ix_(black_hole_mask, black_hole_mask)]

        target_chunks.extend([stellar_mean_target, black_hole_mean_target, stellar_scatter_target, black_hole_scatter_target])
        covariance_blocks.extend([stellar_mean_cov, black_hole_mean_cov, stellar_scatter_cov, black_hole_scatter_cov])

    target_vector = np.concatenate([
        *target_chunks[0::4],
        *target_chunks[1::4],
        *target_chunks[2::4],
        *target_chunks[3::4],
    ])
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
        covariance[index:index + block_size, index:index + block_size] = block
        index += block_size
    return target_vector, covariance


def _load_models(summary_path: Path) -> dict[str, dict]:
    summary = json.loads(summary_path.read_text())
    bundles = {}
    for item in summary["models"]:
        bundle = load_emulator_bundle(item["model_path"])
        bundles[bundle["output_column"]] = bundle
    missing = [name for name in TRINITY_TRAINABLE_OUTPUT_COLUMNS if name not in bundles]
    if missing:
        raise FileNotFoundError(f"Missing serialized emulator bundles for: {missing}")
    return bundles


def _output_index_map() -> dict[str, int]:
    return {name: index for index, name in enumerate(TRINITY_TRAINABLE_OUTPUT_COLUMNS)}


def _target_column(output_name: str) -> str:
    prefix, bin_index = output_name.rsplit("_", 1)
    return f"{prefix}_target_{bin_index}"


def _target_error_column(output_name: str) -> str:
    prefix, bin_index = output_name.rsplit("_", 1)
    return f"{prefix}_target_error_{bin_index}"


def _best_training_row_for_case(table: pd.DataFrame, outputs: list[str]) -> pd.Series:
    chi2 = np.zeros(len(table), dtype=float)
    for output_name in outputs:
        target_col = _target_column(output_name)
        error_col = _target_error_column(output_name)
        residual = table[output_name].to_numpy(dtype=float) - float(table[target_col].iloc[0])
        sigma = float(table[error_col].iloc[0])
        chi2 += (residual / sigma) ** 2
    return table.iloc[int(np.argmin(chi2))]


def _best_training_theta_for_case(table: pd.DataFrame, input_columns: list[str], outputs: list[str]) -> np.ndarray:
    row = _best_training_row_for_case(table, outputs)
    return row[input_columns].to_numpy(dtype=float)


def _select_top_parameters(table: pd.DataFrame, input_columns: list[str], selected_outputs: list[str], top_n: int) -> list[str]:
    corr = table[input_columns + selected_outputs].corr(method="spearman").loc[input_columns, selected_outputs]
    max_abs = corr.abs().max(axis=1).sort_values(ascending=False)
    return max_abs.head(top_n).index.tolist()


class SubsetPosterior:
    def __init__(
        self,
        parameter_specs,
        input_columns: list[str],
        model_bundles: dict[str, dict],
        output_columns: list[str],
        target_vector: np.ndarray,
        target_covariance: np.ndarray,
    ):
        self.parameter_specs = parameter_specs
        self.input_columns = input_columns
        self.model_bundles = model_bundles
        self.output_columns = output_columns
        self.target_vector = target_vector
        self.target_covariance = target_covariance

    def predict_batch(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        x = transform_to_prior_quantiles(self.parameter_specs, theta)
        mean_columns: list[np.ndarray] = []
        variance_columns: list[np.ndarray] = []
        for output_name in self.output_columns:
            bundle = self.model_bundles[output_name]
            pred, pred_std = predict_scaled_gp(bundle["model"], bundle["y_mean"], bundle["y_std"], x)
            mean_columns.append(np.asarray(pred, dtype=float))
            variance_columns.append(np.asarray(pred_std, dtype=float) ** 2)
        return np.column_stack(mean_columns), np.column_stack(variance_columns)

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = self.predict_batch(theta[None, :])
        return means[0], variances[0]

    def log_probability_batch(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        log_prior = np.asarray(log_prior_density(self.parameter_specs, theta), dtype=float)
        result = np.full(theta.shape[0], -np.inf, dtype=float)
        valid = np.isfinite(log_prior)
        if not np.any(valid):
            return result

        means, variances = self.predict_batch(theta[valid])
        residuals = self.target_vector[None, :] - means
        normalizer = len(self.target_vector) * np.log(2.0 * np.pi)
        valid_indices = np.where(valid)[0]
        for local_index, sample_index in enumerate(valid_indices):
            covariance = self.target_covariance + np.diag(variances[local_index] + 1.0e-8)
            try:
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0:
                    continue
                quadratic = float(residuals[local_index] @ np.linalg.solve(covariance, residuals[local_index]))
            except np.linalg.LinAlgError:
                continue
            result[sample_index] = log_prior[sample_index] - 0.5 * (quadratic + logdet + normalizer)
        return result


def _make_trace_plot(
    samples: np.ndarray,
    labels: list[str],
    output_path: Path,
    max_steps_plot: int = 4000,
    max_walkers_plot: int = 64,
) -> None:
    if samples.shape[0] > max_steps_plot:
        step_stride = int(np.ceil(samples.shape[0] / max_steps_plot))
        samples = samples[::step_stride]
    if samples.shape[1] > max_walkers_plot:
        walker_stride = int(np.ceil(samples.shape[1] / max_walkers_plot))
        samples = samples[:, ::walker_stride, :]
    n_steps, _, n_dim = samples.shape
    fig, axes = plt.subplots(n_dim, 1, figsize=(9, 1.8 * n_dim), sharex=True, constrained_layout=True)
    if n_dim == 1:
        axes = [axes]
    for axis, label, dim_index in zip(axes, labels, range(n_dim), strict=True):
        values = samples[:, :, dim_index]
        if label in LOG_PARAMETER_ALIASES:
            values = np.log10(values)
            axis.set_ylabel(f"log10({DISPLAY_LABELS.get(label, label)})")
        else:
            axis.set_ylabel(DISPLAY_LABELS.get(label, label))
        axis.plot(values, alpha=0.18, linewidth=0.5)
    axes[-1].set_xlabel("Step")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _overlay_corner(samples_by_case: dict[str, np.ndarray], parameter_columns: list[str], output_path: Path) -> None:
    if corner is None:
        return
    transformed_samples_by_case: dict[str, np.ndarray] = {}
    labels: list[str] = []
    for case_name, samples in samples_by_case.items():
        transformed_columns = []
        if not labels:
            labels = []
        for dim_index, parameter in enumerate(parameter_columns):
            values = samples[:, dim_index]
            if parameter in LOG_PARAMETER_ALIASES:
                values = np.log10(values)
                if len(labels) < len(parameter_columns):
                    labels.append(f"log10({DISPLAY_LABELS.get(parameter, parameter)})")
            else:
                if len(labels) < len(parameter_columns):
                    labels.append(DISPLAY_LABELS.get(parameter, parameter))
            transformed_columns.append(values)
        transformed_samples_by_case[case_name] = np.column_stack(transformed_columns)

    stacked = np.vstack(list(transformed_samples_by_case.values()))
    ranges = []
    for dim_index in range(stacked.shape[1]):
        lower = float(np.min(stacked[:, dim_index]))
        upper = float(np.max(stacked[:, dim_index]))
        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        ranges.append((lower - margin, upper + margin))

    fig = None
    for case_name, samples in transformed_samples_by_case.items():
        config = CASE_DEFINITIONS[case_name]
        fig = corner.corner(
            samples,
            fig=fig,
            labels=labels,
            range=ranges,
            color=config["color"],
            plot_datapoints=False,
            fill_contours=False,
            no_fill_contours=True,
            plot_density=False,
            levels=(0.393, 0.865),
            hist_kwargs={"density": True, "histtype": "step", "linewidth": 1.8},
            contour_kwargs={"linewidths": 1.4},
        )
    handles = [
        plt.Line2D([0], [0], color=CASE_DEFINITIONS[name]["color"], lw=2, label=CASE_DEFINITIONS[name]["label"])
        for name in samples_by_case
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    if emcee is None:
        raise ModuleNotFoundError("The 'emcee' package is required for run_trinity_mcmc_ablation_suite.py.")

    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / args.figures_dir_name
    ablation_root = campaign_root / args.ablation_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    ablation_root.mkdir(parents=True, exist_ok=True)
    case_definitions = CASE_DEFINITIONS_NO1E11_MBH if args.exclude_low_mbh_bin else CASE_DEFINITIONS

    summary_path = (
        Path(args.models_summary).resolve()
        if args.models_summary is not None
        else campaign_root / "emulator" / "gp_fit_summary_all_outputs_all36_opt_gw.json"
    )

    table = load_or_build_emulator_table(campaign_root)
    parameter_specs = trinity_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    model_bundles = _load_models(summary_path)
    full_target_vector, full_target_covariance = _load_target_vector_and_covariance(campaign_root)
    index_map = _output_index_map()
    rng = np.random.default_rng(args.seed)

    overlay_output_names = [
        *case_definitions["z0_mstar_mean"]["outputs"],
        *case_definitions["z2_mstar_mean"]["outputs"],
        *case_definitions["z0_mbh_mean"]["outputs"],
        *case_definitions["z2_mbh_mean"]["outputs"],
    ]
    top_parameters = _select_top_parameters(table, input_columns, overlay_output_names, args.top_n_params)
    (ablation_root / "overlay_parameter_selection.json").write_text(
        json.dumps(
            {
                "selection_method": "top_max_abs_spearman_across_selected_outputs",
                "parameter_columns": top_parameters,
                "selected_outputs": overlay_output_names,
            },
            indent=2,
        )
        + "\n"
    )

    samples_by_case: dict[str, np.ndarray] = {}

    for case_name, config in case_definitions.items():
        case_root = ablation_root / case_name
        case_root.mkdir(parents=True, exist_ok=True)
        output_columns = config["outputs"]
        indices = [index_map[name] for name in output_columns]
        target_vector = full_target_vector[indices]
        target_covariance = full_target_covariance[np.ix_(indices, indices)]

        posterior = SubsetPosterior(
            parameter_specs=parameter_specs,
            input_columns=input_columns,
            model_bundles=model_bundles,
            output_columns=output_columns,
            target_vector=target_vector,
            target_covariance=target_covariance,
        )

        if args.init_center == "prior_center":
            center_quantiles = np.full(len(parameter_specs), 0.5, dtype=float)
        else:
            center_theta = _best_training_theta_for_case(table, input_columns, output_columns)
            center_quantiles = transform_to_prior_quantiles(parameter_specs, center_theta[None, :])[0]
        initial_quantiles = np.clip(
            center_quantiles + args.init_quantile_sigma * rng.normal(size=(args.n_walkers, len(parameter_specs))),
            1.0e-4,
            1.0 - 1.0e-4,
        )
        initial_positions = transform_from_prior_quantiles(parameter_specs, initial_quantiles)

        sampler = emcee.EnsembleSampler(
            nwalkers=args.n_walkers,
            ndim=len(parameter_specs),
            log_prob_fn=posterior.log_probability_batch,
            vectorize=True,
        )
        sampler.run_mcmc(initial_positions, args.n_steps, progress=args.progress)

        chain = sampler.get_chain()
        log_prob = sampler.get_log_prob()
        flat_samples = sampler.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
        flat_log_prob = sampler.get_log_prob(discard=args.burn_in, thin=args.thin, flat=True)
        acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=float)

        np.save(case_root / "chain.npy", chain)
        np.save(case_root / "log_prob.npy", log_prob)

        posterior_df = pd.DataFrame(flat_samples, columns=input_columns)
        posterior_df["log_probability"] = flat_log_prob
        posterior_df.to_csv(case_root / "posterior_samples.csv", index=False)

        best_index = int(np.argmax(flat_log_prob))
        best_theta = flat_samples[best_index]
        best_pred_mean, best_pred_var = posterior.predict(best_theta)
        write_changes_file(
            [(spec.path, value) for spec, value in zip(parameter_specs, best_theta, strict=True)],
            case_root / "maximum_a_posteriori_model_changes.xml",
        )

        summary = {
            "case": case_name,
            "label": config["label"],
            "output_columns": output_columns,
            "models_summary": str(summary_path),
            "n_walkers": args.n_walkers,
            "n_steps": args.n_steps,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "init_center": args.init_center,
            "init_quantile_sigma": args.init_quantile_sigma,
            "seed": args.seed,
            "mean_acceptance_fraction": float(np.mean(acceptance_fraction)),
            "acceptance_fraction_by_walker": [float(x) for x in acceptance_fraction],
            "best_log_probability": float(flat_log_prob[best_index]),
            "best_theta": {name: float(value) for name, value in zip(input_columns, best_theta, strict=True)},
            "best_prediction_mean": {name: float(value) for name, value in zip(output_columns, best_pred_mean, strict=True)},
            "best_prediction_std": {name: float(np.sqrt(value)) for name, value in zip(output_columns, best_pred_var, strict=True)},
            "target_vector": {name: float(value) for name, value in zip(output_columns, target_vector, strict=True)},
        }
        (case_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        trace_path = figures_root / f"emulator_mcmc_ablation_trace_{case_name}.png"
        _make_trace_plot(chain, input_columns, trace_path)
        samples_by_case[case_name] = posterior_df[top_parameters].to_numpy(dtype=float)

    overlay_path = figures_root / "emulator_mcmc_ablation_corner_overlay_top_params.png"
    _overlay_corner(samples_by_case, top_parameters, overlay_path)

    print(ablation_root)
    for case_name in case_definitions:
        print(ablation_root / case_name / "posterior_samples.csv")
        print(figures_root / f"emulator_mcmc_ablation_trace_{case_name}.png")
    print(overlay_path)


if __name__ == "__main__":
    main()
