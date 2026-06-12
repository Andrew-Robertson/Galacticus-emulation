from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
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

from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp
from galacticus_emu.lhs import (
    TruncatedLogNormalPrior,
    log_prior_density,
    transform_from_prior_quantiles,
    transform_to_prior_quantiles,
)
from galacticus_emu.mcmc_corner import corner_with_log10_priors
from galacticus_emu.mcmc_trace import make_trace_plot
from galacticus_emu.specs import trinity_parameter_specs


DEFAULT_FIXED_PARAMETERS = {
    "spheroidExponent",
    "spheroidSFRExponentVelocity",
    "bondiTemperatureSpheroid",
    "barStabilityThresholdGaseous",
    "barStabilityThresholdStellar",
}

FREE_PARAMETER_ORDER = [
    "diskVelocityCharacteristic",
    "diskExponent",
    "coolingMultiplier",
    "BHefficiencyRadioMode",
    "spheroidSFREfficiency",
    "diskSFRFrequencyNorm",
    "blackHoleSeedMass",
    "spheroidVelocityCharacteristic",
    "henriquesGamma",
    "henriquesDelta1",
    "henriquesDelta2",
    "massRatioMajorMerger",
    "energyOrbital",
    "bondiEnhancementSpheroid",
    "bondiEnhancementHotHalo",
    "coreRadiusOverVirialRadius",
    "barFractionAngularMomentumRetainedSpheroid",
    "BHefficiencyWind",
    "thinDiskMaximum",
]

ANALYSIS_CONFIGS = {
    "smf_z0": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz0",
        "label": "Tomczak z~0",
        "use_training_alpha": True,
        "bad_training_condition": "nonfinite",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
    "smf_z3": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz3",
        "label": "Tomczak z~2",
        "use_training_alpha": True,
        "bad_training_condition": "nonfinite",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
    "sfr": {
        "analysis": "starFormationRateFunctionRobotham2011",
        "label": "Robotham 2011 SFRF",
        "use_training_alpha": True,
        "bad_training_condition": "nonfinite",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
    "size_sf": {
        "analysis": "stellarSizeMassRelationvanDerWel2014Sample1",
        "label": "van der Wel 2014 SF",
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
    "size_q": {
        "analysis": "stellarSizeMassRelationvanDerWel2014Sample7",
        "label": "van der Wel 2014 Q",
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
}

LIKELIHOOD_CASES = {
    "smf_pair": ["smf_z0", "smf_z3"],
    "sfr": ["sfr"],
    "smf_pair_sfr": ["smf_z0", "smf_z3", "sfr"],
    "size_pair": ["size_sf", "size_q"],
    "combined": ["smf_z0", "smf_z3", "sfr", "size_sf", "size_q"],
}

DEFAULT_PCA_COMPONENTS_BY_ANALYSIS_KEY = {
    "smf_z0": 5,
    "smf_z3": 4,
    "sfr": 5,
    "size_sf": 3,
    "size_q": 4,
}


def _parameter_specs():
    spec_map = {spec.short_name: spec for spec in trinity_parameter_specs()}
    return [spec_map[name] for name in FREE_PARAMETER_ORDER if name not in DEFAULT_FIXED_PARAMETERS]


INPUT_PARAMETER_SPECS = _parameter_specs()
INPUT_COLUMNS = [spec.short_name for spec in INPUT_PARAMETER_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emcee MCMC using direct or PCA observable GP emulators for selected mass-function campaign observables."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument(
        "--likelihood-case",
        choices=sorted(LIKELIHOOD_CASES),
        required=True,
        help="Which observable combination to include in the likelihood.",
    )
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-walkers", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--burn-in", type=int, default=800)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--init-center", choices=["prior_center", "best_training"], default="best_training")
    parser.add_argument("--init-quantile-sigma", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--mcmc-dir-name", default=None)
    parser.add_argument("--figures-dir-name", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--emulator-mode", choices=["direct", "pca"], default="direct")
    parser.add_argument(
        "--include-emulator-variance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include GP predictive variance in the likelihood covariance.",
    )
    parser.add_argument(
        "--thin-disk-maximum-prior-median",
        type=float,
        default=None,
        help="Override the thinDiskMaximum prior median (x0) while keeping the original support.",
    )
    parser.add_argument(
        "--thin-disk-maximum-prior-sigma-dex",
        type=float,
        default=None,
        help="Override the thinDiskMaximum prior scatter in dex.",
    )
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="standardized")
    parser.add_argument("--default-pca-components", type=int, default=None)
    parser.add_argument(
        "--pca-components",
        action="append",
        default=[],
        help="Override PCA components as analysis_key=n, e.g. smf_z0=6",
    )
    return parser.parse_args()


def _parameter_specs_with_overrides(args: argparse.Namespace):
    specs = list(_parameter_specs())
    if args.thin_disk_maximum_prior_median is None and args.thin_disk_maximum_prior_sigma_dex is None:
        return specs

    median = args.thin_disk_maximum_prior_median
    sigma_dex = args.thin_disk_maximum_prior_sigma_dex
    for index, spec in enumerate(specs):
        if spec.short_name != "thinDiskMaximum":
            continue
        if not isinstance(spec.prior, TruncatedLogNormalPrior):
            raise TypeError("thinDiskMaximum prior is no longer TruncatedLogNormalPrior; override logic needs updating.")
        updated_prior = TruncatedLogNormalPrior(
            lower=spec.prior.lower,
            upper=spec.prior.upper,
            x0=spec.prior.x0 if median is None else float(median),
            sigma=spec.prior.sigma if sigma_dex is None else float(sigma_dex) * float(np.log(10.0)),
        )
        specs[index] = replace(spec, prior=updated_prior)
        break
    return specs


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _block_diag(arrays: list[np.ndarray]) -> np.ndarray:
    total = sum(array.shape[0] for array in arrays)
    block = np.zeros((total, total), dtype=float)
    start = 0
    for array in arrays:
        n = array.shape[0]
        block[start : start + n, start : start + n] = array
        start += n
    return block


def _transform_x(values: np.ndarray, *, is_log: bool) -> np.ndarray:
    if is_log:
        if np.any(values <= 0.0):
            raise ValueError("Cannot log10-transform non-positive x values")
        return np.log10(values)
    return values


def _transform_y(
    y_linear: np.ndarray,
    y_noise_linear: np.ndarray | None,
    target_linear: np.ndarray,
    target_cov_linear: np.ndarray,
    *,
    is_log: bool,
    min_log10_y: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, dict]:
    if not is_log:
        target_noise = np.sqrt(np.maximum(np.diag(target_cov_linear), 0.0))
        return (
            y_linear,
            y_noise_linear,
            target_linear,
            target_noise,
            {
                "y_transform": "identity",
            },
        )

    positive = np.concatenate([y_linear[y_linear > 0.0], target_linear[target_linear > 0.0]])
    if positive.size == 0:
        raise ValueError("Analysis contains no positive values to log-transform")
    effective_floor_linear = 10.0 ** min_log10_y
    safe_y_linear = np.where(y_linear > 0.0, y_linear, effective_floor_linear)
    y_log10_raw = np.log10(safe_y_linear)
    y_plot = np.maximum(y_log10_raw, min_log10_y)

    safe_target_linear = np.where(target_linear > 0.0, target_linear, effective_floor_linear)
    target_log10 = np.maximum(np.log10(safe_target_linear), min_log10_y)
    jacobian = np.diag(1.0 / (np.log(10.0) * safe_target_linear))
    target_cov_log10 = jacobian @ target_cov_linear @ jacobian
    target_std_log10 = np.sqrt(np.maximum(np.diag(target_cov_log10), 0.0))

    if y_noise_linear is None:
        y_noise = None
    else:
        y_noise = np.maximum(y_noise_linear / (safe_y_linear * np.log(10.0)), 1.0e-6)
        y_noise = np.where(y_log10_raw > min_log10_y, y_noise, np.nan)

    return (
        y_plot,
        y_noise,
        target_log10,
        target_std_log10,
        {
            "y_transform": "log10",
            "effective_floor_linear": float(effective_floor_linear),
            "min_log10_y": float(min_log10_y),
        },
    )


def _prepare_training_targets(
    y_values: np.ndarray,
    y_noise: np.ndarray | None,
    *,
    bad_training_condition: str,
    use_training_alpha: bool,
    bad_training_value_fill: str | float,
    bad_training_sigma: float,
    min_training_sigma: float,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    y_fit = np.asarray(y_values, dtype=float).copy()
    metadata = {
        "training_alpha_used": bool(use_training_alpha),
        "bad_training_condition": bad_training_condition,
        "bad_training_value_fill": bad_training_value_fill,
        "bad_training_sigma": float(bad_training_sigma),
        "min_training_sigma": float(min_training_sigma),
        "n_bad_training_points": 0,
    }
    if not use_training_alpha:
        return y_fit, None, metadata

    sigma = np.asarray(y_noise, dtype=float).copy() if y_noise is not None else np.zeros_like(y_fit)
    sigma = np.where(np.isfinite(sigma), sigma, np.nan)
    bad_mask = ~np.isfinite(y_fit)
    if bad_training_condition == "nonpositive":
        bad_mask |= (y_fit <= 0.0)
    elif bad_training_condition == "zero_only":
        bad_mask |= (y_fit == 0.0)
    metadata["n_bad_training_points"] = int(np.sum(bad_mask))

    if bad_training_value_fill == "bin_median":
        finite = y_fit[np.isfinite(y_fit) & ~bad_mask]
        fallback_value = float(np.nanmedian(finite)) if finite.size else 0.0
        for bin_index in range(y_fit.shape[1]):
            valid = np.isfinite(y_fit[:, bin_index]) & ~bad_mask[:, bin_index]
            replacement = float(np.nanmedian(y_fit[valid, bin_index])) if np.any(valid) else fallback_value
            y_fit[bad_mask[:, bin_index], bin_index] = replacement
    else:
        y_fit[bad_mask] = float(bad_training_value_fill)

    sigma = np.where(bad_mask, bad_training_sigma, sigma)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, min_training_sigma)
    return y_fit, sigma, metadata


class MeanCenterer:
    def fit(self, y: np.ndarray) -> "MeanCenterer":
        self.mean_ = np.mean(y, axis=0)
        self.scale_ = np.ones(y.shape[1])
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y - self.mean_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y + self.mean_


def _make_preprocessor(pca_scaling: str):
    if pca_scaling == "standardized":
        return StandardScaler()
    if pca_scaling == "unscaled":
        return MeanCenterer()
    raise ValueError(f"Unknown pca_scaling={pca_scaling!r}")


def _parse_pca_component_overrides(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --pca-components value {value!r}; expected analysis_key=n")
        key, raw_count = value.split("=", 1)
        key = key.strip()
        if key not in ANALYSIS_CONFIGS:
            raise ValueError(f"Unknown analysis key {key!r} in --pca-components")
        result[key] = int(raw_count)
    return result


def _pca_component_count(
    analysis_key: str,
    *,
    default_pca_components: int | None,
    overrides: dict[str, int],
) -> int:
    if analysis_key in overrides:
        return overrides[analysis_key]
    if default_pca_components is not None:
        return default_pca_components
    return DEFAULT_PCA_COMPONENTS_BY_ANALYSIS_KEY.get(analysis_key, 5)


def _load_analysis_bundle(
    campaign_root: Path,
    analysis_key: str,
    hdf5_filename: str,
    *,
    n_restarts_optimizer: int,
    emulator_mode: str,
    pca_scaling: str,
    pca_components: int,
) -> dict[str, object]:
    config = ANALYSIS_CONFIGS[analysis_key]
    analysis = config["analysis"]
    samples = pd.read_csv(campaign_root / "samples.csv")
    x_quantiles = samples[[f"{name}_quantile" for name in INPUT_COLUMNS]].to_numpy(dtype=float)

    first_path = campaign_root / "evaluations" / samples.iloc[0]["evaluation_id"] / hdf5_filename
    with h5py.File(first_path, "r") as handle:
        group = handle[f"/analyses/{analysis}"]
        attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
        x_dataset = attrs["xDataset"]
        y_dataset = attrs["yDataset"]
        target_dataset = attrs["yDatasetTarget"]
        covariance_dataset = attrs.get("yCovariance")
        target_covariance_dataset = attrs.get("yCovarianceTarget")
        x_is_log = bool(attrs.get("xAxisIsLog", False))
        y_is_log = bool(attrs.get("yAxisIsLog", False))

    x_bins = None
    target_linear = None
    target_cov_linear = None
    y_rows = []
    y_noise_rows = []
    for sample in samples.itertuples(index=False):
        path = campaign_root / "evaluations" / sample.evaluation_id / hdf5_filename
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            current_x = np.asarray(group[x_dataset][...], dtype=float)
            current_target = np.asarray(group[target_dataset][...], dtype=float)
            current_y = np.asarray(group[y_dataset][...], dtype=float)
            current_covariance = np.asarray(group[covariance_dataset][...], dtype=float) if covariance_dataset else None
            current_noise = (
                np.sqrt(np.maximum(np.diag(current_covariance), 0.0))
                if current_covariance is not None
                else None
            )
            current_target_covariance = (
                np.asarray(group[target_covariance_dataset][...], dtype=float)
                if target_covariance_dataset
                else None
            )
        if x_bins is None:
            x_bins = current_x
            target_linear = current_target
            target_cov_linear = current_target_covariance
        y_rows.append(current_y)
        if current_noise is not None:
            y_noise_rows.append(current_noise)

    assert x_bins is not None
    assert target_linear is not None
    assert target_cov_linear is not None
    y_linear = np.vstack(y_rows)
    y_noise_linear = np.vstack(y_noise_rows) if y_noise_rows else None

    x_plot = _transform_x(x_bins, is_log=x_is_log)
    y_fit_space, y_noise_fit_space, target_fit_space, target_fit_std, transform_metadata = _transform_y(
        y_linear,
        y_noise_linear,
        target_linear,
        target_cov_linear,
        is_log=y_is_log,
        min_log10_y=-7.0,
    )
    y_fit, alpha_sigma, training_metadata = _prepare_training_targets(
        y_fit_space,
        y_noise_fit_space,
        bad_training_condition=str(config["bad_training_condition"]),
        use_training_alpha=bool(config["use_training_alpha"]),
        bad_training_value_fill=config["bad_training_value_fill"],
        bad_training_sigma=float(config["bad_training_sigma"]),
        min_training_sigma=float(config["min_training_sigma"]),
    )
    alpha = alpha_sigma**2 if alpha_sigma is not None else None

    bundle = {
        "analysis_key": analysis_key,
        "analysis": analysis,
        "label": config["label"],
        "attrs": attrs,
        "transform_metadata": transform_metadata,
        "training_target_metadata": training_metadata,
        "x_plot": x_plot,
        "target_linear": target_linear,
        "target_covariance_linear": target_cov_linear,
        "target_fit_space": target_fit_space,
        "target_fit_std": target_fit_std,
        "y_linear": y_linear,
        "y_fit": y_fit,
        "y_is_log": y_is_log,
        "x_quantiles": x_quantiles,
        "samples": samples,
    }
    if emulator_mode == "direct":
        models: list[dict[str, object]] = []
        for bin_index in range(y_fit.shape[1]):
            print(
                f"{analysis}: fitting bin {bin_index + 1}/{y_fit.shape[1]} "
                f"(x={x_plot[bin_index]:.3f})",
                flush=True,
            )
            model, y_mean, y_std = fit_scaled_gp(
                x_quantiles,
                y_fit[:, bin_index],
                n_restarts_optimizer=n_restarts_optimizer,
                optimize_hyperparameters=True,
                alpha=alpha[:, bin_index] if alpha is not None else None,
            )
            models.append(
                {
                    "model": model,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "kernel": str(model.kernel_),
                }
            )
        bundle["emulator_mode"] = "direct"
        bundle["models"] = models
        bundle["emulator_metadata"] = {
            "kernel_summaries": [model_bundle["kernel"] for model_bundle in models],
        }
        return bundle

    scaler = _make_preprocessor(pca_scaling)
    y_fit_scaled = scaler.fit_transform(y_fit)
    n_components_actual = min(pca_components, y_fit_scaled.shape[0], y_fit_scaled.shape[1])
    pca = PCA(n_components=n_components_actual)
    coefficients = pca.fit_transform(y_fit_scaled)

    coefficient_alpha = None
    if alpha is not None:
        alpha_scaled = np.asarray(alpha, dtype=float) / (scaler.scale_[None, :] ** 2)
        coefficient_alpha = alpha_scaled @ (pca.components_.T ** 2)
        coefficient_alpha = np.maximum(coefficient_alpha, 1.0e-12)

    component_models: list[dict[str, object]] = []
    for component_index in range(n_components_actual):
        print(
            f"{analysis}: fitting PCA component {component_index + 1}/{n_components_actual}",
            flush=True,
        )
        model, y_mean, y_std = fit_scaled_gp(
            x_quantiles,
            coefficients[:, component_index],
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=True,
            alpha=coefficient_alpha[:, component_index] if coefficient_alpha is not None else None,
        )
        component_models.append(
            {
                "model": model,
                "y_mean": y_mean,
                "y_std": y_std,
                "kernel": str(model.kernel_),
            }
        )

    bundle["emulator_mode"] = "pca"
    bundle["models"] = component_models
    bundle["pca"] = {
        "scaler": scaler,
        "components": pca.components_,
        "mean": pca.mean_,
        "n_components_actual": int(n_components_actual),
    }
    bundle["emulator_metadata"] = {
        "pca_scaling": pca_scaling,
        "pca_components_requested": int(pca_components),
        "pca_components_actual": int(n_components_actual),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "kernel_summaries": [model_bundle["kernel"] for model_bundle in component_models],
    }
    return bundle


def _predict_bundle(bundle: dict[str, object], x_quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_quantiles = np.atleast_2d(x_quantiles)
    if bundle.get("emulator_mode", "direct") == "direct":
        pred = np.zeros((x_quantiles.shape[0], len(bundle["models"])), dtype=float)
        pred_std = np.zeros_like(pred)
        for bin_index, model_bundle in enumerate(bundle["models"]):
            value, value_std = predict_scaled_gp(
                model_bundle["model"],
                model_bundle["y_mean"],
                model_bundle["y_std"],
                x_quantiles,
            )
            pred[:, bin_index] = value
            pred_std[:, bin_index] = value_std
        return pred, pred_std

    n_components = len(bundle["models"])
    coefficient_predictions = np.zeros((x_quantiles.shape[0], n_components), dtype=float)
    coefficient_stds = np.zeros_like(coefficient_predictions)
    for component_index, model_bundle in enumerate(bundle["models"]):
        value, value_std = predict_scaled_gp(
            model_bundle["model"],
            model_bundle["y_mean"],
            model_bundle["y_std"],
            x_quantiles,
        )
        coefficient_predictions[:, component_index] = value
        coefficient_stds[:, component_index] = value_std

    pca_meta = bundle["pca"]
    scaler = pca_meta["scaler"]
    components = np.asarray(pca_meta["components"], dtype=float)
    mean = np.asarray(pca_meta["mean"], dtype=float)
    y_pred_scaled = coefficient_predictions @ components + mean[None, :]
    y_pred = scaler.inverse_transform(y_pred_scaled)
    component_variance = coefficient_stds ** 2
    y_var_scaled = component_variance @ (components ** 2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * scaler.scale_[None, :]
    return y_pred, y_std


def _log10_normal_to_linear_moments(mean_log10: np.ndarray, variance_log10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ln10 = float(np.log(10.0))
    mu_ln = ln10 * mean_log10
    sigma2_ln = (ln10**2) * variance_log10
    mean_linear = np.exp(mu_ln + 0.5 * sigma2_ln)
    variance_linear = (np.exp(sigma2_ln) - 1.0) * np.exp(2.0 * mu_ln + sigma2_ln)
    return mean_linear, variance_linear


def _best_training_theta(
    bundles: list[dict[str, object]],
) -> np.ndarray:
    samples = bundles[0]["samples"]
    n_rows = len(samples)
    scores = np.zeros(n_rows, dtype=float)
    for bundle in bundles:
        residual = bundle["y_linear"] - bundle["target_linear"][None, :]
        covariance = bundle["target_covariance_linear"]
        solve = np.linalg.solve(covariance, residual.T).T
        scores += np.sum(residual * solve, axis=1)
    best_index = int(np.argmin(scores))
    return samples.iloc[best_index][INPUT_COLUMNS].to_numpy(dtype=float)


def _gaussian_log_likelihood(residual: np.ndarray, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -np.inf
    quadratic = float(residual @ np.linalg.solve(covariance, residual))
    return -0.5 * (quadratic + logdet + len(residual) * np.log(2.0 * np.pi))


class ObservableEmulatorPosterior:
    def __init__(
        self,
        *,
        parameter_specs,
        bundles: list[dict[str, object]],
        output_slices: dict[str, slice],
        include_emulator_variance: bool,
    ):
        self.parameter_specs = parameter_specs
        self.bundles = bundles
        self.output_slices = output_slices
        self.include_emulator_variance = include_emulator_variance
        self.target_vector = np.concatenate([bundle["target_linear"] for bundle in bundles])
        self.target_covariance = _block_diag([bundle["target_covariance_linear"] for bundle in bundles])

    def predict_batch(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        x_quantiles = transform_to_prior_quantiles(self.parameter_specs, theta)
        mean_blocks = []
        var_blocks = []
        for bundle in self.bundles:
            pred, pred_std = _predict_bundle(bundle, x_quantiles)
            if bundle["y_is_log"]:
                mean_linear, var_linear = _log10_normal_to_linear_moments(pred, pred_std**2)
            else:
                mean_linear, var_linear = pred, pred_std**2
            mean_blocks.append(mean_linear)
            var_blocks.append(var_linear)
        return np.concatenate(mean_blocks, axis=1), np.concatenate(var_blocks, axis=1)

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, var = self.predict_batch(theta[None, :])
        return mean[0], var[0]

    def log_probability(self, theta: np.ndarray) -> float:
        log_prior = float(log_prior_density(self.parameter_specs, theta[None, :])[0])
        if not np.isfinite(log_prior):
            return -np.inf
        mean, variance = self.predict(theta)
        if self.include_emulator_variance:
            covariance = self.target_covariance + np.diag(variance + 1.0e-8)
        else:
            covariance = self.target_covariance
        residual = self.target_vector - mean
        try:
            log_like = _gaussian_log_likelihood(residual, covariance)
        except np.linalg.LinAlgError:
            return -np.inf
        return log_prior + log_like if np.isfinite(log_like) else -np.inf

    def log_probability_batch(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        log_prior = np.asarray(log_prior_density(self.parameter_specs, theta), dtype=float)
        result = np.full(theta.shape[0], -np.inf, dtype=float)
        valid = np.isfinite(log_prior)
        if not np.any(valid):
            return result
        means, variances = self.predict_batch(theta[valid])
        residuals = self.target_vector[None, :] - means
        valid_indices = np.where(valid)[0]
        for local_index, sample_index in enumerate(valid_indices):
            if self.include_emulator_variance:
                covariance = self.target_covariance + np.diag(variances[local_index] + 1.0e-8)
            else:
                covariance = self.target_covariance
            try:
                log_like = _gaussian_log_likelihood(residuals[local_index], covariance)
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(log_like):
                result[sample_index] = log_prior[sample_index] + log_like
        return result


def _display_y_label(bundle: dict[str, object]) -> str:
    if bundle["y_is_log"]:
        return r"$\log_{10}(\Phi)$"
    return str(bundle["attrs"].get("yAxisLabel", "Observable"))


def _display_x_label(bundle: dict[str, object]) -> str:
    return str(bundle["attrs"].get("xAxisLabel", "x"))


def _plot_best_fit_observables(
    bundles: list[dict[str, object]],
    prediction_mean_by_key: dict[str, np.ndarray],
    prediction_std_by_key: dict[str, np.ndarray],
    slow_theta_quantiles: np.ndarray,
    output_path: Path,
) -> None:
    y_limits_override = {
        "smf_z0": (-7.0, -1.0),
        "smf_z3": (-7.0, -1.0),
        "sfr": (-7.0, -1.0),
    }
    n_panels = len(bundles)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, bundle in zip(axes_flat, bundles, strict=False):
        x = bundle["x_plot"]
        target = bundle["target_linear"]
        target_std = np.sqrt(np.maximum(np.diag(bundle["target_covariance_linear"]), 0.0))
        pred = prediction_mean_by_key[bundle["analysis_key"]]
        pred_std = prediction_std_by_key[bundle["analysis_key"]]
        if bundle["y_is_log"]:
            pred_log10, pred_std_log10 = _predict_bundle(bundle, np.atleast_2d(slow_theta_quantiles))
            pred_plot = pred_log10[0]
            pred_std_plot = pred_std_log10[0]
            lower_target = np.maximum(target - target_std, 1.0e-30)
            upper_target = target + target_std
            target_plot = np.log10(np.maximum(target, 1.0e-30))
            yerr_lower = target_plot - np.log10(lower_target)
            yerr_upper = np.log10(upper_target) - target_plot
            pred_lower_plot = pred_plot - pred_std_plot
            pred_upper_plot = pred_plot + pred_std_plot
            override = y_limits_override.get(bundle["analysis_key"])
            if override is None:
                central_values = np.concatenate([target_plot, pred_plot])
                central_values = central_values[np.isfinite(central_values)]
                lower = float(np.min(central_values))
                upper = float(np.max(central_values))
                margin = 0.08 * (upper - lower if upper > lower else 1.0)
                y_limits = (lower - margin, upper + margin)
            else:
                y_limits = override
            axis.errorbar(x, target_plot, yerr=np.vstack([yerr_lower, yerr_upper]), fmt="o", color="0.15")
            axis.plot(x, pred_plot, color="tab:blue", lw=1.8)
            axis.fill_between(x, pred_lower_plot, pred_upper_plot, color="tab:blue", alpha=0.2)
            axis.set_ylim(*y_limits)
            axis.set_ylabel(r"$\log_{10}(\Phi)$")
        else:
            central_values = np.concatenate([target, pred])
            central_values = central_values[np.isfinite(central_values)]
            lower = float(np.min(central_values))
            upper = float(np.max(central_values))
            margin = 0.08 * (upper - lower if upper > lower else 1.0)
            axis.errorbar(x, target, yerr=target_std, fmt="o", color="0.15")
            axis.plot(x, pred, color="tab:blue", lw=1.8)
            axis.fill_between(x, pred - pred_std, pred + pred_std, color="tab:blue", alpha=0.2)
            axis.set_ylim(lower - margin, upper + margin)
            axis.set_ylabel(_display_y_label(bundle))
        axis.set_title(bundle["label"])
        axis.set_xlabel(_display_x_label(bundle))
        axis.grid(alpha=0.22)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _make_trace_plot(
    samples: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    parameter_specs=None,
    burn_in: int = 0,
) -> dict[str, float]:
    return make_trace_plot(
        samples,
        labels,
        output_path,
        parameter_specs=parameter_specs,
        burn_in=burn_in,
        row_height=2.0,
        alpha=0.25,
        linewidth=0.6,
    )


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    parameter_specs = _parameter_specs_with_overrides(args)
    pca_component_overrides = _parse_pca_component_overrides(args.pca_components)
    default_output_prefix = f"observable_{args.likelihood_case}"
    mcmc_dir_name = args.mcmc_dir_name or f"emulator_observable_mcmc_{args.likelihood_case}"
    figures_dir_name = args.figures_dir_name or f"figures_observable_mcmc_{args.likelihood_case}"
    output_prefix = args.output_prefix or default_output_prefix
    mcmc_root = campaign_root / mcmc_dir_name
    figures_root = campaign_root / figures_dir_name
    mcmc_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    if emcee is None:
        raise ModuleNotFoundError(
            "The 'emcee' package is required for run_mass_function_observable_emulator_mcmc.py. "
            "Activate the galacticus-workspace environment or install emcee."
        )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    analysis_keys = LIKELIHOOD_CASES[args.likelihood_case]
    bundles = [
        _load_analysis_bundle(
            campaign_root,
            analysis_key,
            args.hdf5_filename,
            n_restarts_optimizer=args.n_restarts_optimizer,
            emulator_mode=args.emulator_mode,
            pca_scaling=args.pca_scaling,
            pca_components=_pca_component_count(
                analysis_key,
                default_pca_components=args.default_pca_components,
                overrides=pca_component_overrides,
            ),
        )
        for analysis_key in analysis_keys
    ]

    output_slices: dict[str, slice] = {}
    start = 0
    for bundle in bundles:
        n_out = len(bundle["x_plot"])
        output_slices[bundle["analysis_key"]] = slice(start, start + n_out)
        start += n_out

    posterior = ObservableEmulatorPosterior(
        parameter_specs=parameter_specs,
        bundles=bundles,
        output_slices=output_slices,
        include_emulator_variance=bool(args.include_emulator_variance),
    )

    if args.init_center == "prior_center":
        center_quantiles = np.full(len(INPUT_COLUMNS), 0.5, dtype=float)
        center_theta = transform_from_prior_quantiles(parameter_specs, center_quantiles[None, :])[0]
    else:
        center_theta = _best_training_theta(bundles)
        center_quantiles = transform_to_prior_quantiles(parameter_specs, center_theta[None, :])[0]

    rng = np.random.default_rng(args.seed)
    initial_quantiles = np.clip(
        center_quantiles + args.init_quantile_sigma * rng.normal(size=(args.n_walkers, len(INPUT_COLUMNS))),
        1.0e-4,
        1.0 - 1.0e-4,
    )
    initial_positions = transform_from_prior_quantiles(parameter_specs, initial_quantiles)

    sampler_kwargs = {"nwalkers": args.n_walkers, "ndim": len(INPUT_COLUMNS)}
    if args.n_processes > 1:
        from multiprocessing import Pool

        with Pool(processes=args.n_processes) as pool:
            sampler = emcee.EnsembleSampler(
                log_prob_fn=posterior.log_probability,
                pool=pool,
                vectorize=False,
                **sampler_kwargs,
            )
            sampler.run_mcmc(initial_positions, args.n_steps, progress=args.progress, skip_initial_state_check=True)
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
        sampler.run_mcmc(initial_positions, args.n_steps, progress=args.progress, skip_initial_state_check=True)
        chain = sampler.get_chain()
        log_prob = sampler.get_log_prob()
        flat_samples = sampler.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
        flat_log_prob = sampler.get_log_prob(discard=args.burn_in, thin=args.thin, flat=True)
        acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=float)

    np.save(mcmc_root / f"{output_prefix}_chain.npy", chain)
    np.save(mcmc_root / f"{output_prefix}_log_prob.npy", log_prob)

    posterior_df = pd.DataFrame(flat_samples, columns=INPUT_COLUMNS)
    posterior_df["log_probability"] = flat_log_prob
    posterior_path = mcmc_root / f"{output_prefix}_posterior_samples.csv"
    posterior_df.to_csv(posterior_path, index=False)

    best_index = int(np.argmax(flat_log_prob))
    best_theta = flat_samples[best_index]
    best_mean, best_var = posterior.predict(best_theta)
    best_std = np.sqrt(np.maximum(best_var, 0.0))

    prediction_rows = []
    prediction_mean_by_key: dict[str, np.ndarray] = {}
    prediction_std_by_key: dict[str, np.ndarray] = {}
    for bundle in bundles:
        slc = output_slices[bundle["analysis_key"]]
        pred = best_mean[slc]
        pred_std = best_std[slc]
        prediction_mean_by_key[bundle["analysis_key"]] = pred
        prediction_std_by_key[bundle["analysis_key"]] = pred_std
        for bin_index, (x_value, target_value, target_std_value, pred_value, pred_std_value) in enumerate(
            zip(
                bundle["x_plot"],
                bundle["target_linear"],
                np.sqrt(np.maximum(np.diag(bundle["target_covariance_linear"]), 0.0)),
                pred,
                pred_std,
                strict=True,
            )
        ):
            prediction_rows.append(
                {
                    "analysis_key": bundle["analysis_key"],
                    "analysis": bundle["analysis"],
                    "label": bundle["label"],
                    "bin_index": int(bin_index),
                    "x_plot": float(x_value),
                    "target_value": float(target_value),
                    "target_std": float(target_std_value),
                    "prediction_value": float(pred_value),
                    "prediction_std": float(pred_std_value),
                }
            )
    prediction_path = mcmc_root / f"{output_prefix}_best_fit_observables.csv"
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)

    summary = {
        "campaign_root": str(campaign_root),
        "likelihood_case": args.likelihood_case,
        "emulator_mode": args.emulator_mode,
        "include_emulator_variance": bool(args.include_emulator_variance),
        "analysis_keys": analysis_keys,
        "input_columns": INPUT_COLUMNS,
        "parameter_priors": {
            spec.short_name: {
                "prior_type": type(spec.prior).__name__,
                **spec.prior.__dict__,
            }
            for spec in parameter_specs
        },
        "n_restarts_optimizer": args.n_restarts_optimizer,
        "n_walkers": args.n_walkers,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "seed": args.seed,
        "init_center": args.init_center,
        "init_quantile_sigma": args.init_quantile_sigma,
        "n_processes": args.n_processes,
        "progress": args.progress,
        "pca_scaling": args.pca_scaling,
        "default_pca_components": args.default_pca_components,
        "pca_component_overrides": pca_component_overrides,
        "thin_disk_maximum_prior_median": args.thin_disk_maximum_prior_median,
        "thin_disk_maximum_prior_sigma_dex": args.thin_disk_maximum_prior_sigma_dex,
        "mean_acceptance_fraction": float(np.mean(acceptance_fraction)),
        "acceptance_fraction_by_walker": [float(value) for value in acceptance_fraction],
        "best_log_probability": float(flat_log_prob[best_index]),
        "best_theta": {name: float(value) for name, value in zip(INPUT_COLUMNS, best_theta, strict=True)},
        "analysis_metadata": {
            bundle["analysis_key"]: {
                "analysis": bundle["analysis"],
                "label": bundle["label"],
                "training_target_metadata": bundle["training_target_metadata"],
                "n_bins": int(len(bundle["x_plot"])),
                "emulator_metadata": bundle["emulator_metadata"],
            }
            for bundle in bundles
        },
    }
    summary_path = mcmc_root / f"{output_prefix}_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    emulator_bundle_path = mcmc_root / f"{output_prefix}_fitted_emulators.joblib"
    joblib.dump(
        {
            "bundle_type": "observable_mcmc_fitted_emulators",
            "campaign_root": str(campaign_root),
            "likelihood_case": args.likelihood_case,
            "emulator_mode": args.emulator_mode,
            "include_emulator_variance": bool(args.include_emulator_variance),
            "input_columns": INPUT_COLUMNS,
            "parameter_specs": parameter_specs,
            "analysis_keys": analysis_keys,
            "bundles": bundles,
            "output_slices": output_slices,
            "best_theta": {name: float(value) for name, value in zip(INPUT_COLUMNS, best_theta, strict=True)},
        },
        emulator_bundle_path,
    )

    trace_path = figures_root / f"{output_prefix}_trace.png"
    trace_tau = _make_trace_plot(
        chain,
        INPUT_COLUMNS,
        trace_path,
        parameter_specs=parameter_specs,
        burn_in=args.burn_in,
    )
    summary["trace_autocorrelation_time"] = trace_tau
    summary["trace_autocorrelation_time_burn_in"] = int(args.burn_in)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    corner_path = figures_root / f"{output_prefix}_corner.png"
    if corner is not None:
        corner_fig = corner_with_log10_priors(
            corner,
            posterior_df,
            INPUT_COLUMNS,
            parameter_specs,
            truths=best_theta,
        )
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)

    best_fit_fig_path = figures_root / f"{output_prefix}_best_fit_observables.png"
    best_theta_quantiles = transform_to_prior_quantiles(parameter_specs, best_theta[None, :])[0]
    _plot_best_fit_observables(
        bundles,
        prediction_mean_by_key,
        prediction_std_by_key,
        best_theta_quantiles,
        best_fit_fig_path,
    )

    print(posterior_path)
    print(summary_path)
    print(emulator_bundle_path)
    print(trace_path)
    if corner is not None:
        print(corner_path)
    print(best_fit_fig_path)


if __name__ == "__main__":
    main()
