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

from fit_halpha_sobol_pca_gp_cv import DUST_INPUT_PARAMETER_SPECS
from run_mass_function_observable_emulator_mcmc import (
    INPUT_COLUMNS as SLOW_INPUT_COLUMNS,
    INPUT_PARAMETER_SPECS as SLOW_INPUT_PARAMETER_SPECS,
    LIKELIHOOD_CASES,
    _best_training_theta,
    _block_diag,
    _gaussian_log_likelihood,
    _load_analysis_bundle,
    _make_trace_plot,
    _parameter_specs_with_overrides,
    _pca_component_count,
    _parse_pca_component_overrides,
    _plot_best_fit_observables,
    _predict_bundle,
    _log10_normal_to_linear_moments,
    _transform_x,
)
from galacticus_emu.gp import predict_scaled_gp
from galacticus_emu.lhs import log_prior_density, transform_from_prior_quantiles, transform_to_prior_quantiles
from galacticus_emu.mcmc_corner import corner_with_log10_priors
from galacticus_emu.observable_plot_metadata import (
    DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET,
    HALPHA_LF_X_AXIS_LABEL,
    HALPHA_LF_Y_AXIS_LABEL,
    STANDARD_OBSERVABLE_PLOT_METADATA,
    sidecar_lf_axis_label,
    sidecar_lf_target_label,
    standard_observable_y_display_offset,
)


DEFAULT_HAPHA_KEYS = ["Z1", "Z2", "Z3", "Z4"]
HALPHA_ORDER = ["Z1", "Z2", "Z3", "Z4"]
DUST_DEFAULTS = {
    "delta_0": 0.0,
    "delta_z": 0.0,
    "delta_M": 0.0,
    "delta_Mz": 0.0,
    "attenuation_scatter": 0.25,
}
ALL_PARAMETER_SPECS = list(SLOW_INPUT_PARAMETER_SPECS) + list(DUST_INPUT_PARAMETER_SPECS)
ALL_INPUT_COLUMNS = [spec.short_name for spec in ALL_PARAMETER_SPECS]
SLOW_DIM = len(SLOW_INPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a joint emcee MCMC over 19 slow Galacticus parameters plus 5 dust parameters, "
            "combining the SMF/SFR observable emulators with saved Sobol Halpha PCA bundles."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument(
        "--observable-likelihood-case",
        choices=["smf_pair_sfr"],
        default="smf_pair_sfr",
        help="Which non-Halpha observable combination to include.",
    )
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--halpha-bundle", action="append", required=True, help="Repeat as Z1=/path/to/bundle.joblib")
    parser.add_argument("--extra-observable-bundle", type=Path, default=None)
    parser.add_argument("--extra-observable-analysis-key", default="bh_halo_mass_trinity_z1")
    parser.add_argument("--extra-observable-label", default="TRINITY z~1 BHHMR")
    parser.add_argument("--extra-observable-xmin", type=float, default=None)
    parser.add_argument("--extra-observable-xmax", type=float, default=None)
    parser.add_argument("--observable-emulator-mode", choices=["direct", "pca"], default="direct")
    parser.add_argument(
        "--include-emulator-variance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include GP predictive variance from both observable and Halpha emulators.",
    )
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-walkers", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--init-center", choices=["prior_center", "observable_best_fit"], default="observable_best_fit")
    parser.add_argument("--init-quantile-sigma", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--mcmc-dir-name", default="emulator_joint_observable_halpha_mcmc")
    parser.add_argument("--figures-dir-name", default="figures_joint_observable_halpha_mcmc")
    parser.add_argument("--output-prefix", default="joint_observable_halpha")
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="standardized")
    parser.add_argument("--default-pca-components", type=int, default=None)
    parser.add_argument("--pca-components", action="append", default=[])
    parser.add_argument("--thin-disk-maximum-prior-median", type=float, default=None)
    parser.add_argument("--thin-disk-maximum-prior-sigma-dex", type=float, default=None)
    return parser.parse_args()


def _parse_halpha_bundle_args(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --halpha-bundle value {value!r}; expected Z1=/path/to/bundle.joblib")
        key, path = value.split("=", 1)
        key = key.strip()
        if key not in HALPHA_ORDER:
            raise ValueError(f"Unknown Sobral label {key!r} in --halpha-bundle")
        result[key] = Path(path).expanduser().resolve()
    return result


def _load_halpha_bundle(path: Path) -> dict[str, object]:
    bundle = joblib.load(path)
    if bundle.get("bundle_type") != "halpha_sobol_pca_gp":
        raise ValueError(f"Unexpected Halpha bundle type in {path}: {bundle.get('bundle_type')}")
    return bundle


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_external_observable_bundle(
    path: Path,
    *,
    analysis_key: str,
    label: str,
    xmin: float | None,
    xmax: float | None,
) -> dict[str, object]:
    raw = joblib.load(path)
    if raw.get("bundle_type") != "multid_function_direct_gp":
        raise ValueError(f"Unexpected external observable bundle type in {path}: {raw.get('bundle_type')}")
    campaign_root = Path(raw["campaign_root"])
    hdf5_filename = raw.get("hdf5_filename", "galacticus_reduced.hdf5")
    analysis = raw["analysis"]
    samples = pd.read_csv(campaign_root / "samples.csv")
    first_eval = samples.iloc[0]["evaluation_id"]
    first_path = campaign_root / "evaluations" / first_eval / hdf5_filename
    with h5py.File(first_path, "r") as handle:
        group = handle[f"/analyses/{analysis}"]
        attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
        x_dataset = attrs["xDataset"]
        y_dataset = attrs["yDataset"]
        target_dataset = attrs["yDatasetTarget"]
        target_cov_dataset = attrs.get("yCovarianceTarget")
        x_full_linear = np.asarray(group[x_dataset][...], dtype=float)
        target_full = np.asarray(group[target_dataset][...], dtype=float)
        target_cov_full = np.asarray(group[target_cov_dataset][...], dtype=float)
    x_full = _transform_x(x_full_linear, is_log=bool(attrs.get("xAxisIsLog", False)))

    x_plot = np.asarray(raw["x_plot"], dtype=float)
    target_plot = np.asarray(raw["target_plot"], dtype=float)
    target_std_plot = np.asarray(raw["target_std_plot"], dtype=float)
    models = [
        {
            "model": model,
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "kernel": str(model.kernel_),
        }
        for model, y_mean, y_std in zip(raw["models"], raw["y_means"], raw["y_stds"], strict=True)
    ]

    mask = np.ones_like(x_plot, dtype=bool)
    if xmin is not None:
        mask &= x_plot > xmin
    if xmax is not None:
        mask &= x_plot < xmax
    if not np.any(mask):
        raise ValueError("External observable x-range cut removed all bins.")

    selected_x = x_plot[mask]
    selected_target = target_plot[mask]
    selected_target_std = target_std_plot[mask]
    selected_models = [model for model, keep in zip(models, mask.tolist(), strict=True) if keep]

    selected_indices = []
    for value in selected_x:
        matches = np.flatnonzero(np.isclose(x_full, value, rtol=0.0, atol=1.0e-8))
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely match x={value} back to source target covariance.")
        selected_indices.append(int(matches[0]))
    selected_indices = np.asarray(selected_indices, dtype=int)
    target_linear = target_full[selected_indices]
    target_covariance_linear = target_cov_full[np.ix_(selected_indices, selected_indices)]

    y_rows = []
    for evaluation_id in samples["evaluation_id"]:
        eval_path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        with h5py.File(eval_path, "r") as handle:
            current_y = np.asarray(handle[f"/analyses/{analysis}/{y_dataset}"][...], dtype=float)
        y_rows.append(current_y[selected_indices])
    y_linear = np.vstack(y_rows)

    return {
        "analysis_key": analysis_key,
        "analysis": analysis,
        "label": label,
        "attrs": attrs,
        "transform_metadata": raw.get("transform_metadata", {"y_transform": "identity"}),
        "training_target_metadata": raw.get("training_metadata", {}),
        "x_plot": selected_x,
        "target_linear": target_linear,
        "target_covariance_linear": target_covariance_linear,
        "target_fit_space": selected_target,
        "target_fit_std": selected_target_std,
        "y_linear": y_linear,
        "y_is_log": bool(attrs.get("yAxisIsLog", False)),
        "x_quantiles": samples[[f"{name}_quantile" for name in SLOW_INPUT_COLUMNS]].to_numpy(dtype=float),
        "samples": samples,
        "emulator_mode": "direct",
        "models": selected_models,
        "emulator_metadata": {
            "source_bundle_path": str(path),
            "x_range_cut": [xmin, xmax],
            "kernel_summaries": [model_bundle["kernel"] for model_bundle in selected_models],
        },
    }


def _predict_halpha_bundle(bundle: dict[str, object], x_quantiles_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_quantiles_all = np.atleast_2d(x_quantiles_all)
    coefficient_predictions = np.zeros((x_quantiles_all.shape[0], len(bundle["models"])), dtype=float)
    coefficient_stds = np.zeros_like(coefficient_predictions)
    for index, model in enumerate(bundle["models"]):
        pred, pred_std = predict_scaled_gp(
            model,
            float(bundle["y_means"][index]),
            float(bundle["y_stds"][index]),
            x_quantiles_all,
        )
        coefficient_predictions[:, index] = pred
        coefficient_stds[:, index] = pred_std
    components = np.asarray(bundle["pca_components_matrix"], dtype=float)
    pca_mean = np.asarray(bundle["pca_mean"], dtype=float)
    preprocessor_mean = np.asarray(bundle["preprocessor_mean"], dtype=float)
    preprocessor_scale = np.asarray(bundle["preprocessor_scale"], dtype=float)
    y_pred_scaled = coefficient_predictions @ components + pca_mean[None, :]
    y_pred = y_pred_scaled * preprocessor_scale[None, :] + preprocessor_mean[None, :]
    component_variance = coefficient_stds**2
    y_var_scaled = component_variance @ (components**2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * preprocessor_scale[None, :]
    return y_pred, y_std


def _combined_center(observable_bundles: list[dict[str, object]]) -> np.ndarray:
    best_slow = _best_training_theta(observable_bundles)
    dust_values = np.array([DUST_DEFAULTS[spec.short_name] for spec in DUST_INPUT_PARAMETER_SPECS], dtype=float)
    return np.concatenate([best_slow, dust_values])


class JointPosterior:
    def __init__(
        self,
        *,
        parameter_specs,
        observable_bundles: list[dict[str, object]],
        halpha_bundles: list[dict[str, object]],
        include_emulator_variance: bool,
    ):
        self.parameter_specs = parameter_specs
        self.observable_bundles = observable_bundles
        self.halpha_bundles = halpha_bundles
        self.include_emulator_variance = include_emulator_variance

        observable_target_vector = []
        observable_cov_blocks = []
        for bundle in observable_bundles:
            observable_target_vector.append(bundle["target_linear"])
            observable_cov_blocks.append(bundle["target_covariance_linear"])
        halpha_target_vector = []
        halpha_cov_blocks = []
        for bundle in halpha_bundles:
            halpha_target_vector.append(np.asarray(bundle["target_log10"], dtype=float))
            halpha_cov_blocks.append(np.asarray(bundle["target_covariance_log10"], dtype=float))
        self.observable_target_vector = np.concatenate(observable_target_vector) if observable_target_vector else np.array([], dtype=float)
        self.observable_target_covariance = _block_diag(observable_cov_blocks) if observable_cov_blocks else np.zeros((0, 0), dtype=float)
        self.halpha_target_vector = np.concatenate(halpha_target_vector) if halpha_target_vector else np.array([], dtype=float)
        self.halpha_target_covariance = _block_diag(halpha_cov_blocks) if halpha_cov_blocks else np.zeros((0, 0), dtype=float)
        self.target_vector = np.concatenate([self.observable_target_vector, self.halpha_target_vector])
        self.target_covariance = _block_diag([self.observable_target_covariance, self.halpha_target_covariance])

    def predict_batch(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        x_quantiles_all = transform_to_prior_quantiles(self.parameter_specs, theta)
        observable_mean_blocks = []
        observable_var_blocks = []
        slow_quantiles = x_quantiles_all[:, :SLOW_DIM]
        for bundle in self.observable_bundles:
            pred, pred_std = _predict_bundle(bundle, slow_quantiles)
            if bundle["y_is_log"]:
                mean_linear, var_linear = _log10_normal_to_linear_moments(pred, pred_std**2)
            else:
                mean_linear, var_linear = pred, pred_std**2
            observable_mean_blocks.append(mean_linear)
            observable_var_blocks.append(var_linear)
        halpha_mean_blocks = []
        halpha_var_blocks = []
        for bundle in self.halpha_bundles:
            pred, pred_std = _predict_halpha_bundle(bundle, x_quantiles_all)
            halpha_mean_blocks.append(pred)
            halpha_var_blocks.append(pred_std**2)
        mean = np.concatenate(observable_mean_blocks + halpha_mean_blocks, axis=1)
        variance = np.concatenate(observable_var_blocks + halpha_var_blocks, axis=1)
        return mean, variance

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, var = self.predict_batch(theta[None, :])
        return mean[0], var[0]

    def log_probability(self, theta: np.ndarray) -> float:
        log_prior = float(log_prior_density(self.parameter_specs, theta[None, :])[0])
        if not np.isfinite(log_prior):
            return -np.inf
        mean, variance = self.predict(theta)
        covariance = self.target_covariance + np.diag(variance + 1.0e-8) if self.include_emulator_variance else self.target_covariance
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
            covariance = self.target_covariance + np.diag(variances[local_index] + 1.0e-8) if self.include_emulator_variance else self.target_covariance
            try:
                log_like = _gaussian_log_likelihood(residuals[local_index], covariance)
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(log_like):
                result[sample_index] = log_prior[sample_index] + log_like
        return result


def _plot_best_fit_halpha(
    halpha_bundles: list[dict[str, object]],
    x_quantiles_all: np.ndarray,
    actual_by_label: dict[str, dict[str, np.ndarray]] | None,
    output_path: Path,
) -> None:
    n_panels = len(halpha_bundles)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, bundle in zip(axes_flat, halpha_bundles, strict=False):
        pred, pred_std = _predict_halpha_bundle(bundle, x_quantiles_all[None, :])
        x = np.asarray(bundle["luminosity_centers"], dtype=float)
        target = np.asarray(bundle["target_log10"], dtype=float) + DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET
        target_std = np.sqrt(np.maximum(np.diag(np.asarray(bundle["target_covariance_log10"], dtype=float)), 0.0))
        pred_plot = np.asarray(pred[0], dtype=float) + DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET
        target_linear = 10.0 ** target
        sigma_linear = np.log(10.0) * target_linear * target_std
        lower_target = np.maximum(target_linear - sigma_linear, 1.0e-30)
        upper_target = target_linear + sigma_linear
        yerr_lower = target - np.log10(lower_target)
        yerr_upper = np.log10(upper_target) - target
        axis.errorbar(
            x,
            target,
            yerr=np.vstack([yerr_lower, yerr_upper]),
            fmt="o",
            color="0.15",
            label=sidecar_lf_target_label("halpha_sobral", bundle["sobral_label"]),
        )
        axis.plot(x, pred_plot, color="tab:red", lw=1.8)
        axis.fill_between(x, pred_plot - pred_std[0], pred_plot + pred_std[0], color="tab:red", alpha=0.2)
        if actual_by_label is not None and bundle["sobral_label"] in actual_by_label:
            actual = actual_by_label[bundle["sobral_label"]]
            axis.plot(
                np.asarray(actual["x_plot"], dtype=float),
                np.asarray(actual["y_plot"], dtype=float),
                color="#c23b22",
                lw=2.0,
                ls="--",
            )
        axis.set_ylim(-6.0 + np.log10(np.log(10.0)), -2.0 + np.log10(np.log(10.0)))
        axis.set_title(sidecar_lf_axis_label("halpha_sobral", bundle["sobral_label"], f"Halpha {bundle['sobral_label']}"))
        axis.set_xlabel(HALPHA_LF_X_AXIS_LABEL)
        axis.set_ylabel(HALPHA_LF_Y_AXIS_LABEL)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_joint_best_fit_observables(
    bundles: list[dict[str, object]],
    prediction_mean_by_key: dict[str, np.ndarray],
    prediction_std_by_key: dict[str, np.ndarray],
    slow_theta_quantiles: np.ndarray,
    actual_predictions: dict[str, dict[str, np.ndarray]] | None,
    output_path: Path,
) -> None:
    from run_mass_function_observable_emulator_mcmc import _display_x_label, _display_y_label

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
        observable_key = bundle["analysis_key"]
        plot_metadata = STANDARD_OBSERVABLE_PLOT_METADATA.get(observable_key, {})
        y_display_offset = standard_observable_y_display_offset(observable_key)
        x = bundle["x_plot"]
        x = np.asarray(x, dtype=float)
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x_margin = 0.08 * (x_max - x_min if x_max > x_min else 1.0)
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
            target_plot = target_plot + y_display_offset
            pred_plot = pred_plot + y_display_offset
            pred_lower_plot = pred_lower_plot + y_display_offset
            pred_upper_plot = pred_upper_plot + y_display_offset
            override = y_limits_override.get(bundle["analysis_key"])
            if override is None:
                central_values = np.concatenate([target_plot, pred_plot])
                central_values = central_values[np.isfinite(central_values)]
                lower = float(np.min(central_values))
                upper = float(np.max(central_values))
                margin = 0.08 * (upper - lower if upper > lower else 1.0)
                y_limits = (lower - margin, upper + margin)
            else:
                y_limits = (override[0] + y_display_offset, override[1] + y_display_offset)
            axis.errorbar(x, target_plot, yerr=np.vstack([yerr_lower, yerr_upper]), fmt="o", color="0.15")
            axis.plot(x, pred_plot, color="tab:blue", lw=1.8)
            axis.fill_between(x, pred_lower_plot, pred_upper_plot, color="tab:blue", alpha=0.2)
            if actual_predictions is not None and bundle["analysis_key"] in actual_predictions:
                actual = actual_predictions[bundle["analysis_key"]]
                axis.plot(
                    np.asarray(actual["x_plot"], dtype=float),
                    np.log10(np.maximum(np.asarray(actual["y_value"], dtype=float), 1.0e-30)) + y_display_offset,
                    color="#c23b22",
                    lw=2.0,
                    ls="--",
                )
            axis.set_ylim(*y_limits)
            axis.set_ylabel(plot_metadata.get("y_axis_label", r"$\log_{10}(\Phi)$"))
        else:
            target_plot = target + y_display_offset
            pred_plot = pred + y_display_offset
            central_values = np.concatenate([target_plot, pred_plot])
            central_values = central_values[np.isfinite(central_values)]
            lower = float(np.min(central_values))
            upper = float(np.max(central_values))
            margin = 0.08 * (upper - lower if upper > lower else 1.0)
            axis.errorbar(x, target_plot, yerr=target_std, fmt="o", color="0.15")
            axis.plot(x, pred_plot, color="tab:blue", lw=1.8)
            axis.fill_between(x, pred_plot - pred_std, pred_plot + pred_std, color="tab:blue", alpha=0.2)
            if actual_predictions is not None and bundle["analysis_key"] in actual_predictions:
                actual = actual_predictions[bundle["analysis_key"]]
                axis.plot(
                    np.asarray(actual["x_plot"], dtype=float),
                    np.asarray(actual["y_value"], dtype=float) + y_display_offset,
                    color="#c23b22",
                    lw=2.0,
                    ls="--",
                )
            axis.set_ylim(lower - margin, upper + margin)
            axis.set_ylabel(plot_metadata.get("y_axis_label", _display_y_label(bundle)))
        axis.set_xlim(x_min - x_margin, x_max + x_margin)
        axis.set_title(plot_metadata.get("label", bundle["label"]))
        axis.set_xlabel(plot_metadata.get("x_axis_label", _display_x_label(bundle)))
        axis.grid(alpha=0.22)
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if emcee is None:
        raise ModuleNotFoundError("The 'emcee' package is required. Activate galacticus-workspace or install emcee.")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    campaign_root = args.campaign_root.resolve()
    pca_component_overrides = _parse_pca_component_overrides(args.pca_components)
    parameter_specs = _parameter_specs_with_overrides(args) + list(DUST_INPUT_PARAMETER_SPECS)
    halpha_bundle_paths = _parse_halpha_bundle_args(args.halpha_bundle)
    observable_keys = LIKELIHOOD_CASES[args.observable_likelihood_case]
    observable_bundles = [
        _load_analysis_bundle(
            campaign_root,
            analysis_key,
            args.hdf5_filename,
            n_restarts_optimizer=args.n_restarts_optimizer,
            emulator_mode=args.observable_emulator_mode,
            pca_scaling=args.pca_scaling,
            pca_components=_pca_component_count(
                analysis_key,
                default_pca_components=args.default_pca_components,
                overrides=pca_component_overrides,
            ),
        )
        for analysis_key in observable_keys
    ]
    if args.extra_observable_bundle is not None:
        observable_bundles.append(
            _load_external_observable_bundle(
                args.extra_observable_bundle.expanduser().resolve(),
                analysis_key=args.extra_observable_analysis_key,
                label=args.extra_observable_label,
                xmin=args.extra_observable_xmin,
                xmax=args.extra_observable_xmax,
            )
        )
    halpha_bundles = [_load_halpha_bundle(halpha_bundle_paths[label]) for label in HALPHA_ORDER if label in halpha_bundle_paths]

    posterior = JointPosterior(
        parameter_specs=parameter_specs,
        observable_bundles=observable_bundles,
        halpha_bundles=halpha_bundles,
        include_emulator_variance=bool(args.include_emulator_variance),
    )

    if args.init_center == "prior_center":
        center_quantiles = np.full(len(ALL_INPUT_COLUMNS), 0.5, dtype=float)
        center_theta = transform_from_prior_quantiles(parameter_specs, center_quantiles[None, :])[0]
    else:
        center_theta = _combined_center(observable_bundles)
        center_quantiles = transform_to_prior_quantiles(parameter_specs, center_theta[None, :])[0]

    rng = np.random.default_rng(args.seed)
    initial_quantiles = np.clip(
        center_quantiles + args.init_quantile_sigma * rng.normal(size=(args.n_walkers, len(ALL_INPUT_COLUMNS))),
        1.0e-4,
        1.0 - 1.0e-4,
    )
    initial_positions = transform_from_prior_quantiles(parameter_specs, initial_quantiles)

    mcmc_root = campaign_root / args.mcmc_dir_name
    figures_root = campaign_root / args.figures_dir_name
    mcmc_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    sampler_kwargs = {"nwalkers": args.n_walkers, "ndim": len(ALL_INPUT_COLUMNS)}
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

    np.save(mcmc_root / f"{args.output_prefix}_chain.npy", chain)
    np.save(mcmc_root / f"{args.output_prefix}_log_prob.npy", log_prob)

    posterior_df = {name: flat_samples[:, index] for index, name in enumerate(ALL_INPUT_COLUMNS)}
    posterior_df["log_probability"] = flat_log_prob
    import pandas as pd
    posterior_path = mcmc_root / f"{args.output_prefix}_posterior_samples.csv"
    pd.DataFrame(posterior_df).to_csv(posterior_path, index=False)

    best_index = int(np.argmax(flat_log_prob))
    best_theta = flat_samples[best_index]
    best_mean, best_var = posterior.predict(best_theta)
    best_std = np.sqrt(np.maximum(best_var, 0.0))

    # Observable best-fit CSV
    observable_rows = []
    offset = 0
    prediction_mean_by_key: dict[str, np.ndarray] = {}
    prediction_std_by_key: dict[str, np.ndarray] = {}
    for bundle in observable_bundles:
        n_out = len(bundle["x_plot"])
        pred = best_mean[offset : offset + n_out]
        pred_std = best_std[offset : offset + n_out]
        offset += n_out
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
            observable_rows.append(
                {
                    "kind": "observable",
                    "analysis_key": bundle["analysis_key"],
                    "label": bundle["label"],
                    "bin_index": int(bin_index),
                    "x_plot": float(x_value),
                    "target_value": float(target_value),
                    "target_std": float(target_std_value),
                    "prediction_value": float(pred_value),
                    "prediction_std": float(pred_std_value),
                }
            )
    for bundle in halpha_bundles:
        n_out = len(bundle["luminosity_centers"])
        pred = best_mean[offset : offset + n_out]
        pred_std = best_std[offset : offset + n_out]
        offset += n_out
        for bin_index, (x_value, target_value, target_std_value, pred_value, pred_std_value) in enumerate(
            zip(
                bundle["luminosity_centers"],
                bundle["target_log10"],
                np.sqrt(np.maximum(np.diag(bundle["target_covariance_log10"]), 0.0)),
                pred,
                pred_std,
                strict=True,
            )
        ):
            observable_rows.append(
                {
                    "kind": "halpha",
                    "analysis_key": bundle["sobral_label"],
                    "label": f"Halpha {bundle['sobral_label']}",
                    "bin_index": int(bin_index),
                    "x_plot": float(x_value),
                    "target_value": float(target_value),
                    "target_std": float(target_std_value),
                    "prediction_value": float(pred_value),
                    "prediction_std": float(pred_std_value),
                }
            )
    prediction_path = mcmc_root / f"{args.output_prefix}_best_fit_observables.csv"
    pd.DataFrame(observable_rows).to_csv(prediction_path, index=False)

    summary = {
        "campaign_root": str(campaign_root),
        "observable_likelihood_case": args.observable_likelihood_case,
        "observable_emulator_mode": args.observable_emulator_mode,
        "include_emulator_variance": bool(args.include_emulator_variance),
        "extra_observable_bundle": str(args.extra_observable_bundle.resolve()) if args.extra_observable_bundle is not None else None,
        "extra_observable_analysis_key": args.extra_observable_analysis_key if args.extra_observable_bundle is not None else None,
        "extra_observable_label": args.extra_observable_label if args.extra_observable_bundle is not None else None,
        "extra_observable_xmin": args.extra_observable_xmin,
        "extra_observable_xmax": args.extra_observable_xmax,
        "halpha_bundles": {label: str(path) for label, path in halpha_bundle_paths.items()},
        "input_columns": ALL_INPUT_COLUMNS,
        "n_restarts_optimizer": args.n_restarts_optimizer,
        "n_walkers": args.n_walkers,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "seed": args.seed,
        "init_center": args.init_center,
        "init_quantile_sigma": args.init_quantile_sigma,
        "pca_scaling": args.pca_scaling,
        "default_pca_components": args.default_pca_components,
        "pca_component_overrides": pca_component_overrides,
        "thin_disk_maximum_prior_median": args.thin_disk_maximum_prior_median,
        "thin_disk_maximum_prior_sigma_dex": args.thin_disk_maximum_prior_sigma_dex,
        "mean_acceptance_fraction": float(np.mean(acceptance_fraction)),
        "acceptance_fraction_by_walker": [float(value) for value in acceptance_fraction],
        "best_log_probability": float(flat_log_prob[best_index]),
        "best_theta": {name: float(value) for name, value in zip(ALL_INPUT_COLUMNS, best_theta, strict=True)},
    }
    summary_path = mcmc_root / f"{args.output_prefix}_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    emulator_bundle_path = mcmc_root / f"{args.output_prefix}_fitted_emulators.joblib"
    joblib.dump(
        {
            "bundle_type": "joint_observable_halpha_mcmc_fitted_emulators",
            "campaign_root": str(campaign_root),
            "observable_likelihood_case": args.observable_likelihood_case,
            "observable_emulator_mode": args.observable_emulator_mode,
            "include_emulator_variance": bool(args.include_emulator_variance),
            "extra_observable_bundle": str(args.extra_observable_bundle.resolve()) if args.extra_observable_bundle is not None else None,
            "input_columns": ALL_INPUT_COLUMNS,
            "parameter_specs": parameter_specs,
            "observable_bundles": observable_bundles,
            "halpha_bundles": halpha_bundles,
            "best_theta": {name: float(value) for name, value in zip(ALL_INPUT_COLUMNS, best_theta, strict=True)},
        },
        emulator_bundle_path,
    )

    trace_path = figures_root / f"{args.output_prefix}_trace.png"
    trace_tau = _make_trace_plot(
        chain,
        ALL_INPUT_COLUMNS,
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
            pd.DataFrame(posterior_df),
            ALL_INPUT_COLUMNS,
            parameter_specs,
            truths=best_theta,
        )
        corner_fig.savefig(corner_path, dpi=180)
        plt.close(corner_fig)

    best_fit_fig_path = figures_root / f"{args.output_prefix}_best_fit_observables.png"
    best_theta_quantiles = transform_to_prior_quantiles(parameter_specs, best_theta[None, :])[0]
    _plot_joint_best_fit_observables(
        observable_bundles,
        prediction_mean_by_key,
        prediction_std_by_key,
        best_theta_quantiles[:SLOW_DIM],
        None,
        best_fit_fig_path,
    )
    best_fit_halpha_path = figures_root / f"{args.output_prefix}_best_fit_halpha.png"
    _plot_best_fit_halpha(halpha_bundles, best_theta_quantiles, None, best_fit_halpha_path)

    print(posterior_path)
    print(summary_path)
    print(emulator_bundle_path)
    print(trace_path)
    if corner is not None:
        print(corner_path)
    print(best_fit_fig_path)
    print(best_fit_halpha_path)


if __name__ == "__main__":
    main()
