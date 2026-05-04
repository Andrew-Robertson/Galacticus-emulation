from __future__ import annotations

from pathlib import Path
import time
import json

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .gp import fit_scaled_gp, predict_scaled_gp
from .lhs import ParameterSpec, TruncatedLogNormalPrior, transform_to_prior_quantiles
from .persistence import load_emulator_bundle
from .specs import trinity_parameter_specs


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

FIXED_PARAMETER_NAMES = {
    "spheroidExponent",
    "spheroidSFRExponentVelocity",
    "bondiTemperatureSpheroid",
    "barStabilityThresholdGaseous",
    "barStabilityThresholdStellar",
}

DEFAULT_HDF5_FILENAME = "galacticus_reduced.hdf5"

OBSERVABLE_CONFIGS = {
    "smf_z0": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz0",
        "label": "Tomczak+14 SMF z~0",
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": -6.0,
        "y_plot_max": -1.0,
    },
    "smf_z3": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz3",
        "label": "Tomczak+14 SMF z~2",
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": -6.0,
        "y_plot_max": -1.0,
    },
    "mzr_blanc2019": {
        "analysis": "massMetallicityBlanc2019",
        "label": "Blanc+19 MZR",
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": 7.5,
        "y_plot_max": 10.0,
    },
    "sfr_function_robotham2011": {
        "analysis": "starFormationRateFunctionRobotham2011",
        "label": "Robotham+11 SFRF",
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": -6.0,
        "y_plot_max": -1.0,
    },
    "size_mass_vdw2014_star_forming_z0": {
        "analysis": "stellarSizeMassRelationvanDerWel2014Sample1",
        "label": "van der Wel+14 Size SF",
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": -3.5,
        "y_plot_max": -1.5,
    },
    "size_mass_vdw2014_quiescent_z0": {
        "analysis": "stellarSizeMassRelationvanDerWel2014Sample7",
        "label": "van der Wel+14 Size Q",
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": -3.5,
        "y_plot_max": -1.5,
    },
}

DEFAULT_PCA_COMPONENTS = {
    "smf_z0": 5,
    "smf_z3": 4,
    "mzr_blanc2019": 5,
    "sfr_function_robotham2011": 5,
    "size_mass_vdw2014_star_forming_z0": 3,
    "size_mass_vdw2014_quiescent_z0": 4,
}


def _input_parameter_specs() -> list[ParameterSpec]:
    specs_by_name = {
        spec.short_name: spec
        for spec in trinity_parameter_specs()
        if spec.short_name not in FIXED_PARAMETER_NAMES
    }
    return [specs_by_name[name] for name in FREE_PARAMETER_ORDER]


INPUT_COLUMNS = list(FREE_PARAMETER_ORDER)
INPUT_PARAMETER_SPECS = _input_parameter_specs()


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


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _default_input_ranges(samples: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges = {}
    for column in INPUT_COLUMNS:
        min_value = float(samples[column].min())
        max_value = float(samples[column].max())
        default = float(samples[column].median())
        ranges[column] = {
            "min": min_value,
            "max": max_value,
            "default": min(max(default, min_value), max_value),
            "scale": _slider_scale_for_column(column),
        }
    return ranges


def _slider_scale_for_column(column: str) -> str:
    specs_by_name = {spec.short_name: spec for spec in INPUT_PARAMETER_SPECS}
    return "log" if isinstance(specs_by_name[column].prior, TruncatedLogNormalPrior) else "linear"


def _input_ranges_with_scale(input_ranges: dict[str, dict[str, float]]) -> dict[str, dict[str, float | str]]:
    enriched = {}
    for column, values in input_ranges.items():
        enriched[column] = dict(values)
        enriched[column].setdefault("scale", _slider_scale_for_column(column))
    return enriched


def _default_best_fit_params(campaign_root: str | Path) -> dict[str, float] | None:
    campaign_root = Path(campaign_root)
    summary_path = (
        campaign_root
        / "emulator_observable_mcmc_combined"
        / "observable_combined_run_summary.json"
    )
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text())
    columns = data.get("input_columns")
    theta = data.get("best_theta")
    if isinstance(theta, dict):
        return {str(column): float(theta[column]) for column in columns if column in theta}
    if isinstance(columns, list) and isinstance(theta, list) and len(columns) == len(theta):
        return {str(column): float(value) for column, value in zip(columns, theta, strict=True)}
    return None


def _training_preview(y_values: np.ndarray, max_rows: int) -> np.ndarray:
    if y_values.shape[0] <= max_rows:
        return np.asarray(y_values, dtype=float)
    indices = np.asarray(
        sorted({int(round(value)) for value in np.linspace(0, y_values.shape[0] - 1, max_rows)}),
        dtype=int,
    )
    return np.asarray(y_values[indices], dtype=float)


def _resolve_pca_components(
    observable_keys: list[str],
    pca_components: int | dict[str, int] | None,
) -> dict[str, int]:
    if pca_components is None:
        return {
            observable_key: DEFAULT_PCA_COMPONENTS.get(observable_key, 5)
            for observable_key in observable_keys
        }
    if isinstance(pca_components, int):
        return {observable_key: int(pca_components) for observable_key in observable_keys}
    return {
        observable_key: int(pca_components.get(observable_key, DEFAULT_PCA_COMPONENTS.get(observable_key, 5)))
        for observable_key in observable_keys
    }


def _transform_x(values: np.ndarray, *, is_log: bool) -> np.ndarray:
    if is_log:
        if np.any(values <= 0.0):
            raise ValueError("Cannot log10-transform non-positive x values")
        return np.log10(values)
    return np.asarray(values, dtype=float)


def _format_y_axis_label(attrs: dict, *, is_log: bool) -> str:
    base = str(attrs.get("yAxisLabel", "y"))
    if is_log:
        return f"log10({base})"
    return base


def _format_x_axis_label(attrs: dict) -> str:
    return str(attrs.get("xAxisLabel", "x"))


def _transform_y(
    y_linear: np.ndarray,
    y_noise_linear: np.ndarray | None,
    target_linear: np.ndarray,
    target_noise_linear: np.ndarray | None,
    *,
    is_log: bool,
    min_log10_y: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict]:
    if not is_log:
        return (
            np.asarray(y_linear, dtype=float),
            None if y_noise_linear is None else np.asarray(y_noise_linear, dtype=float),
            np.asarray(target_linear, dtype=float),
            None if target_noise_linear is None else np.asarray(target_noise_linear, dtype=float),
            {"y_transform": "identity"},
        )

    effective_floor_linear = 10.0 ** float(min_log10_y)
    y_nonpositive = np.asarray(y_linear <= 0.0, dtype=bool)
    target_nonpositive = np.asarray(target_linear <= 0.0, dtype=bool)

    safe_y_linear = np.where(y_linear > 0.0, y_linear, effective_floor_linear)
    safe_target_linear = np.where(target_linear > 0.0, target_linear, effective_floor_linear)

    y_log10_raw = np.log10(safe_y_linear)
    target_log10_raw = np.log10(safe_target_linear)
    y_plot = np.maximum(y_log10_raw, min_log10_y)
    target_plot = np.maximum(target_log10_raw, min_log10_y)

    if y_noise_linear is None:
        y_noise_plot = None
    else:
        y_noise_plot = np.maximum(y_noise_linear / (safe_y_linear * np.log(10.0)), 1.0e-6)
        y_noise_plot = np.where(y_nonpositive, np.nan, y_noise_plot)

    if target_noise_linear is None:
        target_noise_plot = None
    else:
        target_noise_plot = np.maximum(target_noise_linear / (safe_target_linear * np.log(10.0)), 1.0e-6)
        target_noise_plot = np.where(target_nonpositive, np.nan, target_noise_plot)

    return (
        y_plot,
        y_noise_plot,
        target_plot,
        target_noise_plot,
        {
            "y_transform": "log10",
            "min_log10_y": float(min_log10_y),
            "effective_floor_linear": float(effective_floor_linear),
            "nonpositive_mask": y_nonpositive,
            "target_nonpositive_mask": target_nonpositive,
            "n_nonpositive": int(np.count_nonzero(y_nonpositive)),
        },
    )


def _prepare_training_targets(
    y_values: np.ndarray,
    y_noise: np.ndarray | None,
    *,
    analysis: str,
    use_training_alpha: bool,
    bad_training_condition: str,
    bad_training_value_fill,
    bad_training_sigma: float,
    min_training_sigma: float,
    bad_mask_override: np.ndarray | None = None,
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

    if bad_mask_override is not None:
        bad_mask = np.asarray(bad_mask_override, dtype=bool).copy()
    else:
        bad_mask = ~np.isfinite(y_fit)
    if bad_mask_override is None and bad_training_condition == "nonpositive":
        bad_mask |= (y_fit <= 0.0)
    elif bad_mask_override is None and bad_training_condition == "zero_only":
        bad_mask |= (y_fit == 0.0)
    elif bad_mask_override is None and bad_training_condition == "at_floor":
        bad_mask |= np.isclose(y_fit, np.nanmin(y_fit), atol=1.0e-12)

    metadata["n_bad_training_points"] = int(np.sum(bad_mask))

    if bad_training_value_fill == "bin_median":
        finite_values = y_fit[np.isfinite(y_fit)]
        fallback_value = 8.5 if analysis == "massMetallicityBlanc2019" else float(np.nanmedian(finite_values))
        for bin_index in range(y_fit.shape[1]):
            valid = np.isfinite(y_fit[:, bin_index]) & ~bad_mask[:, bin_index]
            replacement = float(np.nanmedian(y_fit[valid, bin_index])) if np.any(valid) else fallback_value
            y_fit[bad_mask[:, bin_index], bin_index] = replacement
    else:
        y_fit[bad_mask] = float(bad_training_value_fill)

    sigma = np.where(bad_mask, bad_training_sigma, sigma)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, min_training_sigma)
    return y_fit, sigma, metadata


def _infer_analysis_attributes(campaign_root: Path, analysis: str, hdf5_filename: str) -> dict:
    samples = pd.read_csv(campaign_root / "samples.csv")
    first_evaluation = samples.iloc[0]["evaluation_id"]
    path = campaign_root / "evaluations" / first_evaluation / hdf5_filename
    with h5py.File(path, "r") as handle:
        group = handle[f"/analyses/{analysis}"]
        return {key: _decode_attr(value) for key, value in group.attrs.items()}


def _load_observable_campaign(
    campaign_root: Path,
    *,
    analysis: str,
    hdf5_filename: str,
    min_log10_y: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict, dict]:
    samples = pd.read_csv(campaign_root / "samples.csv")
    x_raw = samples[INPUT_COLUMNS].to_numpy(dtype=float)
    x_quantile = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)

    attrs = _infer_analysis_attributes(campaign_root, analysis, hdf5_filename)
    x_dataset = attrs["xDataset"]
    y_dataset = attrs["yDataset"]
    target_dataset = attrs["yDatasetTarget"]
    covariance_dataset = attrs.get("yCovariance")
    target_covariance_dataset = attrs.get("yCovarianceTarget")
    x_is_log = bool(attrs.get("xAxisIsLog", False))
    y_is_log = bool(attrs.get("yAxisIsLog", False))

    x_bins = None
    target_linear = None
    target_noise_linear = None
    y_rows = []
    y_noise_rows = []
    for sample in samples.itertuples(index=False):
        evaluation_id = sample.evaluation_id
        path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            current_x = np.asarray(group[x_dataset][...], dtype=float)
            current_y = np.asarray(group[y_dataset][...], dtype=float)
            current_target = np.asarray(group[target_dataset][...], dtype=float)
            if covariance_dataset and covariance_dataset in group:
                current_covariance = np.asarray(group[covariance_dataset][...], dtype=float)
                current_noise = np.sqrt(np.maximum(np.diag(current_covariance), 0.0))
            else:
                current_noise = None
            if target_covariance_dataset and target_covariance_dataset in group:
                current_target_covariance = np.asarray(group[target_covariance_dataset][...], dtype=float)
                current_target_noise = np.sqrt(np.maximum(np.diag(current_target_covariance), 0.0))
            else:
                current_target_noise = None

        if x_bins is None:
            x_bins = current_x
            target_linear = current_target
            target_noise_linear = current_target_noise
        else:
            if not np.allclose(current_x, x_bins, equal_nan=True):
                raise ValueError(f"x bins differ for {evaluation_id}")
            if not np.allclose(current_target, target_linear, equal_nan=True):
                raise ValueError(f"target differs for {evaluation_id}")
        y_rows.append(current_y)
        if current_noise is not None:
            y_noise_rows.append(current_noise)

    assert x_bins is not None
    assert target_linear is not None
    y_linear = np.vstack(y_rows)
    y_noise_linear = np.vstack(y_noise_rows) if y_noise_rows else None

    x_plot = _transform_x(x_bins, is_log=x_is_log)
    y_plot, y_noise_plot, target_plot, target_noise_plot, transform_metadata = _transform_y(
        y_linear,
        y_noise_linear,
        target_linear,
        target_noise_linear,
        is_log=y_is_log,
        min_log10_y=min_log10_y,
    )
    return (
        samples,
        x_quantile,
        x_plot,
        target_plot,
        target_noise_plot,
        y_plot,
        y_noise_plot,
        attrs,
        transform_metadata,
    )


def train_observables_bundle(
    campaign_root: str | Path,
    *,
    hdf5_filename: str = DEFAULT_HDF5_FILENAME,
    observable_keys: list[str] | None = None,
    n_restarts_optimizer: int = 0,
    optimize_hyperparameters: bool = True,
    min_log10_y: float = -7.0,
    training_preview_rows: int = 64,
    emulator_mode: str = "bin_by_bin",
    pca_components: int | dict[str, int] | None = None,
    pca_scaling: str = "standardized",
) -> dict:
    campaign_root = Path(campaign_root)
    samples = pd.read_csv(campaign_root / "samples.csv")
    x_raw = samples[INPUT_COLUMNS].to_numpy(dtype=float)
    x_quantile = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)
    input_ranges = _default_input_ranges(samples)
    observable_keys = observable_keys or list(OBSERVABLE_CONFIGS)
    pca_components_by_observable = _resolve_pca_components(observable_keys, pca_components)

    observables = {}
    for observable_key in observable_keys:
        config = OBSERVABLE_CONFIGS[observable_key]
        analysis = config["analysis"]
        print(f"Training interactive observable bundle for {observable_key} ({analysis})", flush=True)
        (
            _samples,
            loaded_x_quantile,
            x_plot,
            target_plot,
            target_noise_plot,
            y_plot,
            y_noise_plot,
            attrs,
            transform_metadata,
        ) = _load_observable_campaign(
            campaign_root,
            analysis=analysis,
            hdf5_filename=hdf5_filename,
            min_log10_y=min_log10_y,
        )
        if not np.allclose(loaded_x_quantile, x_quantile):
            raise ValueError(f"Input quantiles differ unexpectedly for {observable_key}")

        y_fit, alpha_train, training_metadata = _prepare_training_targets(
            y_plot,
            y_noise_plot,
            analysis=analysis,
            use_training_alpha=bool(config.get("use_training_alpha", True)),
            bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
            bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
            bad_training_sigma=float(config.get("bad_training_sigma", 5.0)),
            min_training_sigma=float(config.get("min_training_sigma", 1.0e-3)),
            bad_mask_override=transform_metadata.get("nonpositive_mask")
            if str(config.get("bad_training_condition", "")) == "nonpositive"
            and isinstance(transform_metadata.get("nonpositive_mask"), np.ndarray)
            else None,
        )

        observable_bundle = {
            "observable_key": observable_key,
            "analysis": analysis,
            "label": str(config["label"]),
            "x_plot": np.asarray(x_plot, dtype=float),
            "target_plot": np.asarray(target_plot, dtype=float),
            "target_sigma_plot": None if target_noise_plot is None else np.asarray(target_noise_plot, dtype=float),
            "y_train_preview": _training_preview(y_plot, training_preview_rows),
            "x_axis_label": _format_x_axis_label(attrs),
            "y_axis_label": _format_y_axis_label(attrs, is_log=transform_metadata["y_transform"] == "log10"),
            "y_plot_min": float(config.get("y_plot_min", np.nanmin(y_plot))),
            "y_plot_max": float(config.get("y_plot_max", np.nanmax(y_plot))),
            "attrs": attrs,
            "transform_metadata": {
                key: value
                for key, value in transform_metadata.items()
                if not isinstance(value, np.ndarray)
            },
            "training_metadata": training_metadata,
            "n_bins": int(y_fit.shape[1]),
        }
        if emulator_mode == "bin_by_bin":
            models = []
            y_means = []
            y_stds = []
            kernels = []
            for bin_index in range(y_fit.shape[1]):
                print(
                    f"  {observable_key}: fitting bin {bin_index + 1}/{y_fit.shape[1]} "
                    f"(x={x_plot[bin_index]:.3f})",
                    flush=True,
                )
                model, y_mean, y_std = fit_scaled_gp(
                    x_quantile,
                    y_fit[:, bin_index],
                    n_restarts_optimizer=n_restarts_optimizer,
                    optimize_hyperparameters=optimize_hyperparameters,
                    alpha=alpha_train[:, bin_index] if alpha_train is not None else None,
                )
                models.append(model)
                y_means.append(y_mean)
                y_stds.append(y_std)
                kernels.append(str(model.kernel_))
            observable_bundle.update(
                {
                    "emulator_mode": "bin_by_bin",
                    "models": models,
                    "y_means": np.asarray(y_means, dtype=float),
                    "y_stds": np.asarray(y_stds, dtype=float),
                    "kernels": kernels,
                }
            )
        elif emulator_mode == "pca":
            scaler = _make_preprocessor(pca_scaling)
            y_fit_scaled = scaler.fit_transform(y_fit)
            n_components_requested = pca_components_by_observable[observable_key]
            n_components_actual = min(n_components_requested, y_fit_scaled.shape[0], y_fit_scaled.shape[1])
            pca = PCA(n_components=n_components_actual)
            coefficients = pca.fit_transform(y_fit_scaled)

            coefficient_alpha = None
            if alpha_train is not None:
                alpha_scaled = np.asarray(alpha_train, dtype=float) / (scaler.scale_[None, :] ** 2)
                coefficient_alpha = alpha_scaled @ (pca.components_.T ** 2)
                coefficient_alpha = np.maximum(coefficient_alpha, 1.0e-12)

            models = []
            y_means = []
            y_stds = []
            kernels = []
            for component_index in range(n_components_actual):
                print(
                    f"  {observable_key}: fitting PCA component {component_index + 1}/{n_components_actual}",
                    flush=True,
                )
                model, y_mean, y_std = fit_scaled_gp(
                    x_quantile,
                    coefficients[:, component_index],
                    n_restarts_optimizer=n_restarts_optimizer,
                    optimize_hyperparameters=optimize_hyperparameters,
                    alpha=coefficient_alpha[:, component_index] if coefficient_alpha is not None else None,
                )
                models.append(model)
                y_means.append(y_mean)
                y_stds.append(y_std)
                kernels.append(str(model.kernel_))
            observable_bundle.update(
                {
                    "emulator_mode": "pca",
                    "models": models,
                    "y_means": np.asarray(y_means, dtype=float),
                    "y_stds": np.asarray(y_stds, dtype=float),
                    "kernels": kernels,
                    "pca_scaling": pca_scaling,
                    "pca_components_requested": int(n_components_requested),
                    "pca_components_actual": int(n_components_actual),
                    "pca_mean": np.asarray(pca.mean_, dtype=float),
                    "pca_components_matrix": np.asarray(pca.components_, dtype=float),
                    "pca_explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=float),
                    "preprocessor_mean": np.asarray(scaler.mean_, dtype=float),
                    "preprocessor_scale": np.asarray(scaler.scale_, dtype=float),
                }
            )
        else:
            raise ValueError(f"Unknown emulator_mode={emulator_mode!r}")
        observables[observable_key] = observable_bundle

    return {
        "bundle_type": (
            "interactive_observables_pca_gp"
            if emulator_mode == "pca"
            else "interactive_observables_bin_by_bin_gp"
        ),
        "campaign_root": str(campaign_root.resolve()),
        "best_fit_params": _default_best_fit_params(campaign_root),
        "input_columns": list(INPUT_COLUMNS),
        "input_ranges": input_ranges,
        "evaluation_ids": samples["evaluation_id"].tolist(),
        "x_train_quantile": x_quantile,
        "observable_keys": list(observable_keys),
        "observables": observables,
        "n_restarts_optimizer": int(n_restarts_optimizer),
        "optimize_hyperparameters": bool(optimize_hyperparameters),
        "training_preview_rows": int(training_preview_rows),
        "min_log10_y": float(min_log10_y),
        "emulator_mode": emulator_mode,
        "pca_scaling": pca_scaling if emulator_mode == "pca" else None,
    }


def predict_observables_bundle(bundle: dict, params: dict[str, float]) -> dict[str, dict[str, np.ndarray]]:
    x_raw = np.asarray([[float(params[column]) for column in bundle["input_columns"]]], dtype=float)
    x_quantile = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)
    predictions = {}
    for observable_key in bundle["observable_keys"]:
        observable = bundle["observables"][observable_key]
        if observable.get("emulator_mode", bundle.get("emulator_mode", "bin_by_bin")) == "bin_by_bin":
            y_pred = np.zeros(observable["n_bins"], dtype=float)
            y_std = np.zeros_like(y_pred)
            for index, model in enumerate(observable["models"]):
                pred, pred_std = predict_scaled_gp(
                    model,
                    float(observable["y_means"][index]),
                    float(observable["y_stds"][index]),
                    x_quantile,
                )
                y_pred[index] = float(pred[0])
                y_std[index] = float(pred_std[0])
        else:
            n_components = len(observable["models"])
            coefficient_predictions = np.zeros(n_components, dtype=float)
            coefficient_stds = np.zeros(n_components, dtype=float)
            for index, model in enumerate(observable["models"]):
                pred, pred_std = predict_scaled_gp(
                    model,
                    float(observable["y_means"][index]),
                    float(observable["y_stds"][index]),
                    x_quantile,
                )
                coefficient_predictions[index] = float(pred[0])
                coefficient_stds[index] = float(pred_std[0])
            components = np.asarray(observable["pca_components_matrix"], dtype=float)
            pca_mean = np.asarray(observable["pca_mean"], dtype=float)
            preprocessor_mean = np.asarray(observable["preprocessor_mean"], dtype=float)
            preprocessor_scale = np.asarray(observable["preprocessor_scale"], dtype=float)
            y_pred_scaled = coefficient_predictions @ components + pca_mean
            y_pred = y_pred_scaled * preprocessor_scale + preprocessor_mean
            component_variance = coefficient_stds ** 2
            y_var_scaled = component_variance @ (components ** 2)
            y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * preprocessor_scale
        predictions[observable_key] = {
            "x_plot": np.asarray(observable["x_plot"], dtype=float),
            "y_pred_plot": y_pred,
            "y_std_plot": y_std,
        }
    return predictions


def bundle_meta(bundle: dict) -> dict:
    best_fit_params = bundle.get("best_fit_params")
    if best_fit_params is None:
        best_fit_params = _default_best_fit_params(bundle["campaign_root"])
    input_ranges = _input_ranges_with_scale(bundle["input_ranges"])
    return {
        "bundle_type": bundle["bundle_type"],
        "emulator_mode": bundle.get("emulator_mode", "bin_by_bin"),
        "input_columns": bundle["input_columns"],
        "input_ranges": input_ranges,
        "observable_keys": bundle["observable_keys"],
        "best_fit_params": best_fit_params,
        "observables": {
            observable_key: {
                "analysis": observable["analysis"],
                "label": observable["label"],
                "x_plot": np.asarray(observable["x_plot"], dtype=float).tolist(),
                "target_plot": np.asarray(observable["target_plot"], dtype=float).tolist(),
                "target_sigma_plot": None
                if observable["target_sigma_plot"] is None
                else np.asarray(observable["target_sigma_plot"], dtype=float).tolist(),
                "y_train_preview": np.asarray(observable["y_train_preview"], dtype=float).tolist(),
                "x_axis_label": observable["x_axis_label"],
                "y_axis_label": observable["y_axis_label"],
                "y_plot_min": float(observable["y_plot_min"]),
                "y_plot_max": float(observable["y_plot_max"]),
                "training_metadata": observable["training_metadata"],
                "emulator_mode": observable.get("emulator_mode", bundle.get("emulator_mode", "bin_by_bin")),
                "pca_components_actual": observable.get("pca_components_actual"),
                "pca_explained_variance_ratio_sum": None
                if observable.get("pca_explained_variance_ratio") is None
                else float(np.sum(np.asarray(observable["pca_explained_variance_ratio"], dtype=float))),
            }
            for observable_key, observable in bundle["observables"].items()
        },
        "prediction_benchmark": bundle.get("prediction_benchmark"),
    }


def benchmark_observables_bundle(bundle: dict, *, n_predictions: int = 50) -> dict[str, float]:
    defaults = {
        column: float(bundle["input_ranges"][column]["default"])
        for column in bundle["input_columns"]
    }
    start = time.perf_counter()
    for _ in range(n_predictions):
        predict_observables_bundle(bundle, defaults)
    elapsed = time.perf_counter() - start
    return {
        "n_predictions": int(n_predictions),
        "total_seconds": float(elapsed),
        "milliseconds_per_prediction": float(1000.0 * elapsed / max(n_predictions, 1)),
    }


def load_observables_bundle(path: str | Path) -> dict:
    bundle = load_emulator_bundle(path)
    if bundle.get("bundle_type") not in {
        "interactive_observables_bin_by_bin_gp",
        "interactive_observables_pca_gp",
    }:
        raise ValueError(f"Unexpected bundle_type: {bundle.get('bundle_type')}")
    return bundle
