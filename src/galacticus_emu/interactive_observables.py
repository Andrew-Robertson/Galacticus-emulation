from __future__ import annotations

from pathlib import Path
import time
import json
from collections.abc import Mapping
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .gp import fit_scaled_gp, predict_scaled_gp
from .lhs import ParameterSpec, TruncatedLogNormalPrior, transform_to_prior_quantiles
from .observable_plot_metadata import (
    STANDARD_OBSERVABLE_PLOT_METADATA,
    apply_observable_plot_metadata_overrides,
    apply_y_display_offset,
)
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
    "stellarPopulationMetalYield",
]

FIXED_PARAMETER_NAMES = {
    "spheroidExponent",
    "spheroidSFRExponentVelocity",
    "bondiTemperatureSpheroid",
    "barStabilityThresholdGaseous",
    "barStabilityThresholdStellar",
}

DEFAULT_HDF5_FILENAME = "galacticus_reduced.hdf5"

DEFAULT_OBSERVABLE_CONFIG = {
    "use_training_alpha": True,
    "bad_training_condition": "nonfinite",
    "bad_training_value_fill": "bin_median",
    "bad_training_sigma": 5.0,
    "min_training_sigma": 1.0e-3,
}

OBSERVABLE_CONFIGS = {
    "bh_halo_mass_trinity_z1": {
        "analysis": "blackHoleHaloMassRelationTRINITYz1",
        "label": "TRINITY z~1 BH-halo mass",
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "drop_unsupported_bins": True,
        "y_plot_min": 5.0,
        "y_plot_max": 9.5,
    },
    "bh_velocity_dispersion": {
        "analysis": "blackHoleVelocityDispersionRelation",
        "label": "BH mass-velocity dispersion",
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "drop_unsupported_bins": True,
        "y_plot_min": 5.0,
        "y_plot_max": 10.0,
    },
    "smf_liwhite2009_sdss": {
        "analysis": "massFunctionStellarLiWhite2009SDSS",
        "label": "Li & White 2009 SMF",
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "keep",
        "bad_training_sigma": 0.5,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": -7.0,
        "y_plot_max": -1.0,
    },
    "smf_z0": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz0",
        "label": STANDARD_OBSERVABLE_PLOT_METADATA["smf_z0"]["label"],
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "keep",
        "bad_training_sigma": 0.5,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": -6.0,
        "y_plot_max": -1.0,
    },
    "smf_z3": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz3",
        "label": STANDARD_OBSERVABLE_PLOT_METADATA["smf_z3"]["label"],
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "keep",
        "bad_training_sigma": 0.5,
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
    "morphological_fraction_gama_moffett2016": {
        "analysis": "morphologicalFractionGAMAMoffett2016",
        "label": STANDARD_OBSERVABLE_PLOT_METADATA["morphological_fraction_gama_moffett2016"]["label"],
        "use_training_alpha": True,
        "bad_training_condition": "zero_and_zero_noise",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "y_plot_min": 0.0,
        "y_plot_max": 1.0,
    },
    "sfr_function_robotham2011": {
        "analysis": "starFormationRateFunctionRobotham2011",
        "label": "Robotham+11 SFRF",
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "keep",
        "bad_training_sigma": 0.5,
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
    "bh_halo_mass_trinity_z1": 4,
    "bh_velocity_dispersion": 4,
    "smf_liwhite2009_sdss": 8,
    "smf_z0": 5,
    "smf_z3": 4,
    "mzr_blanc2019": 5,
    "morphological_fraction_gama_moffett2016": 11,
    "sfr_function_robotham2011": 5,
    "size_mass_vdw2014_star_forming_z0": 3,
    "size_mass_vdw2014_quiescent_z0": 4,
}


def _parameter_specs_by_name() -> dict[str, ParameterSpec]:
    return {
        spec.short_name: spec
        for spec in trinity_parameter_specs()
        if spec.short_name not in FIXED_PARAMETER_NAMES
    }


def _parameter_specs_for_columns(columns: list[str]) -> list[ParameterSpec]:
    specs_by_name = _parameter_specs_by_name()
    missing = [column for column in columns if column not in specs_by_name]
    if missing:
        raise ValueError(f"No parameter specification found for input column(s): {missing}")
    return [specs_by_name[column] for column in columns]


def _input_parameter_specs() -> list[ParameterSpec]:
    specs_by_name = {
        spec.short_name: spec
        for spec in trinity_parameter_specs()
        if spec.short_name not in FIXED_PARAMETER_NAMES
    }
    return [specs_by_name[name] for name in FREE_PARAMETER_ORDER if name in specs_by_name]


INPUT_COLUMNS = list(FREE_PARAMETER_ORDER)
INPUT_PARAMETER_SPECS = _input_parameter_specs()


def _normalize_observable_config(key: str, config: Mapping) -> dict:
    normalized = dict(DEFAULT_OBSERVABLE_CONFIG)
    normalized.update(dict(config))
    if not normalized.get("analysis"):
        raise ValueError(f"Observable config {key!r} must define an 'analysis' value")
    normalized.setdefault("label", key)
    return normalized


def _observable_configs(extra_configs: Mapping[str, Mapping] | None = None) -> dict[str, dict]:
    configs = {
        key: _normalize_observable_config(key, config)
        for key, config in OBSERVABLE_CONFIGS.items()
    }
    if extra_configs is not None:
        for key, config in extra_configs.items():
            configs[str(key)] = _normalize_observable_config(str(key), config)
    return configs


def _input_columns_for_samples(
    samples: pd.DataFrame,
    input_columns: list[str] | None = None,
) -> list[str]:
    if input_columns is not None:
        missing = [
            column
            for column in input_columns
            if column not in samples.columns and f"{column}_quantile" not in samples.columns
        ]
        if missing:
            raise ValueError(f"Requested input column(s) missing from samples.csv: {missing}")
        return list(input_columns)

    quantile_columns = [
        column.removesuffix("_quantile")
        for column in samples.columns
        if column.endswith("_quantile")
    ]
    if quantile_columns:
        ordered_columns = [column for column in FREE_PARAMETER_ORDER if column in quantile_columns]
        ordered_columns.extend(column for column in quantile_columns if column not in ordered_columns)
        return ordered_columns

    ordered_columns = [column for column in FREE_PARAMETER_ORDER if column in samples.columns]
    if not ordered_columns:
        raise ValueError("Could not infer emulator input columns from samples.csv")
    return ordered_columns


def _input_quantile_matrix(
    samples: pd.DataFrame,
    input_columns: list[str],
    input_parameter_specs: list[ParameterSpec],
) -> np.ndarray:
    quantile_columns = [f"{column}_quantile" for column in input_columns]
    if all(column in samples.columns for column in quantile_columns):
        return samples[quantile_columns].to_numpy(dtype=float)
    x_raw = samples[input_columns].to_numpy(dtype=float)
    return transform_to_prior_quantiles(input_parameter_specs, x_raw)


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


def _target_noise_from_analysis_group(group: h5py.Group, attrs: Mapping) -> np.ndarray | None:
    target_covariance_dataset = attrs.get("yCovarianceTarget")
    if target_covariance_dataset and target_covariance_dataset in group:
        target_covariance = np.asarray(group[target_covariance_dataset][...], dtype=float)
        return np.sqrt(np.maximum(np.diag(target_covariance), 0.0))

    lower_dataset = attrs.get("yErrorLowerTarget")
    upper_dataset = attrs.get("yErrorUpperTarget")
    if lower_dataset and upper_dataset and lower_dataset in group and upper_dataset in group:
        lower = np.abs(np.asarray(group[lower_dataset][...], dtype=float))
        upper = np.abs(np.asarray(group[upper_dataset][...], dtype=float))
        return np.maximum(lower, upper)
    if lower_dataset and lower_dataset in group:
        return np.abs(np.asarray(group[lower_dataset][...], dtype=float))
    if upper_dataset and upper_dataset in group:
        return np.abs(np.asarray(group[upper_dataset][...], dtype=float))
    return None


def _select_bins(values: np.ndarray, supported_bin_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    mask = np.asarray(supported_bin_mask, dtype=bool)
    if array.ndim == 2 and array.shape[0] == 2 and array.shape[1] == mask.size:
        return array[:, mask]
    return array[mask]


def _default_input_ranges(
    samples: pd.DataFrame,
    input_columns: list[str],
    input_parameter_specs: list[ParameterSpec],
) -> dict[str, dict[str, float]]:
    ranges = {}
    for column in input_columns:
        if column in samples.columns:
            values = samples[column].to_numpy(dtype=float)
        elif f"{column}_quantile" in samples.columns:
            values = samples[f"{column}_quantile"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Input column {column!r} is missing from samples.csv")
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        default = float(np.nanmedian(values))
        ranges[column] = {
            "min": min_value,
            "max": max_value,
            "default": min(max(default, min_value), max_value),
            "scale": _slider_scale_for_column(column, input_parameter_specs),
        }
    return ranges


def _slider_scale_for_column(
    column: str,
    input_parameter_specs: list[ParameterSpec] | None = None,
) -> str:
    if input_parameter_specs is None:
        input_parameter_specs = _parameter_specs_for_columns([column])
    specs_by_name = {spec.short_name: spec for spec in input_parameter_specs}
    return "log" if isinstance(specs_by_name[column].prior, TruncatedLogNormalPrior) else "linear"


def _input_ranges_with_scale(input_ranges: dict[str, dict[str, float]]) -> dict[str, dict[str, float | str]]:
    enriched = {}
    for column, values in input_ranges.items():
        enriched[column] = dict(values)
        enriched[column].setdefault("scale", _slider_scale_for_column(column))
    return enriched


def _campaign_design_reference_defaults(
    campaign_root: str | Path,
    input_columns: list[str],
    input_ranges: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, float] | None, str | None]:
    """Read campaign reference values for the sliders from campaign_design.json.

    For the current campaign design these are the prior x0 values. Uniform
    priors do not carry an x0, so use their midpoint as the least surprising
    slider reset value.
    """
    design_path = Path(campaign_root) / "campaign_design.json"
    if not design_path.exists():
        return None, None

    data = json.loads(design_path.read_text())
    by_short_name = {
        str(parameter["short_name"]): parameter
        for parameter in data.get("free_parameters", [])
        if "short_name" in parameter
    }
    defaults: dict[str, float] = {}
    used_midpoint_fallback = False
    used_range_fallback = False

    for column in input_columns:
        parameter = by_short_name.get(column)
        prior_parameters = {}
        if parameter is not None:
            prior = parameter.get("prior") or {}
            prior_parameters = prior.get("parameters") or {}

        if "x0" in prior_parameters:
            value = float(prior_parameters["x0"])
        elif "lower" in prior_parameters and "upper" in prior_parameters:
            value = 0.5 * (float(prior_parameters["lower"]) + float(prior_parameters["upper"]))
            used_midpoint_fallback = True
        elif input_ranges is not None and column in input_ranges:
            value = float(input_ranges[column]["default"])
            used_range_fallback = True
        else:
            return None, None

        if input_ranges is not None and column in input_ranges:
            lower = float(input_ranges[column]["min"])
            upper = float(input_ranges[column]["max"])
            value = min(max(value, lower), upper)
        defaults[column] = value

    source = "campaign_design.json prior x0"
    if used_midpoint_fallback:
        source += "; prior midpoint for parameters without x0"
    if used_range_fallback:
        source += "; training median where campaign defaults were unavailable"
    return defaults, source


def _campaign_design_free_parameters(campaign_root: str | Path) -> dict[str, Mapping]:
    design_path = Path(campaign_root) / "campaign_design.json"
    if not design_path.exists():
        return {}
    data = json.loads(design_path.read_text())
    return {
        str(parameter["short_name"]): parameter
        for parameter in data.get("free_parameters", [])
        if "short_name" in parameter
    }


def _default_parameter_xml_candidates(campaign_root: str | Path) -> list[Path]:
    campaign_root = Path(campaign_root)
    return [
        campaign_root
        / "MCMCs"
        / "all_standard_observables"
        / "getDefaultParams"
        / "defaultParams.xml",
        campaign_root
        / "emulator_interactive_observables_mcmc"
        / "all_standard_observables"
        / "getDefaultParams"
        / "defaultParams.xml",
        campaign_root / "getDefaultParams" / "defaultParams.xml",
        campaign_root / "defaultParams.xml",
    ]


def _params_from_default_parameter_xml(
    campaign_root: str | Path,
    input_columns: list[str],
    input_ranges: dict[str, dict[str, float]] | None = None,
    *,
    default_params_path: str | Path | None = None,
) -> tuple[dict[str, float] | None, str | None, str | None]:
    candidates = []
    if default_params_path is not None:
        candidates.append(Path(default_params_path).expanduser())
    candidates.extend(_default_parameter_xml_candidates(campaign_root))
    xml_path = next((path for path in candidates if path.exists()), None)
    if xml_path is None:
        return None, None, None

    free_parameters = _campaign_design_free_parameters(campaign_root)
    if not free_parameters:
        return None, None, None

    root = ET.parse(xml_path).getroot()
    defaults: dict[str, float] = {}
    for column in input_columns:
        parameter = free_parameters.get(column)
        if parameter is None or "path" not in parameter:
            return None, None, None
        element = root.find(str(parameter["path"]))
        if element is None or "value" not in element.attrib:
            return None, None, None
        value = float(element.attrib["value"])
        if input_ranges is not None and column in input_ranges:
            lower = float(input_ranges[column]["min"])
            upper = float(input_ranges[column]["max"])
            value = min(max(value, lower), upper)
        defaults[column] = value

    source = f"{xml_path.name} parameter tree via campaign_design.json paths"
    return defaults, source, str(xml_path)


def _galacticus_default_params(
    campaign_root: str | Path,
    input_columns: list[str],
    input_ranges: dict[str, dict[str, float]] | None = None,
    *,
    default_params_path: str | Path | None = None,
) -> tuple[dict[str, float] | None, str | None, str | None]:
    defaults, source, source_path = _params_from_default_parameter_xml(
        campaign_root,
        input_columns,
        input_ranges=input_ranges,
        default_params_path=default_params_path,
    )
    if defaults is not None:
        return defaults, source, source_path

    defaults, source = _campaign_design_reference_defaults(
        campaign_root,
        input_columns,
        input_ranges=input_ranges,
    )
    return defaults, source, None


def _best_fit_params_from_summary(
    summary_path: str | Path,
    input_columns: list[str] | None = None,
) -> dict[str, float] | None:
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text())
    columns = input_columns or data.get("input_columns")
    theta = data.get("best_theta")
    if isinstance(theta, dict):
        if columns is None:
            return {str(column): float(value) for column, value in theta.items()}
        return {str(column): float(theta[column]) for column in columns if column in theta}
    if isinstance(columns, list) and isinstance(theta, list) and len(columns) == len(theta):
        return {str(column): float(value) for column, value in zip(columns, theta, strict=True)}
    return None


def _default_best_fit_params(
    campaign_root: str | Path,
    input_columns: list[str] | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, float] | None:
    campaign_root = Path(campaign_root)
    candidate_paths = []
    if summary_path is not None:
        candidate_paths.append(Path(summary_path).expanduser())
    candidate_paths.extend([
        campaign_root
        / "MCMCs"
        / "all_standard_observables"
        / "mcmc_all_standard_observables_run_summary.json",
        campaign_root
        / "emulator_interactive_observables_mcmc"
        / "all_standard_observables"
        / "mcmc_all_standard_observables_run_summary.json",
        campaign_root
        / "emulator_observable_mcmc_combined"
        / "observable_combined_run_summary.json",
    ])
    for candidate_path in candidate_paths:
        params = _best_fit_params_from_summary(candidate_path, input_columns=input_columns)
        if params is not None:
            return params
    return None


def _training_preview(y_values: np.ndarray, max_rows: int) -> np.ndarray:
    if y_values.shape[0] <= max_rows:
        return np.asarray(y_values, dtype=float)
    indices = np.asarray(
        sorted({int(round(value)) for value in np.linspace(0, y_values.shape[0] - 1, max_rows)}),
        dtype=int,
    )
    return np.asarray(y_values[indices], dtype=float)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _training_preview_row_count(
    rows: int | str | None,
    *,
    n_total: int,
    fallback: int,
) -> int:
    if rows is None:
        return int(fallback)
    if isinstance(rows, str):
        value = rows.strip().lower()
        if value in {"all", "*"}:
            return int(n_total)
        return int(value)
    return int(rows)


def _infer_campaign_hdf5_filename(campaign_root: Path, evaluation_ids: list[str]) -> str:
    candidates = ["galacticus.hdf5", DEFAULT_HDF5_FILENAME]
    first_evaluation = evaluation_ids[0] if evaluation_ids else None
    if first_evaluation is None:
        return DEFAULT_HDF5_FILENAME
    evaluation_dir = campaign_root / "evaluations" / first_evaluation
    for candidate in candidates:
        if (evaluation_dir / candidate).exists():
            return candidate
    return DEFAULT_HDF5_FILENAME


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
    y_floor_mask = np.asarray(y_log10_raw <= min_log10_y, dtype=bool)
    target_floor_mask = np.asarray(target_log10_raw <= min_log10_y, dtype=bool)

    if y_noise_linear is None:
        y_noise_plot = None
    else:
        y_noise_plot = np.maximum(y_noise_linear / (safe_y_linear * np.log(10.0)), 1.0e-6)
        y_noise_plot = np.where(y_floor_mask, np.nan, y_noise_plot)

    if target_noise_linear is None:
        target_noise_plot = None
    else:
        target_noise_plot = np.maximum(target_noise_linear / (safe_target_linear * np.log(10.0)), 1.0e-6)
        target_noise_plot = np.where(target_floor_mask, np.nan, target_noise_plot)

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
            "floor_mask": y_floor_mask,
            "target_floor_mask": target_floor_mask,
            "n_nonpositive": int(np.count_nonzero(y_nonpositive)),
            "n_at_floor": int(np.count_nonzero(y_floor_mask)),
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
        bad_mask = _bad_training_mask(
            y_fit,
            y_noise=sigma if y_noise is not None else None,
            bad_training_condition=bad_training_condition,
        )

    metadata["n_bad_training_points"] = int(np.sum(bad_mask))

    if bad_training_value_fill in {"keep", "as_is"}:
        pass
    elif bad_training_value_fill == "bin_median":
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


def _bad_training_mask(
    y_values: np.ndarray,
    y_noise: np.ndarray | None = None,
    *,
    bad_training_condition: str,
) -> np.ndarray:
    bad_mask = ~np.isfinite(y_values)
    if bad_training_condition == "nonpositive":
        bad_mask |= y_values <= 0.0
    elif bad_training_condition == "zero_only":
        bad_mask |= y_values == 0.0
    elif bad_training_condition == "zero_and_zero_noise" and y_noise is not None:
        noise = np.asarray(y_noise, dtype=float)
        bad_mask |= (y_values == 0.0) & (~np.isfinite(noise) | (noise <= 0.0))
    elif bad_training_condition == "at_floor":
        bad_mask |= np.isclose(y_values, np.nanmin(y_values), atol=1.0e-12)
    return bad_mask


def _supported_bin_mask(
    y_values: np.ndarray,
    y_noise: np.ndarray | None = None,
    *,
    bad_training_condition: str,
) -> np.ndarray:
    bad_mask = _bad_training_mask(
        y_values,
        y_noise=y_noise,
        bad_training_condition=bad_training_condition,
    )
    return np.any(~bad_mask, axis=0)


def training_bad_mask_override_from_transform(
    transform_metadata: Mapping,
    *,
    bad_training_condition: str,
    bad_training_value_fill,
) -> np.ndarray | None:
    if bad_training_condition != "nonpositive":
        return None
    if str(bad_training_value_fill) in {"keep", "as_is"}:
        mask = transform_metadata.get("floor_mask")
    else:
        mask = transform_metadata.get("nonpositive_mask")
    if isinstance(mask, np.ndarray):
        return np.asarray(mask, dtype=bool)
    return None


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
    input_columns: list[str],
    input_parameter_specs: list[ParameterSpec],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict, dict]:
    samples = pd.read_csv(campaign_root / "samples.csv")
    x_quantile = _input_quantile_matrix(samples, input_columns, input_parameter_specs)

    attrs = _infer_analysis_attributes(campaign_root, analysis, hdf5_filename)
    x_dataset = attrs["xDataset"]
    y_dataset = attrs["yDataset"]
    target_dataset = attrs["yDatasetTarget"]
    covariance_dataset = attrs.get("yCovariance")
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
            current_target_noise = _target_noise_from_analysis_group(group, attrs)

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
    pca_variance_threshold: float | None = None,
    observable_configs: Mapping[str, Mapping] | None = None,
    input_columns: list[str] | None = None,
) -> dict:
    campaign_root = Path(campaign_root)
    samples = pd.read_csv(campaign_root / "samples.csv")
    configs = _observable_configs(observable_configs)
    input_columns = _input_columns_for_samples(samples, input_columns)
    input_parameter_specs = _parameter_specs_for_columns(input_columns)
    x_quantile = _input_quantile_matrix(samples, input_columns, input_parameter_specs)
    input_ranges = _default_input_ranges(samples, input_columns, input_parameter_specs)
    observable_keys = observable_keys or list(configs)
    unknown_observables = [observable_key for observable_key in observable_keys if observable_key not in configs]
    if unknown_observables:
        raise ValueError(f"Unknown observable key(s): {unknown_observables}")
    pca_components_by_observable = _resolve_pca_components(observable_keys, pca_components)

    observables = {}
    for observable_key in observable_keys:
        config = configs[observable_key]
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
            input_columns=input_columns,
            input_parameter_specs=input_parameter_specs,
        )
        if not np.allclose(loaded_x_quantile, x_quantile):
            raise ValueError(f"Input quantiles differ unexpectedly for {observable_key}")

        supported_bin_mask = np.ones(y_plot.shape[1], dtype=bool)
        if bool(config.get("drop_unsupported_bins", False)):
            supported_bin_mask = _supported_bin_mask(
                y_plot,
                y_noise=y_noise_plot,
                bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
            )
            if not np.any(supported_bin_mask):
                raise ValueError(f"All bins are unsupported for {observable_key}")
            x_plot = x_plot[supported_bin_mask]
            target_plot = target_plot[supported_bin_mask]
            if target_noise_plot is not None:
                target_noise_plot = _select_bins(target_noise_plot, supported_bin_mask)
            y_plot = y_plot[:, supported_bin_mask]
            if y_noise_plot is not None:
                y_noise_plot = y_noise_plot[:, supported_bin_mask]

        y_fit, alpha_sigma, training_metadata = _prepare_training_targets(
            y_plot,
            y_noise_plot,
            analysis=analysis,
            use_training_alpha=bool(config.get("use_training_alpha", True)),
            bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
            bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
            bad_training_sigma=float(config.get("bad_training_sigma", 5.0)),
            min_training_sigma=float(config.get("min_training_sigma", 1.0e-3)),
            bad_mask_override=training_bad_mask_override_from_transform(
                transform_metadata,
                bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
                bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
            ),
        )
        alpha_train = alpha_sigma**2 if alpha_sigma is not None else None

        observable_bundle = {
            "observable_key": observable_key,
            "analysis": analysis,
            "label": str(config["label"]),
            "x_plot": np.asarray(x_plot, dtype=float),
            "target_plot": np.asarray(target_plot, dtype=float),
            "target_sigma_plot": None if target_noise_plot is None else np.asarray(target_noise_plot, dtype=float),
            "y_train_preview": _training_preview(y_fit, training_preview_rows),
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
            "supported_bin_mask": supported_bin_mask,
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
            config_variance_threshold = config.get("pca_variance_threshold", pca_variance_threshold)
            if config_variance_threshold is not None:
                max_components = min(y_fit_scaled.shape[0], y_fit_scaled.shape[1])
                pca_probe = PCA(n_components=max_components)
                pca_probe.fit(y_fit_scaled)
                cumulative_variance = np.cumsum(pca_probe.explained_variance_ratio_)
                threshold = float(config_variance_threshold)
                if not 0.0 < threshold <= 1.0:
                    raise ValueError(f"pca_variance_threshold must be in (0, 1], got {threshold}")
                n_components_requested = int(np.searchsorted(cumulative_variance, threshold) + 1)
            else:
                threshold = None
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
                    "pca_variance_threshold": None if threshold is None else float(threshold),
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

    galacticus_default_params, galacticus_default_source, galacticus_default_params_path = _galacticus_default_params(
        campaign_root,
        list(input_columns),
        input_ranges=input_ranges,
    )

    return apply_observable_plot_metadata_overrides({
        "bundle_type": (
            "interactive_observables_pca_gp"
            if emulator_mode == "pca"
            else "interactive_observables_bin_by_bin_gp"
        ),
        "campaign_root": str(campaign_root.resolve()),
        "hdf5_filename": hdf5_filename,
        "best_fit_params": _default_best_fit_params(campaign_root, input_columns=list(input_columns)),
        "galacticus_default_params": galacticus_default_params,
        "galacticus_default_source": galacticus_default_source,
        "galacticus_default_params_path": galacticus_default_params_path,
        "input_columns": list(input_columns),
        "input_ranges": input_ranges,
        "evaluation_ids": samples["evaluation_id"].tolist(),
        "x_train_quantile": x_quantile,
        "observable_keys": list(observable_keys),
        "observables": observables,
        "observable_configs": {key: configs[key] for key in observable_keys},
        "n_restarts_optimizer": int(n_restarts_optimizer),
        "optimize_hyperparameters": bool(optimize_hyperparameters),
        "training_preview_rows": int(training_preview_rows),
        "min_log10_y": float(min_log10_y),
        "emulator_mode": emulator_mode,
        "pca_scaling": pca_scaling if emulator_mode == "pca" else None,
        "pca_variance_threshold": None if pca_variance_threshold is None else float(pca_variance_threshold),
    })


def predict_observables_bundle(bundle: dict, params: dict[str, float]) -> dict[str, dict[str, np.ndarray]]:
    x_raw = np.asarray([[float(params[column]) for column in bundle["input_columns"]]], dtype=float)
    input_parameter_specs = _parameter_specs_for_columns(list(bundle["input_columns"]))
    x_quantile = transform_to_prior_quantiles(input_parameter_specs, x_raw)
    return_std = bool(bundle.get("supports_predictive_uncertainty", True))
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
                    return_std=return_std,
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
                    return_std=return_std,
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
            "y_pred_plot": apply_y_display_offset(observable, y_pred),
            "y_std_plot": y_std,
        }
    return predictions


def refresh_observables_training_preview(
    bundle: dict,
    *,
    training_preview_rows: int | str | None = None,
    hdf5_filename: str | None = None,
    use_training_targets: bool = True,
) -> dict:
    campaign_root = Path(bundle["campaign_root"])
    samples = pd.read_csv(campaign_root / "samples.csv")
    input_columns = list(bundle["input_columns"])
    input_parameter_specs = _parameter_specs_for_columns(input_columns)
    evaluation_ids = list(bundle.get("evaluation_ids") or samples["evaluation_id"].tolist())
    hdf5_filename = hdf5_filename or bundle.get("hdf5_filename") or _infer_campaign_hdf5_filename(campaign_root, evaluation_ids)
    preview_rows = _training_preview_row_count(
        training_preview_rows,
        n_total=len(samples),
        fallback=int(bundle.get("training_preview_rows", 64)),
    )
    configs = _observable_configs(bundle.get("observable_configs"))

    for observable_key in bundle["observable_keys"]:
        observable = bundle["observables"][observable_key]
        config = configs[observable_key]
        (
            _samples,
            _x_quantile,
            _x_plot,
            _target_plot,
            _target_noise_plot,
            y_plot,
            y_noise_plot,
            _attrs,
            transform_metadata,
        ) = _load_observable_campaign(
            campaign_root,
            analysis=observable["analysis"],
            hdf5_filename=hdf5_filename,
            min_log10_y=float(bundle.get("min_log10_y", -7.0)),
            input_columns=input_columns,
            input_parameter_specs=input_parameter_specs,
        )
        supported_bin_mask = np.asarray(
            observable.get("supported_bin_mask", np.ones(y_plot.shape[1], dtype=bool)),
            dtype=bool,
        )
        if supported_bin_mask.shape[0] == y_plot.shape[1]:
            y_plot = y_plot[:, supported_bin_mask]
            if y_noise_plot is not None:
                y_noise_plot = y_noise_plot[:, supported_bin_mask]
            bad_mask_override = training_bad_mask_override_from_transform(
                transform_metadata,
                bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
                bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
            )
            if bad_mask_override is not None:
                bad_mask_override = bad_mask_override[:, supported_bin_mask]
        else:
            bad_mask_override = None

        if use_training_targets:
            y_preview_source, _alpha_sigma, _training_metadata = _prepare_training_targets(
                y_plot,
                y_noise_plot,
                analysis=observable["analysis"],
                use_training_alpha=bool(config.get("use_training_alpha", True)),
                bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
                bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
                bad_training_sigma=float(config.get("bad_training_sigma", 5.0)),
                min_training_sigma=float(config.get("min_training_sigma", 1.0e-3)),
                bad_mask_override=bad_mask_override,
            )
        else:
            y_preview_source = y_plot
        observable["y_train_preview"] = apply_y_display_offset(
            observable,
            _training_preview(y_preview_source, preview_rows),
        )

    bundle["hdf5_filename"] = hdf5_filename
    bundle["training_preview_rows"] = int(min(preview_rows, len(samples)))
    bundle["training_preview_source"] = "training_targets" if use_training_targets else "raw_outputs"
    return bundle


def bundle_meta(
    bundle: dict,
    *,
    best_fit_summary_path: str | Path | None = None,
    default_params_path: str | Path | None = None,
) -> dict:
    best_fit_params = bundle.get("best_fit_params")
    if best_fit_summary_path is not None or best_fit_params is None:
        best_fit_params = _default_best_fit_params(
            bundle["campaign_root"],
            input_columns=list(bundle["input_columns"]),
            summary_path=best_fit_summary_path,
        )
    input_ranges = _input_ranges_with_scale(bundle["input_ranges"])
    galacticus_default_params = bundle.get("galacticus_default_params")
    galacticus_default_source = bundle.get("galacticus_default_source")
    galacticus_default_params_path = bundle.get("galacticus_default_params_path")
    if default_params_path is not None or galacticus_default_params is None:
        (
            galacticus_default_params,
            galacticus_default_source,
            galacticus_default_params_path,
        ) = _galacticus_default_params(
            bundle["campaign_root"],
            list(bundle["input_columns"]),
            input_ranges=input_ranges,
            default_params_path=default_params_path,
        )
    return _json_ready({
        "bundle_type": bundle["bundle_type"],
        "emulator_mode": bundle.get("emulator_mode", "bin_by_bin"),
        "supports_predictive_uncertainty": bool(
            bundle.get("supports_predictive_uncertainty", True)
        ),
        "hdf5_filename": bundle.get("hdf5_filename"),
        "input_columns": bundle["input_columns"],
        "input_ranges": input_ranges,
        "observable_keys": bundle["observable_keys"],
        "best_fit_params": best_fit_params,
        "galacticus_default_params": galacticus_default_params,
        "galacticus_default_source": galacticus_default_source,
        "galacticus_default_params_path": galacticus_default_params_path,
        "n_training_rows": int(len(bundle.get("evaluation_ids", []))) if bundle.get("evaluation_ids") else None,
        "training_preview_rows": bundle.get("training_preview_rows"),
        "training_preview_source": bundle.get("training_preview_source"),
        "pca_variance_threshold": bundle.get("pca_variance_threshold"),
        "observables": {
            observable_key: {
                "analysis": observable["analysis"],
                "label": observable["label"],
                "target_label": observable.get("target_label"),
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
                "pca_variance_threshold": observable.get("pca_variance_threshold"),
                "pca_components_actual": observable.get("pca_components_actual"),
                "pca_explained_variance_ratio_sum": None
                if observable.get("pca_explained_variance_ratio") is None
                else float(np.sum(np.asarray(observable["pca_explained_variance_ratio"], dtype=float))),
            }
            for observable_key, observable in bundle["observables"].items()
        },
        "prediction_benchmark": bundle.get("prediction_benchmark"),
    })


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
    return apply_observable_plot_metadata_overrides(bundle)
