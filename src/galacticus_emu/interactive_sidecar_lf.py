from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .gp import predict_scaled_gp
from .lhs import ParameterSpec, TruncatedLogNormalPrior, transform_to_prior_quantiles
from .observable_plot_metadata import (
    apply_sidecar_lf_plot_metadata_overrides,
    apply_sidecar_lf_y_display_offset,
)
from .persistence import load_emulator_bundle


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


def _prior_bounds(spec: ParameterSpec) -> tuple[float, float]:
    prior = spec.prior
    lower = getattr(prior, "lower", None)
    upper = getattr(prior, "upper", None)
    if lower is not None and upper is not None and math.isfinite(float(lower)) and math.isfinite(float(upper)):
        return float(lower), float(upper)
    bounds = np.asarray(prior.inverse_cdf(np.asarray([1.0e-3, 1.0 - 1.0e-3], dtype=float)), dtype=float)
    return float(bounds[0]), float(bounds[1])


def _prior_default(spec: ParameterSpec) -> float:
    prior = spec.prior
    if hasattr(prior, "x0"):
        return float(prior.x0)
    if hasattr(prior, "mean"):
        lower, upper = _prior_bounds(spec)
        return min(max(float(prior.mean), lower), upper)
    lower, upper = _prior_bounds(spec)
    if lower > 0.0 and isinstance(prior, TruncatedLogNormalPrior):
        return float(np.sqrt(lower * upper))
    return 0.5 * (lower + upper)


def _slider_scale(spec: ParameterSpec) -> str:
    return "log" if isinstance(spec.prior, TruncatedLogNormalPrior) else "linear"


def _samples_path(bundle: dict) -> Path:
    return Path(bundle["campaign_root"]) / "samples.csv"


def _input_ranges(bundle: dict) -> dict[str, dict[str, float | str]]:
    specs_by_name = {spec.short_name: spec for spec in bundle["parameter_specs"]}
    samples = pd.read_csv(_samples_path(bundle)) if _samples_path(bundle).exists() else None
    ranges: dict[str, dict[str, float | str]] = {}
    for name in bundle["parameter_names"]:
        spec = specs_by_name[name]
        lower, upper = _prior_bounds(spec)
        if samples is not None and name in samples.columns:
            values = samples[name].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size > 0:
                lower = float(np.nanmin(values))
                upper = float(np.nanmax(values))
        default = min(max(_prior_default(spec), lower), upper)
        ranges[name] = {
            "min": lower,
            "max": upper,
            "default": default,
            "scale": _slider_scale(spec),
        }
    return ranges


def _best_fit_params_from_summary(summary_path: str | Path, parameter_names: list[str]) -> dict[str, float] | None:
    path = Path(summary_path).expanduser()
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    theta = data.get("best_theta")
    if not isinstance(theta, dict):
        return None
    return {name: float(theta[name]) for name in parameter_names if name in theta}


def _default_params(bundle: dict) -> dict[str, float]:
    specs_by_name = {spec.short_name: spec for spec in bundle["parameter_specs"]}
    return {name: _prior_default(specs_by_name[name]) for name in bundle["parameter_names"]}


def _feature_matrix(bundle: dict, params: dict[str, float]) -> np.ndarray:
    parameter_names = list(bundle["parameter_names"])
    slow_count = int(bundle["slow_count"])
    slow_names = parameter_names[:slow_count]
    sidecar_names = parameter_names[slow_count:]
    slow_values = np.asarray([[float(params[name]) for name in slow_names]], dtype=float)
    slow_quantiles = transform_to_prior_quantiles(bundle["parameter_specs"][:slow_count], slow_values)
    if not sidecar_names:
        return slow_quantiles
    sidecar_values = np.asarray([[float(params[name]) for name in sidecar_names]], dtype=float)
    return np.column_stack([slow_quantiles, sidecar_values])


def _predict_observable(observable: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_components = len(observable["models"])
    coefficients = np.zeros((x.shape[0], n_components), dtype=float)
    coefficient_stds = np.zeros_like(coefficients)
    for index, model in enumerate(observable["models"]):
        pred, pred_std = predict_scaled_gp(
            model,
            float(observable["y_means"][index]),
            float(observable["y_stds"][index]),
            x,
        )
        coefficients[:, index] = pred
        coefficient_stds[:, index] = pred_std

    components = np.asarray(observable["pca_components_matrix"], dtype=float)
    pca_mean = np.asarray(observable["pca_mean"], dtype=float)
    preprocessor_mean = np.asarray(observable["preprocessor_mean"], dtype=float)
    preprocessor_scale = np.asarray(observable["preprocessor_scale"], dtype=float)
    y_scaled = coefficients @ components + pca_mean[None, :]
    y_pred = y_scaled * preprocessor_scale[None, :] + preprocessor_mean[None, :]
    y_var_scaled = (coefficient_stds**2) @ (components**2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * preprocessor_scale[None, :]
    return y_pred[0], y_std[0]


def predict_sidecar_lf_bundle(bundle: dict, params: dict[str, float]) -> dict[str, dict[str, np.ndarray]]:
    x = _feature_matrix(bundle, params)
    predictions = {}
    for observable_key in bundle["observables"]:
        observable = bundle["observables"][observable_key]
        y_pred, y_std = _predict_observable(observable, x)
        predictions[observable_key] = {
            "x_plot": np.asarray(observable["x_plot"], dtype=float),
            "y_pred_plot": apply_sidecar_lf_y_display_offset(observable, y_pred),
            "y_std_plot": y_std,
        }
    return predictions


def _training_preview_row_count(rows: int | str | None, fallback: int) -> int | None:
    if rows is None:
        return int(fallback)
    if isinstance(rows, str):
        value = rows.strip().lower()
        if value in {"all", "*"}:
            return None
        return int(value)
    return int(rows)


def _log10_with_floor(values: np.ndarray, min_log10_lf: float) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return np.full_like(values, np.nan, dtype=float)
    floor = 0.5 * float(np.min(positive))
    safe = np.where(np.isfinite(values) & (values > 0.0), values, floor)
    return np.maximum(np.log10(safe), float(min_log10_lf))


def _filter_preview_table(table: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    options = bundle.get("training_options", {})
    dust_draw_index = options.get("dust_draw_index")
    if dust_draw_index is not None and "dust_draw_index" in table.columns:
        table = table.loc[table["dust_draw_index"].isin(dust_draw_index)]
    max_draws = options.get("max_dust_draws_per_eval")
    if max_draws is not None and "evaluation_id" in table.columns:
        table = (
            table.sort_values(["evaluation_id", "dust_draw_index"])
            .groupby("evaluation_id", group_keys=False)
            .head(int(max_draws))
        )
    return table


def _training_preview(bundle: dict, observable: dict, max_rows: int | None) -> np.ndarray:
    output_columns = list(observable["output_columns"])
    use_columns = ["evaluation_id", "dust_draw_index", *output_columns]
    frames = []
    n_rows = 0
    root = Path(bundle["campaign_root"])
    pattern = f"*/{bundle['sidecar_dir']}/{bundle['table_filename']}"
    for path in sorted((root / "evaluations").glob(pattern)):
        table = pd.read_csv(path, usecols=lambda column: column in use_columns)
        table = _filter_preview_table(table, bundle)
        if table.empty:
            continue
        frames.append(table[output_columns])
        n_rows += len(table)
        if max_rows is not None and n_rows >= max_rows:
            break
    if not frames:
        return np.empty((0, len(output_columns)), dtype=float)
    values = pd.concat(frames, ignore_index=True).to_numpy(dtype=float)
    if max_rows is not None:
        values = values[:max_rows]
    min_log10_lf = float(bundle.get("training_options", {}).get("min_log10_lf", -8.0))
    return _log10_with_floor(values, min_log10_lf)


def _y_limits(target: np.ndarray, preview: np.ndarray, min_log10_lf: float) -> tuple[float, float]:
    arrays = [np.asarray(target, dtype=float).ravel()]
    if preview.size:
        arrays.append(np.asarray(preview, dtype=float).ravel())
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(min_log10_lf), -1.0
    return float(min(min_log10_lf, np.nanmin(values) - 0.2)), float(max(-1.0, np.nanmax(values) + 0.2))


def bundle_meta(
    bundle: dict,
    *,
    best_fit_summary_path: str | Path | None = None,
    training_preview_rows: int | str | None = None,
) -> dict:
    bundle = apply_sidecar_lf_plot_metadata_overrides(bundle)
    max_preview_rows = _training_preview_row_count(training_preview_rows, fallback=64)
    parameter_names = list(bundle["parameter_names"])
    input_ranges = _input_ranges(bundle)
    best_fit_params = (
        None
        if best_fit_summary_path is None
        else _best_fit_params_from_summary(best_fit_summary_path, parameter_names)
    )
    min_log10_lf = float(bundle.get("training_options", {}).get("min_log10_lf", -8.0))
    observables = {}
    for observable_key, observable in bundle["observables"].items():
        preview = _training_preview(bundle, observable, max_preview_rows)
        target = apply_sidecar_lf_y_display_offset(observable, observable["target_plot"])
        preview = apply_sidecar_lf_y_display_offset(observable, preview)
        y_plot_min, y_plot_max = _y_limits(target, preview, min_log10_lf)
        observables[observable_key] = {
            "analysis": f"{observable.get('observable', observable_key)}:{observable.get('sample_label', '')}",
            "label": observable.get("label", observable_key),
            "target_label": observable.get("target_label"),
            "x_plot": np.asarray(observable["x_plot"], dtype=float),
            "target_plot": target,
            "target_sigma_plot": None
            if observable["target_sigma_plot"] is None
            else np.asarray(observable["target_sigma_plot"], dtype=float),
            "y_train_preview": preview,
            "x_axis_label": observable.get("x_axis_label", "log10 line luminosity [erg/s]"),
            "y_axis_label": observable.get("y_axis_label", "log10 Phi [Mpc^-3 dex^-1]"),
            "y_plot_min": y_plot_min,
            "y_plot_max": y_plot_max,
            "training_metadata": {
                "n_training_rows": int(bundle.get("n_training_rows", 0)),
                "pca_components_actual": int(observable.get("pca_components_actual", 0)),
                "pca_explained_variance_ratio_sum": float(
                    np.sum(np.asarray(observable.get("pca_explained_variance_ratio", []), dtype=float))
                ),
            },
            "emulator_mode": "pca",
            "pca_variance_threshold": observable.get("pca_variance_threshold"),
            "pca_components_actual": observable.get("pca_components_actual"),
            "pca_explained_variance_ratio_sum": float(
                np.sum(np.asarray(observable.get("pca_explained_variance_ratio", []), dtype=float))
            ),
        }

    return _json_ready(
        {
            "bundle_type": bundle["bundle_type"],
            "emulator_mode": "pca",
            "input_columns": parameter_names,
            "input_ranges": input_ranges,
            "observable_keys": list(bundle["observables"]),
            "observable_groups": bundle.get("observable_groups", {}),
            "best_fit_params": best_fit_params,
            "galacticus_default_params": _default_params(bundle),
            "galacticus_default_source": "parameter prior defaults",
            "galacticus_default_params_path": None,
            "training_preview_rows": "all" if max_preview_rows is None else int(max_preview_rows),
            "training_preview_source": "sidecar luminosity-function tables",
            "observables": observables,
            "prediction_benchmark": bundle.get("prediction_benchmark"),
        }
    )


def benchmark_sidecar_lf_bundle(bundle: dict, *, n_predictions: int = 50) -> dict[str, float]:
    defaults = _default_params(bundle)
    start = time.perf_counter()
    for _ in range(n_predictions):
        predict_sidecar_lf_bundle(bundle, defaults)
    elapsed = time.perf_counter() - start
    return {
        "n_predictions": int(n_predictions),
        "total_seconds": float(elapsed),
        "milliseconds_per_prediction": float(1000.0 * elapsed / max(n_predictions, 1)),
    }


def load_sidecar_lf_bundle(path: str | Path) -> dict:
    bundle = load_emulator_bundle(path)
    if bundle.get("bundle_type") != "sidecar_lf_pca_gp":
        raise ValueError(f"Unexpected bundle_type: {bundle.get('bundle_type')}")
    return apply_sidecar_lf_plot_metadata_overrides(bundle)
