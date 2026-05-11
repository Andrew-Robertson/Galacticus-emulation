from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .lhs import (
    NormalPrior,
    ParameterSpec,
    TruncatedLogNormalPrior,
    TruncatedNormalPrior,
    transform_to_prior_quantiles,
)
from .persistence import load_emulator_bundle


INPUT_COLUMNS = [
    "diskVelocityCharacteristic",
    "delta_0",
    "delta_z",
    "delta_M",
    "delta_Mz",
    "attenuation_scatter",
]

INPUT_PARAMETER_SPECS = [
    ParameterSpec(
        path="nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic",
        short_name="diskVelocityCharacteristic",
        prior=TruncatedLogNormalPrior(lower=25.0, upper=300.0, x0=150.0, sigma=0.5),
    ),
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

SOBRAL_LABELS = ["Z1", "Z2", "Z3", "Z4"]


class MeanCenterer:
    def fit(self, y: np.ndarray) -> "MeanCenterer":
        self.mean_ = np.mean(y, axis=0)
        self.scale_ = np.ones(y.shape[1], dtype=float)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y - self.mean_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y + self.mean_


def _kernel(n_dim: int):
    return ConstantKernel(4.0, (1.0e-3, 1.0e3)) * Matern(
        length_scale=np.asarray([1.0, 0.5, 1.5, 1.0, 3.0, 10.0], dtype=float)[:n_dim],
        length_scale_bounds=(1.0e-2, 1.0e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=3.0e-2, noise_level_bounds=(1.0e-8, 1.0))


def _fit_scaled_gp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_restarts_optimizer: int,
    random_state: int,
) -> tuple[GaussianProcessRegressor, float, float]:
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    if y_std == 0.0:
        y_std = 1.0
    model = GaussianProcessRegressor(
        kernel=_kernel(x.shape[1]),
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    model.fit(x, (y - y_mean) / y_std)
    return model, y_mean, y_std


def _predict_scaled_gp(
    model: GaussianProcessRegressor,
    y_mean: float,
    y_std: float,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_scaled, pred_std_scaled = model.predict(x, return_std=True)
    return y_mean + y_std * pred_scaled, y_std * pred_std_scaled


def _apply_filters(table: pd.DataFrame, max_abs_delta: float | None, max_attenuation_scatter: float | None) -> pd.DataFrame:
    mask = np.ones(len(table), dtype=bool)
    if max_abs_delta is not None:
        for column in ["delta_0", "delta_z", "delta_M", "delta_Mz"]:
            mask &= np.abs(table[column].to_numpy(dtype=float)) <= max_abs_delta
    if max_attenuation_scatter is not None:
        mask &= table["attenuation_scatter"].to_numpy(dtype=float) <= max_attenuation_scatter
    return table.loc[mask].reset_index(drop=True)


def _limit_draws_per_eval(table: pd.DataFrame, train_draws_per_eval: int | None) -> pd.DataFrame:
    if train_draws_per_eval is None:
        return table.reset_index(drop=True)
    return (
        table.sort_values(["evaluation_id", "dust_draw_index"])
        .groupby("evaluation_id", group_keys=False)
        .head(train_draws_per_eval)
        .reset_index(drop=True)
    )


def _bin_sort_key(column: str) -> int:
    match = re.search(r"_bin(\d+)$", column)
    if match is None:
        raise ValueError(f"Could not parse bin index from column '{column}'")
    return int(match.group(1))


def _output_columns(table: pd.DataFrame, sobral_label: str) -> list[str]:
    return sorted(
        [
            column
            for column in table.columns
            if column.startswith(f"halpha_sobral_{sobral_label.lower()}_bin")
            and not column.endswith("_target")
            and not column.endswith("_target_std")
        ],
        key=_bin_sort_key,
    )


def _log10_with_floor(values: np.ndarray, min_log10_lf: float) -> tuple[np.ndarray, float, int, int]:
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("No positive LF values found for log10 transform.")
    floor = 0.5 * float(np.min(positive))
    n_floored = int(np.count_nonzero(values <= 0.0))
    safe = np.where(values > 0.0, values, floor)
    log_values = np.log10(safe)
    n_clipped = int(np.count_nonzero(log_values < min_log10_lf))
    log_values = np.maximum(log_values, min_log10_lf)
    return log_values, floor, n_floored, n_clipped


def _make_preprocessor(pca_scaling: str):
    if pca_scaling == "standardized":
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()
    if pca_scaling == "unscaled":
        return MeanCenterer()
    raise ValueError(f"Unknown pca_scaling={pca_scaling!r}")


def _default_input_ranges(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    defaults = {
        "diskVelocityCharacteristic": float(table["diskVelocityCharacteristic"].median()),
        "delta_0": 0.0,
        "delta_z": 0.0,
        "delta_M": 0.0,
        "delta_Mz": 0.0,
        "attenuation_scatter": 0.25,
    }
    ranges = {}
    for column in INPUT_COLUMNS:
        min_value = float(table[column].min())
        max_value = float(table[column].max())
        default = defaults[column]
        default = min(max(default, min_value), max_value)
        ranges[column] = {
            "min": min_value,
            "max": max_value,
            "default": default,
        }
    return ranges


def _training_preview(y_log10: np.ndarray, max_rows: int) -> np.ndarray:
    if y_log10.shape[0] <= max_rows:
        return y_log10
    indices = np.asarray(sorted({int(round(v)) for v in np.linspace(0, y_log10.shape[0] - 1, max_rows)}), dtype=int)
    return y_log10[indices]


def train_halpha_bundle(
    campaign_root: str | Path,
    *,
    table_filename: str = "halpha_dust_lf_emulator_table.csv",
    long_filename: str = "halpha_dust_lf_long.csv",
    sobral_labels: list[str] | None = None,
    train_draws_per_eval: int | None = 32,
    pca_components: int = 5,
    pca_scaling: str = "unscaled",
    n_restarts_optimizer: int = 0,
    random_state: int = 42,
    min_log10_lf: float = -7.0,
    max_abs_delta: float | None = 2.0,
    max_attenuation_scatter: float | None = 0.45,
    training_preview_rows: int = 64,
) -> dict:
    campaign_root = Path(campaign_root)
    sobral_labels = sobral_labels or SOBRAL_LABELS
    table = pd.read_csv(campaign_root / table_filename)
    long_table = pd.read_csv(campaign_root / long_filename)
    table = _apply_filters(table, max_abs_delta, max_attenuation_scatter)
    table = _limit_draws_per_eval(table, train_draws_per_eval)

    x_raw = table[INPUT_COLUMNS].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)
    input_ranges = _default_input_ranges(table)

    sobral_bundles: dict[str, dict] = {}
    all_y_values: list[np.ndarray] = []
    for sobral_label in sobral_labels:
        output_columns = _output_columns(table, sobral_label)
        centers = np.array(
            [
                float(
                    long_table.loc[
                        (long_table["sobral_label"] == sobral_label)
                        & (long_table["bin_index"] == bin_index),
                        "log10_luminosity_center",
                    ]
                    .drop_duplicates()
                    .iloc[0]
                )
                for bin_index in range(len(output_columns))
            ],
            dtype=float,
        )
        y_linear = table[output_columns].to_numpy(dtype=float)
        y_log10, floor, n_floored, n_clipped = _log10_with_floor(y_linear, min_log10_lf)
        scaler = _make_preprocessor(pca_scaling)
        y_scaled = scaler.fit_transform(y_log10)
        n_components_actual = min(pca_components, y_scaled.shape[0], y_scaled.shape[1])
        pca = PCA(n_components=n_components_actual)
        coefficients = pca.fit_transform(y_scaled)

        component_models = []
        component_y_means = []
        component_y_stds = []
        kernels = []
        for component_index in range(n_components_actual):
            model, y_mean, y_std = _fit_scaled_gp(
                x,
                coefficients[:, component_index],
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=random_state,
            )
            component_models.append(model)
            component_y_means.append(y_mean)
            component_y_stds.append(y_std)
            kernels.append(str(model.kernel_))

        target_columns = [f"{column}_target" for column in output_columns]
        target_std_columns = [f"{column}_target_std" for column in output_columns]
        target_linear = table[target_columns].iloc[0].to_numpy(dtype=float)
        target_std_linear = table[target_std_columns].iloc[0].to_numpy(dtype=float)
        target_sigma_log10 = np.maximum(target_std_linear / np.maximum(target_linear, 1.0e-12) / np.log(10.0), 1.0e-6)

        sobral_bundles[sobral_label] = {
            "output_columns": output_columns,
            "log10_luminosity_center": centers,
            "models": component_models,
            "component_y_means": np.asarray(component_y_means, dtype=float),
            "component_y_stds": np.asarray(component_y_stds, dtype=float),
            "pca": pca,
            "scaler": scaler,
            "kernels": kernels,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components_actual": int(n_components_actual),
            "target_log10": np.log10(np.maximum(target_linear, 1.0e-12)),
            "target_sigma_log10": target_sigma_log10,
            "y_train_log10_preview": _training_preview(y_log10, training_preview_rows),
            "floor_linear": float(floor),
            "n_floored": int(n_floored),
            "n_clipped": int(n_clipped),
            "n_training_rows": int(y_log10.shape[0]),
        }
        all_y_values.extend([y_log10.ravel(), np.log10(np.maximum(target_linear, 1.0e-12))])

    y_all = np.concatenate(all_y_values)
    bundle = {
        "bundle_type": "interactive_halpha_pca_gp",
        "campaign_root": str(campaign_root.resolve()),
        "input_columns": INPUT_COLUMNS,
        "input_ranges": input_ranges,
        "sobral_labels": sobral_labels,
        "sobral_bundles": sobral_bundles,
        "pca_components_requested": int(pca_components),
        "pca_scaling": pca_scaling,
        "train_draws_per_eval": None if train_draws_per_eval is None else int(train_draws_per_eval),
        "n_restarts_optimizer": int(n_restarts_optimizer),
        "random_state": int(random_state),
        "min_log10_lf": float(min_log10_lf),
        "max_abs_delta": None if max_abs_delta is None else float(max_abs_delta),
        "max_attenuation_scatter": None if max_attenuation_scatter is None else float(max_attenuation_scatter),
        "x_axis_label": r"log10(L_Halpha / erg s^-1)",
        "y_axis_label": r"log10(Phi / Mpc^-3 dex^-1)",
        "y_plot_min": float(np.floor(np.min(y_all) - 0.25)),
        "y_plot_max": float(np.ceil(np.max(y_all) + 0.25)),
    }
    return bundle


def predict_halpha_bundle(bundle: dict, params: dict[str, float]) -> dict[str, dict[str, np.ndarray]]:
    x_raw = np.asarray([[float(params[column]) for column in bundle["input_columns"]]], dtype=float)
    x = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)
    predictions = {}
    for sobral_label in bundle["sobral_labels"]:
        sobral_bundle = bundle["sobral_bundles"][sobral_label]
        n_components_actual = sobral_bundle["n_components_actual"]
        coefficient_predictions = np.zeros((1, n_components_actual), dtype=float)
        coefficient_stds = np.zeros_like(coefficient_predictions)
        for component_index, model in enumerate(sobral_bundle["models"]):
            pred, pred_std = _predict_scaled_gp(
                model,
                float(sobral_bundle["component_y_means"][component_index]),
                float(sobral_bundle["component_y_stds"][component_index]),
                x,
            )
            coefficient_predictions[:, component_index] = pred
            coefficient_stds[:, component_index] = pred_std
        y_pred_scaled = sobral_bundle["pca"].inverse_transform(coefficient_predictions)
        y_pred = sobral_bundle["scaler"].inverse_transform(y_pred_scaled)[0]
        y_var_scaled = (coefficient_stds**2) @ (sobral_bundle["pca"].components_**2)
        y_std = (np.sqrt(np.maximum(y_var_scaled, 0.0)) * sobral_bundle["scaler"].scale_[None, :])[0]
        predictions[sobral_label] = {
            "y_pred_log10": y_pred,
            "y_std_log10": y_std,
            "log10_luminosity_center": np.asarray(sobral_bundle["log10_luminosity_center"], dtype=float),
        }
    return predictions


def bundle_meta(bundle: dict) -> dict:
    sobral_meta = {}
    for sobral_label in bundle["sobral_labels"]:
        sobral_bundle = bundle["sobral_bundles"][sobral_label]
        sobral_meta[sobral_label] = {
            "log10_luminosity_center": np.asarray(sobral_bundle["log10_luminosity_center"], dtype=float).tolist(),
            "target_log10": np.asarray(sobral_bundle["target_log10"], dtype=float).tolist(),
            "target_sigma_log10": np.asarray(sobral_bundle["target_sigma_log10"], dtype=float).tolist(),
            "y_train_log10_preview": np.asarray(sobral_bundle["y_train_log10_preview"], dtype=float).tolist(),
            "n_training_rows": int(sobral_bundle["n_training_rows"]),
        }
    meta = {
        "bundle_type": bundle["bundle_type"],
        "input_columns": bundle["input_columns"],
        "input_ranges": bundle["input_ranges"],
        "sobral_labels": bundle["sobral_labels"],
        "sobral_meta": sobral_meta,
        "pca_components_requested": bundle["pca_components_requested"],
        "train_draws_per_eval": bundle["train_draws_per_eval"],
        "x_axis_label": bundle["x_axis_label"],
        "y_axis_label": bundle["y_axis_label"],
        "y_plot_min": bundle["y_plot_min"],
        "y_plot_max": bundle["y_plot_max"],
    }
    if "prediction_benchmark" in bundle:
        meta["prediction_benchmark"] = bundle["prediction_benchmark"]
    return meta


def benchmark_halpha_bundle(bundle: dict, *, n_predictions: int = 200) -> dict[str, float]:
    params = {column: float(bundle["input_ranges"][column]["default"]) for column in bundle["input_columns"]}
    start = time.perf_counter()
    for _ in range(n_predictions):
        predict_halpha_bundle(bundle, params)
    elapsed = time.perf_counter() - start
    return {
        "n_predictions": int(n_predictions),
        "total_seconds": float(elapsed),
        "milliseconds_per_prediction": float(1000.0 * elapsed / n_predictions),
    }


def load_halpha_bundle(path: str | Path) -> dict:
    bundle = load_emulator_bundle(path)
    if bundle.get("bundle_type") != "interactive_halpha_pca_gp":
        raise ValueError(f"Unexpected bundle_type: {bundle.get('bundle_type')}")
    return bundle
