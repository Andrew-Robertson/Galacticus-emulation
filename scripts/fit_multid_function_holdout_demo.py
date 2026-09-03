from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp
from galacticus_emu.interactive_observables import (
    OBSERVABLE_CONFIGS,
    _bad_training_mask as _shared_bad_training_mask,
    _prepare_training_targets as _prepare_shared_training_targets,
    _select_bins,
    _supported_bin_mask as _shared_supported_bin_mask,
    _target_noise_from_analysis_group,
    training_bad_mask_override_from_transform,
)
from galacticus_emu.observable_plot_metadata import STANDARD_OBSERVABLE_PLOT_METADATA


PRESETS = {
    "bh_halo_mass_trinity_z1": {
        "analysis": "blackHoleHaloMassRelationTRINITYz1",
        "output_prefix": "bh_halo_mass_trinity_z1_holdout_demo",
        "figures_dir_name": "figures_bh_halo_mass_trinity_z1_holdout_demo",
        "emulator_dir_name": "emulator_bh_halo_mass_trinity_z1_holdout_demo",
        "overlay_ymin": 5.0,
        "overlay_ymax": 9.5,
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
        "drop_unsupported_bins": True,
    },
    "smf_zfourge_z0": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz0",
        "output_prefix": "smf_zfourge_z0_holdout_demo",
        "figures_dir_name": "figures_smf_zfourge_z0_holdout_demo",
        "emulator_dir_name": "emulator_smf_zfourge_z0_holdout_demo",
        "overlay_ymin": -6.0,
        "overlay_ymax": -1.0,
    },
    "smf_zfourge_z3": {
        "analysis": "massFunctionStellarTomczak2014ZFOURGEz3",
        "output_prefix": "smf_zfourge_z3_holdout_demo",
        "figures_dir_name": "figures_smf_zfourge_z3_holdout_demo",
        "emulator_dir_name": "emulator_smf_zfourge_z3_holdout_demo",
        "overlay_ymin": -6.0,
        "overlay_ymax": -1.0,
    },
    "smf_liwhite2009_sdss": {
        "analysis": "massFunctionStellarLiWhite2009SDSS",
        "output_prefix": "smf_liwhite2009_sdss_holdout_demo",
        "figures_dir_name": "figures_smf_liwhite2009_sdss_holdout_demo",
        "emulator_dir_name": "emulator_smf_liwhite2009_sdss_holdout_demo",
        "overlay_ymin": -7.0,
        "overlay_ymax": -1.0,
    },
    "mzr_blanc2019": {
        "analysis": "massMetallicityBlanc2019",
        "output_prefix": "mzr_blanc2019_holdout_demo",
        "figures_dir_name": "figures_mzr_blanc2019_holdout_demo",
        "emulator_dir_name": "emulator_mzr_blanc2019_holdout_demo",
        "overlay_ymin": 7.5,
        "overlay_ymax": 10.0,
        "use_training_alpha": True,
        "bad_training_condition": "nonpositive",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
    "morphological_fraction_gama_moffett2016": {
        "analysis": "morphologicalFractionGAMAMoffett2016",
        "output_prefix": "morphological_fraction_gama_moffett2016_holdout_demo",
        "figures_dir_name": "figures_morphological_fraction_gama_moffett2016_holdout_demo",
        "emulator_dir_name": "emulator_morphological_fraction_gama_moffett2016_holdout_demo",
        "overlay_ymin": 0.0,
        "overlay_ymax": 1.0,
        "use_training_alpha": True,
        "bad_training_condition": "zero_and_zero_noise",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
    "hi_mass_function_alfalfa": {
        "analysis": "massFunctionHIMartin2010ALFALFA",
        "output_prefix": "hi_mass_function_alfalfa_holdout_demo",
        "figures_dir_name": "figures_hi_mass_function_alfalfa_holdout_demo",
        "emulator_dir_name": "emulator_hi_mass_function_alfalfa_holdout_demo",
        "overlay_ymin": -6.0,
        "overlay_ymax": -1.0,
    },
    "bh_velocity_dispersion": {
        "analysis": "blackHoleVelocityDispersionRelation",
        "output_prefix": "bh_velocity_dispersion_holdout_demo",
        "figures_dir_name": "figures_bh_velocity_dispersion_holdout_demo",
        "emulator_dir_name": "emulator_bh_velocity_dispersion_holdout_demo",
        "overlay_ymin": 5.0,
        "overlay_ymax": 10.0,
    },
    "sfr_function_robotham2011": {
        "analysis": "starFormationRateFunctionRobotham2011",
        "output_prefix": "sfr_function_robotham2011_holdout_demo",
        "figures_dir_name": "figures_sfr_function_robotham2011_holdout_demo",
        "emulator_dir_name": "emulator_sfr_function_robotham2011_holdout_demo",
        "overlay_ymin": -6.0,
        "overlay_ymax": -1.0,
    },
    "size_mass_vdw2014_star_forming_z0": {
        "analysis": "stellarSizeMassRelationvanDerWel2014Sample1",
        "output_prefix": "size_mass_vdw2014_star_forming_z0_holdout_demo",
        "figures_dir_name": "figures_size_mass_vdw2014_star_forming_z0_holdout_demo",
        "emulator_dir_name": "emulator_size_mass_vdw2014_star_forming_z0_holdout_demo",
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
    "size_mass_vdw2014_quiescent_z0": {
        "analysis": "stellarSizeMassRelationvanDerWel2014Sample7",
        "output_prefix": "size_mass_vdw2014_quiescent_z0_holdout_demo",
        "figures_dir_name": "figures_size_mass_vdw2014_quiescent_z0_holdout_demo",
        "emulator_dir_name": "emulator_size_mass_vdw2014_quiescent_z0_holdout_demo",
        "use_training_alpha": True,
        "bad_training_condition": "zero_only",
        "bad_training_value_fill": "bin_median",
        "bad_training_sigma": 5.0,
        "min_training_sigma": 1.0e-3,
    },
}


PRESET_OBSERVABLE_KEYS = {
    "bh_halo_mass_trinity_z1": "bh_halo_mass_trinity_z1",
    "bh_velocity_dispersion": "bh_velocity_dispersion",
    "smf_liwhite2009_sdss": "smf_liwhite2009_sdss",
    "smf_zfourge_z0": "smf_z0",
    "smf_zfourge_z3": "smf_z3",
    "mzr_blanc2019": "mzr_blanc2019",
    "morphological_fraction_gama_moffett2016": "morphological_fraction_gama_moffett2016",
    "sfr_function_robotham2011": "sfr_function_robotham2011",
    "size_mass_vdw2014_star_forming_z0": "size_mass_vdw2014_star_forming_z0",
    "size_mass_vdw2014_quiescent_z0": "size_mass_vdw2014_quiescent_z0",
}


def _apply_shared_observable_policy() -> None:
    policy_keys = [
        "analysis",
        "use_training_alpha",
        "bad_training_condition",
        "bad_training_value_fill",
        "bad_training_sigma",
        "min_training_sigma",
        "drop_unsupported_bins",
    ]
    for preset_key, observable_key in PRESET_OBSERVABLE_KEYS.items():
        if preset_key not in PRESETS or observable_key not in OBSERVABLE_CONFIGS:
            continue
        shared_config = OBSERVABLE_CONFIGS[observable_key]
        preset = PRESETS[preset_key]
        for key in policy_keys:
            if key in shared_config:
                preset[key] = shared_config[key]
        if "y_plot_min" in shared_config:
            preset["overlay_ymin"] = shared_config["y_plot_min"]
        if "y_plot_max" in shared_config:
            preset["overlay_ymax"] = shared_config["y_plot_max"]


_apply_shared_observable_policy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a multi-parameter GP holdout demo for a 1D Galacticus analysis."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--analysis", default=None)
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--min-log10-y", type=float, default=-7.0)
    parser.add_argument("--n-example-curves", type=int, default=5)
    parser.add_argument("--overlay-ymin", type=float, default=None)
    parser.add_argument("--overlay-ymax", type=float, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--figures-dir-name", default=None)
    parser.add_argument("--emulator-dir-name", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--replot-only", action="store_true")
    parser.add_argument(
        "--use-training-alpha",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--bad-training-condition", default=None)
    parser.add_argument("--bad-training-value-fill", default=None)
    parser.add_argument("--bad-training-sigma", type=float, default=None)
    parser.add_argument("--min-training-sigma", type=float, default=None)
    parser.add_argument(
        "--drop-unsupported-bins",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--optimize-hyperparameters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fit-white-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit an additional constant WhiteKernel noise term on top of any supplied training alpha.",
    )
    return parser.parse_args()


def _setting(args: argparse.Namespace, key: str, default=None):
    normalized_key = key.replace("-", "_")
    value = getattr(args, normalized_key)
    if value is not None:
        return value
    if args.preset is not None:
        preset = PRESETS[args.preset]
        if key in preset:
            return preset[key]
        if normalized_key in preset:
            return preset[normalized_key]
    return default


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quantile_columns(samples: pd.DataFrame) -> list[str]:
    columns = [column for column in samples.columns if column.endswith("_quantile")]
    if not columns:
        raise ValueError("No quantile columns found in samples.csv")
    return columns


def _infer_analysis_attributes(campaign_root: Path, analysis: str, hdf5_filename: str) -> dict:
    samples = pd.read_csv(campaign_root / "samples.csv")
    first_evaluation = samples.iloc[0]["evaluation_id"]
    path = campaign_root / "evaluations" / first_evaluation / hdf5_filename
    with h5py.File(path, "r") as handle:
        group = handle[f"/analyses/{analysis}"]
        attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
    return attrs


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
    target_noise_linear: np.ndarray | None,
    *,
    is_log: bool,
    min_log10_y: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict]:
    if not is_log:
        return (
            y_linear,
            y_noise_linear,
            target_linear,
            target_noise_linear,
            {"y_transform": "identity"},
        )

    positive = np.concatenate([y_linear[y_linear > 0.0], target_linear[target_linear > 0.0]])
    if positive.size == 0:
        raise ValueError("Analysis contains no positive values to log-transform")

    effective_floor_linear = 10.0 ** min_log10_y
    y_nonpositive = np.asarray(y_linear <= 0.0, dtype=bool)
    target_nonpositive = np.asarray(target_linear <= 0.0, dtype=bool)
    n_floored = int(np.sum(y_nonpositive))
    safe_y_linear = np.where(y_linear > 0.0, y_linear, effective_floor_linear)
    y_log10_raw = np.log10(safe_y_linear)
    n_clipped = int(np.sum(y_log10_raw < min_log10_y))
    y_plot = np.maximum(y_log10_raw, min_log10_y)
    y_floor_mask = np.asarray(y_log10_raw <= min_log10_y, dtype=bool)

    safe_target_linear = np.where(target_linear > 0.0, target_linear, effective_floor_linear)
    target_log10_raw = np.log10(safe_target_linear)
    target_plot = np.maximum(target_log10_raw, min_log10_y)
    target_floor_mask = np.asarray(target_log10_raw <= min_log10_y, dtype=bool)

    if y_noise_linear is None:
        y_noise = None
    else:
        y_noise = np.maximum(y_noise_linear / (safe_y_linear * np.log(10.0)), 1.0e-6)
        y_noise = np.where(y_floor_mask, np.nan, y_noise)

    if target_noise_linear is None:
        target_noise = None
    else:
        target_noise = np.maximum(target_noise_linear / (safe_target_linear * np.log(10.0)), 1.0e-6)
        target_noise = np.where(target_floor_mask, np.nan, target_noise)

    return (
        y_plot,
        y_noise,
        target_plot,
        target_noise,
        {
            "y_transform": "log10",
            "effective_floor_linear": float(effective_floor_linear),
            "min_log10_y": float(min_log10_y),
            "n_floored": n_floored,
            "n_clipped": n_clipped,
            "nonpositive_mask": y_nonpositive,
            "target_nonpositive_mask": target_nonpositive,
            "floor_mask": y_floor_mask,
            "target_floor_mask": target_floor_mask,
            "n_at_floor": int(np.count_nonzero(y_floor_mask)),
        },
    )


def _load_campaign(
    campaign_root: Path,
    *,
    analysis: str,
    hdf5_filename: str,
    min_log10_y: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, dict, dict]:
    samples = pd.read_csv(campaign_root / "samples.csv")
    quantile_columns = _quantile_columns(samples)
    x_train = samples[quantile_columns].to_numpy(dtype=float)

    attrs = _infer_analysis_attributes(campaign_root, analysis, hdf5_filename)
    x_dataset = attrs["xDataset"]
    y_dataset = attrs["yDataset"]
    target_dataset = attrs["yDatasetTarget"]
    covariance_dataset = attrs.get("yCovariance")
    x_is_log = bool(attrs.get("xAxisIsLog", False))
    y_is_log = bool(attrs.get("yAxisIsLog", False))

    x_bins: np.ndarray | None = None
    target_linear: np.ndarray | None = None
    target_noise_linear: np.ndarray | None = None
    y_rows = []
    y_noise_rows = []
    for sample in samples.itertuples(index=False):
        evaluation_id = sample.evaluation_id
        path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            current_x = np.asarray(group[x_dataset][...], dtype=float)
            current_target = np.asarray(group[target_dataset][...], dtype=float)
            current_y = np.asarray(group[y_dataset][...], dtype=float)
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
    return samples, x_train, x_plot, target_plot, target_noise_plot, y_plot, y_noise_plot, attrs, transform_metadata


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    analysis: str,
    x_bins_plot: np.ndarray,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    alpha_train: np.ndarray | None,
    fit_white_noise: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pred = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=float)
    pred_std = np.zeros_like(pred)
    kernels: list[str] = []
    for bin_index in range(y_train.shape[1]):
        print(
            f"{analysis}: fitting bin {bin_index + 1}/{y_train.shape[1]} "
            f"(x={x_bins_plot[bin_index]:.3f})"
        )
        model, y_mean, y_std = fit_scaled_gp(
            x_train,
            y_train[:, bin_index],
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=optimize_hyperparameters,
            alpha=alpha_train[:, bin_index] if alpha_train is not None else None,
            fit_white_noise=fit_white_noise,
        )
        pred[:, bin_index], pred_std[:, bin_index] = predict_scaled_gp(model, y_mean, y_std, x_test)
        kernels.append(str(model.kernel_))
        print(f"{analysis}: finished bin {bin_index + 1}/{y_train.shape[1]}")
    return pred, pred_std, kernels


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
    return _prepare_shared_training_targets(
        y_values,
        y_noise,
        analysis=analysis,
        use_training_alpha=use_training_alpha,
        bad_training_condition=bad_training_condition,
        bad_training_value_fill=bad_training_value_fill,
        bad_training_sigma=bad_training_sigma,
        min_training_sigma=min_training_sigma,
        bad_mask_override=bad_mask_override,
    )


def _supported_bin_mask(
    y_values: np.ndarray,
    y_noise: np.ndarray | None = None,
    *,
    bad_training_condition: str,
) -> np.ndarray:
    return _shared_supported_bin_mask(
        y_values,
        y_noise=y_noise,
        bad_training_condition=bad_training_condition,
    )


def _bad_training_mask(
    y_values: np.ndarray,
    y_noise: np.ndarray | None = None,
    *,
    bad_training_condition: str,
) -> np.ndarray:
    return _shared_bad_training_mask(
        y_values,
        y_noise=y_noise,
        bad_training_condition=bad_training_condition,
    )


def _bad_mask_override_for_transform(
    transform_metadata: dict,
    *,
    bad_training_condition: str,
    bad_training_value_fill="bin_median",
) -> np.ndarray | None:
    return training_bad_mask_override_from_transform(
        transform_metadata,
        bad_training_condition=bad_training_condition,
        bad_training_value_fill=bad_training_value_fill,
    )


def _json_safe_metadata(metadata: dict) -> dict:
    safe = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            safe[key] = {
                "shape": list(value.shape),
                "n_true": int(np.count_nonzero(value)) if value.dtype == bool else None,
            }
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def _plot_keeps_bad_values(bad_training_value_fill) -> bool:
    return str(bad_training_value_fill) in {"keep", "as_is"}


def _valid_observed_mask(
    observed: np.ndarray,
    *,
    bad_training_condition: str,
    bad_training_value_fill,
    observed_std: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(observed)
    if _plot_keeps_bad_values(bad_training_value_fill):
        return valid
    if bad_training_condition == "nonpositive":
        return valid & (observed > 0.0)
    if bad_training_condition == "zero_only":
        return valid & (observed != 0.0)
    if bad_training_condition == "zero_and_zero_noise":
        return valid & ~_shared_bad_training_mask(
            observed,
            y_noise=observed_std,
            bad_training_condition=bad_training_condition,
        )
    return valid


def _metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    valid = valid & np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_std)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return {
            "n_valid": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "coverage_1sigma": np.nan,
            "coverage_2sigma": np.nan,
        }
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]
    y_std_valid = y_std[valid]
    return {
        "n_valid": n_valid,
        "rmse": float(np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))),
        "mae": float(mean_absolute_error(y_true_valid, y_pred_valid)),
        "r2": float(r2_score(y_true_valid, y_pred_valid)) if n_valid > 1 else np.nan,
        "coverage_1sigma": float(np.mean(np.abs(y_pred_valid - y_true_valid) <= y_std_valid)),
        "coverage_2sigma": float(np.mean(np.abs(y_pred_valid - y_true_valid) <= 2.0 * y_std_valid)),
    }


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    x_bins_plot: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    for bin_index in range(y_true.shape[1]):
        valid = np.ones(y_true.shape[0], dtype=bool)
        if valid_mask is not None:
            valid &= valid_mask[:, bin_index]
        metric_values = _metric_values(
            y_true[:, bin_index],
            y_pred[:, bin_index],
            y_std[:, bin_index],
            valid,
        )
        rows.append(
            {
                "bin": int(bin_index),
                "x_bin_plot": float(x_bins_plot[bin_index]),
                **metric_values,
            }
        )
    all_valid = np.ones(y_true.shape, dtype=bool)
    if valid_mask is not None:
        all_valid &= valid_mask
    all_metric_values = _metric_values(y_true, y_pred, y_std, all_valid)
    rows.append(
        {
            "bin": "all",
            "x_bin_plot": np.nan,
            **all_metric_values,
        }
    )
    return pd.DataFrame(rows)


def _space_filling_examples(x_test: np.ndarray, n_examples: int) -> np.ndarray:
    if x_test.shape[0] <= n_examples:
        return np.arange(x_test.shape[0], dtype=int)
    order = np.argsort(x_test[:, 0])
    values = np.linspace(0, len(order) - 1, n_examples)
    return np.asarray(sorted({int(order[int(round(v))]) for v in values}), dtype=int)


def _metadata_for_analysis(analysis: str | None) -> dict:
    if analysis is None:
        return {}
    for observable_key, config in OBSERVABLE_CONFIGS.items():
        if str(config.get("analysis")) == str(analysis):
            return STANDARD_OBSERVABLE_PLOT_METADATA.get(observable_key, {})
    return {}


def _y_label(attrs: dict, transform_metadata: dict, analysis: str | None = None) -> str:
    metadata = _metadata_for_analysis(analysis)
    if metadata.get("y_axis_label"):
        return str(metadata["y_axis_label"])
    if transform_metadata["y_transform"] == "log10":
        return r"$\log_{10}(\mathrm{model\ output})$"
    return str(attrs.get("yAxisLabel", "output"))


def _x_label(attrs: dict, analysis: str | None = None) -> str:
    metadata = _metadata_for_analysis(analysis)
    if metadata.get("x_axis_label"):
        return str(metadata["x_axis_label"])
    return str(attrs.get("xAxisLabel", "x"))


def _title(attrs: dict, analysis: str) -> str:
    metadata = _metadata_for_analysis(analysis)
    target = str(metadata.get("target_label") or attrs.get("targetLabel", "target"))
    description = str(metadata.get("label") or attrs.get("description", analysis))
    description = (
        description.replace("$", "")
        .replace(r"\le", "<=")
        .replace(r"\ge", ">=")
    )
    return f"{target}: {description}"


def _plot_overlay(
    *,
    x_plot: np.ndarray,
    target_plot: np.ndarray,
    target_std_plot: np.ndarray | None,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_test_std: np.ndarray | None,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    example_indices: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    ymin: float | None,
    ymax: float | None,
    path: Path,
    bad_training_condition: str = "nonfinite",
    bad_training_value_fill="bin_median",
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    for curve in y_train:
        ax.plot(x_plot, curve, color="0.65", alpha=0.10, lw=1.0, zorder=1)

    sanitized_y_test_std = None
    if y_test_std is not None:
        finite_std = y_test_std[np.isfinite(y_test_std) & (y_test_std >= 0.0)]
        finite_y = np.concatenate(
            [
                y_train[np.isfinite(y_train)],
                y_test[np.isfinite(y_test)],
                target_plot[np.isfinite(target_plot)],
            ]
        )
        robust_span = float(
            np.percentile(finite_y, 95.0) - np.percentile(finite_y, 5.0)
        ) if finite_y.size else 1.0
        robust_span = max(robust_span, 1.0)
        typical_std = float(np.percentile(finite_std, 90.0)) if finite_std.size else 0.0
        absurd_threshold = max(10.0 * robust_span, 10.0 * typical_std, 1.0)
        sanitized_y_test_std = np.where(
            np.isfinite(y_test_std) & (y_test_std >= 0.0) & (y_test_std <= absurd_threshold),
            y_test_std,
            np.nan,
        )

    cmap = plt.get_cmap("tab10")
    x_spacing = np.diff(x_plot[np.isfinite(x_plot)])
    x_spacing = x_spacing[x_spacing > 0.0]
    offset_half_width = 0.15 * float(np.median(x_spacing)) if x_spacing.size else 0.0
    x_offsets = np.linspace(-offset_half_width, offset_half_width, len(example_indices)) if len(example_indices) else []
    for color_index, row_index in enumerate(example_indices):
        color = cmap(color_index % 10)
        lower = y_pred[row_index] - y_pred_std[row_index]
        upper = y_pred[row_index] + y_pred_std[row_index]
        ax.fill_between(x_plot, lower, upper, color=color, alpha=0.18, zorder=2)
        ax.plot(x_plot, y_pred[row_index], color=color, lw=2.0, zorder=3)

        observed = y_test[row_index]
        valid_observed = _valid_observed_mask(
            observed,
            bad_training_condition=bad_training_condition,
            bad_training_value_fill=bad_training_value_fill,
            observed_std=None if y_test_std is None else y_test_std[row_index],
        )

        if sanitized_y_test_std is None:
            x_observed = x_plot[valid_observed] + x_offsets[color_index]
            ax.scatter(x_observed, observed[valid_observed], color=color, s=18, alpha=0.95, zorder=5)
        else:
            observed_std = sanitized_y_test_std[row_index]
            x_observed = x_plot[valid_observed] + x_offsets[color_index]
            observed_valid = observed[valid_observed]
            observed_std_valid = observed_std[valid_observed]
            finite_std = np.isfinite(observed_std_valid)
            if np.any(finite_std):
                ax.errorbar(
                    x_observed[finite_std],
                    observed_valid[finite_std],
                    yerr=observed_std_valid[finite_std],
                    fmt="o",
                    ms=3.5,
                    lw=1.0,
                    capsize=2.0,
                    color=color,
                    alpha=0.95,
                    zorder=5,
                )
            if np.any(~finite_std):
                ax.scatter(
                    x_observed[~finite_std],
                    observed_valid[~finite_std],
                    color=color,
                    s=18,
                    alpha=0.95,
                    zorder=5,
                )

    ax.plot(x_plot, target_plot, color="k", lw=2.2, zorder=9)
    if target_std_plot is None:
        ax.scatter(x_plot, target_plot, color="k", s=22, zorder=10)
    else:
        ax.errorbar(
            x_plot,
            target_plot,
            yerr=target_std_plot,
            fmt="o",
            color="k",
            ms=4.3,
            lw=1.3,
            capsize=2.5,
            zorder=10,
        )

    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_saved_predictions(predictions_path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    predictions = pd.read_csv(predictions_path)
    x_cols = sorted(
        [column for column in predictions.columns if column.endswith("_x_plot")],
        key=lambda column: int(column.split("_")[0].replace("bin", "")),
    )
    x_plot = predictions.loc[0, x_cols].to_numpy(dtype=float)

    def _matrix(suffix: str) -> np.ndarray:
        cols = sorted(
            [column for column in predictions.columns if column.endswith(suffix)],
            key=lambda column: int(column.split("_")[0].replace("bin", "")),
        )
        return predictions[cols].to_numpy(dtype=float)

    y_true = _matrix("_true")
    y_true_std = _matrix("_true_std")
    if np.all(np.isnan(y_true_std)):
        y_true_std = None
    y_pred = _matrix("_pred")
    y_pred_std = _matrix("_pred_std")
    example_indices = np.flatnonzero(predictions["is_example_curve"].to_numpy(dtype=bool))
    return x_plot, y_true_std, y_true, y_pred, y_pred_std, example_indices


def _plot_parity(
    *,
    x_bins_plot: np.ndarray,
    y_test: np.ndarray,
    y_test_std: np.ndarray | None,
    y_pred: np.ndarray,
    y_pred_std: np.ndarray,
    path: Path,
    target_plot: np.ndarray | None = None,
    bad_training_condition: str = "nonfinite",
    bad_training_value_fill="bin_median",
) -> None:
    n_bins = y_test.shape[1]
    ncols = min(4, n_bins)
    nrows = int(np.ceil(n_bins / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for bin_index in range(n_bins):
        axis = axes_flat[bin_index]
        observed = y_test[:, bin_index]
        predicted = y_pred[:, bin_index]
        valid = _valid_observed_mask(
            observed,
            bad_training_condition=bad_training_condition,
            bad_training_value_fill=bad_training_value_fill,
            observed_std=None if y_test_std is None else y_test_std[:, bin_index],
        ) & np.isfinite(predicted)

        if not np.any(valid):
            axis.set_title(f"x={x_bins_plot[bin_index]:.2f}")
            axis.text(0.5, 0.5, "no valid held-out data", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue

        observed_valid = observed[valid]
        predicted_valid = predicted[valid]
        predicted_std_valid = y_pred_std[valid, bin_index]
        target_value = (
            float(target_plot[bin_index])
            if target_plot is not None and bin_index < len(target_plot) and np.isfinite(target_plot[bin_index])
            else None
        )
        lower_candidates = [float(observed_valid.min()), float(predicted_valid.min())]
        upper_candidates = [float(observed_valid.max()), float(predicted_valid.max())]
        if target_value is not None:
            lower_candidates.append(target_value)
            upper_candidates.append(target_value)
        lower = min(lower_candidates)
        upper = max(upper_candidates)
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(observed_valid, predicted_valid, yerr=predicted_std_valid, fmt="o", ms=3.0, alpha=0.75)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        if target_value is not None:
            axis.scatter(
                [target_value],
                [target_value],
                marker="*",
                s=110,
                color="k",
                edgecolor="white",
                linewidth=0.45,
                zorder=8,
            )
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(f"x={x_bins_plot[bin_index]:.2f}")
        axis.set_xlabel("Held-out Galacticus")
        axis.set_ylabel("GP prediction")
        axis.grid(alpha=0.2)

    for axis in axes_flat[n_bins:]:
        axis.axis("off")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    analysis = _setting(args, "analysis")
    if analysis is None:
        raise ValueError("Specify --preset or --analysis")
    output_prefix = _setting(args, "output-prefix", f"{analysis}_holdout_demo")
    figures_dir = campaign_root / _setting(args, "figures-dir-name", f"figures_{output_prefix}")
    emulator_dir = campaign_root / _setting(args, "emulator-dir-name", f"emulator_{output_prefix}")
    overlay_ymin = _setting(args, "overlay-ymin", None)
    overlay_ymax = _setting(args, "overlay-ymax", None)
    use_training_alpha = _setting(args, "use-training-alpha", True)
    bad_training_condition = _setting(args, "bad_training_condition", "nonfinite")
    bad_training_value_fill = _setting(args, "bad_training_value_fill", "bin_median")
    bad_training_sigma = _setting(args, "bad_training_sigma", 5.0)
    min_training_sigma = _setting(args, "min_training_sigma", 1.0e-3)
    drop_unsupported_bins = bool(_setting(args, "drop_unsupported_bins", False))
    figures_dir.mkdir(parents=True, exist_ok=True)
    emulator_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = emulator_dir / f"{output_prefix}_metrics.csv"
    predictions_path = emulator_dir / f"{output_prefix}_predictions.csv"
    summary_path = emulator_dir / f"{output_prefix}_summary.json"
    overlay_path = figures_dir / f"{output_prefix}_heldout_curves.png"
    parity_path = figures_dir / f"{output_prefix}_parity.png"
    expected_outputs = [metrics_path, predictions_path, summary_path, overlay_path, parity_path]
    if args.skip_existing and all(path.exists() for path in expected_outputs):
        print(f"{analysis}: all outputs already exist; skipping.")
        for path in expected_outputs:
            print(path)
        return

    print(f"{analysis}: loading campaign data from {campaign_root}")
    (
        samples,
        x_train_all,
        x_bins_plot,
        target_plot,
        target_std_plot,
        y_plot_all,
        y_std_all,
        attrs,
        transform_metadata,
    ) = _load_campaign(
        campaign_root,
        analysis=analysis,
        hdf5_filename=args.hdf5_filename,
        min_log10_y=args.min_log10_y,
    )
    supported_bin_mask = np.ones(y_plot_all.shape[1], dtype=bool)
    bad_mask_override = _bad_mask_override_for_transform(
        transform_metadata,
        bad_training_condition=str(bad_training_condition),
        bad_training_value_fill=bad_training_value_fill,
    )
    if drop_unsupported_bins:
        if bad_mask_override is not None:
            supported_bin_mask = np.any(~bad_mask_override, axis=0)
        else:
            supported_bin_mask = _supported_bin_mask(
                y_plot_all,
                y_noise=y_std_all,
                bad_training_condition=str(bad_training_condition),
            )
        if not np.any(supported_bin_mask):
            raise ValueError("All bins are unsupported after applying the bad-training mask.")
        x_bins_plot = x_bins_plot[supported_bin_mask]
        target_plot = target_plot[supported_bin_mask]
        if target_std_plot is not None:
            target_std_plot = _select_bins(target_std_plot, supported_bin_mask)
        y_plot_all = y_plot_all[:, supported_bin_mask]
        if y_std_all is not None:
            y_std_all = y_std_all[:, supported_bin_mask]
    if bad_mask_override is not None:
        bad_mask_override = bad_mask_override[:, supported_bin_mask]
    bad_metric_mask_all = (
        bad_mask_override
        if bad_mask_override is not None
        else _bad_training_mask(
            y_plot_all,
            y_noise=y_std_all,
            bad_training_condition=str(bad_training_condition),
        )
    )
    y_fit_all, alpha_sigma_all, training_target_metadata = _prepare_training_targets(
        y_plot_all,
        y_std_all,
        analysis=analysis,
        use_training_alpha=bool(use_training_alpha),
        bad_training_condition=str(bad_training_condition),
        bad_training_value_fill=bad_training_value_fill,
        bad_training_sigma=float(bad_training_sigma),
        min_training_sigma=float(min_training_sigma),
        bad_mask_override=bad_mask_override,
    )
    alpha_all = alpha_sigma_all**2 if alpha_sigma_all is not None else None

    if args.replot_only:
        summary = json.loads(summary_path.read_text())
        prediction_x_plot, saved_true_std, y_test, y_pred, y_pred_std, example_indices = _load_saved_predictions(predictions_path)
        if not np.allclose(prediction_x_plot, x_bins_plot, equal_nan=True):
            raise ValueError("Saved prediction x-bins do not match campaign data")
        train_id_set = set(summary["train_ids"])
        train_mask = samples["evaluation_id"].isin(train_id_set).to_numpy()
        y_train = y_fit_all[train_mask]
        _plot_overlay(
            x_plot=x_bins_plot,
            target_plot=target_plot,
            target_std_plot=target_std_plot,
            y_train=y_train,
            y_test=y_test,
            y_test_std=saved_true_std,
            y_pred=y_pred,
            y_pred_std=y_pred_std,
            example_indices=example_indices,
            x_label=_x_label(attrs, analysis),
            y_label=_y_label(attrs, transform_metadata, analysis),
            title=_title(attrs, analysis),
            ymin=overlay_ymin,
            ymax=overlay_ymax,
            path=overlay_path,
            bad_training_condition=str(bad_training_condition),
            bad_training_value_fill=bad_training_value_fill,
        )
        _plot_parity(
            x_bins_plot=x_bins_plot,
            target_plot=target_plot,
            y_test=y_test,
            y_test_std=y_test_std,
            y_pred=y_pred,
            y_pred_std=y_pred_std,
            path=parity_path,
            bad_training_condition=str(bad_training_condition),
            bad_training_value_fill=bad_training_value_fill,
        )
        print(overlay_path)
        print(parity_path)
        return

    n_samples = x_train_all.shape[0]
    n_splits = int(round(1.0 / (1.0 - args.train_fraction)))
    if n_splits < 2:
        raise ValueError("--train-fraction must be less than 1")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    train_index, test_index = next(splitter.split(x_train_all))

    print(
        f"{analysis}: train {len(train_index)} evals / test {len(test_index)} evals "
        f"({len(train_index)/n_samples:.3f}/{len(test_index)/n_samples:.3f})"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        y_pred, y_pred_std, kernels = _fit_predict(
            x_train_all[train_index],
            y_fit_all[train_index],
            x_train_all[test_index],
            analysis=analysis,
            x_bins_plot=x_bins_plot,
            n_restarts_optimizer=args.n_restarts_optimizer,
            optimize_hyperparameters=args.optimize_hyperparameters,
            alpha_train=alpha_all[train_index] if alpha_all is not None else None,
            fit_white_noise=args.fit_white_noise,
        )

    metrics = _metrics(
        y_plot_all[test_index],
        y_pred,
        y_pred_std,
        x_bins_plot,
        valid_mask=~bad_metric_mask_all[test_index],
    )
    metrics.to_csv(metrics_path, index=False)

    example_indices = _space_filling_examples(x_train_all[test_index], args.n_example_curves)

    _plot_overlay(
        x_plot=x_bins_plot,
        target_plot=target_plot,
        target_std_plot=target_std_plot,
        y_train=y_fit_all[train_index],
        y_test=y_plot_all[test_index],
        y_test_std=y_std_all[test_index] if y_std_all is not None else None,
        y_pred=y_pred,
        y_pred_std=y_pred_std,
        example_indices=example_indices,
        x_label=_x_label(attrs, analysis),
        y_label=_y_label(attrs, transform_metadata, analysis),
        title=_title(attrs, analysis),
        ymin=overlay_ymin,
        ymax=overlay_ymax,
        path=overlay_path,
        bad_training_condition=str(bad_training_condition),
        bad_training_value_fill=bad_training_value_fill,
    )

    _plot_parity(
        x_bins_plot=x_bins_plot,
        target_plot=target_plot,
        y_test=y_plot_all[test_index],
        y_test_std=y_std_all[test_index] if y_std_all is not None else None,
        y_pred=y_pred,
        y_pred_std=y_pred_std,
        path=parity_path,
        bad_training_condition=str(bad_training_condition),
        bad_training_value_fill=bad_training_value_fill,
    )

    prediction_rows = []
    for local_row, global_row in enumerate(test_index):
        row = {
            "evaluation_id": samples.iloc[global_row]["evaluation_id"],
            "is_example_curve": bool(local_row in set(example_indices.tolist())),
        }
        for bin_index, x_value in enumerate(x_bins_plot):
            prefix = f"bin{bin_index}"
            row[f"{prefix}_x_plot"] = float(x_value)
            row[f"{prefix}_true"] = float(y_plot_all[global_row, bin_index])
            row[f"{prefix}_true_std"] = float(y_std_all[global_row, bin_index]) if y_std_all is not None else np.nan
            row[f"{prefix}_true_missing"] = bool(bad_metric_mask_all[global_row, bin_index])
            row[f"{prefix}_pred"] = float(y_pred[local_row, bin_index])
            row[f"{prefix}_pred_std"] = float(y_pred_std[local_row, bin_index])
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    summary = {
        "campaign_root": str(campaign_root),
        "analysis": analysis,
        "n_total_evaluations": int(n_samples),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "train_fraction": float(len(train_index) / n_samples),
        "test_fraction": float(len(test_index) / n_samples),
        "train_ids": samples.iloc[train_index]["evaluation_id"].tolist(),
        "test_ids": samples.iloc[test_index]["evaluation_id"].tolist(),
        "example_curve_ids": samples.iloc[test_index[example_indices]]["evaluation_id"].tolist(),
        "quantile_columns": _quantile_columns(samples),
        "kernels": kernels,
        "fit_white_noise": bool(args.fit_white_noise),
        "analysis_attrs": attrs,
        "transform_metadata": _json_safe_metadata(transform_metadata),
        "training_target_metadata": training_target_metadata,
        "supported_bin_mask": supported_bin_mask.tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(overlay_path)
    print(parity_path)


if __name__ == "__main__":
    main()
