from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from fit_sidecar_lf_pca_holdout_demo import (
    TARGET_ERROR_MODE_LEGACY,
    TARGET_ERROR_MODES,
    _alpha_log10,
    _bin_metadata,
    _input_columns as _sidecar_input_columns,
    _log10_with_floor as _sidecar_log10_with_floor,
    _output_columns as _sidecar_output_columns,
    _read_first_sidecar_table,
    _read_sidecar_tables,
    _sample_labels,
)
from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp
from galacticus_emu.interactive_observables import (
    OBSERVABLE_CONFIGS,
    _input_columns_for_samples,
    _load_observable_campaign,
    _make_preprocessor,
    _parameter_specs_for_columns,
    _prepare_training_targets,
    _supported_bin_mask,
    training_bad_mask_override_from_transform,
)
from run_sidecar_lf_pca_emulator_mcmc import (
    _feature_matrix_from_table,
    _filter_table,
    _parameter_specs_for_table,
)
from run_standard_observable_training_size_cv import (
    _metric_rows,
    _training_sigma_rms,
)


FINAL_STANDARD_OBSERVABLES = [
    "smf_z0",
    "smf_z3",
    "sfr_function_robotham2011",
    "size_mass_vdw2014_star_forming_z0",
    "size_mass_vdw2014_quiescent_z0",
    "bh_velocity_dispersion",
    "mzr_blanc2019",
]

FINAL_SIDECAR_SAMPLE_LABELS = ["Z1", "Z2", "Z3", "Z4"]

DISPLAY_LABELS = {
    "smf_z0": ("Stellar mass function", "0.20<z<0.50"),
    "smf_z3": ("Stellar mass function", "1.00<z<1.25"),
    "sfr_function_robotham2011": ("Star-formation-rate function", "0.013<z<0.1"),
    "size_mass_vdw2014_star_forming_z0": ("Size-mass relation (star-forming)", "0<z<0.5"),
    "size_mass_vdw2014_quiescent_z0": ("Size-mass relation (quiescent)", "0<z<0.5"),
    "bh_velocity_dispersion": ("Black-hole-velocity-dispersion relation", "z~0"),
    "mzr_blanc2019": ("Gas-phase mass-metallicity relation", "z~0"),
    "morphological_fraction_gama_moffett2016": ("Early-type fraction", "GAMA"),
    "halpha_sobral_z1": ("H-alpha luminosity function", "z=0.40"),
    "halpha_sobral_z2": ("H-alpha luminosity function", "z=0.84"),
    "halpha_sobral_z3": ("H-alpha luminosity function", "z=1.47"),
    "halpha_sobral_z4": ("H-alpha luminosity function", "z=2.23"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 99 percent PCA-GP emulators with independent bin-by-bin GP "
            "emulators on the final calibration observables, using matched CV folds."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--hdf5-filename", default="galacticus.hdf5")
    parser.add_argument("--standard-observable", action="append", default=None)
    parser.add_argument("--sidecar-sample-label", action="append", default=None)
    parser.add_argument("--no-standard", action="store_true")
    parser.add_argument("--no-sidecar", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=12345)
    parser.add_argument("--min-log10-y", type=float, default=-7.0)
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="standardized")
    parser.add_argument("--pca-variance-threshold", type=float, default=0.99)
    parser.add_argument("--standard-n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--sidecar-n-restarts-optimizer", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--optimize-hyperparameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-white-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sidecar-dir", default="emission_line_dust")
    parser.add_argument("--sidecar-table-filename", default="emission_line_dust_lf_emulator_table.csv")
    parser.add_argument("--sidecar-long-filename", default="emission_line_dust_lf_long.csv")
    parser.add_argument("--sidecar-observable", default="halpha_sobral")
    parser.add_argument("--dust-draw-index", type=int, action="append", default=None)
    parser.add_argument("--max-dust-draws-per-eval", type=int, default=None)
    parser.add_argument("--sidecar-min-log10-lf", type=float, default=-8.0)
    parser.add_argument(
        "--target-error-mode",
        choices=TARGET_ERROR_MODES,
        default=TARGET_ERROR_MODE_LEGACY,
    )
    parser.add_argument("--use-training-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-training-log10-sigma", type=float, default=1.0e-4)
    parser.add_argument("--max-training-log10-sigma", type=float, default=2.0)
    parser.add_argument("--write-predictions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _folds_for_rows(n_rows: int, n_folds: int, random_state: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    indices = np.arange(n_rows, dtype=int)
    splitter = KFold(n_splits=min(n_folds, n_rows), shuffle=True, random_state=random_state)
    return [
        (fold_index, indices[train_local], indices[test_local])
        for fold_index, (train_local, test_local) in enumerate(splitter.split(indices), start=1)
    ]


def _group_kfold_indices(groups: np.ndarray, n_folds: int, random_state: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    unique_groups = np.asarray(sorted(pd.unique(groups)), dtype=object)
    if n_folds > unique_groups.size:
        raise ValueError(f"--n-folds={n_folds} exceeds the number of groups ({unique_groups.size}).")
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_index, (train_group_index, test_group_index) in enumerate(splitter.split(unique_groups), start=1):
        train_groups = set(unique_groups[train_group_index])
        test_groups = set(unique_groups[test_group_index])
        train_index = np.flatnonzero(np.asarray([group in train_groups for group in groups], dtype=bool))
        test_index = np.flatnonzero(np.asarray([group in test_groups for group in groups], dtype=bool))
        folds.append((fold_index, train_index, test_index))
    return folds


def _choose_components(cumulative: np.ndarray, threshold: float) -> int:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("--pca-variance-threshold must be in (0, 1]")
    return int(np.searchsorted(cumulative, threshold) + 1)


def _fit_predict_one_gp(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha_train: np.ndarray | None,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    fit_white_noise: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    model, y_mean, y_std = fit_scaled_gp(
        x_train,
        y_train,
        n_restarts_optimizer=n_restarts_optimizer,
        optimize_hyperparameters=optimize_hyperparameters,
        alpha=alpha_train,
        fit_white_noise=fit_white_noise,
    )
    pred, pred_std = predict_scaled_gp(model, y_mean, y_std, x_test)
    return pred, pred_std, str(model.kernel_)


def _predict_pca_gp_parallel(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha_train: np.ndarray | None,
    pca_scaling: str,
    pca_variance_threshold: float,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    fit_white_noise: bool,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scaler = _make_preprocessor(pca_scaling)
    y_scaled = scaler.fit_transform(y_train)
    max_components = min(y_scaled.shape)
    pca = PCA(n_components=max_components)
    coefficients = pca.fit_transform(y_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = min(_choose_components(cumulative, pca_variance_threshold), max_components)

    coefficient_alpha = None
    if alpha_train is not None:
        alpha_scaled = np.asarray(alpha_train, dtype=float) / (scaler.scale_[None, :] ** 2)
        coefficient_alpha = alpha_scaled @ (pca.components_.T ** 2)
        coefficient_alpha = np.maximum(coefficient_alpha, 1.0e-12)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_predict_one_gp)(
            x_train=x_train,
            y_train=coefficients[:, component_index],
            x_test=x_test,
            alpha_train=(
                coefficient_alpha[:, component_index]
                if coefficient_alpha is not None
                else None
            ),
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=optimize_hyperparameters,
            fit_white_noise=fit_white_noise,
        )
        for component_index in range(n_components)
    )
    coefficient_predictions = np.column_stack([result[0] for result in results])
    coefficient_stds = np.column_stack([result[1] for result in results])
    components = pca.components_[:n_components]
    y_pred_scaled = coefficient_predictions @ components + pca.mean_[None, :]
    y_pred = y_pred_scaled * scaler.scale_[None, :] + scaler.mean_[None, :]
    y_var_scaled = (coefficient_stds**2) @ (components**2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * scaler.scale_[None, :]
    return y_pred, y_std, {
        "pca_components": int(n_components),
        "pca_explained_variance": float(cumulative[n_components - 1]),
        "kernels": [result[2] for result in results],
    }


def _predict_bin_by_bin_gp_parallel(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha_train: np.ndarray | None,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    fit_white_noise: bool,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_predict_one_gp)(
            x_train=x_train,
            y_train=y_train[:, bin_index],
            x_test=x_test,
            alpha_train=alpha_train[:, bin_index] if alpha_train is not None else None,
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=optimize_hyperparameters,
            fit_white_noise=fit_white_noise,
        )
        for bin_index in range(y_train.shape[1])
    )
    y_pred = np.column_stack([result[0] for result in results])
    y_std = np.column_stack([result[1] for result in results])
    return y_pred, y_std, {
        "pca_components": np.nan,
        "pca_explained_variance": np.nan,
        "kernels": [result[2] for result in results],
    }


def _method_metadata_rows(
    metric_rows: list[dict[str, object]],
    *,
    component_counts: list[float],
    explained_variances: list[float],
    n_gp_fit_total: int,
    n_bins: int,
    family: str,
    observable_key: str,
    sample_label: str,
    training_bad_points: int | None = None,
) -> list[dict[str, object]]:
    component_array = np.asarray(component_counts, dtype=float)
    variance_array = np.asarray(explained_variances, dtype=float)
    label, redshift = DISPLAY_LABELS.get(observable_key, (observable_key, sample_label))
    for row in metric_rows:
        row["family"] = family
        row["sample_label"] = sample_label
        row["observable_label"] = label
        row["redshift_label"] = redshift
        row["n_bins"] = int(n_bins)
        row["n_gp_fit_total"] = int(n_gp_fit_total)
        row["pca_components_mean"] = (
            float(np.nanmean(component_array)) if np.any(np.isfinite(component_array)) else np.nan
        )
        row["pca_components_min"] = (
            float(np.nanmin(component_array)) if np.any(np.isfinite(component_array)) else np.nan
        )
        row["pca_components_max"] = (
            float(np.nanmax(component_array)) if np.any(np.isfinite(component_array)) else np.nan
        )
        row["pca_explained_variance_mean"] = (
            float(np.nanmean(variance_array)) if np.any(np.isfinite(variance_array)) else np.nan
        )
        row["training_bad_points"] = np.nan if training_bad_points is None else int(training_bad_points)
    return metric_rows


def _standard_prediction_rows(
    *,
    observable_key: str,
    method: str,
    x_plot: np.ndarray,
    fold_predictions: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_index, test_index, y_true, y_pred, y_std, heldout_sigma in fold_predictions:
        for local_index, sample_index in enumerate(test_index):
            for bin_index, x_value in enumerate(x_plot):
                rows.append(
                    {
                        "family": "standard",
                        "observable_key": observable_key,
                        "sample_label": "",
                        "emulator_type": method,
                        "fold_index": int(fold_index),
                        "sample_index": int(sample_index),
                        "bin": int(bin_index),
                        "x_plot": float(x_value),
                        "y_true": float(y_true[local_index, bin_index]),
                        "y_pred": float(y_pred[local_index, bin_index]),
                        "y_pred_std": float(y_std[local_index, bin_index]),
                        "heldout_sigma": (
                            np.nan
                            if heldout_sigma is None
                            else float(heldout_sigma[local_index, bin_index])
                        ),
                    }
                )
    return rows


def _sidecar_prediction_rows(
    *,
    table: pd.DataFrame,
    observable_key: str,
    sample_label: str,
    method: str,
    output_columns: list[str],
    x_plot: np.ndarray,
    fold_by_row: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    heldout_sigma: np.ndarray | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index, source in table.reset_index(drop=True).iterrows():
        base = {
            "family": "sidecar",
            "observable_key": observable_key,
            "sample_label": sample_label,
            "emulator_type": method,
            "fold_index": int(fold_by_row[row_index]),
            "evaluation_id": source["evaluation_id"],
            "dust_draw_index": int(source["dust_draw_index"]),
        }
        for bin_index, column in enumerate(output_columns):
            rows.append(
                {
                    **base,
                    "bin": int(bin_index),
                    "x_plot": float(x_plot[bin_index]),
                    "column": column,
                    "y_true": float(y_true[row_index, bin_index]),
                    "y_pred": float(y_pred[row_index, bin_index]),
                    "y_pred_std": float(y_std[row_index, bin_index]),
                    "heldout_sigma": (
                        np.nan
                        if heldout_sigma is None
                        else float(heldout_sigma[row_index, bin_index])
                    ),
                }
            )
    return rows


def _heldout_sigma_from_sidecar_log_columns(
    table: pd.DataFrame,
    output_columns: list[str],
    *,
    min_sigma: float,
    max_sigma: float,
) -> np.ndarray | None:
    sigma_columns = [
        column.removesuffix("_phi_mpc3_dex") + "_shot_noise_log10_std_conservative"
        for column in output_columns
    ]
    if not all(column in table.columns for column in sigma_columns):
        return None
    sigma = table[sigma_columns].to_numpy(dtype=float)
    sigma = np.where(np.isfinite(sigma), sigma, max_sigma)
    return np.clip(sigma, min_sigma, max_sigma)


def run_standard_observable(
    *,
    campaign_root: Path,
    observable_key: str,
    args: argparse.Namespace,
    samples: pd.DataFrame,
    input_columns: list[str],
    input_parameter_specs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = OBSERVABLE_CONFIGS[observable_key]
    (
        _samples,
        x_all,
        x_plot,
        _target_plot,
        _target_noise_plot,
        y_plot,
        y_noise,
        _attrs,
        transform_metadata,
    ) = _load_observable_campaign(
        campaign_root,
        analysis=config["analysis"],
        hdf5_filename=args.hdf5_filename,
        min_log10_y=args.min_log10_y,
        input_columns=input_columns,
        input_parameter_specs=input_parameter_specs,
    )

    supported_bin_mask = np.ones(y_plot.shape[1], dtype=bool)
    if bool(config.get("drop_unsupported_bins", False)):
        supported_bin_mask = _supported_bin_mask(
            y_plot,
            y_noise=y_noise,
            bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
        )
        x_plot = x_plot[supported_bin_mask]
        y_plot = y_plot[:, supported_bin_mask]
        if y_noise is not None:
            y_noise = y_noise[:, supported_bin_mask]

    training_bad_mask_all = training_bad_mask_override_from_transform(
        transform_metadata,
        bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
        bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
    )
    if training_bad_mask_all is not None and bool(config.get("drop_unsupported_bins", False)):
        training_bad_mask_all = training_bad_mask_all[:, supported_bin_mask]

    metric_bad_mask_all = None
    if (
        training_bad_mask_all is not None
        and str(config.get("bad_training_value_fill", "")) not in {"keep", "as_is"}
    ):
        metric_bad_mask_all = np.asarray(training_bad_mask_all, dtype=bool)

    y_fit, alpha_sigma, training_metadata = _prepare_training_targets(
        y_plot,
        y_noise,
        analysis=config["analysis"],
        use_training_alpha=bool(config.get("use_training_alpha", True)),
        bad_training_condition=str(config.get("bad_training_condition", "nonfinite")),
        bad_training_value_fill=config.get("bad_training_value_fill", "bin_median"),
        bad_training_sigma=float(config.get("bad_training_sigma", 5.0)),
        min_training_sigma=float(config.get("min_training_sigma", 1.0e-3)),
        bad_mask_override=training_bad_mask_all,
    )
    alpha = alpha_sigma**2 if alpha_sigma is not None else None
    folds = _folds_for_rows(x_all.shape[0], args.n_folds, args.random_state)
    methods = ["pca", "bin_by_bin"]
    method_blocks = {
        method: {"pred": [], "std": [], "component_counts": [], "explained_variances": [], "pred_rows": []}
        for method in methods
    }
    true_blocks = []
    heldout_sigma_blocks = []
    bad_mask_blocks = []
    n_train_values = []
    train_indices_by_split = []

    for fold_index, train_index, test_index in folds:
        print(
            f"{observable_key}: fold {fold_index}/{len(folds)}, "
            f"train={len(train_index)}, test={len(test_index)}",
            flush=True,
        )
        true_block = y_plot[test_index]
        true_blocks.append(true_block)
        if alpha_sigma is not None:
            heldout_sigma_blocks.append(alpha_sigma[test_index])
        if metric_bad_mask_all is not None:
            bad_mask_blocks.append(metric_bad_mask_all[test_index])
        n_train_values.append(int(len(train_index)))
        train_indices_by_split.append(train_index)

        pca_pred, pca_std, pca_metadata = _predict_pca_gp_parallel(
            x_train=x_all[train_index],
            y_train=y_fit[train_index],
            x_test=x_all[test_index],
            alpha_train=alpha[train_index] if alpha is not None else None,
            pca_scaling=args.pca_scaling,
            pca_variance_threshold=args.pca_variance_threshold,
            n_restarts_optimizer=args.standard_n_restarts_optimizer,
            optimize_hyperparameters=args.optimize_hyperparameters,
            fit_white_noise=args.fit_white_noise,
            n_jobs=args.n_jobs,
        )
        bin_pred, bin_std, _bin_fit_metadata = _predict_bin_by_bin_gp_parallel(
            x_train=x_all[train_index],
            y_train=y_fit[train_index],
            x_test=x_all[test_index],
            alpha_train=alpha[train_index] if alpha is not None else None,
            n_restarts_optimizer=args.standard_n_restarts_optimizer,
            optimize_hyperparameters=args.optimize_hyperparameters,
            fit_white_noise=args.fit_white_noise,
            n_jobs=args.n_jobs,
        )
        fold_heldout_sigma = alpha_sigma[test_index] if alpha_sigma is not None else None
        for method, pred, std, metadata in [
            ("pca", pca_pred, pca_std, pca_metadata),
            ("bin_by_bin", bin_pred, bin_std, {"pca_components": np.nan, "pca_explained_variance": np.nan}),
        ]:
            method_blocks[method]["pred"].append(pred)
            method_blocks[method]["std"].append(std)
            method_blocks[method]["component_counts"].append(metadata["pca_components"])
            method_blocks[method]["explained_variances"].append(metadata["pca_explained_variance"])
            if args.write_predictions:
                method_blocks[method]["pred_rows"].extend(
                    _standard_prediction_rows(
                        observable_key=observable_key,
                        method=method,
                        x_plot=x_plot,
                        fold_predictions=[
                            (fold_index, test_index, true_block, pred, std, fold_heldout_sigma)
                        ],
                    )
                )

    y_true_stack = np.vstack(true_blocks)
    heldout_sigma_stack = np.vstack(heldout_sigma_blocks) if heldout_sigma_blocks else None
    bad_mask_stack = np.vstack(bad_mask_blocks) if bad_mask_blocks else None
    training_sigma_rms_by_bin, training_sigma_rms_all = _training_sigma_rms(alpha_sigma, train_indices_by_split)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for method in methods:
        rows = _metric_rows(
            observable_key=observable_key,
            emulator_type=method,
            subset_size=int(x_all.shape[0]),
            validation_mode="kfold",
            n_train_values=n_train_values,
            x_plot=x_plot,
            y_true=y_true_stack,
            y_pred=np.vstack(method_blocks[method]["pred"]),
            y_std=np.vstack(method_blocks[method]["std"]),
            heldout_sigma=heldout_sigma_stack,
            uncertainty_denominator="emulator_plus_heldout",
            bad_training_condition=(
                "nonfinite"
                if str(config.get("bad_training_value_fill", "")) in {"keep", "as_is"}
                else str(config.get("bad_training_condition", "nonfinite"))
            ),
            bad_mask=bad_mask_stack,
            training_sigma_rms_by_bin=training_sigma_rms_by_bin,
            training_sigma_rms_all=training_sigma_rms_all,
        )
        n_gp_fit_total = (
            int(np.nansum(method_blocks[method]["component_counts"]))
            if method == "pca"
            else int(len(folds) * y_plot.shape[1])
        )
        metric_rows.extend(
            _method_metadata_rows(
                rows,
                component_counts=method_blocks[method]["component_counts"],
                explained_variances=method_blocks[method]["explained_variances"],
                n_gp_fit_total=n_gp_fit_total,
                n_bins=y_plot.shape[1],
                family="standard",
                observable_key=observable_key,
                sample_label="",
                training_bad_points=int(training_metadata["n_bad_training_points"]),
            )
        )
        prediction_rows.extend(method_blocks[method]["pred_rows"])

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def run_sidecar_observable(
    *,
    campaign_root: Path,
    requested_sample_label: str,
    table: pd.DataFrame,
    long_table: pd.DataFrame,
    x: np.ndarray,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_columns = _sidecar_output_columns(table, args.sidecar_observable, [requested_sample_label])
    sample_labels = _sample_labels(output_columns)
    if not sample_labels or any(label != sample_labels[0] for label in sample_labels):
        raise ValueError(f"Could not infer a single sample label for {requested_sample_label!r}")
    sample_label = sample_labels[0]
    observable_key = f"{args.sidecar_observable}_{sample_label}"
    bin_metadata = _bin_metadata(long_table, args.sidecar_observable, sample_label, output_columns)
    x_plot = bin_metadata["log10_luminosity_center"].to_numpy(dtype=float)
    y, _floor = _sidecar_log10_with_floor(
        table[output_columns].to_numpy(dtype=float),
        args.sidecar_min_log10_lf,
    )
    alpha_sigma = None
    alpha = None
    if args.use_training_alpha:
        alpha_sigma = _alpha_log10(
            table,
            output_columns,
            min_sigma=args.min_training_log10_sigma,
            max_sigma=args.max_training_log10_sigma,
        )
        alpha = alpha_sigma**2
    heldout_sigma = _heldout_sigma_from_sidecar_log_columns(
        table,
        output_columns,
        min_sigma=args.min_training_log10_sigma,
        max_sigma=args.max_training_log10_sigma,
    )
    if heldout_sigma is None:
        heldout_sigma = alpha_sigma

    methods = ["pca", "bin_by_bin"]
    method_blocks = {
        method: {"pred": np.full_like(y, np.nan), "std": np.full_like(y, np.nan), "pred_rows": []}
        for method in methods
    }
    component_counts = {method: [] for method in methods}
    explained_variances = {method: [] for method in methods}
    n_train_values = []
    train_indices_by_split = []
    fold_by_row = np.zeros(len(table), dtype=int)

    for fold_index, train_index, test_index in folds:
        print(
            f"{observable_key}: fold {fold_index}/{len(folds)}, "
            f"train={len(train_index)}, test={len(test_index)}",
            flush=True,
        )
        pca_pred, pca_std, pca_metadata = _predict_pca_gp_parallel(
            x_train=x[train_index],
            y_train=y[train_index],
            x_test=x[test_index],
            alpha_train=alpha[train_index] if alpha is not None else None,
            pca_scaling="standardized",
            pca_variance_threshold=args.pca_variance_threshold,
            n_restarts_optimizer=args.sidecar_n_restarts_optimizer,
            optimize_hyperparameters=args.optimize_hyperparameters,
            fit_white_noise=args.fit_white_noise,
            n_jobs=args.n_jobs,
        )
        bin_pred, bin_std, _bin_fit_metadata = _predict_bin_by_bin_gp_parallel(
            x_train=x[train_index],
            y_train=y[train_index],
            x_test=x[test_index],
            alpha_train=alpha[train_index] if alpha is not None else None,
            n_restarts_optimizer=args.sidecar_n_restarts_optimizer,
            optimize_hyperparameters=args.optimize_hyperparameters,
            fit_white_noise=args.fit_white_noise,
            n_jobs=args.n_jobs,
        )
        method_blocks["pca"]["pred"][test_index] = pca_pred
        method_blocks["pca"]["std"][test_index] = pca_std
        method_blocks["bin_by_bin"]["pred"][test_index] = bin_pred
        method_blocks["bin_by_bin"]["std"][test_index] = bin_std
        fold_by_row[test_index] = fold_index
        n_train_values.append(int(len(train_index)))
        train_indices_by_split.append(train_index)
        component_counts["pca"].append(int(pca_metadata["pca_components"]))
        explained_variances["pca"].append(float(pca_metadata["pca_explained_variance"]))
        component_counts["bin_by_bin"].append(np.nan)
        explained_variances["bin_by_bin"].append(np.nan)

    training_sigma_rms_by_bin, training_sigma_rms_all = _training_sigma_rms(
        alpha_sigma,
        train_indices_by_split,
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for method in methods:
        rows = _metric_rows(
            observable_key=observable_key,
            emulator_type=method,
            subset_size=int(pd.Series(table["evaluation_id"]).nunique()),
            validation_mode="grouped_kfold",
            n_train_values=n_train_values,
            x_plot=x_plot,
            y_true=y,
            y_pred=method_blocks[method]["pred"],
            y_std=method_blocks[method]["std"],
            heldout_sigma=heldout_sigma,
            uncertainty_denominator="emulator_plus_heldout",
            bad_training_condition="nonfinite",
            bad_mask=None,
            training_sigma_rms_by_bin=training_sigma_rms_by_bin,
            training_sigma_rms_all=training_sigma_rms_all,
        )
        n_gp_fit_total = (
            int(np.nansum(component_counts[method]))
            if method == "pca"
            else int(len(folds) * y.shape[1])
        )
        metric_rows.extend(
            _method_metadata_rows(
                rows,
                component_counts=component_counts[method],
                explained_variances=explained_variances[method],
                n_gp_fit_total=n_gp_fit_total,
                n_bins=y.shape[1],
                family="sidecar",
                observable_key=observable_key,
                sample_label=sample_label,
            )
        )
        if args.write_predictions:
            prediction_rows.extend(
                _sidecar_prediction_rows(
                    table=table,
                    observable_key=observable_key,
                    sample_label=sample_label,
                    method=method,
                    output_columns=output_columns,
                    x_plot=x_plot,
                    fold_by_row=fold_by_row,
                    y_true=y,
                    y_pred=method_blocks[method]["pred"],
                    y_std=method_blocks[method]["std"],
                    heldout_sigma=heldout_sigma,
                )
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    bin_rows = metrics.loc[metrics["bin"].astype(str) != "all"].copy()
    group_columns = [
        "family",
        "observable_key",
        "sample_label",
        "observable_label",
        "redshift_label",
        "emulator_type",
    ]
    summary_rows = []
    for keys, group in bin_rows.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row["n_bins"] = int(group["bin"].nunique())
        row["n_valid_total"] = int(group["n_valid"].sum())
        row["n_gp_fit_total"] = int(group["n_gp_fit_total"].iloc[0])
        row["rmse_median"] = float(group["rmse"].median(skipna=True))
        row["r2_median"] = float(group["r2"].median(skipna=True))
        row["normalized_rmse_median"] = float(group["normalized_rmse"].median(skipna=True))
        row["coverage_1sigma_median"] = float(group["coverage_1sigma"].median(skipna=True))
        row["coverage_2sigma_median"] = float(group["coverage_2sigma"].median(skipna=True))
        row["pca_components_mean"] = float(group["pca_components_mean"].mean(skipna=True))
        row["pca_explained_variance_mean"] = float(group["pca_explained_variance_mean"].mean(skipna=True))
        all_row = metrics.loc[
            (metrics["observable_key"] == row["observable_key"])
            & (metrics["emulator_type"] == row["emulator_type"])
            & (metrics["bin"].astype(str) == "all")
        ]
        if not all_row.empty:
            row["rmse_all_bins"] = float(all_row["rmse"].iloc[0])
            row["r2_all_bins"] = float(all_row["r2"].iloc[0])
            row["normalized_rmse_all_bins"] = float(all_row["normalized_rmse"].iloc[0])
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _comparison_delta(summary: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["family", "observable_key", "sample_label", "observable_label", "redshift_label"]
    pca = summary.loc[summary["emulator_type"] == "pca"].set_index(index_columns)
    bbb = summary.loc[summary["emulator_type"] == "bin_by_bin"].set_index(index_columns)
    shared = pca.index.intersection(bbb.index)
    rows = []
    for key in shared:
        pca_row = pca.loc[key]
        bbb_row = bbb.loc[key]
        row = dict(zip(index_columns, key if isinstance(key, tuple) else (key,), strict=True))
        row["n_bins"] = int(pca_row["n_bins"])
        row["pca_n_gp_fit_total"] = int(pca_row["n_gp_fit_total"])
        row["bin_by_bin_n_gp_fit_total"] = int(bbb_row["n_gp_fit_total"])
        row["pca_components_mean"] = float(pca_row["pca_components_mean"])
        row["pca_rmse_median"] = float(pca_row["rmse_median"])
        row["bin_by_bin_rmse_median"] = float(bbb_row["rmse_median"])
        row["delta_rmse_bin_minus_pca"] = row["bin_by_bin_rmse_median"] - row["pca_rmse_median"]
        row["rmse_ratio_bin_over_pca"] = row["bin_by_bin_rmse_median"] / row["pca_rmse_median"]
        row["pca_r2_median"] = float(pca_row["r2_median"])
        row["bin_by_bin_r2_median"] = float(bbb_row["r2_median"])
        row["delta_r2_bin_minus_pca"] = row["bin_by_bin_r2_median"] - row["pca_r2_median"]
        if row["rmse_ratio_bin_over_pca"] > 1.02:
            row["rmse_preferred"] = "pca"
        elif row["rmse_ratio_bin_over_pca"] < 0.98:
            row["rmse_preferred"] = "bin_by_bin"
        else:
            row["rmse_preferred"] = "similar"
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    order = (
        summary.loc[summary["emulator_type"] == "pca", ["observable_key", "redshift_label"]]
        .drop_duplicates()
        .assign(label=lambda frame: frame["observable_key"] + " (" + frame["redshift_label"] + ")")
    )
    labels = order["label"].tolist()
    key_order = order["observable_key"].tolist()
    y_positions = np.arange(len(key_order), dtype=float)
    styles = {
        "pca": {"marker": "o", "color": "#1f77b4", "label": "PCA 99%"},
        "bin_by_bin": {"marker": "s", "color": "#222222", "label": "bin-by-bin"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.8, 0.38 * len(key_order))), sharey=True)
    for method, style in styles.items():
        rows = summary.loc[summary["emulator_type"] == method].set_index("observable_key")
        x_rmse = [float(rows.loc[key, "rmse_median"]) if key in rows.index else np.nan for key in key_order]
        x_r2 = [float(rows.loc[key, "r2_median"]) if key in rows.index else np.nan for key in key_order]
        axes[0].scatter(x_rmse, y_positions, **style)
        axes[1].scatter(x_r2, y_positions, **style)
    axes[0].set_xlabel("median per-bin RMSE")
    axes[1].set_xlabel("median per-bin R^2")
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[1].axvline(0.0, color="0.7", lw=1.0, ls=":")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    handles, labels_for_legend = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_for_legend, frameon=False, loc="upper center", ncol=2)
    fig.subplots_adjust(left=0.37, right=0.98, bottom=0.09, top=0.91, wspace=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _write_outputs(
    *,
    output_dir: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    args: argparse.Namespace,
    command: list[str],
) -> None:
    metrics_path = output_dir / "pca_vs_bin_by_bin_cv_metrics.csv"
    summary_path = output_dir / "pca_vs_bin_by_bin_cv_summary_by_observable.csv"
    delta_path = output_dir / "pca_vs_bin_by_bin_cv_delta_by_observable.csv"
    run_summary_path = output_dir / "pca_vs_bin_by_bin_cv_run_summary.json"
    metrics.to_csv(metrics_path, index=False)
    summary = _summarize_metrics(metrics)
    summary.to_csv(summary_path, index=False)
    delta = _comparison_delta(summary)
    delta.to_csv(delta_path, index=False)
    if args.write_predictions and not predictions.empty:
        predictions.to_csv(output_dir / "pca_vs_bin_by_bin_cv_predictions.csv", index=False)
    _plot_summary(summary, output_dir / "figures" / "pca_vs_bin_by_bin_rmse_r2_summary.png")
    payload = {
        "command": " ".join(command),
        "argv": command,
        "campaign_root": str(args.campaign_root.resolve()),
        "n_folds": int(args.n_folds),
        "random_state": int(args.random_state),
        "pca_variance_threshold": float(args.pca_variance_threshold),
        "pca_scaling": args.pca_scaling,
        "standard_n_restarts_optimizer": int(args.standard_n_restarts_optimizer),
        "sidecar_n_restarts_optimizer": int(args.sidecar_n_restarts_optimizer),
        "optimize_hyperparameters": bool(args.optimize_hyperparameters),
        "fit_white_noise": bool(args.fit_white_noise),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "delta_path": str(delta_path),
    }
    run_summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(metrics_path)
    print(summary_path)
    print(delta_path)
    print(run_summary_path)
    print(output_dir / "figures" / "pca_vs_bin_by_bin_rmse_r2_summary.png")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    campaign_root = args.campaign_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else campaign_root / "cross_validation" / f"pca_vs_bin_by_bin_all_observables_kfold{args.n_folds}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_output = output_dir / "pca_vs_bin_by_bin_cv_metrics.csv"
    if metrics_output.exists() and not args.force:
        print(f"Using existing {metrics_output}; pass --force to refit.", flush=True)
        metrics = pd.read_csv(metrics_output)
        predictions = pd.DataFrame()
        _write_outputs(output_dir=output_dir, metrics=metrics, predictions=predictions, args=args, command=[sys.executable, *sys.argv])
        return

    all_metrics = []
    all_predictions = []

    if not args.no_standard:
        samples = pd.read_csv(campaign_root / "samples.csv")
        input_columns = _input_columns_for_samples(samples)
        input_parameter_specs = _parameter_specs_for_columns(input_columns)
        for observable_key in args.standard_observable or FINAL_STANDARD_OBSERVABLES:
            metrics, predictions = run_standard_observable(
                campaign_root=campaign_root,
                observable_key=observable_key,
                args=args,
                samples=samples,
                input_columns=input_columns,
                input_parameter_specs=input_parameter_specs,
            )
            metrics.to_csv(output_dir / f"{observable_key}_metrics.csv", index=False)
            if args.write_predictions and not predictions.empty:
                predictions.to_csv(output_dir / f"{observable_key}_predictions.csv", index=False)
            all_metrics.append(metrics)
            all_predictions.append(predictions)

    if not args.no_sidecar:
        sidecar_filter_args = SimpleNamespace(
            dust_draw_index=args.dust_draw_index if args.dust_draw_index is not None else [0],
            max_dust_draws_per_eval=args.max_dust_draws_per_eval,
        )
        table = _filter_table(
            _read_sidecar_tables(campaign_root, args.sidecar_dir, args.sidecar_table_filename),
            sidecar_filter_args,
        )
        long_table = _read_first_sidecar_table(campaign_root, args.sidecar_dir, args.sidecar_long_filename)
        input_columns = _sidecar_input_columns(table)
        _parameter_specs, _parameter_names, dust_names = _parameter_specs_for_table(input_columns)
        x = _feature_matrix_from_table(table, input_columns, dust_names)
        folds = _group_kfold_indices(table["evaluation_id"].to_numpy(), args.n_folds, args.random_state)
        for sample_label in args.sidecar_sample_label or FINAL_SIDECAR_SAMPLE_LABELS:
            metrics, predictions = run_sidecar_observable(
                campaign_root=campaign_root,
                requested_sample_label=sample_label,
                table=table,
                long_table=long_table,
                x=x,
                folds=folds,
                args=args,
            )
            safe_label = str(sample_label).lower().replace(".", "p")
            metrics.to_csv(output_dir / f"{args.sidecar_observable}_{safe_label}_metrics.csv", index=False)
            if args.write_predictions and not predictions.empty:
                predictions.to_csv(output_dir / f"{args.sidecar_observable}_{safe_label}_predictions.csv", index=False)
            all_metrics.append(metrics)
            all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    predictions = pd.concat(all_predictions, ignore_index=True) if args.write_predictions and all_predictions else pd.DataFrame()
    _write_outputs(
        output_dir=output_dir,
        metrics=metrics,
        predictions=predictions,
        args=args,
        command=[sys.executable, *sys.argv],
    )


if __name__ == "__main__":
    main()
