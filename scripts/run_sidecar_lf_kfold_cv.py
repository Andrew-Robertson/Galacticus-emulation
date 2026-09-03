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

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from fit_sidecar_lf_pca_holdout_demo import (
    TARGET_ERROR_MODE_LEGACY,
    TARGET_ERROR_MODES,
    _alpha_log10,
    _bin_metadata,
    _input_columns,
    _log10_with_floor,
    _output_columns,
    _read_first_sidecar_table,
    _read_sidecar_tables,
    _sample_labels,
    _target_curve,
)
from run_sidecar_lf_pca_emulator_mcmc import (
    _feature_matrix_from_table,
    _filter_table,
    _fit_pca_bundle,
    _parameter_specs_for_table,
    _predict_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run grouped K-fold CV for sidecar emission-line LF PCA-GP emulators. "
            "The grouped split holds out entire Galacticus evaluations."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--sidecar-dir", default="emission_line_dust")
    parser.add_argument("--table-filename", default="emission_line_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="emission_line_dust_lf_long.csv")
    parser.add_argument("--observable", required=True)
    parser.add_argument("--sample-label", action="append", required=True)
    parser.add_argument("--dust-draw-index", type=int, action="append", default=None)
    parser.add_argument(
        "--max-dust-draws-per-eval",
        type=int,
        default=None,
        help="After other filters, keep the first N dust rows per Galacticus evaluation.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=12345)
    parser.add_argument("--min-log10-lf", type=float, default=-8.0)
    parser.add_argument(
        "--target-error-mode",
        choices=TARGET_ERROR_MODES,
        default=TARGET_ERROR_MODE_LEGACY,
        help=(
            "How to convert H-alpha Sobral target LF errors into log10(Phi) space. "
            "The default preserves legacy runs; use sobral_log_table for the "
            "Sobral table's log-space errors."
        ),
    )
    parser.add_argument("--use-training-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-training-log10-sigma", type=float, default=1.0e-4)
    parser.add_argument("--max-training-log10-sigma", type=float, default=2.0)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--optimize-hyperparameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument(
        "--pca-variance-threshold",
        type=float,
        default=0.99,
        help="Choose enough PCA components to explain this variance fraction.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def _output_prefix(args: argparse.Namespace) -> str:
    if args.output_prefix:
        return args.output_prefix
    if args.dust_draw_index is None:
        dust_tag = "all_dustdraws"
    else:
        dust_tag = "dustdraw" + "_".join(str(index) for index in args.dust_draw_index)
    pca_tag = f"pca{str(args.pca_variance_threshold).replace('.', 'p')}"
    return f"{args.observable}_{dust_tag}_{pca_tag}_kfold{args.n_folds}"


def _group_kfold_indices(groups: np.ndarray, n_folds: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.asarray(sorted(pd.unique(groups)), dtype=object)
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if n_folds > unique_groups.size:
        raise ValueError(f"--n-folds={n_folds} exceeds the number of groups ({unique_groups.size}).")
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    for train_group_index, test_group_index in splitter.split(unique_groups):
        train_groups = set(unique_groups[train_group_index])
        test_groups = set(unique_groups[test_group_index])
        train_index = np.flatnonzero(np.asarray([group in train_groups for group in groups], dtype=bool))
        test_index = np.flatnonzero(np.asarray([group in test_groups for group in groups], dtype=bool))
        folds.append((train_index, test_index))
    return folds


def _safe_metric_values(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {
            "n_valid": 0,
            "n_valid_uncertainty": 0,
            "rmse": np.nan,
            "bias": np.nan,
            "r2": np.nan,
            "normalized_rmse": np.nan,
            "normalized_mae": np.nan,
            "normalized_bias": np.nan,
            "normalized_std": np.nan,
            "coverage_1sigma": np.nan,
            "coverage_2sigma": np.nan,
        }

    residual = y_pred[valid] - y_true[valid]
    valid_uncertainty = valid & np.isfinite(y_std) & (y_std > 0.0)
    if np.any(valid_uncertainty):
        residual_uncertainty = y_pred[valid_uncertainty] - y_true[valid_uncertainty]
        std_valid = y_std[valid_uncertainty]
        normalized_residual = residual_uncertainty / std_valid
        normalized_rmse = float(np.sqrt(np.mean(normalized_residual**2)))
        normalized_mae = float(np.mean(np.abs(normalized_residual)))
        normalized_bias = float(np.mean(normalized_residual))
        normalized_std = float(np.std(normalized_residual))
        coverage_1sigma = float(np.mean(np.abs(residual_uncertainty) <= std_valid))
        coverage_2sigma = float(np.mean(np.abs(residual_uncertainty) <= 2.0 * std_valid))
    else:
        normalized_rmse = np.nan
        normalized_mae = np.nan
        normalized_bias = np.nan
        normalized_std = np.nan
        coverage_1sigma = np.nan
        coverage_2sigma = np.nan

    return {
        "n_valid": int(np.count_nonzero(valid)),
        "n_valid_uncertainty": int(np.count_nonzero(valid_uncertainty)),
        "rmse": float(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))),
        "bias": float(np.mean(residual)),
        "r2": float(r2_score(y_true[valid], y_pred[valid])) if np.count_nonzero(valid) > 1 else np.nan,
        "normalized_rmse": normalized_rmse,
        "normalized_mae": normalized_mae,
        "normalized_bias": normalized_bias,
        "normalized_std": normalized_std,
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma,
    }


def _metric_rows(
    *,
    observable_key: str,
    sample_label: str,
    x_plot: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_train_values: list[int],
    pca_components: list[int],
    pca_explained_variance: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    common = {
        "observable_key": observable_key,
        "sample_label": sample_label,
        "emulator_type": "pca",
        "subset_size": int(y_true.shape[0]),
        "validation_mode": "grouped_kfold",
        "n_train_mean": float(np.mean(n_train_values)),
        "pca_components_mean": float(np.mean(pca_components)),
        "pca_components_min": int(np.min(pca_components)),
        "pca_components_max": int(np.max(pca_components)),
        "pca_explained_variance_mean": float(np.mean(pca_explained_variance)),
    }
    for bin_index, x_value in enumerate(x_plot):
        rows.append(
            {
                **common,
                "n_test_total": int(y_true.shape[0]),
                "bin": int(bin_index),
                "x_plot": float(x_value),
                **_safe_metric_values(y_true[:, bin_index], y_pred[:, bin_index], y_std[:, bin_index]),
            }
        )
    rows.append(
        {
            **common,
            "n_test_total": int(y_true.size),
            "bin": "all",
            "x_plot": np.nan,
            **_safe_metric_values(y_true.ravel(), y_pred.ravel(), y_std.ravel()),
        }
    )
    return rows


def _prediction_rows(
    *,
    table: pd.DataFrame,
    observable_key: str,
    sample_label: str,
    output_columns: list[str],
    x_plot: np.ndarray,
    fold_by_row: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_index, source in table.reset_index(drop=True).iterrows():
        base = {
            "observable_key": observable_key,
            "sample_label": sample_label,
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
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    campaign_root = args.campaign_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = _output_prefix(args)

    table = _filter_table(_read_sidecar_tables(campaign_root, args.sidecar_dir, args.table_filename), args)
    long_table = _read_first_sidecar_table(campaign_root, args.sidecar_dir, args.long_filename)
    input_columns = _input_columns(table)
    _parameter_specs, _parameter_names, dust_names = _parameter_specs_for_table(input_columns)
    x = _feature_matrix_from_table(table, input_columns, dust_names)
    groups = table["evaluation_id"].to_numpy()
    folds = _group_kfold_indices(groups, args.n_folds, args.random_state)

    fit_args = SimpleNamespace(
        n_restarts_optimizer=args.n_restarts_optimizer,
        optimize_hyperparameters=args.optimize_hyperparameters,
        pca_components=args.pca_components,
        pca_variance_threshold=args.pca_variance_threshold,
    )

    all_metric_rows: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "campaign_root": str(campaign_root),
        "sidecar_dir": args.sidecar_dir,
        "table_filename": args.table_filename,
        "long_filename": args.long_filename,
        "observable": args.observable,
        "sample_labels": args.sample_label,
        "dust_draw_index": args.dust_draw_index,
        "max_dust_draws_per_eval": args.max_dust_draws_per_eval,
        "n_rows": int(len(table)),
        "n_unique_evaluations": int(pd.Series(groups).nunique()),
        "n_folds": int(args.n_folds),
        "random_state": int(args.random_state),
        "input_columns": input_columns,
        "training_options": {
            "min_log10_lf": args.min_log10_lf,
            "target_error_mode": args.target_error_mode,
            "use_training_alpha": args.use_training_alpha,
            "min_training_log10_sigma": args.min_training_log10_sigma,
            "max_training_log10_sigma": args.max_training_log10_sigma,
            "n_restarts_optimizer": args.n_restarts_optimizer,
            "optimize_hyperparameters": args.optimize_hyperparameters,
            "pca_components": args.pca_components,
            "pca_variance_threshold": args.pca_variance_threshold,
        },
        "observables": {},
    }

    for requested_sample_label in args.sample_label:
        output_columns = _output_columns(table, args.observable, [requested_sample_label])
        sample_labels = _sample_labels(output_columns)
        if not sample_labels or any(label != sample_labels[0] for label in sample_labels):
            raise ValueError(f"Could not infer a single sample label for {requested_sample_label!r}")
        sample_label = sample_labels[0]
        bin_metadata = _bin_metadata(long_table, args.observable, sample_label, output_columns)
        x_plot = bin_metadata["log10_luminosity_center"].to_numpy(dtype=float)
        target_plot, target_std_plot = _target_curve(
            campaign_root,
            args.observable,
            sample_label,
            x_plot,
            args.min_log10_lf,
            target_error_mode=args.target_error_mode,
        )
        y, floor = _log10_with_floor(table[output_columns].to_numpy(dtype=float), args.min_log10_lf)
        alpha = None
        if args.use_training_alpha:
            alpha_sigma = _alpha_log10(
                table,
                output_columns,
                min_sigma=args.min_training_log10_sigma,
                max_sigma=args.max_training_log10_sigma,
            )
            alpha = alpha_sigma**2

        y_pred = np.full_like(y, np.nan, dtype=float)
        y_std = np.full_like(y, np.nan, dtype=float)
        fold_by_row = np.zeros(len(table), dtype=int)
        n_train_values: list[int] = []
        pca_components: list[int] = []
        pca_explained_variance: list[float] = []
        fold_summaries: list[dict[str, object]] = []

        for fold_index, (train_index, test_index) in enumerate(folds, start=1):
            print(
                f"{args.observable} {sample_label}: fold {fold_index}/{args.n_folds}, "
                f"train {len(train_index)} rows, test {len(test_index)} rows",
                flush=True,
            )
            bundle = _fit_pca_bundle(
                x=x[train_index],
                y=y[train_index],
                alpha=alpha[train_index] if alpha is not None else None,
                observable_key=args.observable,
                sample_label=sample_label,
                output_columns=output_columns,
                x_plot=x_plot,
                target_plot=target_plot,
                target_std_plot=target_std_plot,
                args=fit_args,
            )
            bundle["target_error_mode"] = args.target_error_mode
            fold_pred, fold_std = _predict_bundle(bundle, x[test_index])
            y_pred[test_index] = fold_pred
            y_std[test_index] = fold_std
            fold_by_row[test_index] = fold_index
            n_train_values.append(int(len(train_index)))
            pca_components.append(int(bundle["pca_components_actual"]))
            pca_explained_variance.append(float(np.sum(bundle["pca_explained_variance_ratio"])))
            fold_summaries.append(
                {
                    "fold_index": int(fold_index),
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "pca_components_actual": int(bundle["pca_components_actual"]),
                    "pca_explained_variance_ratio_sum": float(np.sum(bundle["pca_explained_variance_ratio"])),
                    "heldout_evaluations": sorted(pd.unique(table.iloc[test_index]["evaluation_id"]).tolist()),
                }
            )

        all_metric_rows.extend(
            _metric_rows(
                observable_key=args.observable,
                sample_label=sample_label,
                x_plot=x_plot,
                y_true=y,
                y_pred=y_pred,
                y_std=y_std,
                n_train_values=n_train_values,
                pca_components=pca_components,
                pca_explained_variance=pca_explained_variance,
            )
        )
        all_prediction_rows.extend(
            _prediction_rows(
                table=table,
                observable_key=args.observable,
                sample_label=sample_label,
                output_columns=output_columns,
                x_plot=x_plot,
                fold_by_row=fold_by_row,
                y_true=y,
                y_pred=y_pred,
                y_std=y_std,
            )
        )
        summary["observables"][f"{args.observable}_{sample_label}"] = {
            "sample_label": sample_label,
            "output_columns": output_columns,
            "n_bins": int(len(output_columns)),
            "x_plot": x_plot.tolist(),
            "target_log10_phi": target_plot.tolist(),
            "target_log10_phi_std": None if target_std_plot is None else target_std_plot.tolist(),
            "target_error_mode": args.target_error_mode,
            "log10_floor_linear": float(floor),
            "folds": fold_summaries,
        }

    metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    predictions_path = output_dir / f"{output_prefix}_predictions.csv"
    summary_path = output_dir / f"{output_prefix}_summary.json"
    pd.DataFrame(all_metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(all_prediction_rows).to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(metrics_path)
    print(predictions_path)
    print(summary_path)


if __name__ == "__main__":
    main()
