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

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from galacticus_emu.gp import fit_scaled_gp

from fit_multid_function_holdout_demo import (
    PRESETS,
    _bad_mask_override_for_transform,
    _bad_training_mask,
    _fit_predict,
    _json_safe_metadata,
    _load_campaign,
    _metrics,
    _plot_overlay,
    _plot_parity,
    _prepare_training_targets,
    _quantile_columns,
    _select_bins,
    _setting,
    _space_filling_examples,
    _supported_bin_mask,
    _title,
    _x_label,
    _y_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a multi-parameter GP emulator with K-fold CV for a 1D Galacticus analysis."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--analysis", default=None)
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--min-log10-y", type=float, default=-7.0)
    parser.add_argument("--n-example-curves", type=int, default=5)
    parser.add_argument("--overlay-ymin", type=float, default=None)
    parser.add_argument("--overlay-ymax", type=float, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--figures-dir-name", default=None)
    parser.add_argument("--emulator-dir-name", default=None)
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
    parser.add_argument(
        "--save-full-emulator",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _fit_full_emulator_bundle(
    x_train: np.ndarray,
    y_fit: np.ndarray,
    alpha_train: np.ndarray | None,
    *,
    analysis: str,
    x_bins_plot: np.ndarray,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    fit_white_noise: bool,
) -> dict[str, object]:
    models = []
    y_means = []
    y_stds = []
    kernel_summaries = []
    for bin_index in range(y_fit.shape[1]):
        print(
            f"{analysis}: fitting full emulator bin {bin_index + 1}/{y_fit.shape[1]} "
            f"(x={x_bins_plot[bin_index]:.3f})"
        )
        model, y_mean, y_std = fit_scaled_gp(
            x_train,
            y_fit[:, bin_index],
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=optimize_hyperparameters,
            alpha=alpha_train[:, bin_index] if alpha_train is not None else None,
            fit_white_noise=fit_white_noise,
        )
        models.append(model)
        y_means.append(float(y_mean))
        y_stds.append(float(y_std))
        kernel_summaries.append(str(model.kernel_))
    return {
        "bundle_type": "multid_function_direct_gp",
        "analysis": analysis,
        "models": models,
        "y_means": np.asarray(y_means, dtype=float),
        "y_stds": np.asarray(y_stds, dtype=float),
        "kernel_summaries": kernel_summaries,
    }


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    analysis = _setting(args, "analysis")
    if analysis is None:
        raise ValueError("Specify --preset or --analysis")

    output_prefix = _setting(args, "output-prefix", f"{analysis}_gp_cv")
    figures_dir = campaign_root / _setting(args, "figures-dir-name", f"figures_{output_prefix}")
    emulator_dir = campaign_root / _setting(args, "emulator-dir-name", f"emulator_{output_prefix}")
    overlay_ymin = _setting(args, "overlay-ymin", None)
    overlay_ymax = _setting(args, "overlay-ymax", None)
    use_training_alpha = bool(_setting(args, "use-training-alpha", True))
    bad_training_condition = _setting(args, "bad_training_condition", "nonfinite")
    bad_training_value_fill = _setting(args, "bad_training_value_fill", "bin_median")
    bad_training_sigma = float(_setting(args, "bad_training_sigma", 5.0))
    min_training_sigma = float(_setting(args, "min_training_sigma", 1.0e-3))
    drop_unsupported_bins = bool(_setting(args, "drop_unsupported_bins", False))
    optimize_hyperparameters = bool(args.optimize_hyperparameters)
    figures_dir.mkdir(parents=True, exist_ok=True)
    emulator_dir.mkdir(parents=True, exist_ok=True)

    (
        samples,
        x_train,
        x_bins_plot,
        target_plot,
        target_std_plot,
        y_plot,
        y_noise_plot,
        attrs,
        transform_metadata,
    ) = _load_campaign(
        campaign_root,
        analysis=analysis,
        hdf5_filename=args.hdf5_filename,
        min_log10_y=args.min_log10_y,
    )
    supported_bin_mask = np.ones(y_plot.shape[1], dtype=bool)
    bad_mask_override = _bad_mask_override_for_transform(
        transform_metadata,
        bad_training_condition=bad_training_condition,
        bad_training_value_fill=bad_training_value_fill,
    )
    if drop_unsupported_bins:
        if bad_mask_override is not None:
            supported_bin_mask = np.any(~bad_mask_override, axis=0)
        else:
            supported_bin_mask = _supported_bin_mask(
                y_plot,
                y_noise=y_noise_plot,
                bad_training_condition=bad_training_condition,
            )
        if not np.any(supported_bin_mask):
            raise ValueError("All bins are unsupported after applying the bad-training mask.")
        x_bins_plot = x_bins_plot[supported_bin_mask]
        target_plot = target_plot[supported_bin_mask]
        if target_std_plot is not None:
            target_std_plot = _select_bins(target_std_plot, supported_bin_mask)
        y_plot = y_plot[:, supported_bin_mask]
        if y_noise_plot is not None:
            y_noise_plot = y_noise_plot[:, supported_bin_mask]
    if bad_mask_override is not None:
        bad_mask_override = bad_mask_override[:, supported_bin_mask]
    bad_metric_mask = (
        bad_mask_override
        if bad_mask_override is not None
        else _bad_training_mask(
            y_plot,
            y_noise=y_noise_plot,
            bad_training_condition=bad_training_condition,
        )
    )
    quantile_columns = _quantile_columns(samples)

    n_samples = x_train.shape[0]
    if args.n_splits < 2 or args.n_splits > n_samples:
        raise ValueError(f"--n-splits must be in [2, {n_samples}]")

    predictions = np.full_like(y_plot, np.nan, dtype=float)
    prediction_stds = np.full_like(y_plot, np.nan, dtype=float)
    fold_index_per_row = np.full(n_samples, -1, dtype=int)
    fold_kernel_summaries: dict[str, list[str]] = {}
    fold_training_metadata: dict[str, dict] = {}

    splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for fold_index, (train_index, test_index) in enumerate(splitter.split(x_train), start=1):
            print(
                f"{analysis}: fold {fold_index}/{args.n_splits}, "
                f"train={len(train_index)}, test={len(test_index)}",
                flush=True,
            )
            y_train_fit, alpha_sigma, training_metadata = _prepare_training_targets(
                y_plot[train_index],
                y_noise_plot[train_index] if y_noise_plot is not None else None,
                analysis=analysis,
                use_training_alpha=use_training_alpha,
                bad_training_condition=bad_training_condition,
                bad_training_value_fill=bad_training_value_fill,
                bad_training_sigma=bad_training_sigma,
                min_training_sigma=min_training_sigma,
                bad_mask_override=bad_mask_override[train_index] if bad_mask_override is not None else None,
            )
            alpha_train = alpha_sigma**2 if alpha_sigma is not None else None
            y_pred_fold, y_std_fold, kernels = _fit_predict(
                x_train[train_index],
                y_train_fit,
                x_train[test_index],
                analysis=analysis,
                x_bins_plot=x_bins_plot,
                n_restarts_optimizer=args.n_restarts_optimizer,
                optimize_hyperparameters=optimize_hyperparameters,
                alpha_train=alpha_train,
                fit_white_noise=args.fit_white_noise,
            )
            predictions[test_index] = y_pred_fold
            prediction_stds[test_index] = y_std_fold
            fold_index_per_row[test_index] = fold_index
            fold_kernel_summaries[f"fold_{fold_index}"] = kernels
            fold_training_metadata[f"fold_{fold_index}"] = training_metadata
            print(f"{analysis}: finished fold {fold_index}/{args.n_splits}", flush=True)

    metrics = _metrics(
        y_plot,
        predictions,
        prediction_stds,
        x_bins_plot,
        valid_mask=~bad_metric_mask,
    )
    metrics_path = emulator_dir / f"{output_prefix}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    example_indices = _space_filling_examples(x_train, args.n_example_curves)
    predictions_table = samples[["evaluation_id", *quantile_columns]].copy()
    predictions_table["fold_index"] = fold_index_per_row
    predictions_table["is_example_curve"] = False
    predictions_table.loc[example_indices, "is_example_curve"] = True
    for bin_index, x_value in enumerate(x_bins_plot):
        prefix = f"bin{bin_index:02d}"
        predictions_table[f"{prefix}_x_plot"] = x_value
        predictions_table[f"{prefix}_true"] = y_plot[:, bin_index]
        if y_noise_plot is None:
            predictions_table[f"{prefix}_true_std"] = np.nan
        else:
            predictions_table[f"{prefix}_true_std"] = y_noise_plot[:, bin_index]
        predictions_table[f"{prefix}_true_missing"] = bad_metric_mask[:, bin_index]
        predictions_table[f"{prefix}_pred"] = predictions[:, bin_index]
        predictions_table[f"{prefix}_pred_std"] = prediction_stds[:, bin_index]
    predictions_path = emulator_dir / f"{output_prefix}_predictions.csv"
    predictions_table.to_csv(predictions_path, index=False)

    summary = {
        "analysis": analysis,
        "n_samples": int(n_samples),
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
        "n_restarts_optimizer": int(args.n_restarts_optimizer),
        "optimize_hyperparameters": bool(optimize_hyperparameters),
        "fit_white_noise": bool(args.fit_white_noise),
        "use_training_alpha": bool(use_training_alpha),
        "bad_training_condition": bad_training_condition,
        "bad_training_value_fill": bad_training_value_fill,
        "bad_training_sigma": bad_training_sigma,
        "min_training_sigma": min_training_sigma,
        "example_curve_ids": samples.iloc[example_indices]["evaluation_id"].tolist(),
        "fold_kernel_summaries": fold_kernel_summaries,
        "fold_training_metadata": fold_training_metadata,
        "supported_bin_mask": supported_bin_mask.tolist(),
        "transform_metadata": _json_safe_metadata(transform_metadata),
        "attrs": attrs,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }
    summary_path = emulator_dir / f"{output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    parity_path = figures_dir / f"{output_prefix}_parity.png"
    _plot_parity(
        x_bins_plot=x_bins_plot,
        y_test=y_plot,
        y_test_std=y_noise_plot,
        y_pred=predictions,
        y_pred_std=prediction_stds,
        path=parity_path,
        target_plot=target_plot,
        bad_training_condition=bad_training_condition,
        bad_training_value_fill=bad_training_value_fill,
    )

    overlay_path = figures_dir / f"{output_prefix}_heldout_curves.png"
    _plot_overlay(
        x_plot=x_bins_plot,
        target_plot=target_plot,
        target_std_plot=target_std_plot,
        y_train=y_plot,
        y_test=y_plot,
        y_test_std=y_noise_plot,
        y_pred=predictions,
        y_pred_std=prediction_stds,
        example_indices=example_indices,
        x_label=_x_label(attrs, analysis),
        y_label=_y_label(attrs, transform_metadata, analysis),
        title=_title(attrs, analysis),
        ymin=overlay_ymin,
        ymax=overlay_ymax,
        path=overlay_path,
        bad_training_condition=bad_training_condition,
        bad_training_value_fill=bad_training_value_fill,
    )

    if args.save_full_emulator:
        y_fit_all, alpha_sigma_all, training_metadata_all = _prepare_training_targets(
            y_plot,
            y_noise_plot,
            analysis=analysis,
            use_training_alpha=use_training_alpha,
            bad_training_condition=bad_training_condition,
            bad_training_value_fill=bad_training_value_fill,
            bad_training_sigma=bad_training_sigma,
            min_training_sigma=min_training_sigma,
            bad_mask_override=bad_mask_override,
        )
        alpha_all = alpha_sigma_all**2 if alpha_sigma_all is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            bundle = _fit_full_emulator_bundle(
                x_train,
                y_fit_all,
                alpha_all,
                analysis=analysis,
                x_bins_plot=x_bins_plot,
                n_restarts_optimizer=args.n_restarts_optimizer,
                optimize_hyperparameters=optimize_hyperparameters,
                fit_white_noise=args.fit_white_noise,
            )
        bundle.update(
            {
                "campaign_root": str(campaign_root),
                "hdf5_filename": args.hdf5_filename,
                "input_labels": quantile_columns,
                "x_plot": x_bins_plot,
                "target_plot": target_plot,
                "target_std_plot": target_std_plot,
                "attrs": attrs,
                "transform_metadata": _json_safe_metadata(transform_metadata),
                "training_metadata": training_metadata_all,
            }
        )
        bundle_path = emulator_dir / f"{output_prefix}_full_emulator.joblib"
        joblib.dump(bundle, bundle_path)
        summary["full_emulator_bundle_path"] = str(bundle_path)
        summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote parity plot to {parity_path}")
    print(f"Wrote held-out curves plot to {overlay_path}")


if __name__ == "__main__":
    main()
