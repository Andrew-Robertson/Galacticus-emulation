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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from galacticus_emu.gp import fit_scaled_gp, predict_scaled_gp
from galacticus_emu.interactive_observables import DEFAULT_PCA_COMPONENTS as SHARED_DEFAULT_PCA_COMPONENTS

from fit_multid_function_holdout_demo import (
    PRESETS,
    PRESET_OBSERVABLE_KEYS,
    _bad_mask_override_for_transform,
    _bad_training_mask,
    _load_campaign,
    _json_safe_metadata,
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


DEFAULT_PCA_COMPONENTS = {
    preset_key: SHARED_DEFAULT_PCA_COMPONENTS[observable_key]
    for preset_key, observable_key in PRESET_OBSERVABLE_KEYS.items()
    if observable_key in SHARED_DEFAULT_PCA_COMPONENTS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a PCA-decomposed multi-parameter GP holdout demo for a 1D Galacticus analysis."
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
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument(
        "--pca-scaling",
        choices=["standardized", "unscaled"],
        default="standardized",
    )
    parser.add_argument("--plot-max-components", type=int, default=5)
    return parser.parse_args()


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


def _pca_components_for(args: argparse.Namespace) -> int:
    if args.pca_components is not None:
        return args.pca_components
    if args.preset is not None and args.preset in DEFAULT_PCA_COMPONENTS:
        return DEFAULT_PCA_COMPONENTS[args.preset]
    return 5


def _with_pca_suffix(name: str) -> str:
    return name if "_pca" in name else f"{name}_pca"


def _fit_predict_pca(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    analysis: str,
    n_components: int,
    pca_scaling: str,
    n_restarts_optimizer: int,
    optimize_hyperparameters: bool,
    alpha_train: np.ndarray | None,
    fit_white_noise: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scaler = _make_preprocessor(pca_scaling)
    y_train_scaled = scaler.fit_transform(y_train)
    n_components_actual = min(n_components, y_train_scaled.shape[0], y_train_scaled.shape[1])
    pca = PCA(n_components=n_components_actual)
    coefficients = pca.fit_transform(y_train_scaled)
    coefficient_alpha = None
    if alpha_train is not None:
        alpha_train_scaled = np.asarray(alpha_train, dtype=float) / (scaler.scale_[None, :] ** 2)
        coefficient_alpha = alpha_train_scaled @ (pca.components_.T ** 2)
        coefficient_alpha = np.maximum(coefficient_alpha, 1.0e-12)

    coefficient_predictions = np.zeros((x_test.shape[0], n_components_actual), dtype=float)
    coefficient_stds = np.zeros_like(coefficient_predictions)
    kernel_summaries: list[str] = []
    for component_index in range(n_components_actual):
        print(
            f"{analysis}: fitting PCA component {component_index + 1}/{n_components_actual}"
        )
        model, y_mean, y_std = fit_scaled_gp(
            x_train,
            coefficients[:, component_index],
            n_restarts_optimizer=n_restarts_optimizer,
            optimize_hyperparameters=optimize_hyperparameters,
            alpha=coefficient_alpha[:, component_index] if coefficient_alpha is not None else None,
            fit_white_noise=fit_white_noise,
        )
        coefficient_predictions[:, component_index], coefficient_stds[:, component_index] = predict_scaled_gp(
            model,
            y_mean,
            y_std,
            x_test,
        )
        kernel_summaries.append(str(model.kernel_))
        print(
            f"{analysis}: finished PCA component {component_index + 1}/{n_components_actual}"
        )

    y_pred_scaled = pca.inverse_transform(coefficient_predictions)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # This is a first-order propagation that ignores component covariances, but
    # it tracks relative uncertainty well enough for holdout diagnostics.
    component_variance = coefficient_stds ** 2
    y_var_scaled = component_variance @ (pca.components_ ** 2)
    y_std = np.sqrt(np.maximum(y_var_scaled, 0.0)) * scaler.scale_[None, :]

    metadata = {
        "n_components_requested": int(n_components),
        "n_components_actual": int(n_components_actual),
        "pca_scaling": pca_scaling,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "kernel_summaries": kernel_summaries,
    }
    return y_pred, y_std, metadata


def _plot_pca_modes(
    y: np.ndarray,
    x_plot: np.ndarray,
    *,
    n_components: int,
    pca_scaling: str,
    x_label: str,
    analysis: str,
    path: Path,
    max_components_to_plot: int,
) -> dict[str, object]:
    scaler = _make_preprocessor(pca_scaling)
    y_scaled = scaler.fit_transform(y)
    pca = PCA(n_components=min(n_components, y.shape[0], y.shape[1])).fit(y_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    axes[0].plot(
        np.arange(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        "o-",
    )
    axes[0].set_xlabel("PCA component")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.22)

    for component_index, component in enumerate(
        pca.components_[: min(max_components_to_plot, pca.n_components_)],
        start=1,
    ):
        axes[1].plot(x_plot, component, label=f"PC{component_index}")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("scaled PCA loading" if pca_scaling == "standardized" else "PCA loading")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle(f"PCA decomposition of {analysis} ({pca_scaling})")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_cumulative": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "pca_scaling": pca_scaling,
    }


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    analysis = _setting(args, "analysis")
    if analysis is None:
        raise ValueError("Specify --preset or --analysis")

    output_prefix_base = _setting(args, "output-prefix", f"{analysis}_holdout_demo")
    output_prefix = _with_pca_suffix(output_prefix_base)

    figures_dir_name = args.figures_dir_name
    if figures_dir_name is None:
        preset_figures_dir_name = _setting(args, "figures-dir-name", None)
        figures_dir_name = _with_pca_suffix(preset_figures_dir_name) if preset_figures_dir_name is not None else f"figures_{output_prefix}"

    emulator_dir_name = args.emulator_dir_name
    if emulator_dir_name is None:
        preset_emulator_dir_name = _setting(args, "emulator-dir-name", None)
        emulator_dir_name = _with_pca_suffix(preset_emulator_dir_name) if preset_emulator_dir_name is not None else f"emulator_{output_prefix}"

    figures_dir = campaign_root / figures_dir_name
    emulator_dir = campaign_root / emulator_dir_name
    overlay_ymin = _setting(args, "overlay-ymin", None)
    overlay_ymax = _setting(args, "overlay-ymax", None)
    use_training_alpha = _setting(args, "use-training-alpha", True)
    bad_training_condition = _setting(args, "bad-training-condition", "nonfinite")
    bad_training_value_fill = _setting(args, "bad-training-value-fill", "bin_median")
    bad_training_sigma = _setting(args, "bad-training-sigma", 5.0)
    min_training_sigma = _setting(args, "min-training-sigma", 1.0e-3)
    drop_unsupported_bins = bool(_setting(args, "drop_unsupported_bins", False))
    pca_components = _pca_components_for(args)
    figures_dir.mkdir(parents=True, exist_ok=True)
    emulator_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = emulator_dir / f"{output_prefix}_metrics.csv"
    predictions_path = emulator_dir / f"{output_prefix}_predictions.csv"
    summary_path = emulator_dir / f"{output_prefix}_summary.json"
    overlay_path = figures_dir / f"{output_prefix}_heldout_curves.png"
    parity_path = figures_dir / f"{output_prefix}_parity.png"
    pca_modes_path = figures_dir / f"{output_prefix}_modes.png"
    expected_outputs = [metrics_path, predictions_path, summary_path, overlay_path, parity_path, pca_modes_path]
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
        y_pred, y_pred_std, pca_fit_metadata = _fit_predict_pca(
            x_train_all[train_index],
            y_fit_all[train_index],
            x_train_all[test_index],
            analysis=analysis,
            n_components=pca_components,
            pca_scaling=args.pca_scaling,
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
    metrics.insert(2, "pca_components_actual", pca_fit_metadata["n_components_actual"])
    metrics.insert(3, "pca_explained_variance", pca_fit_metadata["explained_variance_ratio_sum"])
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

    pca_summary = _plot_pca_modes(
        y_fit_all[train_index],
        x_bins_plot,
        n_components=pca_components,
        pca_scaling=args.pca_scaling,
        x_label=_x_label(attrs, analysis),
        analysis=analysis,
        path=pca_modes_path,
        max_components_to_plot=args.plot_max_components,
    )

    prediction_rows = []
    example_index_set = set(example_indices.tolist())
    for local_row, global_row in enumerate(test_index):
        row = {
            "evaluation_id": samples.iloc[global_row]["evaluation_id"],
            "is_example_curve": bool(local_row in example_index_set),
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
        "analysis_attrs": attrs,
        "transform_metadata": _json_safe_metadata(transform_metadata),
        "training_target_metadata": training_target_metadata,
        "pca_fit_metadata": pca_fit_metadata,
        "pca_summary": pca_summary,
        "fit_white_noise": bool(args.fit_white_noise),
        "supported_bin_mask": supported_bin_mask.tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(metrics_path)
    print(predictions_path)
    print(summary_path)
    print(overlay_path)
    print(parity_path)
    print(pca_modes_path)


if __name__ == "__main__":
    main()
