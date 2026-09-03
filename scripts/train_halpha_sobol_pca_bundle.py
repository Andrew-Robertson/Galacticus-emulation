from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fit_halpha_sobol_pca_gp_cv import (
    DUST_INPUT_COLUMNS,
    _apply_filters,
    _combo_metadata,
    _input_matrix,
    _limit_draws_per_eval,
    _log10_with_floor,
    _output_columns,
    _pca_component_alpha,
    _training_alpha_log10,
)
from run_halpha_pca_emulator_mcmc import _load_target_block
from galacticus_emu.gp import fit_scaled_gp


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


def _make_preprocessor(pca_scaling: str):
    if pca_scaling == "standardized":
        return StandardScaler()
    if pca_scaling == "unscaled":
        return MeanCenterer()
    raise ValueError(f"Unknown pca_scaling={pca_scaling!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and save a full-data PCA GP emulator bundle for one Sobral Halpha LF "
            "in the 19D Sobol + dust parameter space."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--sobral-label", required=True, choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--train-draws-per-eval", type=int, default=1)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="unscaled")
    parser.add_argument("--use-training-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shot-noise-alpha",
        choices=["conservative", "smoothed_expectation"],
        default="conservative",
    )
    parser.add_argument("--min-training-log10-sigma", type=float, default=1.0e-4)
    parser.add_argument("--max-training-log10-sigma", type=float, default=2.0)
    parser.add_argument("--fit-white-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    table = pd.read_csv(campaign_root / args.table_filename)
    long_table = pd.read_csv(campaign_root / args.long_filename)
    table = _apply_filters(table, args.max_abs_delta, args.max_attenuation_scatter)
    table = _limit_draws_per_eval(table, args.train_draws_per_eval)

    output_columns = _output_columns(table, args.sobral_label)
    luminosity_centers = np.array(
        [_combo_metadata(long_table, args.sobral_label, index)[1] for index in range(len(output_columns))],
        dtype=float,
    )
    y_linear = table[output_columns].to_numpy(dtype=float)
    y_log10, floor_linear, n_floored, n_clipped = _log10_with_floor(y_linear, args.min_log10_lf)
    alpha_log10 = (
        _training_alpha_log10(
            table,
            output_columns,
            y_linear,
            floor_linear,
            shot_noise_alpha=args.shot_noise_alpha,
            min_training_log10_sigma=args.min_training_log10_sigma,
            max_training_log10_sigma=args.max_training_log10_sigma,
        )
        if args.use_training_alpha
        else None
    )
    x_quantile, input_labels = _input_matrix(table)

    scaler = _make_preprocessor(args.pca_scaling)
    y_scaled = scaler.fit_transform(y_log10)
    n_components_actual = min(args.pca_components, y_scaled.shape[0], y_scaled.shape[1])
    pca = PCA(n_components=n_components_actual)
    coefficients = pca.fit_transform(y_scaled)
    coefficient_alpha = _pca_component_alpha(alpha_log10, scaler, pca, n_components_actual)

    models = []
    y_means = []
    y_stds = []
    kernels = []
    for component_index in range(n_components_actual):
        print(
            f"{args.sobral_label}: fitting PCA component {component_index + 1}/{n_components_actual}",
            flush=True,
        )
        model, y_mean, y_std = fit_scaled_gp(
            x_quantile,
            coefficients[:, component_index],
            n_restarts_optimizer=args.n_restarts_optimizer,
            optimize_hyperparameters=True,
            alpha=coefficient_alpha[:, component_index] if coefficient_alpha is not None else None,
            fit_white_noise=args.fit_white_noise,
        )
        models.append(model)
        y_means.append(float(y_mean))
        y_stds.append(float(y_std))
        kernels.append(str(model.kernel_))

    _, _, _, target_log10, target_cov_log10 = _load_target_block(campaign_root, args.sobral_label)
    output_path = args.output_path
    if output_path is None:
        output_dir = campaign_root / "emulator_halpha_sobol_bundles"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"halpha_{args.sobral_label.lower()}_draws{args.train_draws_per_eval}_pca_bundle.joblib"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "bundle_type": "halpha_sobol_pca_gp",
        "campaign_root": str(campaign_root),
        "sobral_label": args.sobral_label,
        "input_labels": input_labels,
        "n_input_dimensions": int(x_quantile.shape[1]),
        "x_train_quantile": x_quantile,
        "evaluation_ids": table["evaluation_id"].tolist(),
        "dust_draw_indices": table["dust_draw_index"].astype(int).tolist(),
        "luminosity_centers": luminosity_centers,
        "target_log10": target_log10,
        "target_covariance_log10": target_cov_log10,
        "y_train_log10": y_log10,
        "models": models,
        "y_means": np.asarray(y_means, dtype=float),
        "y_stds": np.asarray(y_stds, dtype=float),
        "kernels": kernels,
        "pca_scaling": args.pca_scaling,
        "use_training_alpha": bool(args.use_training_alpha),
        "shot_noise_alpha": args.shot_noise_alpha if args.use_training_alpha else None,
        "min_training_log10_sigma": args.min_training_log10_sigma if args.use_training_alpha else None,
        "max_training_log10_sigma": args.max_training_log10_sigma if args.use_training_alpha else None,
        "fit_white_noise": bool(args.fit_white_noise),
        "pca_components_requested": int(args.pca_components),
        "pca_components_actual": int(n_components_actual),
        "pca_mean": np.asarray(pca.mean_, dtype=float),
        "pca_components_matrix": np.asarray(pca.components_, dtype=float),
        "pca_explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=float),
        "preprocessor_mean": np.asarray(scaler.mean_, dtype=float),
        "preprocessor_scale": np.asarray(scaler.scale_, dtype=float),
        "train_draws_per_eval": int(args.train_draws_per_eval),
        "n_restarts_optimizer": int(args.n_restarts_optimizer),
        "random_state": int(args.random_state),
        "min_log10_lf": float(args.min_log10_lf),
        "max_abs_delta": args.max_abs_delta,
        "max_attenuation_scatter": args.max_attenuation_scatter,
        "log10_floor_linear": float(floor_linear),
        "n_floored_values": int(n_floored),
        "n_clipped_values": int(n_clipped),
    }
    joblib.dump(bundle, output_path)

    meta = {
        "bundle_type": bundle["bundle_type"],
        "campaign_root": bundle["campaign_root"],
        "sobral_label": bundle["sobral_label"],
        "n_input_dimensions": bundle["n_input_dimensions"],
        "input_labels": bundle["input_labels"],
        "luminosity_centers": bundle["luminosity_centers"].tolist(),
        "pca_scaling": bundle["pca_scaling"],
        "pca_components_requested": bundle["pca_components_requested"],
        "pca_components_actual": bundle["pca_components_actual"],
        "pca_explained_variance_ratio_sum": float(np.sum(bundle["pca_explained_variance_ratio"])),
        "use_training_alpha": bundle["use_training_alpha"],
        "shot_noise_alpha": bundle["shot_noise_alpha"],
        "fit_white_noise": bundle["fit_white_noise"],
        "train_draws_per_eval": bundle["train_draws_per_eval"],
        "n_training_rows": int(len(table)),
        "n_unique_evaluations": int(pd.Series(table["evaluation_id"]).nunique()),
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(output_path)
    print(meta_path)


if __name__ == "__main__":
    main()
