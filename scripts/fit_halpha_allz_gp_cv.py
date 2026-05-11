from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from galacticus_emu.lhs import (
    NormalPrior,
    ParameterSpec,
    TruncatedLogNormalPrior,
    TruncatedNormalPrior,
    transform_to_prior_quantiles,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run grouped 5-fold-style CV for all Sobral Halpha LF bins across all four redshifts, "
            "holding out whole Galacticus evaluations together."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--train-draws-per-eval", type=int, default=8)
    parser.add_argument("--n-restarts-optimizer", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--figures-dir-name", default="figures_halpha_allz_cv")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_allz_cv")
    parser.add_argument(
        "--parity-bins",
        type=str,
        nargs="*",
        default=["Z1:7", "Z1:16", "Z4:1", "Z4:12"],
        help="Subset of sobral_label:bin_index pairs for a compact parity panel.",
    )
    return parser.parse_args()


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


def _bin_sort_key(column: str) -> int:
    match = re.search(r"_bin(\d+)$", column)
    if match is None:
        raise ValueError(f"Could not parse bin index from column '{column}'")
    return int(match.group(1))


def _apply_filters(table: pd.DataFrame, max_abs_delta: float | None, max_attenuation_scatter: float | None) -> pd.DataFrame:
    mask = np.ones(len(table), dtype=bool)
    if max_abs_delta is not None:
        for column in ["delta_0", "delta_z", "delta_M", "delta_Mz"]:
            mask &= np.abs(table[column].to_numpy(dtype=float)) <= max_abs_delta
    if max_attenuation_scatter is not None:
        mask &= table["attenuation_scatter"].to_numpy(dtype=float) <= max_attenuation_scatter
    return table.loc[mask].reset_index(drop=True)


def _limit_draws_per_eval(table: pd.DataFrame, train_draws_per_eval: int) -> pd.DataFrame:
    return (
        table.sort_values(["evaluation_id", "dust_draw_index"])
        .groupby("evaluation_id", group_keys=False)
        .head(train_draws_per_eval)
        .reset_index(drop=True)
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


def _parse_pair(text: str) -> tuple[str, int]:
    label, bin_text = text.split(":")
    return label.upper(), int(bin_text)


def _combo_metadata(long_table: pd.DataFrame, sobral_label: str, bin_index: int) -> tuple[float, float]:
    row = (
        long_table.loc[
            (long_table["sobral_label"] == sobral_label) & (long_table["bin_index"] == bin_index),
            ["redshift", "log10_luminosity_center"],
        ]
        .drop_duplicates()
        .iloc[0]
    )
    return float(row["redshift"]), float(row["log10_luminosity_center"])


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(campaign_root / args.table_filename)
    long_table = pd.read_csv(campaign_root / args.long_filename)
    table = _apply_filters(table, args.max_abs_delta, args.max_attenuation_scatter)
    table = _limit_draws_per_eval(table, args.train_draws_per_eval)

    x_raw = table[INPUT_COLUMNS].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)
    groups = table["evaluation_id"].to_numpy()
    splitter = GroupKFold(n_splits=args.n_splits)

    pair_predictions: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    floor_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for sobral_label in SOBRAL_LABELS:
            output_columns = _output_columns(table, sobral_label)
            for bin_index, output_column in enumerate(output_columns):
                print(
                    f"starting {sobral_label} bin {bin_index} ({output_column})",
                    flush=True,
                )
                y_linear = table[output_column].to_numpy(dtype=float)
                y, floor, n_floored, n_clipped = _log10_with_floor(y_linear, args.min_log10_lf)
                pred = np.zeros_like(y)
                pred_std = np.zeros_like(y)
                fold_kernel_summaries: list[str] = []
                for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
                    heldout = np.unique(groups[test_index])
                    print(
                        f"{sobral_label} bin {bin_index}: fold {fold_index}/{args.n_splits}, "
                        f"hold out {len(heldout)} evals / {len(test_index)} rows",
                        flush=True,
                    )
                    model, y_mean, y_std = _fit_scaled_gp(
                        x[train_index],
                        y[train_index],
                        n_restarts_optimizer=args.n_restarts_optimizer,
                        random_state=args.random_state,
                    )
                    fold_kernel_summaries.append(str(model.kernel_))
                    kernel_rows.append(
                        {
                            "sobral_label": sobral_label,
                            "bin_index": bin_index,
                            "fold_index": fold_index,
                            "n_train": int(len(train_index)),
                            "n_test": int(len(test_index)),
                            "heldout_evaluations": "|".join(heldout.tolist()),
                            "kernel": str(model.kernel_),
                            "y_mean": float(y_mean),
                            "y_std": float(y_std),
                        }
                    )
                    pred[test_index], pred_std[test_index] = _predict_scaled_gp(model, y_mean, y_std, x[test_index])

                redshift, log10_center = _combo_metadata(long_table, sobral_label, bin_index)
                rows.append(
                    {
                        "sobral_label": sobral_label,
                        "redshift": redshift,
                        "bin_index": bin_index,
                        "log10_luminosity_center": log10_center,
                        "rmse_dex": float(np.sqrt(mean_squared_error(y, pred))),
                        "mae_dex": float(mean_absolute_error(y, pred)),
                        "r2": float(r2_score(y, pred)),
                        "coverage_1sigma": float(np.mean(np.abs(pred - y) <= pred_std)),
                        "coverage_2sigma": float(np.mean(np.abs(pred - y) <= 2.0 * pred_std)),
                        "n_rows": int(len(y)),
                        "kernel_example": fold_kernel_summaries[0] if fold_kernel_summaries else "",
                    }
                )
                floor_rows.append(
                    {
                        "sobral_label": sobral_label,
                        "bin_index": bin_index,
                        "log10_luminosity_center": log10_center,
                        "log10_floor_linear": floor,
                        "n_floored_values": n_floored,
                        "n_clipped_values": n_clipped,
                    }
                )
                pair_predictions[(sobral_label, bin_index)] = {
                    "true": y,
                    "pred": pred,
                    "std": pred_std,
                }
                print(
                    f"finished {sobral_label} bin {bin_index}",
                    flush=True,
                )

    metrics = pd.DataFrame(rows)
    floors = pd.DataFrame(floor_rows)
    kernels = pd.DataFrame(kernel_rows)
    metrics_path = emulator_root / "gp_allz_cv_metrics.csv"
    floors_path = emulator_root / "gp_allz_cv_flooring.csv"
    kernels_path = emulator_root / "gp_allz_cv_kernels.csv"
    metrics.to_csv(metrics_path, index=False)
    floors.to_csv(floors_path, index=False)
    kernels.to_csv(kernels_path, index=False)

    # Summary plot by redshift/bin
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for sobral_label in SOBRAL_LABELS:
        sub = metrics.loc[metrics["sobral_label"] == sobral_label].sort_values("log10_luminosity_center")
        axes[0].plot(sub["log10_luminosity_center"], sub["rmse_dex"], marker="o", lw=1.6, label=sobral_label)
        axes[1].plot(sub["log10_luminosity_center"], sub["r2"], marker="o", lw=1.6, label=sobral_label)
    axes[0].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    axes[0].set_ylabel("RMSE [dex]")
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    curves_path = figures_root / "gp_allz_cv_rmse_r2_vs_luminosity.png"
    fig.savefig(curves_path, dpi=220)
    plt.close(fig)

    # Compact 2x2 parity for representative bins
    chosen_pairs = [_parse_pair(item) for item in args.parity_bins]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    for axis, (sobral_label, bin_index) in zip(axes.flat, chosen_pairs, strict=False):
        result = pair_predictions[(sobral_label, bin_index)]
        observed = result["true"]
        predicted = result["pred"]
        predicted_std = result["std"]
        redshift, log10_center = _combo_metadata(long_table, sobral_label, bin_index)
        lower = float(min(observed.min(), predicted.min()))
        upper = float(max(observed.max(), predicted.max()))
        margin = 0.08 * (upper - lower if upper > lower else 1.0)
        axis.errorbar(observed, predicted, yerr=predicted_std, fmt="o", ms=3.0, alpha=0.65)
        axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="0.25", lw=1.0)
        axis.set_xlim(lower - margin, upper + margin)
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_title(rf"$z={redshift:.2f}$, $\log_{{10}}L_{{\mathrm{{H}}\alpha}}={log10_center:.2f}$")
        axis.set_xlabel("True log10 LF")
        axis.set_ylabel("Predicted log10 LF")
        axis.grid(alpha=0.2)
    parity_path = figures_root / "gp_allz_cv_parity_2x2.png"
    fig.savefig(parity_path, dpi=220)
    plt.close(fig)

    summary = {
        "campaign_root": str(campaign_root),
        "table_filename": args.table_filename,
        "long_filename": args.long_filename,
        "n_splits": args.n_splits,
        "train_draws_per_eval": args.train_draws_per_eval,
        "n_restarts_optimizer": args.n_restarts_optimizer,
        "min_log10_lf": args.min_log10_lf,
        "max_abs_delta": args.max_abs_delta,
        "max_attenuation_scatter": args.max_attenuation_scatter,
        "input_transform": "prior_quantiles",
        "n_rows_total": int(len(table)),
        "n_evaluations": int(table["evaluation_id"].nunique()),
    }
    summary_path = emulator_root / "gp_allz_cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(metrics_path)
    print(floors_path)
    print(kernels_path)
    print(curves_path)
    print(parity_path)
    print(summary_path)


if __name__ == "__main__":
    main()
