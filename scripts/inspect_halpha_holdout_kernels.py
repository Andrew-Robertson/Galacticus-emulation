from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--holdout-evaluation-id", required=True)
    parser.add_argument("--train-draws-per-eval", type=int, default=8)
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _kernel(n_dim: int):
    return ConstantKernel(1.0, (1.0e-3, 1.0e3)) * Matern(
        length_scale=np.full(n_dim, 0.5, dtype=float),
        length_scale_bounds=(1.0e-2, 1.0e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1.0e-4, noise_level_bounds=(1.0e-8, 1.0))


def _log10_with_floor(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    positive = values[values > 0.0]
    floor = 0.5 * float(np.min(positive))
    safe = np.where(values > 0.0, values, floor)
    return np.log10(safe), floor


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.campaign_root / args.table_filename)
    train = (
        table.loc[table["evaluation_id"] != args.holdout_evaluation_id]
        .sort_values(["evaluation_id", "dust_draw_index"])
        .groupby("evaluation_id", group_keys=False)
        .head(args.train_draws_per_eval)
        .reset_index(drop=True)
    )

    x_raw = train[INPUT_COLUMNS].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(INPUT_PARAMETER_SPECS, x_raw)
    print("training rows", len(train))
    print("input raw mins", dict(zip(INPUT_COLUMNS, np.min(x_raw, axis=0), strict=True)))
    print("input raw maxs", dict(zip(INPUT_COLUMNS, np.max(x_raw, axis=0), strict=True)))
    print("input quantile mins", dict(zip(INPUT_COLUMNS, np.min(x, axis=0), strict=True)))
    print("input quantile maxs", dict(zip(INPUT_COLUMNS, np.max(x, axis=0), strict=True)))

    for column in args.columns:
        y, floor = _log10_with_floor(train[column].to_numpy(dtype=float))
        y_mean = float(np.mean(y))
        y_std = float(np.std(y)) or 1.0
        gp = GaussianProcessRegressor(
            kernel=_kernel(x.shape[1]),
            normalize_y=False,
            n_restarts_optimizer=args.n_restarts_optimizer,
            random_state=args.random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(x, (y - y_mean) / y_std)
        print()
        print(column)
        print("floor", floor)
        print("kernel_", gp.kernel_)
        print("log_marginal_likelihood", gp.log_marginal_likelihood(gp.kernel_.theta))


if __name__ == "__main__":
    main()
