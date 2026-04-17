from __future__ import annotations

from pathlib import Path

import joblib


def sanitize_output_name(name: str) -> str:
    return name.replace("/", "_")


def save_emulator_bundle(
    output_path: str | Path,
    *,
    model,
    input_columns: list[str],
    output_column: str,
    y_mean: float,
    y_std: float,
    kernel_optimized: str,
    log_marginal_likelihood: float,
    optimize_hyperparameters: bool,
    n_restarts_optimizer: int,
    cv_folds: int | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "input_columns": input_columns,
        "output_column": output_column,
        "y_mean": y_mean,
        "y_std": y_std,
        "kernel_optimized": kernel_optimized,
        "log_marginal_likelihood": log_marginal_likelihood,
        "optimize_hyperparameters": optimize_hyperparameters,
        "n_restarts_optimizer": n_restarts_optimizer,
        "cv_folds": cv_folds,
    }
    joblib.dump(bundle, output_path)
    return output_path


def load_emulator_bundle(input_path: str | Path):
    return joblib.load(Path(input_path))
