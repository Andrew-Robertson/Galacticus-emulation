from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import joblib


def _patch_numpy_bit_generator_unpickling() -> Callable[[], None]:
    try:
        import numpy.random._pickle as numpy_random_pickle
    except Exception:
        return lambda: None

    original = getattr(numpy_random_pickle, "__bit_generator_ctor", None)
    if original is None or getattr(original, "_galacticus_emu_compat", False):
        return lambda: None

    def compat_bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original(bit_generator_name)

    compat_bit_generator_ctor._galacticus_emu_compat = True  # type: ignore[attr-defined]
    numpy_random_pickle.__bit_generator_ctor = compat_bit_generator_ctor

    def restore() -> None:
        numpy_random_pickle.__bit_generator_ctor = original

    return restore


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
    restore_numpy_unpickling = _patch_numpy_bit_generator_unpickling()
    try:
        return joblib.load(Path(input_path))
    finally:
        restore_numpy_unpickling()
