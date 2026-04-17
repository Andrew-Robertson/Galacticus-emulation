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

import pandas as pd

from galacticus_emu import (
    TRINITY_TRAINABLE_OUTPUT_COLUMNS,
    fit_scaled_gp,
    load_or_build_emulator_table,
    sanitize_output_name,
    save_emulator_bundle,
    transform_to_prior_quantiles,
    trinity_parameter_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and serialize GPs for all 36 Trinity outputs.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument("--output-suffix", default="_all36_opt", help="Suffix appended to saved model filenames.")
    parser.add_argument("--n-restarts-optimizer", type=int, default=1)
    parser.add_argument("--optimize-hyperparameters", action="store_true", help="Optimize GP hyperparameters instead of keeping the initial kernel fixed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")

    campaign_root = Path(args.campaign_root).resolve()
    emulator_root = campaign_root / "emulator"
    models_root = emulator_root / "models"
    emulator_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    table = load_or_build_emulator_table(campaign_root)
    input_columns = [spec.short_name for spec in trinity_parameter_specs()]
    x_raw = table[input_columns].to_numpy(dtype=float)
    x = transform_to_prior_quantiles(trinity_parameter_specs(), x_raw)

    model_summaries = []
    for output_name in TRINITY_TRAINABLE_OUTPUT_COLUMNS:
        y = table[output_name].to_numpy(dtype=float)
        model, y_mean, y_std = fit_scaled_gp(
            x=x,
            y=y,
            n_restarts_optimizer=args.n_restarts_optimizer,
            optimize_hyperparameters=args.optimize_hyperparameters,
        )
        kernel_optimized = str(model.kernel_)
        log_marginal_likelihood = float(model.log_marginal_likelihood(model.kernel_.theta))

        model_filename = f"{sanitize_output_name(output_name)}{args.output_suffix}.joblib"
        model_path = save_emulator_bundle(
            models_root / model_filename,
            model=model,
            input_columns=input_columns,
            output_column=output_name,
            y_mean=y_mean,
            y_std=y_std,
            kernel_optimized=kernel_optimized,
            log_marginal_likelihood=log_marginal_likelihood,
            optimize_hyperparameters=args.optimize_hyperparameters,
            n_restarts_optimizer=args.n_restarts_optimizer,
            cv_folds=None,
        )
        model_summaries.append(
            {
                "output": output_name,
                "model_path": str(model_path),
                "kernel_optimized": kernel_optimized,
                "log_marginal_likelihood": log_marginal_likelihood,
                "y_mean": y_mean,
                "y_std": y_std,
            }
        )

    summary_path = emulator_root / f"gp_fit_summary_all_outputs{args.output_suffix}.json"
    summary_path.write_text(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "input_columns": input_columns,
                "output_columns": TRINITY_TRAINABLE_OUTPUT_COLUMNS,
                "n_restarts_optimizer": args.n_restarts_optimizer,
                "optimize_hyperparameters": args.optimize_hyperparameters,
                "models_root": str(models_root),
                "models": model_summaries,
            },
            indent=2,
        )
        + "\n"
    )

    print(summary_path)
    for item in model_summaries:
        print(item["model_path"])


if __name__ == "__main__":
    main()
