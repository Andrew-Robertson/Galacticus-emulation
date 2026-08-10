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

from galacticus_emu.interactive_halpha import (
    benchmark_halpha_bundle,
    bundle_meta,
    train_halpha_bundle,
)


DEFAULT_CAMPAIGN_ROOT = REPO_ROOT / "runs" / "campaigns" / "disk_feedback_velocity_1d_mass_function_emissionlines_dust-mMax1e14_32_hpc_reduced"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "demo_emulators" / "interactive_halpha_demo" / "halpha_demo_bundle.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save a PCA Halpha bundle for the interactive slider demo."
    )
    parser.add_argument("campaign_root", nargs="?", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--table-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--train-draws-per-eval", type=int, default=32)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--pca-scaling", choices=["unscaled", "standardized"], default="unscaled")
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--training-preview-rows", type=int, default=64)
    parser.add_argument("--benchmark-predictions", type=int, default=200)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = train_halpha_bundle(
        args.campaign_root,
        table_filename=args.table_filename,
        long_filename=args.long_filename,
        train_draws_per_eval=args.train_draws_per_eval,
        pca_components=args.pca_components,
        pca_scaling=args.pca_scaling,
        n_restarts_optimizer=args.n_restarts_optimizer,
        random_state=args.random_state,
        min_log10_lf=args.min_log10_lf,
        max_abs_delta=args.max_abs_delta,
        max_attenuation_scatter=args.max_attenuation_scatter,
        training_preview_rows=args.training_preview_rows,
    )
    benchmark = benchmark_halpha_bundle(bundle, n_predictions=args.benchmark_predictions)
    bundle["prediction_benchmark"] = benchmark

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_path)

    meta = bundle_meta(bundle)
    meta["prediction_benchmark"] = benchmark
    meta_path = args.output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(args.output_path)
    print(meta_path)
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()
