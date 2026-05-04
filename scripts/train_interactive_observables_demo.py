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
from sklearn.exceptions import ConvergenceWarning

from galacticus_emu.interactive_observables import (
    DEFAULT_PCA_COMPONENTS,
    OBSERVABLE_CONFIGS,
    benchmark_observables_bundle,
    bundle_meta,
    train_observables_bundle,
)


DEFAULT_CAMPAIGN_ROOT = REPO_ROOT / "runs" / "campaigns" / "sobol_mass_function_emissionlines_dust_19p_512_reduced"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "playing" / "interactive_observables_demo" / "observables_demo_bundle.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save a multi-observable bundle for the interactive Sobol demo."
    )
    parser.add_argument("campaign_root", nargs="?", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument(
        "--observable",
        dest="observables",
        action="append",
        choices=sorted(OBSERVABLE_CONFIGS),
        help="Train only a selected observable key. Repeat to include multiple.",
    )
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument(
        "--optimize-hyperparameters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-log10-y", type=float, default=-7.0)
    parser.add_argument("--training-preview-rows", type=int, default=64)
    parser.add_argument("--benchmark-predictions", type=int, default=50)
    parser.add_argument("--emulator-mode", choices=["bin_by_bin", "pca"], default="bin_by_bin")
    parser.add_argument("--pca-scaling", choices=["standardized", "unscaled"], default="standardized")
    parser.add_argument("--default-pca-components", type=int, default=None)
    parser.add_argument(
        "--pca-components",
        action="append",
        default=[],
        help="Override PCA components as observable_key=n, e.g. smf_z0=6",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _parse_pca_component_overrides(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --pca-components value {value!r}; expected observable_key=n")
        key, raw_count = value.split("=", 1)
        key = key.strip()
        if key not in OBSERVABLE_CONFIGS:
            raise ValueError(f"Unknown observable key {key!r} in --pca-components")
        result[key] = int(raw_count)
    return result


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    pca_component_overrides = _parse_pca_component_overrides(args.pca_components)
    if args.default_pca_components is not None:
        pca_components = {key: int(args.default_pca_components) for key in (args.observables or OBSERVABLE_CONFIGS.keys())}
        pca_components.update(pca_component_overrides)
    else:
        pca_components = dict(DEFAULT_PCA_COMPONENTS)
        pca_components.update(pca_component_overrides)
    bundle = train_observables_bundle(
        args.campaign_root,
        hdf5_filename=args.hdf5_filename,
        observable_keys=args.observables,
        n_restarts_optimizer=args.n_restarts_optimizer,
        optimize_hyperparameters=args.optimize_hyperparameters,
        min_log10_y=args.min_log10_y,
        training_preview_rows=args.training_preview_rows,
        emulator_mode=args.emulator_mode,
        pca_components=pca_components,
        pca_scaling=args.pca_scaling,
    )
    benchmark = benchmark_observables_bundle(bundle, n_predictions=args.benchmark_predictions)
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
