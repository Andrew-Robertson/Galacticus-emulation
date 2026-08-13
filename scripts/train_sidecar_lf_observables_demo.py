from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from fit_sidecar_lf_pca_holdout_demo import (
    TARGET_ERROR_MODE_LEGACY,
    TARGET_ERROR_MODES,
    _alpha_log10,
    _bin_metadata,
    _input_columns,
    _log10_with_floor,
    _output_columns,
    _read_first_sidecar_table,
    _read_sidecar_tables,
    _sample_labels,
    _target_curve,
)
from run_sidecar_lf_pca_emulator_mcmc import (
    _feature_matrix_from_table,
    _filter_table,
    _fit_pca_bundle,
    _parameter_specs_for_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save PCA GP emulator bundles for sidecar emission-line luminosity functions."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--sidecar-dir", default="emission_line_dust")
    parser.add_argument("--table-filename", default="emission_line_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="emission_line_dust_lf_long.csv")
    parser.add_argument("--observable", action="append", required=True)
    parser.add_argument(
        "--sample-label",
        action="append",
        required=True,
        help="Sample/redshift label to train for each observable, e.g. Z1. Repeat for multiple.",
    )
    parser.add_argument("--dust-draw-index", type=int, action="append", default=None)
    parser.add_argument(
        "--max-dust-draws-per-eval",
        type=int,
        default=None,
        help="After other filters, keep the first N dust rows per Galacticus evaluation.",
    )
    parser.add_argument("--min-log10-lf", type=float, default=-8.0)
    parser.add_argument(
        "--target-error-mode",
        choices=TARGET_ERROR_MODES,
        default=TARGET_ERROR_MODE_LEGACY,
        help=(
            "How to convert H-alpha Sobral target LF errors into log10(Phi) space. "
            "The default preserves legacy runs; use sobral_log_table for the "
            "Sobral table's log-space errors."
        ),
    )
    parser.add_argument("--use-training-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-training-log10-sigma", type=float, default=1.0e-4)
    parser.add_argument("--max-training-log10-sigma", type=float, default=2.0)
    parser.add_argument("--n-restarts-optimizer", type=int, default=0)
    parser.add_argument("--optimize-hyperparameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument(
        "--pca-variance-threshold",
        type=float,
        default=None,
        help="Choose enough PCA components to explain this variance fraction, e.g. 0.99.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def _observable_key(observable: str, sample_label: str) -> str:
    return f"{observable}_{sample_label.lower().replace('.', 'p')}"


def _json_safe_meta(bundle: dict) -> dict:
    return {
        "bundle_type": bundle["bundle_type"],
        "campaign_root": bundle["campaign_root"],
        "sidecar_dir": bundle["sidecar_dir"],
        "table_filename": bundle["table_filename"],
        "long_filename": bundle["long_filename"],
        "input_columns": bundle["input_columns"],
        "parameter_names": bundle["parameter_names"],
        "slow_count": bundle["slow_count"],
        "n_training_rows": bundle["n_training_rows"],
        "observable_groups": bundle["observable_groups"],
        "observables": {
            key: {
                "observable_key": value["observable_key"],
                "observable": value["observable"],
                "sample_label": value["sample_label"],
                "target_error_mode": value.get("target_error_mode"),
                "n_bins": int(len(value["x_plot"])),
                "pca_components_actual": int(value["pca_components_actual"]),
                "pca_variance_threshold": value["pca_variance_threshold"],
                "pca_explained_variance_ratio_sum": float(np.sum(value["pca_explained_variance_ratio"])),
            }
            for key, value in bundle["observables"].items()
        },
        "training_options": bundle["training_options"],
    }


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    campaign_root = args.campaign_root.resolve()
    table = _filter_table(_read_sidecar_tables(campaign_root, args.sidecar_dir, args.table_filename), args)
    long_table = _read_first_sidecar_table(campaign_root, args.sidecar_dir, args.long_filename)
    input_columns = _input_columns(table)
    parameter_specs, parameter_names, dust_names = _parameter_specs_for_table(input_columns)
    slow_count = len([column for column in input_columns if column.endswith("_quantile")])
    x = _feature_matrix_from_table(table, input_columns, dust_names)

    fit_args = SimpleNamespace(
        min_log10_lf=args.min_log10_lf,
        min_training_log10_sigma=args.min_training_log10_sigma,
        max_training_log10_sigma=args.max_training_log10_sigma,
        n_restarts_optimizer=args.n_restarts_optimizer,
        optimize_hyperparameters=args.optimize_hyperparameters,
        pca_components=args.pca_components,
        pca_variance_threshold=args.pca_variance_threshold,
    )

    observables = {}
    observable_groups: dict[str, list[str]] = {}
    for observable in args.observable:
        observable_groups.setdefault(observable, [])
        for requested_sample_label in args.sample_label:
            output_columns = _output_columns(table, observable, [requested_sample_label])
            sample_labels = _sample_labels(output_columns)
            if not sample_labels or any(label != sample_labels[0] for label in sample_labels):
                raise ValueError(f"Could not infer a single sample label for {observable!r} {requested_sample_label!r}")
            sample_label = sample_labels[0]
            bin_metadata = _bin_metadata(long_table, observable, sample_label, output_columns)
            x_plot = bin_metadata["log10_luminosity_center"].to_numpy(dtype=float)
            target_plot, target_std_plot = _target_curve(
                campaign_root,
                observable,
                sample_label,
                x_plot,
                args.min_log10_lf,
                target_error_mode=args.target_error_mode,
            )
            y, _floor = _log10_with_floor(table[output_columns].to_numpy(dtype=float), args.min_log10_lf)
            alpha = None
            if args.use_training_alpha:
                alpha_sigma = _alpha_log10(
                    table,
                    output_columns,
                    min_sigma=args.min_training_log10_sigma,
                    max_sigma=args.max_training_log10_sigma,
                )
                alpha = alpha_sigma**2
            key = _observable_key(observable, sample_label)
            sidecar_bundle = _fit_pca_bundle(
                x=x,
                y=y,
                alpha=alpha,
                observable_key=observable,
                sample_label=sample_label,
                output_columns=output_columns,
                x_plot=x_plot,
                target_plot=target_plot,
                target_std_plot=target_std_plot,
                args=fit_args,
            )
            sidecar_bundle["observable"] = observable
            sidecar_bundle["key"] = key
            sidecar_bundle["label"] = f"{observable} {sample_label}"
            sidecar_bundle["target_error_mode"] = args.target_error_mode
            observables[key] = sidecar_bundle
            observable_groups[observable].append(key)

    bundle = {
        "bundle_type": "sidecar_lf_pca_gp",
        "campaign_root": str(campaign_root),
        "sidecar_dir": args.sidecar_dir,
        "table_filename": args.table_filename,
        "long_filename": args.long_filename,
        "input_columns": input_columns,
        "parameter_names": parameter_names,
        "parameter_specs": parameter_specs,
        "slow_count": slow_count,
        "observables": observables,
        "observable_groups": observable_groups,
        "n_training_rows": int(len(table)),
        "training_options": {
            "dust_draw_index": args.dust_draw_index,
            "max_dust_draws_per_eval": args.max_dust_draws_per_eval,
            "min_log10_lf": args.min_log10_lf,
            "target_error_mode": args.target_error_mode,
            "use_training_alpha": args.use_training_alpha,
            "min_training_log10_sigma": args.min_training_log10_sigma,
            "max_training_log10_sigma": args.max_training_log10_sigma,
            "n_restarts_optimizer": args.n_restarts_optimizer,
            "optimize_hyperparameters": args.optimize_hyperparameters,
            "pca_components": args.pca_components,
            "pca_variance_threshold": args.pca_variance_threshold,
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_path)
    meta_path = args.output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(_json_safe_meta(bundle), indent=2) + "\n")
    print(args.output_path)
    print(meta_path)


if __name__ == "__main__":
    main()
