from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from galacticus_emu.interactive_observables import refresh_observables_training_preview
from galacticus_emu.interactive_sidecar_lf import embed_sidecar_lf_training_preview
from galacticus_emu.persistence import load_emulator_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a compact emulator bundle that preserves mean predictions "
            "but omits the GP state required for predictive uncertainty."
        )
    )
    parser.add_argument("input_bundle", type=Path)
    parser.add_argument("output_bundle", type=Path)
    parser.add_argument("--compress", type=int, default=3)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        help="Campaign root used while embedding training previews and slider ranges.",
    )
    parser.add_argument(
        "--best-fit-summary",
        type=Path,
        help="MCMC run summary whose best_theta values are embedded in the bundle.",
    )
    parser.add_argument(
        "--training-preview-rows",
        type=int,
        help="Embed this many processed training curves in each observable.",
    )
    parser.add_argument(
        "--exclude-observable",
        action="append",
        default=[],
        help="Observable key to omit from the exported bundle. May be repeated.",
    )
    return parser.parse_args()


def _models(bundle: dict):
    for observable in bundle.get("observables", {}).values():
        yield from observable.get("models", [])


def strip_predictive_uncertainty(bundle: dict) -> dict:
    for model in _models(bundle):
        if not hasattr(model, "L_"):
            continue
        probe = np.asarray(model.X_train_[:1], dtype=float)
        expected = model.predict(probe, return_std=False)
        del model.L_
        actual = model.predict(probe, return_std=False)
        if not np.array_equal(expected, actual):
            raise RuntimeError("Removing predictive-variance state changed a GP mean prediction")
    bundle["supports_predictive_uncertainty"] = False
    bundle["deployment_format"] = "mean_only"
    return bundle


def embed_best_fit(bundle: dict, summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    best_theta = summary.get("best_theta")
    if not isinstance(best_theta, dict):
        raise ValueError(f"{summary_path} does not contain a best_theta mapping")
    parameter_names = list(
        bundle.get("parameter_names") or bundle.get("input_columns") or []
    )
    missing = [name for name in parameter_names if name not in best_theta]
    if missing:
        raise ValueError(
            f"{summary_path} is missing best-fit values for: {', '.join(missing)}"
        )
    bundle["best_fit_params"] = {
        name: float(best_theta[name]) for name in parameter_names
    }
    bundle["best_fit_source"] = str(summary_path)
    return bundle


def exclude_observables(bundle: dict, observable_keys: list[str]) -> dict:
    if not observable_keys:
        return bundle
    excluded = set(observable_keys)
    available = set(bundle.get("observables", {}))
    unknown = excluded - available
    if unknown:
        raise ValueError("Unknown observable key(s): " + ", ".join(sorted(unknown)))
    bundle["observables"] = {
        key: observable
        for key, observable in bundle["observables"].items()
        if key not in excluded
    }
    bundle["observable_keys"] = [
        key for key in bundle.get("observable_keys", []) if key not in excluded
    ]
    if "observable_configs" in bundle:
        bundle["observable_configs"] = {
            key: config
            for key, config in bundle["observable_configs"].items()
            if key not in excluded
        }
    return bundle


def embed_training_preview(bundle: dict, rows: int) -> dict:
    bundle_type = bundle.get("bundle_type")
    if bundle_type in {
        "interactive_observables_bin_by_bin_gp",
        "interactive_observables_pca_gp",
    }:
        return refresh_observables_training_preview(
            bundle,
            training_preview_rows=rows,
            use_training_targets=True,
        )
    if bundle_type == "sidecar_lf_pca_gp":
        bundle = embed_sidecar_lf_training_preview(
            bundle,
            training_preview_rows=rows,
        )
        empty = [
            key
            for key, observable in bundle["observables"].items()
            if np.asarray(observable["y_train_preview"]).size == 0
        ]
        if empty:
            raise RuntimeError(
                "No sidecar training previews were found for: " + ", ".join(empty)
            )
        return bundle
    raise ValueError(f"Unsupported bundle_type for training previews: {bundle_type}")


def main() -> None:
    args = parse_args()
    bundle = load_emulator_bundle(args.input_bundle)
    bundle = exclude_observables(bundle, args.exclude_observable)
    if args.campaign_root is not None:
        bundle["campaign_root"] = str(args.campaign_root)
    if args.training_preview_rows is not None:
        bundle = embed_training_preview(bundle, args.training_preview_rows)
    if args.best_fit_summary is not None:
        bundle = embed_best_fit(bundle, args.best_fit_summary)
    bundle["deployment_source_bundle"] = str(args.input_bundle)
    bundle = strip_predictive_uncertainty(bundle)
    args.output_bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_bundle, compress=args.compress)
    size_mib = args.output_bundle.stat().st_size / 1024**2
    print(f"Wrote {args.output_bundle} ({size_mib:.2f} MiB)")


if __name__ == "__main__":
    main()
