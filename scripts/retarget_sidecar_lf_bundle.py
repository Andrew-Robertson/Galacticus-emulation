from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import joblib
import numpy as np

from galacticus_emu.persistence import load_emulator_bundle
from fit_sidecar_lf_pca_holdout_demo import (
    TARGET_ERROR_MODE_SOBRAL_LOG,
    TARGET_ERROR_MODES,
    _target_curve,
)
from train_sidecar_lf_observables_demo import _json_safe_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a saved sidecar LF PCA-GP bundle while replacing its target "
            "curve errors. This is useful when the emulator should be identical "
            "but the likelihood target-error convention changes."
        )
    )
    parser.add_argument("input_bundle", type=Path)
    parser.add_argument("output_bundle", type=Path)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=None,
        help=(
            "Campaign root to use when re-reading target curves. Defaults to the "
            "campaign_root stored in the input bundle. Use this when retargeting "
            "a bundle created on another machine."
        ),
    )
    parser.add_argument(
        "--target-error-mode",
        choices=TARGET_ERROR_MODES,
        default=TARGET_ERROR_MODE_SOBRAL_LOG,
        help="Target error convention to store in the copied bundle.",
    )
    parser.add_argument(
        "--min-log10-lf",
        type=float,
        default=None,
        help="Log10 LF floor used for target values. Defaults to the input bundle training option, then -8.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_emulator_bundle(args.input_bundle)
    if bundle.get("bundle_type") != "sidecar_lf_pca_gp":
        raise ValueError(f"{args.input_bundle} is not a sidecar_lf_pca_gp bundle")

    campaign_root = (
        args.campaign_root.expanduser().resolve()
        if args.campaign_root is not None
        else Path(bundle["campaign_root"]).expanduser().resolve()
    )
    training_options = dict(bundle.get("training_options", {}))
    min_log10_lf = float(args.min_log10_lf if args.min_log10_lf is not None else training_options.get("min_log10_lf", -8.0))

    for observable_key, observable in bundle["observables"].items():
        target_plot, target_sigma_plot = _target_curve(
            campaign_root,
            str(observable["observable"]),
            str(observable["sample_label"]),
            np.asarray(observable["x_plot"], dtype=float),
            min_log10_lf,
            target_error_mode=args.target_error_mode,
        )
        observable["target_plot"] = np.asarray(target_plot, dtype=float)
        observable["target_sigma_plot"] = None if target_sigma_plot is None else np.asarray(target_sigma_plot, dtype=float)
        observable["target_error_mode"] = args.target_error_mode
        print(
            f"{observable_key}: target_error_mode={args.target_error_mode}, "
            f"n_bins={len(target_plot)}",
            flush=True,
        )

    training_options["target_error_mode"] = args.target_error_mode
    training_options["min_log10_lf"] = min_log10_lf
    bundle["training_options"] = training_options
    bundle["campaign_root"] = str(campaign_root)

    args.output_bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_bundle)
    meta_path = args.output_bundle.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(_json_safe_meta(bundle), indent=2) + "\n")
    print(args.output_bundle)
    print(meta_path)


if __name__ == "__main__":
    main()
