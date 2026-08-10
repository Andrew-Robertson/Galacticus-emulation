from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu.interactive_smf import DEFAULT_ANALYSIS, bundle_meta, train_smf_bundle

DEFAULT_CAMPAIGN_ROOT = REPO_ROOT / "runs" / "campaigns" / "disk_feedback_velocity_1d_mass_function_emissionlines_dust-mMax1e14_32_hpc_reduced"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "demo_emulators" / "interactive_smf_demo" / "smf_demo_bundle.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save a tiny 1D SMF GP bundle for the interactive slider demo."
    )
    parser.add_argument(
        "campaign_root",
        nargs="?",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
        help="Reduced campaign root with galacticus_reduced.hdf5 outputs.",
    )
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--input-column", default="diskVelocityCharacteristic")
    parser.add_argument("--hdf5-filename", default="galacticus_reduced.hdf5")
    parser.add_argument("--n-restarts-optimizer", type=int, default=5)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = train_smf_bundle(
        args.campaign_root,
        analysis=args.analysis,
        input_columns=[args.input_column],
        hdf5_filename=args.hdf5_filename,
        n_restarts_optimizer=args.n_restarts_optimizer,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(bundle, args.output_path)
    meta_path = args.output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(bundle_meta(bundle), indent=2))
    print(args.output_path)
    print(meta_path)


if __name__ == "__main__":
    main()
