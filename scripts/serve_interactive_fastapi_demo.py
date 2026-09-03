from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PAPER_EMULATOR_ROOT = (
    REPO_ROOT
    / "runs"
    / "campaigns"
    / "sobol_1024_20p_simpleSizes_moreHalos_reduced"
    / "pipeline"
    / "emulators"
)
PAPER_PIPELINE_ROOT = PAPER_EMULATOR_ROOT.parent
PAPER_STANDARD_BUNDLE_PATH = (
    PAPER_EMULATOR_ROOT
    / "standard_observables"
    / "pca_99"
    / "standard_observables_bundle.joblib"
)
PAPER_SIDECAR_BUNDLE_PATH = (
    PAPER_EMULATOR_ROOT
    / "emission_line_lfs"
    / "pca_99"
    / "halpha_sobral_1dustdraw_pca99.joblib"
)
PAPER_STANDARD_BEST_FIT_SUMMARY_PATH = (
    PAPER_PIPELINE_ROOT
    / "MCMCs"
    / "standard_observables"
    / "all_standard_observables"
    / "mcmc_all_standard_observables_run_summary.json"
)
PAPER_SIDECAR_BEST_FIT_SUMMARY_PATH = (
    PAPER_PIPELINE_ROOT
    / "transformedParams_MCMC_exploratory"
    / "standard_observables_plus_emission_line_lfs"
    / "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99"
    / "mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_run_summary.json"
)
CHECKED_IN_STANDARD_BUNDLE_PATH = (
    REPO_ROOT
    / "demo_emulators"
    / "interactive_observables_demo"
    / "observables_demo_bundle_pca.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the interactive emulator demos with FastAPI/Uvicorn.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def configure_demo_app() -> None:
    os.environ["GALACTICUS_EMU_PROJECT_ROOT"] = str(REPO_ROOT)
    os.environ["INTERACTIVE_SMF_ENABLED"] = "0"
    os.environ["INTERACTIVE_HALPHA_ENABLED"] = "0"

    standard_bundle = (
        PAPER_STANDARD_BUNDLE_PATH
        if PAPER_STANDARD_BUNDLE_PATH.exists()
        else CHECKED_IN_STANDARD_BUNDLE_PATH
    )
    os.environ["INTERACTIVE_OBSERVABLES_ENABLED"] = "1"
    os.environ["INTERACTIVE_OBSERVABLES_BUNDLE_PATH"] = str(standard_bundle)
    if standard_bundle == PAPER_STANDARD_BUNDLE_PATH:
        os.environ["INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH"] = str(
            PAPER_STANDARD_BEST_FIT_SUMMARY_PATH
        )
        os.environ["INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS"] = "64"
    else:
        os.environ.pop("INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH", None)
        os.environ.pop("INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS", None)

    if PAPER_SIDECAR_BUNDLE_PATH.exists():
        os.environ["INTERACTIVE_SIDECAR_LF_ENABLED"] = "1"
        os.environ["INTERACTIVE_SIDECAR_LF_BUNDLE_PATH"] = str(
            PAPER_SIDECAR_BUNDLE_PATH
        )
        os.environ["INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH"] = str(
            PAPER_SIDECAR_BEST_FIT_SUMMARY_PATH
        )
        os.environ["INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS"] = "64"
    else:
        os.environ["INTERACTIVE_SIDECAR_LF_ENABLED"] = "0"
        os.environ.pop("INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH", None)
        os.environ.pop("INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS", None)


def main() -> None:
    args = parse_args()
    configure_demo_app()

    import uvicorn

    uvicorn.run(
        "galacticus_emu.interactive_fastapi:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
