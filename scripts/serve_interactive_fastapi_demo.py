from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PAPER_CAMPAIGN_ROOT = (
    REPO_ROOT
    / "runs"
    / "campaigns"
    / "sobol_1024_20p_simpleSizes_moreHalos_reduced"
)
PAPER_EMULATOR_ROOT = (
    PAPER_CAMPAIGN_ROOT
    / "automatedPipeline_transformedParams_finalPaper_definitive"
    / "emulators"
)
PAPER_STANDARD_BUNDLE_PATH = (
    PAPER_EMULATOR_ROOT
    / "standard_observables"
    / "pca_99"
    / "standard_observables_bundle.joblib"
)
PAPER_SIDECAR_BUNDLE_PATH = (
    PAPER_CAMPAIGN_ROOT
    / "scratch"
    / "sobral_log_errors_local"
    / "emulators"
    / "emission_line_lfs"
    / "pca_99"
    / "halpha_sobral_1dustdraw_pca99_sobralLogErr.joblib"
)
PAPER_BEST_FIT_SUMMARY_PATH = (
    REPO_ROOT
    / "paper"
    / "figure_data"
    / "final_calibration"
    / "mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr_run_summary.json"
)
DEPLOYMENT_BUNDLE_ROOT = REPO_ROOT / "demo_emulators" / "paper"
DEPLOYMENT_STANDARD_BUNDLE_PATH = (
    DEPLOYMENT_BUNDLE_ROOT / "standard_observables_mean_only.joblib"
)
DEPLOYMENT_SIDECAR_BUNDLE_PATH = (
    DEPLOYMENT_BUNDLE_ROOT / "halpha_sobral_log_error_mean_only.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the interactive emulator demos with FastAPI/Uvicorn.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--deployment",
        action="store_true",
        help="Use the compact, self-contained paper bundles committed for hosting.",
    )
    return parser.parse_args()


def configure_demo_app(*, deployment: bool = False) -> None:
    os.environ["GALACTICUS_EMU_PROJECT_ROOT"] = str(REPO_ROOT)
    os.environ["INTERACTIVE_SMF_ENABLED"] = "0"
    os.environ["INTERACTIVE_HALPHA_ENABLED"] = "0"
    os.environ.pop("INTERACTIVE_CAMPAIGN_ROOT", None)

    if deployment:
        standard_bundle = DEPLOYMENT_STANDARD_BUNDLE_PATH
        sidecar_bundle = DEPLOYMENT_SIDECAR_BUNDLE_PATH
    else:
        standard_bundle = (
            PAPER_STANDARD_BUNDLE_PATH
            if PAPER_STANDARD_BUNDLE_PATH.exists()
            else DEPLOYMENT_STANDARD_BUNDLE_PATH
        )
        sidecar_bundle = (
            PAPER_SIDECAR_BUNDLE_PATH
            if PAPER_SIDECAR_BUNDLE_PATH.exists()
            else DEPLOYMENT_SIDECAR_BUNDLE_PATH
        )
        if standard_bundle == PAPER_STANDARD_BUNDLE_PATH:
            os.environ["INTERACTIVE_CAMPAIGN_ROOT"] = str(PAPER_CAMPAIGN_ROOT)

    os.environ["INTERACTIVE_OBSERVABLES_ENABLED"] = (
        "1" if standard_bundle.exists() else "0"
    )
    os.environ["INTERACTIVE_OBSERVABLES_BUNDLE_PATH"] = str(standard_bundle)
    os.environ["INTERACTIVE_SIDECAR_LF_ENABLED"] = (
        "1" if sidecar_bundle.exists() else "0"
    )
    os.environ["INTERACTIVE_SIDECAR_LF_BUNDLE_PATH"] = str(sidecar_bundle)

    if PAPER_BEST_FIT_SUMMARY_PATH.exists():
        summary_path = str(PAPER_BEST_FIT_SUMMARY_PATH)
        os.environ["INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH"] = summary_path
        os.environ["INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH"] = summary_path
    else:
        os.environ.pop("INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH", None)
        os.environ.pop("INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH", None)

    if not deployment and standard_bundle == PAPER_STANDARD_BUNDLE_PATH:
        os.environ["INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS"] = "64"
    else:
        os.environ.pop("INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS", None)
    if not deployment and sidecar_bundle == PAPER_SIDECAR_BUNDLE_PATH:
        os.environ["INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS"] = "64"
    else:
        os.environ.pop("INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS", None)


def main() -> None:
    args = parse_args()
    configure_demo_app(deployment=args.deployment)

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
