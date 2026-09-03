from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CHECKED_IN_DEMOS = {
    "INTERACTIVE_SMF_BUNDLE_PATH": (
        "demo_emulators/interactive_smf_demo/smf_demo_bundle.joblib"
    ),
    "INTERACTIVE_HALPHA_BUNDLE_PATH": (
        "demo_emulators/interactive_halpha_demo/halpha_demo_bundle_8draws.joblib"
    ),
    "INTERACTIVE_OBSERVABLES_BUNDLE_PATH": (
        "demo_emulators/interactive_observables_demo/observables_demo_bundle_pca.joblib"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the interactive emulator demos with FastAPI/Uvicorn.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def configure_checked_in_demos() -> None:
    os.environ["GALACTICUS_EMU_PROJECT_ROOT"] = str(REPO_ROOT)
    for variable, relative_path in CHECKED_IN_DEMOS.items():
        os.environ[variable] = str(REPO_ROOT / relative_path)


def main() -> None:
    args = parse_args()
    configure_checked_in_demos()

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
