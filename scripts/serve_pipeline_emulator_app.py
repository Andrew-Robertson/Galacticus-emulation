from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CAMPAIGN_ROOT = (
    REPO_ROOT
    / "runs"
    / "campaigns"
    / "sobol_1024_20p_simpleSizes_moreHalos_reduced"
)
DEFAULT_EMULATOR_ROOT = (
    DEFAULT_CAMPAIGN_ROOT
    / "pipeline"
    / "emulators"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the interactive emulator app for a campaign pipeline/emulators directory."
    )
    parser.add_argument(
        "emulator_root",
        nargs="?",
        type=Path,
        default=DEFAULT_EMULATOR_ROOT,
        help=(
            "Campaign root, pipeline directory, or pipeline/emulators directory. "
            f"Defaults to {DEFAULT_CAMPAIGN_ROOT.relative_to(REPO_ROOT)}."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--standard-bundle", type=Path, default=None)
    parser.add_argument("--sidecar-lf-bundle", type=Path, default=None)
    parser.add_argument("--standard-best-fit-summary", type=Path, default=None)
    parser.add_argument("--sidecar-lf-best-fit-summary", type=Path, default=None)
    parser.add_argument(
        "--observables-training-preview-rows",
        default="64",
        help="Refresh standard-observable training preview curves from processed training targets.",
    )
    parser.add_argument(
        "--sidecar-lf-training-preview-rows",
        default="64",
        help="Refresh sidecar luminosity-function training preview curves from reduced sidecar tables.",
    )
    parser.add_argument(
        "--include-demo-defaults",
        action="store_true",
        help="Also show any default SMF/Halpha demo emulators present under demo_emulators/.",
    )
    return parser.parse_args()


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _resolve_emulator_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "standard_observables").exists() or (root / "emission_line_lfs").exists():
        return root
    if (root / "emulators").exists():
        return (root / "emulators").resolve()
    if (root / "pipeline" / "emulators").exists():
        return (root / "pipeline" / "emulators").resolve()
    return root


def _discover_standard_bundle(emulator_root: Path) -> Path | None:
    preferred = emulator_root / "standard_observables" / "pca_99" / "standard_observables_bundle.joblib"
    candidates = [preferred, *sorted(emulator_root.glob("standard_observables/*/standard_observables_bundle.joblib"))]
    return _first_existing(candidates)


def _discover_sidecar_lf_bundle(emulator_root: Path) -> Path | None:
    preferred = emulator_root / "emission_line_lfs" / "pca_99" / "halpha_sobral_1dustdraw_pca99.joblib"
    candidates = [preferred, *sorted(emulator_root.glob("emission_line_lfs/*/*.joblib"))]
    return _first_existing(candidates)


def _standard_summary_path(pipeline_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return _first_existing(
        [
            pipeline_root
            / "MCMCs"
            / "standard_observables"
            / "all_standard_observables"
            / "mcmc_all_standard_observables_run_summary.json",
            pipeline_root
            / "transformedParams_MCMCs"
            / "standard_observables"
            / "all_standard_observables"
            / "mcmc_all_standard_observables_run_summary.json",
        ]
    )


def _sidecar_summary_path(pipeline_root: Path, sidecar_bundle: Path | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if sidecar_bundle is None:
        return None
    tag = sidecar_bundle.stem
    mcmc_dir_name = f"all_standard_observables_plus_{tag}"
    summary_name = f"mcmc_all_standard_observables_plus_{tag}_run_summary.json"
    return _first_existing(
        [
            pipeline_root
            / "transformedParams_MCMC_production_from_exploratory_MAP"
            / "standard_observables_plus_emission_line_lfs"
            / mcmc_dir_name
            / summary_name,
            pipeline_root
            / "transformedParams_MCMC_exploratory"
            / "standard_observables_plus_emission_line_lfs"
            / mcmc_dir_name
            / summary_name,
        ]
    )


def _set_app_environment(args: argparse.Namespace) -> None:
    emulator_root = _resolve_emulator_root(args.emulator_root)
    pipeline_root = emulator_root.parent
    os.environ["GALACTICUS_EMU_PROJECT_ROOT"] = str(REPO_ROOT)

    if not args.include_demo_defaults:
        os.environ["INTERACTIVE_SMF_ENABLED"] = "0"
        os.environ["INTERACTIVE_HALPHA_ENABLED"] = "0"

    standard_bundle = args.standard_bundle.expanduser().resolve() if args.standard_bundle else _discover_standard_bundle(emulator_root)
    if standard_bundle is None:
        os.environ["INTERACTIVE_OBSERVABLES_ENABLED"] = "0"
    else:
        os.environ["INTERACTIVE_OBSERVABLES_ENABLED"] = "1"
        os.environ["INTERACTIVE_OBSERVABLES_BUNDLE_PATH"] = str(standard_bundle)
        preview_rows = getattr(args, "observables_training_preview_rows", "64")
        if preview_rows is not None:
            os.environ.setdefault("INTERACTIVE_OBSERVABLES_TRAINING_PREVIEW_ROWS", str(preview_rows))
        standard_summary = _standard_summary_path(pipeline_root, args.standard_best_fit_summary)
        if standard_summary is not None:
            os.environ["INTERACTIVE_OBSERVABLES_BEST_FIT_SUMMARY_PATH"] = str(standard_summary)

    sidecar_bundle = args.sidecar_lf_bundle.expanduser().resolve() if args.sidecar_lf_bundle else _discover_sidecar_lf_bundle(emulator_root)
    if sidecar_bundle is None:
        os.environ["INTERACTIVE_SIDECAR_LF_ENABLED"] = "0"
    else:
        os.environ["INTERACTIVE_SIDECAR_LF_ENABLED"] = "1"
        os.environ["INTERACTIVE_SIDECAR_LF_BUNDLE_PATH"] = str(sidecar_bundle)
        preview_rows = getattr(args, "sidecar_lf_training_preview_rows", "64")
        if preview_rows is not None:
            os.environ.setdefault("INTERACTIVE_SIDECAR_LF_TRAINING_PREVIEW_ROWS", str(preview_rows))
        sidecar_summary = _sidecar_summary_path(pipeline_root, sidecar_bundle, args.sidecar_lf_best_fit_summary)
        if sidecar_summary is not None:
            os.environ["INTERACTIVE_SIDECAR_LF_BEST_FIT_SUMMARY_PATH"] = str(sidecar_summary)


def main() -> None:
    args = parse_args()
    _set_app_environment(args)
    import uvicorn

    print(f"Serving Galacticus emulator app at http://{args.host}:{args.port}/")
    print(f"Standard observables: {os.environ.get('INTERACTIVE_OBSERVABLES_BUNDLE_PATH', 'disabled')}")
    print(f"Sidecar luminosity functions: {os.environ.get('INTERACTIVE_SIDECAR_LF_BUNDLE_PATH', 'disabled')}")
    uvicorn.run(
        "galacticus_emu.interactive_fastapi:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
