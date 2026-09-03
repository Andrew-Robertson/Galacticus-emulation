#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path("runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced")
FINAL_MCMC_ROOT = (
    CAMPAIGN
    / "automatedPipeline_transformedParams_finalPaper_definitive"
    / "MCMCs_production_from_exploratory_MAP"
)
FINAL_JOINT_RUN = (
    FINAL_MCMC_ROOT
    / "standard_observables_plus_emission_line_lfs"
    / "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"
)

GALACTICUS_PARAMETERS = [
    "diskSFRFrequencyNorm",
    "spheroidSFREfficiency",
    "stellarPopulationMetalYield",
    "diskVelocityCharacteristic",
    "diskExponent",
    "spheroidVelocityCharacteristic",
    "coolingMultiplier",
    "coreRadiusOverVirialRadius",
    "henriquesGamma",
    "henriquesDelta1",
    "henriquesDelta2",
    "massRatioMajorMerger",
    "energyOrbital",
    "barFractionAngularMomentumRetainedSpheroid",
    "blackHoleSeedMass",
    "BHefficiencyRadioMode",
    "BHefficiencyWind",
    "bondiEnhancementSpheroid",
    "bondiEnhancementHotHalo",
    "thinDiskMaximum",
]

DUST_PARAMETERS = [
    "delta_0",
    "delta_M",
    "delta_z",
    "delta_Mz",
    "attenuation_scatter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate the final posterior corner figure with a five-parameter dust inset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT,
        help="Root containing the repo-relative runs/ tree, e.g. a checkout after extracting Zenodo archives.",
    )
    parser.add_argument(
        "--standard-results",
        type=Path,
        default=None,
        help="Standard-observable-only *_mcmc_results.hdf5 file.",
    )
    parser.add_argument(
        "--combined-results",
        type=Path,
        default=None,
        help="Final standard-plus-Halpha *_mcmc_results.hdf5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the final joint run paperFigures directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Final overlaid PDF path. Defaults to OUTPUT_DIR/final_posterior_corner_galacticus20_with_dust5_inset_inwards.pdf.",
    )
    parser.add_argument(
        "--base-pdf",
        type=Path,
        default=None,
        help="Intermediate 20-parameter corner PDF.",
    )
    parser.add_argument(
        "--dust-pdf",
        type=Path,
        default=None,
        help="Intermediate five-parameter dust corner PDF.",
    )
    parser.add_argument(
        "--skip-corner-generation",
        action="store_true",
        help="Only overlay existing --base-pdf and --dust-pdf files.",
    )
    parser.add_argument("--base-width-inch", type=float, default=1177.804001 / 72.0)
    parser.add_argument("--dust-width-inch", type=float, default=293.077501 / 72.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--inset-scale", type=float, default=1.001242187)
    parser.add_argument("--inset-x", type=float, default=738.362443)
    parser.add_argument("--inset-y", type=float, default=744.183123)
    parser.add_argument(
        "--keep-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the generated base and dust PDFs next to the final output.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(shlex.join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _default_standard_results(data_root: Path) -> Path:
    return (
        data_root
        / FINAL_MCMC_ROOT
        / "standard_observables"
        / "all_standard_observables"
        / "mcmc_all_standard_observables_mcmc_results.hdf5"
    )


def _default_combined_results(data_root: Path) -> Path:
    return data_root / FINAL_JOINT_RUN / (
        "mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"
        "_mcmc_results.hdf5"
    )


def _corner_command(
    *,
    output_path: Path,
    posterior_args: list[tuple[str, Path]],
    parameters: list[str],
    width_inch: float,
    dpi: int,
    marker_args: list[tuple[str, Path]],
    legend_loc: str,
    axes_labelsize: float,
    tick_labelsize: float,
    linewidth_scale: float,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/plot_observable_mcmc_overlay_getdist.py"),
        "--parameter-set",
        "joint24",
        "--label-style",
        "paper",
        "--label-rotation-style",
        "straight",
        "--filled",
        "--width-inch",
        f"{width_inch:.8g}",
        "--dpi",
        str(dpi),
        "--axes-labelsize",
        f"{axes_labelsize:g}",
        "--tick-labelsize",
        f"{tick_labelsize:g}",
        "--legend-loc",
        legend_loc,
        "--linewidth-scale",
        f"{linewidth_scale:g}",
        "--save-provenance",
        "--output-path",
        str(output_path),
    ]
    for label, path in posterior_args:
        command.extend(["--posterior", f"{label}={path}"])
    for label, path in marker_args:
        command.extend(["--posterior-marker-summary", f"{label}={path}"])
    for parameter in parameters:
        command.extend(["--parameter", parameter])
    return command


def generate_corner_pdfs(
    *,
    standard_results: Path,
    combined_results: Path,
    base_pdf: Path,
    dust_pdf: Path,
    base_width_inch: float,
    dust_width_inch: float,
    dpi: int,
) -> None:
    generate_base = _corner_command(
        output_path=base_pdf,
        posterior_args=[("Standard", standard_results), ("Combined", combined_results)],
        marker_args=[("Standard", standard_results), ("Combined", combined_results)],
        parameters=GALACTICUS_PARAMETERS,
        width_inch=base_width_inch,
        dpi=dpi,
        legend_loc="upper right",
        axes_labelsize=8.5,
        tick_labelsize=6.8,
        linewidth_scale=0.85,
    )
    generate_dust = _corner_command(
        output_path=dust_pdf,
        posterior_args=[("Combined", combined_results)],
        marker_args=[("Combined", combined_results)],
        parameters=DUST_PARAMETERS,
        width_inch=dust_width_inch,
        dpi=dpi,
        legend_loc="none",
        axes_labelsize=7.0,
        tick_labelsize=5.8,
        linewidth_scale=0.75,
    )
    _run(generate_base)
    _run(generate_dust)


def overlay_pdf(*, base_pdf: Path, dust_pdf: Path, output_path: Path, scale: float, x: float, y: float) -> None:
    try:
        from pypdf import PdfReader, PdfWriter, Transformation
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "The 'pypdf' package is required to overlay the dust inset. "
            "Install the paper extras with: python -m pip install -e \".[paper]\""
        ) from exc

    base_reader = PdfReader(str(base_pdf))
    dust_reader = PdfReader(str(dust_pdf))
    writer = PdfWriter()
    transformation = Transformation().scale(scale).translate(tx=x, ty=y)
    for page_index, page in enumerate(base_reader.pages):
        if page_index == 0:
            page.merge_transformed_page(dust_reader.pages[0], transformation)
        writer.add_page(page)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)


def write_provenance(path: Path, args: argparse.Namespace, *, base_pdf: Path, dust_pdf: Path) -> None:
    command = shlex.join([sys.executable, *sys.argv])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# created_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# output_path: {args.output_path}",
        f"# base_pdf: {base_pdf}",
        f"# dust_pdf: {dust_pdf}",
        f"# inset_scale: {args.inset_scale}",
        f"# inset_translate_pts: {args.inset_x}, {args.inset_y}",
        command,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    standard_results = (
        args.standard_results.expanduser().resolve()
        if args.standard_results is not None
        else _default_standard_results(data_root)
    )
    combined_results = (
        args.combined_results.expanduser().resolve()
        if args.combined_results is not None
        else _default_combined_results(data_root)
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / FINAL_JOINT_RUN / "paperFigures"
    )
    base_pdf = (
        args.base_pdf.expanduser().resolve()
        if args.base_pdf is not None
        else output_dir / "final_posterior_corner_galacticus20.pdf"
    )
    dust_pdf = (
        args.dust_pdf.expanduser().resolve()
        if args.dust_pdf is not None
        else output_dir / "final_posterior_corner_dust5.pdf"
    )
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else output_dir / "final_posterior_corner_galacticus20_with_dust5_inset_inwards.pdf"
    )
    args.output_path = output_path

    if not args.skip_corner_generation:
        generate_corner_pdfs(
            standard_results=standard_results,
            combined_results=combined_results,
            base_pdf=base_pdf,
            dust_pdf=dust_pdf,
            base_width_inch=args.base_width_inch,
            dust_width_inch=args.dust_width_inch,
            dpi=args.dpi,
        )

    if not base_pdf.exists() or not dust_pdf.exists():
        missing = [str(path) for path in (base_pdf, dust_pdf) if not path.exists()]
        raise FileNotFoundError("Missing intermediate PDF(s): " + ", ".join(missing))

    overlay_pdf(
        base_pdf=base_pdf,
        dust_pdf=dust_pdf,
        output_path=output_path,
        scale=args.inset_scale,
        x=args.inset_x,
        y=args.inset_y,
    )
    write_provenance(Path(f"{output_path}.command.sh"), args, base_pdf=base_pdf, dust_pdf=dust_pdf)
    if not args.keep_intermediate:
        for path in (base_pdf, dust_pdf):
            path.unlink()
    print(output_path)


if __name__ == "__main__":
    main()
