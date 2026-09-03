#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path("runs/campaigns/sobol_1024_20p_simpleSizes_moreHalos_reduced")
FINAL_PIPELINE = CAMPAIGN / "automatedPipeline_transformedParams_finalPaper_definitive"
FINAL_MCMC_ROOT = FINAL_PIPELINE / "MCMCs_production_from_exploratory_MAP"
FINAL_STANDARD_ROOT = FINAL_MCMC_ROOT / "standard_observables"
FINAL_JOINT_RUN = (
    FINAL_MCMC_ROOT
    / "standard_observables_plus_emission_line_lfs"
    / "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"
)

EXCLUDE_NAMES = {".DS_Store", "__pycache__"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class Entry:
    source: Path
    dest: Path | None = None
    required: bool = True


@dataclass(frozen=True)
class Package:
    name: str
    title: str
    summary: str
    entries: tuple[Entry, ...]


@dataclass(frozen=True)
class ManifestRow:
    package: str
    source: Path
    dest: Path
    size: int
    modified_utc: str
    sha256: str | None


STANDARD_RUNS = {
    "smf_z0_z3": "mcmc_smf_z0_z3",
    "sfr_function_robotham2011": "mcmc_sfr_function_robotham2011",
    "size_mass_vdw2014_sf_q": "mcmc_size_mass_vdw2014_sf_q",
    "smf_sfrf_sizes": "mcmc_smf_sfrf_sizes",
    "all_standard_observables": "mcmc_all_standard_observables",
    "bh_velocity_dispersion": "mcmc_bh_velocity_dispersion",
    "mzr_blanc2019": "mcmc_mzr_blanc2019",
}


def _entry(path: Path | str, dest: Path | str | None = None, *, required: bool = True) -> Entry:
    return Entry(Path(path), None if dest is None else Path(dest), required)


def _standard_mcmc_entries() -> list[Entry]:
    entries: list[Entry] = []
    for run_dir_name, prefix in STANDARD_RUNS.items():
        run_dir = FINAL_STANDARD_ROOT / run_dir_name
        for filename in (
            f"{prefix}_mcmc_results.hdf5",
            f"{prefix}_mcmc_inputs.joblib",
            f"{prefix}_run_summary.json",
            f"{prefix}_best_fit_observables.csv",
            f"{prefix}_best_fit_smf_sfrf_sizes_observable_predictions.csv",
            "maximum_a_posteriori_model_changes.xml",
        ):
            entries.append(_entry(run_dir / filename, required=False))
    return entries


def build_packages() -> dict[str, Package]:
    sobral_log_error_bundle = (
        CAMPAIGN
        / "scratch/sobral_log_errors_local/emulators/emission_line_lfs/pca_99"
        / "halpha_sobral_1dustdraw_pca99_sobralLogErr.joblib"
    )
    sobral_log_error_bundle_dest = (
        FINAL_PIPELINE
        / "emulators/emission_line_lfs/pca_99/halpha_sobral_1dustdraw_pca99_sobralLogErr.joblib"
    )
    sobral_log_error_meta = sobral_log_error_bundle.with_suffix(".meta.json")
    sobral_log_error_meta_dest = sobral_log_error_bundle_dest.with_suffix(".meta.json")

    joint_prefix = "mcmc_all_standard_observables_plus_halpha_sobral_1dustdraw_pca99_sobralLogErr"
    joint_entries = [
        _entry(FINAL_JOINT_RUN / f"{joint_prefix}_mcmc_results.hdf5"),
        _entry(FINAL_JOINT_RUN / f"{joint_prefix}_mcmc_inputs.joblib"),
        _entry(FINAL_JOINT_RUN / f"{joint_prefix}_run_summary.json"),
        _entry(FINAL_JOINT_RUN / f"{joint_prefix}_best_fit_standard_observables.csv"),
        _entry(FINAL_JOINT_RUN / f"{joint_prefix}_best_fit_sidecar_lfs.csv"),
        _entry(FINAL_JOINT_RUN / "maximum_a_posteriori_model_changes.xml"),
        _entry(FINAL_JOINT_RUN / "paperFigures"),
        _entry(
            "paper/figure_data/final_calibration/galacticus_run_standard_observable_actuals.csv",
            FINAL_JOINT_RUN / "bestFitModel_GalacticusRun/standard_observable_actuals.csv",
        ),
        _entry(FINAL_JOINT_RUN / "bestFitModel_GalacticusRun/emission_line_dust"),
        _entry(FINAL_JOINT_RUN / "bestFitModel_GalacticusRun/maximum_a_posteriori_model_changes.xml"),
        _entry(FINAL_JOINT_RUN / "bestFitModels_UNIT10_array"),
    ]

    packages = [
        Package(
            name="galacticemu_sobol1024_reduced_campaign_v1",
            title="GalacticEmu Sobol-1024 Reduced Training Campaign",
            summary=(
                "Reduced Galacticus outputs and design files for the 1024-point, 20-parameter "
                "training campaign used to build the paper emulators."
            ),
            entries=(
                _entry(CAMPAIGN / "campaign_design.json"),
                _entry(CAMPAIGN / "samples.csv"),
                _entry(CAMPAIGN / "commands.txt"),
                _entry(CAMPAIGN / "commands.sh"),
                _entry(CAMPAIGN / "COMMAND_LOG.md"),
                _entry(CAMPAIGN / "submit_slurm_array.sh"),
                _entry(CAMPAIGN / "evaluations"),
            ),
        ),
        Package(
            name="galacticemu_paper_reproduction_products_v1",
            title="GalacticEmu Paper Reproduction Products",
            summary=(
                "MCMC products, emulator bundles, cached cross-validation outputs, and final "
                "direct-run summaries needed by paper/figure_scripts."
            ),
            entries=(
                _entry(CAMPAIGN / "cross_validation/COMMAND_LOG.md", required=False),
                _entry(CAMPAIGN / "cross_validation/pca_threshold_smf_z0_kfold5_64_to_1024"),
                _entry(
                    CAMPAIGN
                    / "automatedPipeline_transformedParams_finalPaper/emulators/standard_observables/pca_99"
                    / "standard_observables_bundle.joblib"
                ),
                _entry(
                    CAMPAIGN
                    / "automatedPipeline_transformedParams_finalPaper/emulators/standard_observables/pca_99"
                    / "standard_observables_bundle.meta.json"
                ),
                _entry(FINAL_PIPELINE / "emulators"),
                _entry(sobral_log_error_bundle, sobral_log_error_bundle_dest),
                _entry(sobral_log_error_meta, sobral_log_error_meta_dest),
                *_standard_mcmc_entries(),
                *joint_entries,
            ),
        ),
        Package(
            name="galacticemu_example_full_unit1_run_v1",
            title="GalacticEmu Example Full UNIT1 Run",
            summary=(
                "One full Galacticus run at the final MAP parameters, including the full galaxy "
                "catalog for constructing additional observables."
            ),
            entries=(_entry(FINAL_JOINT_RUN / "bestFitModel_UNIT1"),),
        ),
    ]
    return {package.name: package for package in packages}


def _is_excluded(path: Path) -> bool:
    return path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES


def _repo_relative(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved


def _iter_entry_files(entry: Entry) -> list[tuple[Path, Path]]:
    source = (REPO_ROOT / entry.source).resolve() if not entry.source.is_absolute() else entry.source.resolve()
    if not source.exists():
        if entry.required:
            raise FileNotFoundError(source)
        return []

    base_dest = entry.dest if entry.dest is not None else _repo_relative(source)
    if source.is_file():
        return [(source, base_dest)]

    if not source.is_dir():
        raise TypeError(f"Unsupported entry type: {source}")

    rows: list[tuple[Path, Path]] = []
    for root, dirnames, filenames in os.walk(source):
        root_path = Path(root)
        dirnames[:] = sorted(dirname for dirname in dirnames if not _is_excluded(Path(dirname)))
        for filename in sorted(filenames):
            file_path = root_path / filename
            if _is_excluded(file_path):
                continue
            relative = file_path.relative_to(source)
            rows.append((file_path, base_dest / relative))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modified_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def build_manifest(package: Package, *, checksums: bool) -> list[ManifestRow]:
    seen: dict[Path, Path] = {}
    rows: list[ManifestRow] = []
    for entry in package.entries:
        for source, dest in _iter_entry_files(entry):
            if dest in seen and seen[dest] != source:
                raise ValueError(f"Duplicate staged path {dest}: {seen[dest]} and {source}")
            seen[dest] = source
            rows.append(
                ManifestRow(
                    package=package.name,
                    source=_repo_relative(source),
                    dest=dest,
                    size=source.stat().st_size,
                    modified_utc=_modified_utc(source),
                    sha256=_sha256(source) if checksums else None,
                )
            )
    return rows


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    raise AssertionError("unreachable")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    header = "package\trelative_path\tsource_path\tbytes\tmodified_utc\tsha256\n"
    lines = [header]
    for row in rows:
        checksum = "" if row.sha256 is None else row.sha256
        lines.append(
            f"{row.package}\t{row.dest.as_posix()}\t{row.source.as_posix()}\t"
            f"{row.size}\t{row.modified_utc}\t{checksum}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_sha256s(path: Path, rows: list[ManifestRow]) -> None:
    lines = [
        f"{row.sha256}  {row.dest.as_posix()}\n"
        for row in rows
        if row.sha256 is not None
    ]
    path.write_text("".join(lines), encoding="utf-8")


def _package_readme(package: Package, rows: list[ManifestRow]) -> str:
    total_bytes = sum(row.size for row in rows)
    return "\n".join(
        [
            f"# {package.title}",
            "",
            package.summary,
            "",
            "The paths in this package are repository-relative. To reproduce the paper figures,",
            "extract the archive from the root of a GalacticEmu checkout so that the `runs/`",
            "tree lands next to `paper/` and `scripts/`.",
            "",
            f"Files: {len(rows)}",
            f"Uncompressed size: {_format_size(total_bytes)}",
            "",
            "After extraction, see `paper/figure_scripts/rebuild_paper_figures.py` in the",
            "repository for the figure-generation entry point.",
            "",
        ]
    )


def _release_readme(release_name: str, packages: list[Package], manifests: dict[str, list[ManifestRow]]) -> str:
    lines = [
        f"# {release_name}",
        "",
        "Local staging directory for the GalacticEmu paper data release candidate.",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Repository commit: {_git_revision()}",
        "",
        "Suggested use:",
        "",
        "```bash",
        "python -m pip install -e \".[paper]\"",
        "python paper/figure_scripts/rebuild_paper_figures.py --data-root . --output-dir paper/tmp/rebuilt_figures",
        "```",
        "",
        "Packages:",
        "",
    ]
    for package in packages:
        rows = manifests[package.name]
        total_bytes = sum(row.size for row in rows)
        lines.append(f"- `{package.name}`: {len(rows)} files, {_format_size(total_bytes)}")
        lines.append(f"  {package.summary}")
    lines.extend(
        [
            "",
            "The reduced campaign package is the canonical training/evaluation data.",
            "The paper reproduction package supplies the MCMC/emulator/direct-run products used by",
            "`paper/figure_scripts`. The example full UNIT1 package is intentionally separate",
            "because it contains a full galaxy catalog useful for deriving new observables.",
            "",
        ]
    )
    return "\n".join(lines)


def _stage_file(source: Path, dest: Path, *, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if mode == "copy":
        shutil.copy2(source, dest)
    elif mode == "symlink":
        dest.symlink_to(source)
    elif mode == "hardlink":
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def stage_package(package_dir: Path, package: Package, rows: list[ManifestRow], *, mode: str) -> None:
    for row in rows:
        source = (REPO_ROOT / row.source).resolve() if not row.source.is_absolute() else row.source
        _stage_file(source, package_dir / row.dest, mode=mode)
    _write_manifest(package_dir / "MANIFEST.tsv", rows)
    _write_sha256s(package_dir / "SHA256SUMS.txt", rows)
    (package_dir / "README.md").write_text(_package_readme(package, rows), encoding="utf-8")


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if info.isdir():
        info.mode = 0o755
    else:
        executable = bool(info.mode & stat.S_IXUSR)
        info.mode = 0o755 if executable else 0o644
    return info


def make_archive(package_dir: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{package_dir.name}.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz", dereference=True) as archive:
        archive.add(package_dir, arcname=package_dir.name, filter=_tar_filter)
    return archive_path


def parse_args() -> argparse.Namespace:
    packages = build_packages()
    parser = argparse.ArgumentParser(
        description="Stage local data packages for the GalacticEmu Zenodo paper-data release."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dist/zenodo_data_release",
        help="Directory under which the release candidate will be staged.",
    )
    parser.add_argument("--release-name", default="galacticemu-paper-data-v1")
    parser.add_argument(
        "--package",
        action="append",
        choices=sorted(packages),
        dest="selected_packages",
        help="Stage only this package. Repeat to select multiple packages.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually create the staging directory. Without this, only print a dry-run summary.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to populate the staging tree when --execute is used.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip SHA256 calculation. Useful for a quick structure-only dry run.",
    )
    parser.add_argument(
        "--make-archives",
        action="store_true",
        help="Create one .tar.gz archive per staged package.",
    )
    parser.add_argument(
        "--archive-output-dir",
        type=Path,
        default=None,
        help="Archive destination. Defaults to OUTPUT_ROOT/RELEASE_NAME/archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing staging directory for this release name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packages_by_name = build_packages()
    selected_names = args.selected_packages or list(packages_by_name)
    selected_packages = [packages_by_name[name] for name in selected_names]

    checksums = bool(args.execute and not args.skip_checksums)
    manifests = {package.name: build_manifest(package, checksums=checksums) for package in selected_packages}

    for package in selected_packages:
        rows = manifests[package.name]
        total_bytes = sum(row.size for row in rows)
        print(f"{package.name}: {len(rows)} files, {_format_size(total_bytes)}")

    release_dir = args.output_root.expanduser().resolve() / args.release_name
    if not args.execute:
        print(f"Dry run only. Add --execute to create {release_dir}")
        return

    if release_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{release_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(release_dir)

    release_dir.mkdir(parents=True)
    for package in selected_packages:
        print(f"staging {package.name}")
        stage_package(release_dir / package.name, package, manifests[package.name], mode=args.link_mode)

    all_rows = [row for package in selected_packages for row in manifests[package.name]]
    _write_manifest(release_dir / "MANIFEST.tsv", all_rows)
    _write_sha256s(release_dir / "SHA256SUMS.txt", all_rows)
    (release_dir / "README.md").write_text(
        _release_readme(args.release_name, selected_packages, manifests),
        encoding="utf-8",
    )

    if args.make_archives:
        archive_dir = (
            args.archive_output_dir.expanduser().resolve()
            if args.archive_output_dir is not None
            else release_dir / "archives"
        )
        for package in selected_packages:
            archive = make_archive(release_dir / package.name, archive_dir)
            print(f"archive {archive}")

    print(release_dir)


if __name__ == "__main__":
    main()
