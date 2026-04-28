from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import h5py


DEFAULT_EXCLUDE = ("Outputs",)
DEFAULT_SIDECARS = (
    "campaign.json",
    "samples.csv",
    "commands.txt",
    "commands.sh",
    "submit_slurm_array.sh",
    "commands_postprocess.txt",
    "submit_slurm_array_postprocess.sh",
    "postprocess_rescue_manifest.json",
    "subset_train_ids_n8.txt",
    "subset_train_ids_n16.txt",
    "subset_train_ids_n32.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract lightweight top-level groups from Galacticus HDF5 outputs. "
            "By default this copies every top-level object except /Outputs."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Campaign directory, evaluations directory, or a single Galacticus HDF5 file.",
    )
    parser.add_argument(
        "--input-filename",
        default="galacticus.hdf5",
        help="HDF5 filename to find within evaluation directories when PATH is a directory.",
    )
    parser.add_argument(
        "--output-filename",
        default="galacticus_reduced.hdf5",
        help="Reduced HDF5 filename to write for each input file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for reduced outputs. If omitted, reduced files are "
            "written next to each input HDF5 file."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUDE),
        help="Top-level HDF5 objects to exclude. Default: Outputs.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of top-level HDF5 objects to copy. If supplied, "
            "--exclude is ignored."
        ),
    )
    parser.add_argument(
        "--copy-sidecars",
        action="store_true",
        help=(
            "When using --output-root with a campaign directory, also copy small campaign "
            "metadata files and per-evaluation XML/params files."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reduced HDF5 files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args()


def _top_level_name(name: str) -> str:
    return name.strip("/")


def _copy_root_attributes(source: h5py.File, destination: h5py.File) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _copy_hdf5_groups(
    input_path: Path,
    output_path: Path,
    *,
    include: set[str] | None,
    exclude: set[str],
    overwrite: bool,
    dry_run: bool,
) -> list[str]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite to replace it")

    with h5py.File(input_path, "r") as source:
        if include is None:
            names = [name for name in source.keys() if name not in exclude]
        else:
            missing = sorted(include.difference(source.keys()))
            if missing:
                raise KeyError(f"{input_path} is missing requested top-level object(s): {missing}")
            names = [name for name in source.keys() if name in include]

        if dry_run:
            return names

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if temporary_path.exists():
            temporary_path.unlink()

        try:
            with h5py.File(temporary_path, "w") as destination:
                _copy_root_attributes(source, destination)
                for name in names:
                    source.copy(name, destination, name=name)
            temporary_path.replace(output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    return names


def _is_hdf5_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}


def _campaign_root(path: Path) -> Path:
    if _is_hdf5_file(path):
        return path.parent
    if path.name == "evaluations":
        return path.parent
    return path


def _find_inputs(path: Path, input_filename: str) -> list[Path]:
    if _is_hdf5_file(path):
        return [path]
    search_root = path / "evaluations" if (path / "evaluations").is_dir() else path
    return sorted(search_root.glob(f"*/{input_filename}"))


def _output_path_for(
    input_path: Path,
    *,
    base_path: Path,
    output_root: Path | None,
    output_filename: str,
) -> Path:
    if output_root is None:
        return input_path.with_name(output_filename)
    root = _campaign_root(base_path)
    try:
        relative_parent = input_path.parent.relative_to(root)
    except ValueError:
        relative_parent = Path(input_path.parent.name)
    return output_root / relative_parent / output_filename


def _copy_if_present(source: Path, destination: Path, *, dry_run: bool) -> bool:
    if not source.exists():
        return False
    if dry_run:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _copy_tree_if_present(source: Path, destination: Path, *, dry_run: bool) -> bool:
    if not source.exists() or not source.is_dir():
        return False
    if dry_run:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return True


def _copy_sidecars(campaign_root: Path, output_root: Path, *, dry_run: bool) -> int:
    copied = 0
    for name in DEFAULT_SIDECARS:
        if _copy_if_present(campaign_root / name, output_root / name, dry_run=dry_run):
            copied += 1

    evaluations_root = campaign_root / "evaluations"
    if not evaluations_root.is_dir():
        return copied

    for evaluation_dir in sorted(path for path in evaluations_root.iterdir() if path.is_dir()):
        for sidecar_name in ("model_changes.xml", "output.xml", "params.xml"):
            source = evaluation_dir / sidecar_name
            destination = output_root / "evaluations" / evaluation_dir.name / sidecar_name
            if _copy_if_present(source, destination, dry_run=dry_run):
                copied += 1
        halpha_dust_source = evaluation_dir / "halpha_dust"
        halpha_dust_destination = output_root / "evaluations" / evaluation_dir.name / "halpha_dust"
        if _copy_tree_if_present(halpha_dust_source, halpha_dust_destination, dry_run=dry_run):
            copied += 1
    return copied


def main() -> None:
    args = parse_args()
    base_path = args.path.resolve()
    output_root = args.output_root.resolve() if args.output_root is not None else None
    include = None if args.include is None else {_top_level_name(name) for name in args.include}
    exclude = {_top_level_name(name) for name in args.exclude}

    inputs = _find_inputs(base_path, args.input_filename)
    if not inputs:
        raise FileNotFoundError(f"No '{args.input_filename}' files found under {base_path}")

    print(f"Found {len(inputs)} input HDF5 file(s).")
    for input_path in inputs:
        output_path = _output_path_for(
            input_path,
            base_path=base_path,
            output_root=output_root,
            output_filename=args.output_filename,
        )
        copied_names = _copy_hdf5_groups(
            input_path,
            output_path,
            include=include,
            exclude=exclude,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        action = "Would copy" if args.dry_run else "Copied"
        names = ", ".join(f"/{name}" for name in copied_names) if copied_names else "(nothing)"
        print(f"{action}: {input_path} -> {output_path} [{names}]")

    if args.copy_sidecars:
        if output_root is None:
            raise ValueError("--copy-sidecars requires --output-root")
        copied_count = _copy_sidecars(_campaign_root(base_path), output_root, dry_run=args.dry_run)
        action = "Would copy" if args.dry_run else "Copied"
        print(f"{action} {copied_count} sidecar file(s).")


if __name__ == "__main__":
    main()
