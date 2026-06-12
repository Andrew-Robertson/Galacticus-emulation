from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Galacticus at an MCMC MAP point and optionally post-process "
            "emission-line luminosity functions in the same output directory."
        )
    )
    parser.add_argument("--mcmc-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-changes-filename", default="maximum_a_posteriori_model_changes.xml")
    parser.add_argument("--galacticus-executable", required=True)
    parser.add_argument(
        "--parameter-file",
        action="append",
        nargs="+",
        required=True,
        help="Galacticus parameter file(s). May be repeated or followed by several files.",
    )
    parser.add_argument("--processed-parameters", default="params.xml")
    parser.add_argument("--galacticus-output", default="romanEPS_massFunction.hdf5")
    parser.add_argument("--python-command", default=sys.executable)
    parser.add_argument("--postprocess-emission-line-dust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluation-id", default="bestFitModel")
    parser.add_argument("--evaluation-index", type=int, default=0)
    parser.add_argument("--fixed-dust-case-label", default="map_dust")
    parser.add_argument("--scatter-mode", default="expected")
    parser.add_argument("--sidecar-output-dir", default="emission_line_dust")
    parser.add_argument("--plot-mass-function-results", action="store_true")
    parser.add_argument(
        "--plot-mass-function-script",
        default="$GALACTICUS_DUST_ROOT/dust_model/scripts/plotMassFunctionResults.py",
    )
    return parser.parse_args()


def _expand_path(text: str) -> str:
    return os.path.expanduser(os.path.expandvars(text))


def _copy_model_changes(source: Path, destination_dir: Path, filename: str) -> Path:
    destination = destination_dir / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _run(command: list[str], *, cwd: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    mcmc_dir = args.mcmc_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_changes = mcmc_dir / args.model_changes_filename
    if not source_changes.exists():
        raise FileNotFoundError(source_changes)
    changes_path = _copy_model_changes(source_changes, output_dir, args.model_changes_filename)

    galacticus_command = [
        _expand_path(args.galacticus_executable),
        *[_expand_path(value) for values in args.parameter_file for value in values],
        str(changes_path.name),
        "--output-processed-parameters",
        args.processed_parameters,
    ]
    _run(galacticus_command, cwd=output_dir)

    if args.postprocess_emission_line_dust:
        _run(
            [
                args.python_command,
                str(REPO_ROOT / "scripts/process_emission_line_dust_evaluation.py"),
                args.galacticus_output,
                "--evaluation-id",
                args.evaluation_id,
                "--evaluation-index",
                str(args.evaluation_index),
                "--map-changes-xml",
                str(changes_path.name),
                "--fixed-dust-case-label",
                args.fixed_dust_case_label,
                "--scatter-mode",
                args.scatter_mode,
                "--output-dir",
                args.sidecar_output_dir,
            ],
            cwd=output_dir,
        )

    if args.plot_mass_function_results:
        _run([args.python_command, _expand_path(args.plot_mass_function_script)], cwd=output_dir)

    print(output_dir)


if __name__ == "__main__":
    main()
