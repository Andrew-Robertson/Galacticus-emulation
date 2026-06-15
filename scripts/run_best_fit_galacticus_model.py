from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET


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
    parser.add_argument("--galacticus-executable", default=None)
    parser.add_argument(
        "--parameter-file",
        action="append",
        nargs="+",
        help="Galacticus parameter file(s). May be repeated or followed by several files.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=None,
        help=(
            "Campaign root from which to derive the Galacticus command. When supplied, "
            "the first Galacticus command in a template evaluation run_eval.sh is used, "
            "with model_changes.xml/output.xml/params.xml replaced for this MAP run."
        ),
    )
    parser.add_argument(
        "--template-evaluation-dir",
        type=Path,
        default=None,
        help="Template evaluation directory containing run_eval.sh. Overrides --template-evaluation-index.",
    )
    parser.add_argument("--template-evaluation-index", type=int, default=0)
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


def _write_output_change(path: Path, output_filename: str) -> Path:
    root = ET.Element("changes")
    ET.SubElement(root, "change", type="update", path="outputFileName", value=output_filename)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return path


def _template_evaluation_dir(campaign_root: Path, evaluation_index: int) -> Path:
    suffix = f"eval-{evaluation_index:04d}"
    matches = sorted((campaign_root / "evaluations").glob(f"*-{suffix}"))
    if matches:
        return matches[0]
    all_evaluations = sorted((campaign_root / "evaluations").glob("*-eval-*"))
    if not all_evaluations:
        raise FileNotFoundError(f"No evaluation directories found in {campaign_root / 'evaluations'}")
    return all_evaluations[0]


def _read_template_galacticus_command(run_eval_path: Path) -> list[str]:
    for line in run_eval_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "Galacticus" not in stripped:
            continue
        parts = shlex.split(stripped)
        if parts and "Galacticus" in parts[0]:
            return parts
    raise ValueError(f"Could not find a Galacticus command in {run_eval_path}")


def _command_from_template(
    *,
    campaign_root: Path,
    template_evaluation_dir: Path | None,
    template_evaluation_index: int,
    galacticus_executable: str | None,
    changes_filename: str,
    output_change_filename: str,
    processed_parameters: str,
) -> list[str]:
    evaluation_dir = (
        template_evaluation_dir.expanduser().resolve()
        if template_evaluation_dir is not None
        else _template_evaluation_dir(campaign_root, template_evaluation_index).resolve()
    )
    command = _read_template_galacticus_command(evaluation_dir / "run_eval.sh")
    if galacticus_executable:
        command[0] = galacticus_executable
    replaced_model_changes = False
    replaced_output_changes = False
    index = 0
    while index < len(command):
        token = command[index]
        if token.endswith("/model_changes.xml") or token == "model_changes.xml":
            command[index] = changes_filename
            replaced_model_changes = True
        elif token.endswith("/output.xml") or token == "output.xml":
            command[index] = output_change_filename
            replaced_output_changes = True
        elif token == "--output-processed-parameters":
            if index + 1 >= len(command):
                raise ValueError("Template command has --output-processed-parameters with no following value")
            command[index + 1] = processed_parameters
            index += 1
        index += 1
    if not replaced_model_changes:
        raise ValueError(f"Template command from {evaluation_dir / 'run_eval.sh'} had no model_changes.xml argument")
    if not replaced_output_changes:
        command.insert(-2 if "--output-processed-parameters" in command else len(command), output_change_filename)
    return [_expand_path(token) for token in command]


def _command_from_parameter_files(args: argparse.Namespace, changes_filename: str) -> list[str]:
    if not args.galacticus_executable:
        raise ValueError("--galacticus-executable is required when --campaign-root is not supplied")
    if not args.parameter_file:
        raise ValueError("--parameter-file is required when --campaign-root is not supplied")
    return [
        _expand_path(args.galacticus_executable),
        *[_expand_path(value) for values in args.parameter_file for value in values],
        changes_filename,
        "--output-processed-parameters",
        args.processed_parameters,
    ]


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
    output_change_path = _write_output_change(output_dir / "output.xml", args.galacticus_output)

    if args.campaign_root is not None:
        galacticus_command = _command_from_template(
            campaign_root=args.campaign_root.expanduser().resolve(),
            template_evaluation_dir=args.template_evaluation_dir,
            template_evaluation_index=args.template_evaluation_index,
            galacticus_executable=args.galacticus_executable,
            changes_filename=changes_path.name,
            output_change_filename=output_change_path.name,
            processed_parameters=args.processed_parameters,
        )
    else:
        galacticus_command = _command_from_parameter_files(args, changes_path.name)
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
