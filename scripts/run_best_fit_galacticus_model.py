from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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
            "with model_changes.xml/output.xml/params.xml replaced for this MAP run. "
            "Prepared campaigns without run_eval.sh use a local copy of input.xml."
        ),
    )
    parser.add_argument(
        "--template-evaluation-dir",
        type=Path,
        default=None,
        help="Template evaluation directory containing run_eval.sh. Overrides --template-evaluation-index.",
    )
    parser.add_argument("--template-evaluation-index", type=int, default=0)
    parser.add_argument("--changes-file", type=Path, action="append", default=[],
                        help="Additional changes applied after base inputs and before MAP/output changes.")
    parser.add_argument("--dry-run-first", action="store_true")
    parser.add_argument("--reduced-output", default=None)
    parser.add_argument("--root-host-mass-minimum", type=float, default=None,
                        help="Override the root-host minimum in a prepared input (Msun).")
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
    # Galacticus compares these names as strings before renaming. An absolute
    # scratch path and relative final path can alias the same inode, causing its
    # overwrite-rename helper to delete the source. Set both in the LAST overlay.
    final_path = str((path.parent / output_filename).resolve())
    for parameter in ("outputFileName", "outputScratchFileName"):
        ET.SubElement(root, "change", type="replaceOrAppend", path=parameter, value=final_path)
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
    output_dir: Path,
    output_filename: str,
) -> list[str]:
    evaluation_dir = (
        template_evaluation_dir.expanduser().resolve()
        if template_evaluation_dir is not None
        else _template_evaluation_dir(campaign_root, template_evaluation_index).resolve()
    )
    if not (evaluation_dir / "run_eval.sh").exists():
        if not galacticus_executable:
            raise ValueError("Prepared inputs require an explicit Galacticus executable")
        original = evaluation_dir / "input.xml"
        tree = ET.parse(original)
        root = tree.getroot()
        if root.tag != "parameters":
            raise ValueError(f"Expected prepared parameters in {original}")
        # Never retain absolute output/scratch paths into a training evaluation.
        for tag in ("outputFileName", "outputScratchFileName"):
            for node in root.findall(tag):
                node.set("value", str(output_dir / output_filename))
        if root.find("outputFileName") is None:
            ET.SubElement(root, "outputFileName", value=str(output_dir / output_filename))
        local_input = output_dir / "template_input.xml"
        tree.write(local_input, encoding="UTF-8", xml_declaration=True)
        return [_expand_path(galacticus_executable), str(local_input),
                changes_filename, output_change_filename,
                "--output-processed-parameters", processed_parameters]
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


def _extend_root_hosts(path: Path, minimum: float) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    masses = root.find("mergerTreeBuildMasses")
    resolution = root.find("mergerTreeMassResolution/massResolution")
    if masses is None or masses.get("value") != "sampledDistributionUniform" or resolution is None:
        raise ValueError("Root-host override requires sampledDistributionUniform and fixed mass resolution")
    lower = masses.find("massTreeMinimum")
    if lower is None:
        raise ValueError("Missing root-host mass minimum")
    old = float(lower.get("value"))
    if not float(resolution.get("value")) <= minimum < old:
        raise ValueError("New minimum must be at least the resolution and below the original minimum")
    lower.set("value", format(minimum, ".16g"))
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return {"old_minimum": old, "new_minimum": minimum,
            "trees_per_decade": masses.find("treesPerDecade").get("value"),
            "note": "Seed retained; changed sampling does not guarantee matching overlapping trees."}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(campaign_root: Path | None, exe: Path, output_dir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    if campaign_root is not None and (campaign_root / "assembly.json").is_file():
        assembly = json.loads((campaign_root / "assembly.json").read_text())
        environment.update(assembly["configuration"]["environment"])
    environment["GALACTICUS_EXEC_PATH"] = str(exe.parent)
    environment["GALACTICUS_DYNAMIC_DATA_PATH"] = str(output_dir / "cache")
    environment["PYTHONPATH"] = str(exe.parent / "python") + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    args = parse_args()
    mcmc_dir = args.mcmc_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_changes = mcmc_dir / args.model_changes_filename
    if not source_changes.exists():
        raise FileNotFoundError(source_changes)
    if (output_dir / args.galacticus_output).exists():
        raise FileExistsError(output_dir / args.galacticus_output)
    # Exclusive claim also protects against accidentally submitting the stage twice.
    with (output_dir / "started.json").open("x") as stream:
        json.dump({"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
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
            output_dir=output_dir,
            output_filename=args.galacticus_output,
        )
    else:
        galacticus_command = _command_from_parameter_files(args, changes_path.name)
    if args.changes_file:
        insertion = galacticus_command.index(changes_path.name)
        extras = []
        for index, source in enumerate(args.changes_file):
            dest = output_dir / f"additional_changes_{index:02d}.xml"
            shutil.copy2(source.expanduser().resolve(), dest)
            extras.append(str(dest))
        galacticus_command[insertion:insertion] = extras
    host_override = None
    if args.root_host_mass_minimum is not None:
        prepared = output_dir / "template_input.xml"
        if not prepared.is_file():
            raise ValueError("Root-host override currently requires a prepared campaign input.xml")
        host_override = _extend_root_hosts(prepared, args.root_host_mass_minimum)
    exe = Path(galacticus_command[0]).resolve()
    environment = _environment(args.campaign_root, exe, output_dir)
    (output_dir / "cache").mkdir(exist_ok=True)
    inputs = {}
    for token in galacticus_command[1:]:
        path = Path(token)
        if not path.is_absolute():
            path = output_dir / path
        if path.suffix == ".xml" and path.is_file():
            inputs[str(path)] = _sha(path)
    if args.campaign_root is not None:
        evaluation = (args.template_evaluation_dir or
                      _template_evaluation_dir(args.campaign_root, args.template_evaluation_index))
        for name in ("input.xml", "run_eval.sh"):
            path = evaluation / name
            if path.is_file():
                inputs[str(path.resolve())] = _sha(path)
    try:
        source_revision = subprocess.check_output(
            ["git", "-C", str(exe.parent), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        source_status = subprocess.check_output(
            ["git", "-C", str(exe.parent), "status", "--short", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision, source_status = None, None
    provenance = {
        "command": galacticus_command, "cwd": str(output_dir),
        "root_host_override": host_override,
        "executable": str(exe), "executable_sha256": _sha(exe),
        "source_checkout_revision": source_revision, "source_checkout_status": source_status,
        "input_sha256": inputs, "map_source": str(source_changes),
        "map_source_sha256": _sha(source_changes),
        "runner_sha256": _sha(Path(__file__)),
        "environment": {k: v for k, v in environment.items()
                        if k.startswith("GALACTICUS_") or k in
                        ("SHMR_PATH", "PYTHONPATH", "OMP_NUM_THREADS")},
    }
    (output_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.dry_run_first:
        _run(galacticus_command + ["--dry-run"], cwd=output_dir, environment=environment)
    _run(galacticus_command, cwd=output_dir, environment=environment)
    import h5py
    with h5py.File(output_dir / args.galacticus_output, "r") as output:
        if output.attrs.get("statusCompletion", -1) != 0:
            raise RuntimeError("Galacticus output does not report successful completion")

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
            environment=environment,
        )

    if args.plot_mass_function_results:
        _run([args.python_command, _expand_path(args.plot_mass_function_script)], cwd=output_dir, environment=environment)

    if args.reduced_output:
        _run([args.python_command, str(REPO_ROOT / "scripts/extract_campaign_hdf5_groups.py"),
              args.galacticus_output, "--output-filename", args.reduced_output],
             cwd=output_dir, environment=environment)
    (output_dir / "complete.json").write_text(json.dumps({
        "statusCompletion": 0, "dust_processed": args.postprocess_emission_line_dust,
        "processed_parameters_sha256": _sha(output_dir / args.processed_parameters),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    print(output_dir)


if __name__ == "__main__":
    main()
