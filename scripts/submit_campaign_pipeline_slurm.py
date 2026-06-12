from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_campaign_pipeline import (  # type: ignore
    _default_profiles,
    _load_config,
    _parse_overrides,
    _profile_variables,
    _resolve_variables,
    _stage_by_name,
)


SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job\s+(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit config-driven campaign pipeline stages to Slurm. "
            "Each selected stage becomes one Slurm job, with dependencies read "
            "from stage['depends_on']."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workflow", default=None, help="Workflow name from the config['workflows'] mapping.")
    parser.add_argument("--stage", action="append", default=[], help="Stage to submit. Repeat for several.")
    parser.add_argument(
        "--include-dependencies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also submit recursive stage dependencies.",
    )
    parser.add_argument("--profile", action="append", default=[], help="Apply a named config profile.")
    parser.add_argument(
        "--no-default-profiles",
        action="store_true",
        help="Ignore any top-level default_profiles listed in the config.",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a scalar variable.")
    parser.add_argument("--python-command", default="python", help="Python executable/command to use inside batch jobs.")
    parser.add_argument(
        "--setup-command",
        action="append",
        default=[],
        help="Shell setup command to place before the pipeline command, e.g. conda activation. Repeat as needed.",
    )
    parser.add_argument("--partition", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--nodes", default="1")
    parser.add_argument("--ntasks", default="1")
    parser.add_argument("--cpus-per-task", default="1")
    parser.add_argument("--mem-per-cpu", default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--job-prefix", default="gp-pipeline")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory for Slurm logs. Defaults to PIPELINE_OUTPUT_ROOT/slurm_logs.",
    )
    parser.add_argument(
        "--script-dir",
        type=Path,
        default=None,
        help="Directory for generated batch scripts. Defaults to PIPELINE_OUTPUT_ROOT/slurm_scripts.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="JSON submission manifest. Defaults to SCRIPT_DIR/slurm_submission_manifest.json.",
    )
    parser.add_argument("--submit", action="store_true", help="Call sbatch. Default is to write scripts and print commands.")
    return parser.parse_args()


def _workflow_stage_names(config: Mapping[str, Any], workflow_name: str) -> list[str]:
    workflows = config.get("workflows", {}) or {}
    if not isinstance(workflows, Mapping):
        raise ValueError("config['workflows'] must be a mapping when present")
    if workflow_name not in workflows:
        raise ValueError(f"Unknown workflow {workflow_name!r}")
    workflow = workflows[workflow_name]
    if isinstance(workflow, list):
        stages = workflow
    elif isinstance(workflow, Mapping):
        stages = workflow.get("stages", [])
    else:
        raise ValueError(f"Workflow {workflow_name!r} must be a list or mapping")
    if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
        raise ValueError(f"Workflow {workflow_name!r} stages must be a list of strings")
    return list(stages)


def _requested_stage_names(config: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    names = []
    if args.workflow is not None:
        names.extend(_workflow_stage_names(config, args.workflow))
    names.extend(args.stage)
    if not names:
        raise ValueError("Pass --workflow NAME or at least one --stage NAME")
    return list(dict.fromkeys(names))


def _stage_dependencies(stage: Mapping[str, Any]) -> list[str]:
    depends_on = stage.get("depends_on", [])
    if depends_on is None:
        return []
    if isinstance(depends_on, str):
        return [depends_on]
    if not isinstance(depends_on, list) or not all(isinstance(value, str) for value in depends_on):
        raise ValueError(f"Stage {stage.get('name')} depends_on must be a string or list of strings")
    return list(depends_on)


def _dependency_closure(stage_names: Sequence[str], by_name: Mapping[str, Mapping[str, Any]]) -> list[str]:
    selected: set[str] = set()
    ordered: list[str] = []

    def visit(name: str, stack: tuple[str, ...] = ()) -> None:
        if name in selected:
            return
        if name in stack:
            raise ValueError("Stage dependency cycle: " + " -> ".join((*stack, name)))
        if name not in by_name:
            raise ValueError(f"Unknown stage {name!r}")
        stage = by_name[name]
        for dependency_name in _stage_dependencies(stage):
            visit(dependency_name, (*stack, name))
        selected.add(name)
        ordered.append(name)

    for stage_name in stage_names:
        visit(stage_name)
    return ordered


def _ordered_without_dependencies(stage_names: Sequence[str], by_name: Mapping[str, Mapping[str, Any]]) -> list[str]:
    missing = [name for name in stage_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown stage(s): {', '.join(missing)}")
    return list(dict.fromkeys(stage_names))


def _resolve_output_path(path: Path | None, variables: Mapping[str, Any], fallback_suffix: str) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    root = Path(str(variables.get("PIPELINE_OUTPUT_ROOT", REPO_ROOT / "pipeline"))).expanduser()
    return (root / fallback_suffix).resolve()


def _slurm_value(stage: Mapping[str, Any], key: str, default: str | None) -> str | None:
    slurm = stage.get("slurm", {}) or {}
    if not isinstance(slurm, Mapping):
        raise ValueError(f"Stage {stage.get('name')} slurm metadata must be a mapping")
    value = slurm.get(key.replace("-", "_"), slurm.get(key, default))
    if value is None:
        return None
    return str(value)


def _sbatch_script(
    *,
    stage: Mapping[str, Any],
    stage_name: str,
    args: argparse.Namespace,
    config_path: Path,
    logs_dir: Path,
) -> str:
    safe_name = stage_name.replace("_", "-")
    job_name = f"{args.job_prefix}-{safe_name}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={_slurm_value(stage, 'nodes', args.nodes)}",
        f"#SBATCH --ntasks={_slurm_value(stage, 'ntasks', args.ntasks)}",
        f"#SBATCH --cpus-per-task={_slurm_value(stage, 'cpus_per_task', args.cpus_per_task)}",
        f"#SBATCH --time={_slurm_value(stage, 'time', args.time)}",
        f"#SBATCH --output={logs_dir}/%x-%j.out",
        f"#SBATCH --error={logs_dir}/%x-%j.err",
    ]
    mem_per_cpu = _slurm_value(stage, "mem_per_cpu", args.mem_per_cpu)
    mem = _slurm_value(stage, "mem", args.mem)
    if mem_per_cpu is not None and mem is not None:
        mem = None
    for key, value in (
        ("partition", _slurm_value(stage, "partition", args.partition)),
        ("qos", _slurm_value(stage, "qos", args.qos)),
        ("account", _slurm_value(stage, "account", args.account)),
        ("mem-per-cpu", mem_per_cpu),
        ("mem", mem),
    ):
        if value:
            lines.append(f"#SBATCH --{key}={value}")
    lines.extend(["", "set -euo pipefail", f"cd {shlex.quote(str(REPO_ROOT))}"])
    for setup_command in args.setup_command:
        lines.append(setup_command)
    command = [
        args.python_command,
        "scripts/run_campaign_pipeline.py",
        "--config",
        str(config_path),
        "--stage",
        stage_name,
        "--execute",
    ]
    for profile_name in args.profile:
        command.extend(["--profile", profile_name])
    if args.no_default_profiles:
        command.append("--no-default-profiles")
    for override in args.set:
        command.extend(["--set", override])
    lines.extend(["", shlex.join(command), ""])
    return "\n".join(lines)


def _write_stage_script(
    *,
    stage: Mapping[str, Any],
    stage_name: str,
    args: argparse.Namespace,
    config_path: Path,
    script_dir: Path,
    logs_dir: Path,
) -> Path:
    script_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = script_dir / f"{stage_name}.sbatch"
    path.write_text(
        _sbatch_script(stage=stage, stage_name=stage_name, args=args, config_path=config_path, logs_dir=logs_dir)
    )
    path.chmod(0o755)
    return path


def _submit_stage(script_path: Path, dependency_job_ids: Sequence[str]) -> str:
    command = ["sbatch"]
    if dependency_job_ids:
        command.append("--dependency=afterok:" + ":".join(dependency_job_ids))
    command.append(str(script_path))
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    match = SBATCH_JOB_ID_RE.search(completed.stdout.strip())
    if match is None:
        raise RuntimeError(f"Could not parse sbatch job id from output: {completed.stdout!r}")
    return match.group(1)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    profile_names = [] if args.no_default_profiles else _default_profiles(config)
    profile_names.extend(args.profile)
    variables = _resolve_variables(
        config.get("variables", {}),
        _profile_variables(config, profile_names),
        _parse_overrides(args.set),
    )
    stages = config.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("config['stages'] must be a list")
    by_name = _stage_by_name(stages)
    requested = _requested_stage_names(config, args)
    stage_names = (
        _dependency_closure(requested, by_name)
        if args.include_dependencies
        else _ordered_without_dependencies(requested, by_name)
    )
    logs_dir = _resolve_output_path(args.logs_dir, variables, "slurm_logs")
    script_dir = _resolve_output_path(args.script_dir, variables, "slurm_scripts")
    manifest_path = (
        args.manifest_path.expanduser().resolve()
        if args.manifest_path is not None
        else script_dir / "slurm_submission_manifest.json"
    )

    job_ids: dict[str, str] = {}
    manifest_rows = []
    for stage_name in stage_names:
        stage = by_name[stage_name]
        if bool(stage.get("manual", False)):
            raise ValueError(f"Refusing to submit manual stage {stage_name!r}")
        dependency_names = [name for name in _stage_dependencies(stage) if name in stage_names]
        missing_dependency_jobs = [name for name in dependency_names if name not in job_ids and args.submit]
        if missing_dependency_jobs:
            raise RuntimeError(f"Internal ordering error; dependencies not yet submitted: {missing_dependency_jobs}")
        script_path = _write_stage_script(
            stage=stage,
            stage_name=stage_name,
            args=args,
            config_path=config_path,
            script_dir=script_dir,
            logs_dir=logs_dir,
        )
        dependency_job_ids = [job_ids[name] for name in dependency_names if name in job_ids]
        sbatch_command = ["sbatch"]
        if dependency_job_ids:
            sbatch_command.append("--dependency=afterok:" + ":".join(dependency_job_ids))
        sbatch_command.append(str(script_path))
        job_id = _submit_stage(script_path, dependency_job_ids) if args.submit else None
        if job_id is not None:
            job_ids[stage_name] = job_id
        print(("submitted " if args.submit else "dry-run ") + stage_name)
        if dependency_names:
            print("depends_on=" + ",".join(dependency_names))
        print(shlex.join(sbatch_command))
        if job_id is not None:
            print(f"job_id={job_id}")
        manifest_rows.append(
            {
                "stage": stage_name,
                "depends_on": dependency_names,
                "dependency_job_ids": dependency_job_ids,
                "script_path": str(script_path),
                "sbatch_command": sbatch_command,
                "job_id": job_id,
            }
        )

    manifest = {
        "config": str(config_path),
        "workflow": args.workflow,
        "requested_stages": requested,
        "submitted_stages": stage_names,
        "submit": bool(args.submit),
        "jobs": manifest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
