from __future__ import annotations

from pathlib import Path

from .config import PathsConfig
from .manifest import EvaluationManifest
from .parameter_changes import write_changes_file as write_parameter_changes_file


def write_changes_file(
    manifest: EvaluationManifest,
    config: PathsConfig,
) -> Path:
    run_dir = config.run_root / manifest.evaluation_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "theta.xml"
    parameter_path_value_pairs = [
        (parameter.name, f"{parameter.value:.16g}")
        for parameter in manifest.parameter_points
    ]
    write_parameter_changes_file(
        parameter_path_value_pairs=parameter_path_value_pairs,
        output_path=output_path,
    )
    return output_path


def write_output_changes_file(
    manifest: EvaluationManifest,
    config: PathsConfig,
    command_root: str | Path | None = None,
    output_stem: str = "galacticus",
) -> Path:
    run_dir = config.run_root / manifest.evaluation_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = run_dir / f"{output_stem}.hdf5"
    output_changes_path = run_dir / f"{output_stem}_output.xml"
    output_file_value = str(output_file_path)
    if command_root is not None:
        output_file_value = str(output_file_path.relative_to(Path(command_root)))
    write_parameter_changes_file(
        parameter_path_value_pairs=[
            ("outputFileName", output_file_value),
        ],
        output_path=output_changes_path,
    )
    return output_changes_path


def build_galacticus_commands(
    manifest: EvaluationManifest,
    config: PathsConfig,
    command_root: str | Path | None = None,
) -> list[list[str]]:
    """Return one or more Galacticus commands for a single manifest evaluation."""
    changes_path = write_changes_file(manifest, config)
    changes_arg = str(changes_path)
    command_root_path = Path(command_root) if command_root is not None else None
    if command_root_path is not None:
        changes_arg = str(changes_path.relative_to(command_root_path))

    run_groups = manifest.run_groups or [
        {
            "name": "galacticus",
            "run_definition_changes": manifest.run_definition_changes,
        }
    ]
    commands: list[list[str]] = []
    for run_group in run_groups:
        group_name = run_group["name"] if isinstance(run_group, dict) else run_group.name
        group_changes = (
            run_group["run_definition_changes"]
            if isinstance(run_group, dict)
            else run_group.run_definition_changes
        )
        output_changes_path = write_output_changes_file(
            manifest,
            config,
            command_root=command_root,
            output_stem=group_name,
        )
        output_changes_arg = str(output_changes_path)
        if command_root_path is not None:
            output_changes_arg = str(output_changes_path.relative_to(command_root_path))
        commands.append([
            config.galacticus_binary_expr,
            manifest.base_parameters or config.base_parameters,
            *group_changes,
            changes_arg,
            output_changes_arg,
        ])
    return commands
