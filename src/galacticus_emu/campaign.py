from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import replace
from pathlib import Path
import csv
import json

from .config import PathsConfig
from .lhs import ParameterSpec, sample_parameter_space
from .manifest import EvaluationManifest, ParameterPoint, RunGroup
from .runner import build_galacticus_commands


@dataclass(frozen=True)
class CampaignDefinition:
    campaign_name: str
    n_eval: int
    seed: int
    base_parameters: str
    run_definition_changes: list[str]
    run_groups: list[RunGroup] | None
    mode: str
    notes: str


@dataclass(frozen=True)
class SlurmArrayDefinition:
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int | None = None
    time_limit: str | None = None
    partition: str | None = None
    account: str | None = None
    conda_env: str | None = None
    memory: str | None = None
    memory_per_cpu: str | None = None
    qos: str | None = None
    constraint: str | None = None
    module_commands: tuple[str, ...] = ()
    email: str | None = None
    mail_type: str | None = None


def campaign_root(config: PathsConfig, campaign_name: str) -> Path:
    return config.run_root / "campaigns" / campaign_name


def build_evaluation_manifest(
    campaign: CampaignDefinition,
    parameter_specs: list[ParameterSpec],
    sample_row: list[float],
    evaluation_index: int,
) -> EvaluationManifest:
    evaluation_id = f"{campaign.campaign_name}-eval-{evaluation_index:04d}"
    return EvaluationManifest(
        evaluation_id=evaluation_id,
        base_parameters=campaign.base_parameters,
        run_definition_changes=campaign.run_definition_changes,
        parameter_points=[
            ParameterPoint(spec.path, float(value))
            for spec, value in zip(parameter_specs, sample_row, strict=True)
        ],
        run_groups=campaign.run_groups,
        mode=campaign.mode,
        notes=campaign.notes,
    )


def write_campaign(
    config: PathsConfig,
    campaign: CampaignDefinition,
    parameter_specs: list[ParameterSpec],
    slurm_array: SlurmArrayDefinition | None = None,
) -> Path:
    root = campaign_root(config, campaign.campaign_name)
    manifests_dir = root / "manifests"
    evaluations_root = root / "evaluations"
    commands_path = root / "commands.sh"
    commands_txt_path = root / "commands.txt"
    samples_path = root / "samples.csv"
    definition_path = root / "campaign.json"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    evaluations_root.mkdir(parents=True, exist_ok=True)
    campaign_config = replace(config, run_root=evaluations_root)
    samples = sample_parameter_space(
        parameter_specs=parameter_specs,
        n_eval=campaign.n_eval,
        seed=campaign.seed,
    )

    with samples_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation_id", *[spec.short_name for spec in parameter_specs]])
        for index, row in enumerate(samples):
            writer.writerow([f"{campaign.campaign_name}-eval-{index:04d}", *[f"{value:.16g}" for value in row]])

    command_lines: list[str] = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"export OMP_NUM_THREADS={config.threads}",
        "",
    ]
    plain_commands: list[str] = []
    for index, row in enumerate(samples):
        manifest = build_evaluation_manifest(
            campaign=campaign,
            parameter_specs=parameter_specs,
            sample_row=row.tolist(),
            evaluation_index=index,
        )
        manifest_path = manifests_dir / f"{manifest.evaluation_id}.json"
        manifest.write_json(manifest_path)
        commands = build_galacticus_commands(manifest, campaign_config, command_root=root)
        if len(commands) == 1:
            command_text = " ".join(commands[0])
        else:
            eval_dir = evaluations_root / manifest.evaluation_id
            run_script_path = eval_dir / "run_eval.sh"
            run_script_lines = [
                "#!/bin/bash",
                "set -euo pipefail",
                *[" ".join(command) for command in commands],
                "",
            ]
            run_script_path.write_text("\n".join(run_script_lines))
            command_text = f"bash evaluations/{manifest.evaluation_id}/run_eval.sh"
        command_lines.append(command_text)
        plain_commands.append(command_text)

    commands_path.write_text("\n".join(command_lines) + "\n")
    commands_txt_path.write_text("\n".join(plain_commands) + "\n")

    definition_path.write_text(
        json.dumps(
            {
                "campaign": asdict(campaign),
                "slurm_array": asdict(slurm_array) if slurm_array is not None else None,
                "parameter_specs": [
                    {
                        "path": spec.path,
                        "short_name": spec.short_name,
                        "prior": asdict(spec.prior),
                        "prior_type": type(spec.prior).__name__,
                    }
                    for spec in parameter_specs
                ],
            },
            indent=2,
        )
        + "\n"
    )
    if slurm_array is not None:
        from .slurm import write_slurm_array_script

        write_slurm_array_script(
            campaign_root=root,
            campaign=campaign,
            config=config,
            slurm_array=slurm_array,
        )
    return root
