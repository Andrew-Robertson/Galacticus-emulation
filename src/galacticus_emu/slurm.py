from __future__ import annotations

from pathlib import Path
import sys

from .campaign import CampaignDefinition, SlurmArrayDefinition
from .config import PathsConfig


def _load_platform_config(config: PathsConfig, slurm_array: SlurmArrayDefinition):
    dust_repo = str(config.galacticus_dust_modelling)
    if dust_repo not in sys.path:
        sys.path.insert(0, dust_repo)
    from dust_model.platform_config import get_platform_config

    return get_platform_config(
        platform=slurm_array.platform_config_name,
        config_file=slurm_array.platform_config_file,
    )


def write_slurm_array_script(
    campaign_root: str | Path,
    campaign: CampaignDefinition,
    config: PathsConfig,
    slurm_array: SlurmArrayDefinition,
) -> Path:
    campaign_root = Path(campaign_root)
    commands_txt = campaign_root / "commands.txt"
    script_path = campaign_root / "submit_slurm_array.sh"
    logs_dir = campaign_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    platform_config = _load_platform_config(config, slurm_array)
    slurm_settings = platform_config.get_slurm_settings("single_likelihood")
    env_settings = platform_config.get_environment_setup()
    user_settings = platform_config.get_user_settings()

    cpus_per_task = slurm_array.cpus_per_task or slurm_settings.get("cpus_per_task", config.threads)
    time_limit = slurm_array.time_limit or slurm_settings.get("time_limit", "12:00:00")
    partition = slurm_array.partition if slurm_array.partition is not None else slurm_settings.get("partition", "")
    account = slurm_array.account if slurm_array.account is not None else slurm_settings.get("account", "")
    conda_env = slurm_array.conda_env or env_settings.get("conda_env", "")
    memory = slurm_settings.get("memory", "")
    memory_per_cpu = slurm_settings.get("memory_per_cpu", "")
    qos = slurm_settings.get("qos", "")
    constraint = slurm_settings.get("constraint", "")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={campaign.campaign_name}",
        f"#SBATCH --array=0-{campaign.n_eval - 1}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output=logs/{campaign.campaign_name}-job-%A-eval-%a.out",
        f"#SBATCH --error=logs/{campaign.campaign_name}-job-%A-eval-%a.err",
    ]
    if memory:
        lines.append(f"#SBATCH --mem={memory}")
    elif memory_per_cpu:
        lines.append(f"#SBATCH --mem-per-cpu={memory_per_cpu}")
    if partition:
        lines.append(f"#SBATCH -p {partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    if user_settings.get("email"):
        lines.append(f"#SBATCH --mail-user={user_settings['email']}")
    if slurm_settings.get("mail_type"):
        lines.append(f"#SBATCH --mail-type={slurm_settings['mail_type']}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            'if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then',
            '  CAMPAIGN_DIR="${SLURM_SUBMIT_DIR}"',
            "else",
            '  CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            "fi",
            'cd "${CAMPAIGN_DIR}"',
            "ulimit -c 0",
            "export GFORTRAN_ERROR_DUMPCORE=NO",
            "ulimit -t unlimited",
            f"export OMP_NUM_THREADS={cpus_per_task}",
        ]
    )
    for module_cmd in env_settings.get("module_commands", []):
        lines.append(module_cmd)
    if conda_env:
        lines.append(f"conda activate {conda_env}")
    lines.extend(
        [
            "",
            'COMMAND=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${CAMPAIGN_DIR}/commands.txt")',
            'echo "Running task ${SLURM_ARRAY_TASK_ID}: ${COMMAND}"',
            'bash -lc "${COMMAND}"',
            "",
        ]
    )
    script_path.write_text("\n".join(lines))
    return script_path
