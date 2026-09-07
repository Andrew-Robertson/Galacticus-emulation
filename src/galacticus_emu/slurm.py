from __future__ import annotations

from pathlib import Path

from .campaign import CampaignDefinition, SlurmArrayDefinition
from .config import PathsConfig


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

    cpus_per_task = slurm_array.cpus_per_task or config.threads
    time_limit = slurm_array.time_limit or "12:00:00"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={campaign.campaign_name}",
        f"#SBATCH --array=0-{campaign.n_eval - 1}",
        f"#SBATCH --nodes={slurm_array.nodes}",
        f"#SBATCH --ntasks={slurm_array.ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output=logs/{campaign.campaign_name}-job-%A-eval-%a.out",
        f"#SBATCH --error=logs/{campaign.campaign_name}-job-%A-eval-%a.err",
    ]
    if slurm_array.memory:
        lines.append(f"#SBATCH --mem={slurm_array.memory}")
    elif slurm_array.memory_per_cpu:
        lines.append(f"#SBATCH --mem-per-cpu={slurm_array.memory_per_cpu}")
    if slurm_array.partition:
        lines.append(f"#SBATCH --partition={slurm_array.partition}")
    if slurm_array.account:
        lines.append(f"#SBATCH --account={slurm_array.account}")
    if slurm_array.qos:
        lines.append(f"#SBATCH --qos={slurm_array.qos}")
    if slurm_array.constraint:
        lines.append(f"#SBATCH --constraint={slurm_array.constraint}")
    if slurm_array.email:
        lines.append(f"#SBATCH --mail-user={slurm_array.email}")
    if slurm_array.mail_type:
        lines.append(f"#SBATCH --mail-type={slurm_array.mail_type}")

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
    for module_cmd in slurm_array.module_commands:
        lines.append(module_cmd)
    if slurm_array.conda_env:
        lines.append(f"conda activate {slurm_array.conda_env}")
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
