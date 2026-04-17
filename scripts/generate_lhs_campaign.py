from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu import load_paths_config
from galacticus_emu.campaign import CampaignDefinition, SlurmArrayDefinition, write_campaign
from galacticus_emu.specs import default_parameter_specs


CONFIG_PATH = REPO_ROOT / "config" / "paths.toml"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dry-run Latin hypercube Galacticus campaign."
    )
    parser.add_argument("--campaign-name", default="lhs_z2_mcmc_analog")
    parser.add_argument("--n-eval", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-parameters",
        default="$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/romanEPS_local.xml",
    )
    parser.add_argument(
        "--change-file",
        action="append",
        dest="change_files",
        help="Run-definition change file. May be supplied multiple times.",
    )
    parser.add_argument(
        "--mode",
        default="low_cost_likelihood",
        choices=["low_cost_likelihood", "emulator_training"],
    )
    parser.add_argument(
        "--notes",
        default=(
            "Dry-run Latin hypercube campaign over the 7-parameter MCMC analog. "
            "Samples are Latin-hypercube draws in prior quantile space."
        ),
    )
    parser.add_argument("--write-slurm-array", action="store_true")
    parser.add_argument("--platform-config", default=None)
    parser.add_argument("--platform-config-file", default=None)
    parser.add_argument("--slurm-cpus-per-task", type=int, default=None)
    parser.add_argument("--slurm-time-limit", default=None)
    parser.add_argument("--slurm-partition", default=None)
    parser.add_argument("--slurm-account", default=None)
    parser.add_argument("--slurm-conda-env", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paths_config(CONFIG_PATH)
    change_files = args.change_files or [
        "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/handCalibratedChanges.xml",
        "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/universeMachine/z2_Likelihood.xml",
        "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/seeds/seed1234.xml",
    ]

    campaign = CampaignDefinition(
        campaign_name=args.campaign_name,
        n_eval=args.n_eval,
        seed=args.seed,
        base_parameters=args.base_parameters,
        run_definition_changes=change_files,
        run_groups=None,
        mode=args.mode,
        notes=args.notes,
    )
    slurm_array = None
    if args.write_slurm_array:
        slurm_array = SlurmArrayDefinition(
            cpus_per_task=args.slurm_cpus_per_task,
            time_limit=args.slurm_time_limit,
            partition=args.slurm_partition,
            account=args.slurm_account,
            conda_env=args.slurm_conda_env,
            platform_config_name=args.platform_config,
            platform_config_file=args.platform_config_file,
        )
    output_root = write_campaign(
        config=config,
        campaign=campaign,
        parameter_specs=default_parameter_specs(),
        slurm_array=slurm_array,
    )
    print(output_root)


if __name__ == "__main__":
    main()
