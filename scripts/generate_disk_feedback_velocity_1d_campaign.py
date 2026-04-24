from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import stat
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu.lhs import TruncatedLogNormalPrior


DISK_VELOCITY_PATH = (
    "nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/"
    "stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic"
)
DISK_VELOCITY_PRIOR = TruncatedLogNormalPrior(lower=25.0, upper=300.0, x0=150.0, sigma=0.5)

RUN_DEFINITION_CHANGES = [
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/romanEPS.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/massFunction.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/handCalibratedChanges.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/odeAcc1e-4.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/seeds/seed1234.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/additionalQuantities.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/universeMachine/Trinity/universeMachine_Trinity_z012_outputs.xml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 1D mass-function emulator-demo campaign around a MAP model, "
            "varying only the disk stellar-feedback velocity characteristic."
        )
    )
    parser.add_argument(
        "--campaign-name",
        default="disk_feedback_velocity_1d_mass_function_32",
        help="Directory name under runs/campaigns.",
    )
    parser.add_argument("--n-eval", type=int, default=32, help="Number of evaluations. Use a power of two for nested Sobol subsets.")
    parser.add_argument(
        "--base-model-changes",
        default=(
            "runs/campaigns/lhs_trinity_moreparams_512/"
            "emulator_mcmc_four_mean_families_no1e11_mbh_plus_mzr_z0_upper_gas/"
            "maximum_a_posteriori_model_changes.xml"
        ),
        help="Base MAP model changes file, relative to the repo root unless absolute.",
    )
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--time-limit", default="40:30:00")
    parser.add_argument("--partition", default="expansion")
    parser.add_argument("--qos", default="normal")
    parser.add_argument("--conda-env", default="galacticus-workspace")
    parser.add_argument(
        "--conda-profile",
        default="/resnick/groups/carnegie_poc/arobert2/miniconda3/etc/profile.d/conda.sh",
        help="Path to conda.sh on the target HPC.",
    )
    parser.add_argument(
        "--include-galactic-structure-solver-fixed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include a campaign-root galacticStructureSolver-Fixed.xml argument in every command. "
            "The script writes a minimal change to set galacticStructureSolver=fixed; disable this "
            "if you prefer to provide a custom file on the HPC."
        ),
    )
    parser.add_argument(
        "--extra-run-definition-change",
        action="append",
        default=[],
        help=(
            "Additional Galacticus parameter/change XML file to append to every run command. "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--enable-halpha-dust-postprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Halpha dust-LF post-processing at the end of each evaluation script.",
    )
    parser.add_argument("--halpha-dust-draws", type=int, default=32, help="Number of dust draws per evaluation.")
    parser.add_argument(
        "--halpha-dust-scatter-mode",
        choices=["draw", "expected"],
        default="expected",
        help="How to treat attenuation scatter in the Halpha dust post-processing.",
    )
    parser.add_argument("--halpha-dust-base-seed", type=int, default=12345)
    parser.add_argument("--halpha-dust-z-pivot", type=float, default=1.0)
    parser.add_argument(
        "--halpha-dust-output-dir-name",
        default="halpha_dust",
        help="Per-evaluation output directory name for Halpha dust post-processing products.",
    )
    return parser.parse_args()


def _uniform_1d_quantiles(n_eval: int) -> list[float]:
    if n_eval < 2:
        raise ValueError("n_eval must be at least 2 for a bounded 1D demo design.")
    return np.linspace(0.0, 1.0, n_eval).tolist()


def _space_filling_subset_indices(n_eval: int, subset_size: int) -> list[int]:
    if subset_size > n_eval:
        raise ValueError("subset_size must be <= n_eval")
    if subset_size == n_eval:
        return list(range(n_eval))
    return sorted({int(round(value)) for value in np.linspace(0, n_eval - 1, subset_size)})


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def _update_change_value(base_changes_path: Path, output_path: Path, parameter_path: str, value: float) -> None:
    tree = ET.parse(base_changes_path)
    root = tree.getroot()
    for change in root.findall("change"):
        if change.get("path") == parameter_path:
            change.set("value", f"{value:.17g}")
            _write_xml(tree, output_path)
            return
    ET.SubElement(root, "change", type="update", path=parameter_path, value=f"{value:.17g}")
    _write_xml(tree, output_path)


def _write_single_change(path: Path, parameter_path: str, value: str) -> None:
    root = ET.Element("changes")
    ET.SubElement(root, "change", type="update", path=parameter_path, value=value)
    _write_xml(ET.ElementTree(root), path)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_slurm_script(
    campaign_root: Path,
    campaign_name: str,
    n_eval: int,
    cpus_per_task: int,
    mem_per_cpu: str,
    time_limit: str,
    partition: str,
    qos: str,
    conda_env: str,
    conda_profile: str,
) -> None:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={campaign_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --qos={qos}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem-per-cpu={mem_per_cpu}",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --array=0-" + str(n_eval - 1),
        "#SBATCH --output=logs/slurm-%A_%a.out",
        "#SBATCH --error=logs/slurm-%A_%a.err",
        "",
        'echo "Job ID: $SLURM_JOB_ID"',
        'echo "Job Name: $SLURM_JOB_NAME"',
        'echo "Node: $SLURM_NODELIST"',
        'echo "Partition: $SLURM_PARTITION"',
        'echo "Number of nodes: $SLURM_NNODES"',
        'echo "Number of tasks: $SLURM_NTASKS"',
        'echo "CPUs per task: $SLURM_CPUS_PER_TASK"',
        'echo "Memory per CPU: $SLURM_MEM_PER_CPU"',
        'echo "Time limit: $SLURM_TIME_LIMIT"',
        "",
        "cd $SLURM_SUBMIT_DIR",
        "",
        "ulimit -c 0",
        "export GFORTRAN_ERROR_DUMPCORE=NO",
        "ulimit -t unlimited",
        f"export OMP_NUM_THREADS={cpus_per_task}",
        "",
        f"source {conda_profile}",
        f"conda activate {conda_env}",
        "",
        'COMMAND=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" commands.txt)',
        'echo "Running task ${SLURM_ARRAY_TASK_ID}: ${COMMAND}"',
        'bash -lc "${COMMAND}"',
        "",
    ]
    path = campaign_root / "submit_slurm_array.sh"
    path.write_text("\n".join(lines))
    _make_executable(path)


def main() -> None:
    args = parse_args()
    base_changes_path = Path(args.base_model_changes)
    if not base_changes_path.is_absolute():
        base_changes_path = REPO_ROOT / base_changes_path
    if not base_changes_path.exists():
        raise FileNotFoundError(base_changes_path)
    run_definition_changes = [*RUN_DEFINITION_CHANGES, *args.extra_run_definition_change]

    campaign_root = REPO_ROOT / "runs" / "campaigns" / args.campaign_name
    evaluations_root = campaign_root / "evaluations"
    logs_root = campaign_root / "logs"
    evaluations_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    quantiles = _uniform_1d_quantiles(args.n_eval)
    velocities = np.clip(
        DISK_VELOCITY_PRIOR.inverse_cdf(np.asarray(quantiles, dtype=float)),
        DISK_VELOCITY_PRIOR.lower,
        DISK_VELOCITY_PRIOR.upper,
    )

    commands: list[str] = []
    with (campaign_root / "samples.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation_id", "prior_quantile", "diskVelocityCharacteristic"])
        for index, (quantile, velocity) in enumerate(zip(quantiles, velocities, strict=True)):
            evaluation_id = f"{args.campaign_name}-eval-{index:04d}"
            evaluation_dir = evaluations_root / evaluation_id
            evaluation_dir.mkdir(parents=True, exist_ok=True)

            model_changes = evaluation_dir / "model_changes.xml"
            output_changes = evaluation_dir / "output.xml"
            params_output = evaluation_dir / "params.xml"
            output_hdf5 = evaluation_dir / "galacticus.hdf5"
            run_script = evaluation_dir / "run_eval.sh"

            _update_change_value(base_changes_path, model_changes, DISK_VELOCITY_PATH, float(velocity))
            _write_single_change(output_changes, "outputFileName", str(output_hdf5.relative_to(campaign_root)))

            command_parts = [
                "$GALACTICUS_EXEC_PATH/Galacticus.exe",
                *run_definition_changes,
                str(model_changes.relative_to(campaign_root)),
                str(output_changes.relative_to(campaign_root)),
            ]
            if args.include_galactic_structure_solver_fixed:
                command_parts.append("galacticStructureSolver-Fixed.xml")
            command_parts.extend(["--output-processed-parameters", str(params_output.relative_to(campaign_root))])
            command = " ".join(command_parts)
            run_lines = [
                "#!/bin/bash",
                "set -euo pipefail",
                'cd "$(dirname "$0")/../.."',
                command,
            ]
            if args.enable_halpha_dust_postprocess:
                galacticus_input_json = evaluation_dir / "galacticus_input_values.json"
                galacticus_input_json.write_text(
                    json.dumps(
                        {
                            "prior_quantile": float(quantile),
                            "diskVelocityCharacteristic": float(velocity),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                dust_command_parts = [
                    "python",
                    "scripts/process_halpha_dust_evaluation.py",
                    str(output_hdf5.relative_to(campaign_root)),
                    "--evaluation-id",
                    evaluation_id,
                    "--output-dir",
                    str((evaluation_dir / args.halpha_dust_output_dir_name).relative_to(campaign_root)),
                    "--n-dust-draws",
                    str(args.halpha_dust_draws),
                    "--base-seed",
                    str(args.halpha_dust_base_seed),
                    "--evaluation-index",
                    str(index),
                    "--scatter-mode",
                    args.halpha_dust_scatter_mode,
                    "--z-pivot",
                    str(args.halpha_dust_z_pivot),
                    "--input-json",
                    str(galacticus_input_json.relative_to(campaign_root)),
                ]
                run_lines.append(" ".join(dust_command_parts))
            run_lines.append("")
            run_script.write_text("\n".join(run_lines))
            _make_executable(run_script)
            commands.append(f"bash {run_script.relative_to(campaign_root)}")
            writer.writerow([evaluation_id, f"{quantile:.17g}", f"{float(velocity):.17g}"])

    if args.include_galactic_structure_solver_fixed:
        _write_single_change(campaign_root / "galacticStructureSolver-Fixed.xml", "galacticStructureSolver", "fixed")

    (campaign_root / "commands.txt").write_text("\n".join(commands) + "\n")
    (campaign_root / "commands.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
    _make_executable(campaign_root / "commands.sh")

    for subset_size in (8, 16, args.n_eval):
        if subset_size <= args.n_eval:
            subset_indices = _space_filling_subset_indices(args.n_eval, subset_size)
            with (campaign_root / f"subset_train_ids_n{subset_size}.txt").open("w") as handle:
                for index in subset_indices:
                    handle.write(f"{args.campaign_name}-eval-{index:04d}\n")

    _write_slurm_script(
        campaign_root=campaign_root,
        campaign_name=args.campaign_name,
        n_eval=args.n_eval,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        time_limit=args.time_limit,
        partition=args.partition,
        qos=args.qos,
        conda_env=args.conda_env,
        conda_profile=args.conda_profile,
    )

    (campaign_root / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_name": args.campaign_name,
                "n_eval": args.n_eval,
                "design": "uniform 1D grid in prior quantile space, including prior endpoints",
                "base_model_changes": str(base_changes_path),
                "varied_parameter": {
                    "path": DISK_VELOCITY_PATH,
                    "short_name": "diskVelocityCharacteristic",
                    "prior": {
                        "type": "TruncatedLogNormalPrior",
                        "lower": DISK_VELOCITY_PRIOR.lower,
                        "upper": DISK_VELOCITY_PRIOR.upper,
                        "x0": DISK_VELOCITY_PRIOR.x0,
                        "sigma": DISK_VELOCITY_PRIOR.sigma,
                    },
                },
                "run_definition_changes": run_definition_changes,
                "halpha_dust_postprocess": (
                    {
                        "enabled": True,
                        "n_dust_draws": args.halpha_dust_draws,
                        "scatter_mode": args.halpha_dust_scatter_mode,
                        "base_seed": args.halpha_dust_base_seed,
                        "z_pivot": args.halpha_dust_z_pivot,
                        "output_dir_name": args.halpha_dust_output_dir_name,
                    }
                    if args.enable_halpha_dust_postprocess
                    else {"enabled": False}
                ),
                "subsets": [size for size in (8, 16, args.n_eval) if size <= args.n_eval],
                "notes": (
                    "Mass-function emulator-demo campaign centered on the gas-phase MZR+Mstar+Mbh MAP model. "
                    "Only disk stellar-feedback velocityCharacteristic is varied. "
                    "Subset files choose approximately space-filling 8- and 16-point subsets."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    (campaign_root / "README.md").write_text(
        f"""# {args.campaign_name}

One-dimensional Galacticus mass-function campaign for GP-emulator visualization.

Only this parameter varies:

`{DISK_VELOCITY_PATH}`

The other parameters are inherited from:

`{base_changes_path}`

The 32 design points use a uniform grid in prior-quantile space, including the prior endpoints. The `subset_train_ids_n8.txt` and `subset_train_ids_n16.txt` files identify approximately space-filling subsets for visualization.

Submit on the HPC with:

```bash
sbatch submit_slurm_array.sh
```

Each task writes into its own directory under `evaluations/`.

Halpha dust-LF post-processing is {("enabled" if args.enable_halpha_dust_postprocess else "disabled")} for this campaign.

If you need your custom `galacticStructureSolver-Fixed.xml`, either regenerate this campaign with `--include-galactic-structure-solver-fixed` or place your preferred file in this campaign root and append it to the commands before `--output-processed-parameters`.
"""
    )

    print(campaign_root)


if __name__ == "__main__":
    main()
