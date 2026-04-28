from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a post-processing-only SLURM rescue array for a completed campaign "
            "whose Galacticus runs finished but whose Halpha dust processing needs rerunning."
        )
    )
    parser.add_argument("campaign_root", type=Path, help="Path to the campaign directory on the HPC.")
    parser.add_argument("--hdf5-filename", default="galacticus.hdf5")
    parser.add_argument("--input-json-name", default="galacticus_input_values.json")
    parser.add_argument("--output-dir-name", default="halpha_dust")
    parser.add_argument("--commands-file", default="commands_postprocess.txt")
    parser.add_argument("--slurm-script-name", default="submit_slurm_array_postprocess.sh")
    parser.add_argument("--manifest-name", default="postprocess_rescue_manifest.json")
    parser.add_argument("--n-dust-draws", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--scatter-mode", choices=["draw", "expected"], default="expected")
    parser.add_argument("--z-pivot", type=float, default=1.0)
    parser.add_argument("--partition", default="expansion")
    parser.add_argument("--qos", default="normal")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem-per-cpu", default="4G")
    parser.add_argument("--time-limit", default="04:00:00")
    parser.add_argument("--conda-env", default="galacticus-workspace")
    parser.add_argument(
        "--conda-profile",
        default="/resnick/groups/carnegie_poc/arobert2/miniconda3/etc/profile.d/conda.sh",
    )
    parser.add_argument(
        "--only-missing-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only include evaluations whose post-processing output directory is missing or incomplete.",
    )
    return parser.parse_args()


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _is_postprocess_complete(output_dir: Path) -> bool:
    required = [
        output_dir / "dust_draws.csv",
        output_dir / "halpha_dust_lf_long.csv",
        output_dir / "halpha_dust_lf_emulator_table.csv",
        output_dir / "metadata.json",
    ]
    return all(path.exists() for path in required)


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    evaluations_root = campaign_root / "evaluations"
    if not evaluations_root.exists():
        raise FileNotFoundError(evaluations_root)

    commands: list[str] = []
    manifest_rows: list[dict[str, str | int]] = []

    evaluation_dirs = sorted(path for path in evaluations_root.iterdir() if path.is_dir())
    for evaluation_index, evaluation_dir in enumerate(evaluation_dirs):
        evaluation_id = evaluation_dir.name
        hdf5_path = evaluation_dir / args.hdf5_filename
        input_json_path = evaluation_dir / args.input_json_name
        output_dir = evaluation_dir / args.output_dir_name

        if not hdf5_path.exists():
            continue
        if not input_json_path.exists():
            continue
        if args.only_missing_output and _is_postprocess_complete(output_dir):
            continue

        command = " ".join(
            [
                'python "$GALACTICUS_EMU_ROOT/scripts/process_halpha_dust_evaluation.py"',
                str(hdf5_path.relative_to(campaign_root)),
                "--evaluation-id",
                evaluation_id,
                "--output-dir",
                str(output_dir.relative_to(campaign_root)),
                "--n-dust-draws",
                str(args.n_dust_draws),
                "--base-seed",
                str(args.base_seed),
                "--evaluation-index",
                str(evaluation_index),
                "--scatter-mode",
                args.scatter_mode,
                "--z-pivot",
                str(args.z_pivot),
                "--input-json",
                str(input_json_path.relative_to(campaign_root)),
            ]
        )
        commands.append(command)
        manifest_rows.append(
            {
                "evaluation_id": evaluation_id,
                "evaluation_index": evaluation_index,
                "hdf5_path": str(hdf5_path.relative_to(campaign_root)),
                "input_json_path": str(input_json_path.relative_to(campaign_root)),
                "output_dir": str(output_dir.relative_to(campaign_root)),
            }
        )

    if not commands:
        raise RuntimeError("No evaluations require rescue post-processing.")

    commands_path = campaign_root / args.commands_file
    commands_path.write_text("\n".join(commands) + "\n")

    manifest_path = campaign_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest_rows, indent=2) + "\n")

    slurm_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={campaign_root.name}-post",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --qos={args.qos}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem-per-cpu={args.mem_per_cpu}",
        f"#SBATCH --time={args.time_limit}",
        f"#SBATCH --array=0-{len(commands)-1}",
        "#SBATCH --output=logs/slurm-post-%A_%a.out",
        "#SBATCH --error=logs/slurm-post-%A_%a.err",
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
        f"source {args.conda_profile}",
        f"conda activate {args.conda_env}",
        "",
        'if [ -z "${GALACTICUS_EMU_ROOT:-}" ]; then',
        '  echo "GALACTICUS_EMU_ROOT is not set." >&2',
        "  exit 1",
        "fi",
        'if [ -z "${GALACTICUS_SED_CALC_PATH:-}" ]; then',
        '  echo "GALACTICUS_SED_CALC_PATH is not set." >&2',
        "  exit 1",
        "fi",
        "",
        'COMMAND=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ' + args.commands_file + ")",
        'echo "Running task ${SLURM_ARRAY_TASK_ID}: ${COMMAND}"',
        'echo "Python: $(which python)"',
        'python -c "import sys; print(sys.executable)"',
        'python -c "import h5py; print(h5py.__version__)"',
        'eval "${COMMAND}"',
        "",
    ]
    slurm_path = campaign_root / args.slurm_script_name
    slurm_path.write_text("\n".join(slurm_lines))
    _make_executable(slurm_path)

    print(commands_path)
    print(slurm_path)
    print(manifest_path)
    print(f"{len(commands)} rescue post-processing tasks")


if __name__ == "__main__":
    main()
