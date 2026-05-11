from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from math import log2
from pathlib import Path
import shlex
import shutil
import stat
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu import trinity_parameter_specs
from galacticus_emu.lhs import ParameterSpec


RUN_DEFINITION_CHANGES = [
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/romanEPS.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/massFunction.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/handCalibratedChanges.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/odeAcc1e-4.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/seeds/seed1234.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/additionalQuantities.xml",
    "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/universeMachine/Trinity/universeMachine_Trinity_z012_outputs.xml",
]

DEFAULT_FIXED_PARAMETERS = {
    "spheroidExponent",
    "spheroidSFRExponentVelocity",
    "bondiTemperatureSpheroid",
    "barStabilityThresholdGaseous",
    "barStabilityThresholdStellar",
}

FREE_PARAMETER_ORDER = [
    "diskVelocityCharacteristic",
    "diskExponent",
    "coolingMultiplier",
    "BHefficiencyRadioMode",
    "spheroidSFREfficiency",
    "diskSFRFrequencyNorm",
    "blackHoleSeedMass",
    "spheroidVelocityCharacteristic",
    "henriquesGamma",
    "henriquesDelta1",
    "henriquesDelta2",
    "massRatioMajorMerger",
    "energyOrbital",
    "bondiEnhancementSpheroid",
    "bondiEnhancementHotHalo",
    "coreRadiusOverVirialRadius",
    "barFractionAngularMomentumRetainedSpheroid",
    "BHefficiencyWind",
    "thinDiskMaximum",
    "stellarPopulationMetalYield",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a broad mass-function emission-line Sobol campaign with per-evaluation "
            "Halpha dust post-processing and design-validation plots."
        )
    )
    parser.add_argument(
        "--campaign-name",
        default="sobol_mass_function_emissionlines_dust_19p_512",
        help="Directory name under runs/campaigns.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=512,
        help="Number of slow Galacticus evaluations. Sobol designs work best for powers of two.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sobol scrambling seed.")
    parser.add_argument(
        "--model-changes-template",
        "--base-model-changes",
        dest="model_changes_template",
        default=(
            "runs/campaigns/lhs_trinity_moreparams_512/"
            "emulator_mcmc_four_mean_families_no1e11_mbh_plus_mzr_z0_upper_gas/"
            "maximum_a_posteriori_model_changes.xml"
        ),
        help=(
            "Template model changes XML, relative to the repo root unless absolute. "
            "Sobol-varied paths are overwritten and defaulted template paths are removed."
        ),
    )
    parser.add_argument(
        "--fixed-parameter",
        action="append",
        default=[],
        help=(
            "Additional Galacticus parameter short_name to default by removing it from the template XML. "
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--overwrite-campaign",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete an existing campaign directory before writing new artifacts.",
    )
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem-per-cpu", default="8G")
    parser.add_argument("--time-limit", default="40:30:00")
    parser.add_argument("--partition", default="expansion")
    parser.add_argument("--qos", default="normal")
    parser.add_argument(
        "--galacticus-executable",
        default="$GALACTICUS_EXEC_PATH/Galacticus.exe",
        help="Galacticus executable path used in each generated run script.",
    )
    parser.add_argument("--conda-env", default="galacticus-workspace")
    parser.add_argument(
        "--conda-profile",
        default="/resnick/groups/carnegie_poc/arobert2/miniconda3/etc/profile.d/conda.sh",
        help="Path to conda.sh on the target HPC.",
    )
    parser.add_argument(
        "--extra-run-definition-change",
        action="append",
        default=[],
        help="Additional Galacticus XML change file to append to every command.",
    )
    parser.add_argument(
        "--sfh-parameters-file",
        default="$GALACTICUS_PARAMS_PATH/starFormationHistory/adaptiveSFH-noSave.xml",
        help=(
            "Galacticus XML file that enables SFH evaluation for emission lines. "
            "Defaults to the no-save variant to avoid unnecessarily large outputs."
        ),
    )
    parser.add_argument(
        "--enable-halpha-dust-postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Halpha dust-LF post-processing at the end of each evaluation script.",
    )
    parser.add_argument("--halpha-dust-draws", type=int, default=8)
    parser.add_argument(
        "--halpha-dust-scatter-mode",
        choices=["draw", "expected"],
        default="expected",
    )
    parser.add_argument("--halpha-dust-base-seed", type=int, default=12345)
    parser.add_argument("--halpha-dust-z-pivot", type=float, default=1.0)
    parser.add_argument(
        "--halpha-dust-output-dir-name",
        default="halpha_dust",
    )
    parser.add_argument(
        "--enable-emission-line-dust-postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run common-dust Halpha, [OII], and Hbeta+[OIII] LF post-processing at the end of each evaluation script.",
    )
    parser.add_argument(
        "--emission-line-dust-output-dir-name",
        default="emission_line_dust",
    )
    parser.add_argument(
        "--write-subset-command-scripts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write commands_first_N.{txt,sh} helper files for common training subset sizes.",
    )
    return parser.parse_args()


def _free_parameter_specs(extra_fixed: list[str]) -> list[ParameterSpec]:
    fixed = set(DEFAULT_FIXED_PARAMETERS)
    fixed.update(extra_fixed)
    all_specs = {spec.short_name: spec for spec in trinity_parameter_specs()}
    ordered_names = [name for name in FREE_PARAMETER_ORDER if name not in fixed]
    missing = [name for name in ordered_names if name not in all_specs]
    if missing:
        raise ValueError(f"Unknown Galacticus parameter short_name(s) in FREE_PARAMETER_ORDER: {missing}")
    specs = [all_specs[name] for name in ordered_names]
    expected_free = len(FREE_PARAMETER_ORDER) - len(fixed.intersection(FREE_PARAMETER_ORDER))
    if len(specs) != expected_free:
        raise ValueError(
            f"Expected {expected_free} free parameters after fixing {sorted(fixed)}, found {len(specs)}."
        )
    return specs


def _fixed_parameter_specs(extra_fixed: list[str]) -> list[ParameterSpec]:
    fixed = set(DEFAULT_FIXED_PARAMETERS)
    fixed.update(extra_fixed)
    return [spec for spec in trinity_parameter_specs() if spec.short_name in fixed]


def _sobol_quantiles(n_eval: int, ndim: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=ndim, scramble=True, seed=seed)
    if n_eval > 0 and n_eval & (n_eval - 1) == 0:
        return sampler.random_base2(m=int(round(log2(n_eval))))
    return sampler.random(n=n_eval)


def _sample_parameter_space(parameter_specs: list[ParameterSpec], n_eval: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    quantiles = _sobol_quantiles(n_eval=n_eval, ndim=len(parameter_specs), seed=seed)
    columns = [np.asarray(spec.prior.inverse_cdf(quantiles[:, index]), dtype=float) for index, spec in enumerate(parameter_specs)]
    return quantiles, np.column_stack(columns)


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def _update_change_values(
    base_changes_path: Path,
    output_path: Path,
    parameter_updates: dict[str, float],
    remove_paths: set[str] | None = None,
) -> None:
    tree = ET.parse(base_changes_path)
    root = tree.getroot()
    remove_paths = remove_paths or set()
    for change in list(root.findall("change")):
        path = change.get("path")
        if path in remove_paths:
            root.remove(change)
    seen: set[str] = set()
    for change in root.findall("change"):
        path = change.get("path")
        if path in parameter_updates:
            change.set("value", f"{float(parameter_updates[path]):.17g}")
            seen.add(path)
    for path, value in parameter_updates.items():
        if path not in seen:
            ET.SubElement(root, "change", type="update", path=path, value=f"{float(value):.17g}")
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
        f"#SBATCH --array=0-{n_eval - 1}",
        "#SBATCH --output=logs/slurm-%A_%a.out",
        "#SBATCH --error=logs/slurm-%A_%a.err",
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
        'echo "Python: $(which python)"',
        'python -c "import sys; print(sys.executable)"',
        'eval "${COMMAND}"',
        "",
    ]
    path = campaign_root / "submit_slurm_array.sh"
    path.write_text("\n".join(lines))
    _make_executable(path)


def _write_validation_plots(
    campaign_root: Path,
    quantiles: np.ndarray,
    samples: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> None:
    validation_root = campaign_root / "design_validation"
    validation_root.mkdir(parents=True, exist_ok=True)

    quantile_df = pd.DataFrame(quantiles, columns=[spec.short_name for spec in parameter_specs])
    sample_df = pd.DataFrame(samples, columns=[spec.short_name for spec in parameter_specs])

    quantile_df.to_csv(validation_root / "sobol_quantiles.csv", index=False)

    # Full scatter-matrix in quantile space for a visual corner-like sanity check.
    fig = plt.figure(figsize=(24, 24))
    axes = scatter_matrix(
        quantile_df,
        figsize=(24, 24),
        diagonal="hist",
        marker=".",
        s=6,
        alpha=0.55,
        hist_kwds={"bins": 20},
    )
    for axis_row in axes:
        for axis in axis_row:
            axis.grid(alpha=0.15)
            axis.tick_params(axis="x", labelrotation=0, labelsize=6)
            axis.tick_params(axis="y", labelrotation=0, labelsize=6)
            axis.xaxis.label.set_visible(False)
            axis.yaxis.label.set_visible(False)
    for axis, spec in zip(axes[-1, :], parameter_specs, strict=True):
        axis.set_xlabel(spec.short_name, fontsize=7, rotation=60, ha="right", labelpad=6)
        axis.xaxis.label.set_visible(True)
    for axis, spec in zip(axes[:, 0], parameter_specs, strict=True):
        axis.set_ylabel(spec.short_name, fontsize=7, rotation=0, ha="right", va="center", labelpad=10)
        axis.yaxis.label.set_visible(True)
    fig.subplots_adjust(left=0.18, bottom=0.18, right=0.98, top=0.90, wspace=0.05, hspace=0.05)
    plt.suptitle("Sobol design in prior-quantile space", y=0.92)
    plt.savefig(validation_root / "sobol_quantiles_scatter_matrix.png", dpi=180)
    plt.close(fig)

    # Compact summary diagnostics.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    axes[0].hist(quantiles.ravel(), bins=30, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("prior quantile")
    axes[0].set_ylabel("count")
    axes[0].set_title("All marginal quantiles")
    axes[0].grid(alpha=0.2)

    corr = quantile_df.corr().to_numpy(dtype=float)
    im = axes[1].imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1].set_title("Quantile-space correlation")
    axes[1].set_xticks(range(len(parameter_specs)))
    axes[1].set_xticklabels([spec.short_name for spec in parameter_specs], rotation=90, fontsize=6)
    axes[1].set_yticks(range(len(parameter_specs)))
    axes[1].set_yticklabels([spec.short_name for spec in parameter_specs], fontsize=6)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    distances = qmc.discrepancy(quantiles)
    axes[2].axis("off")
    axes[2].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"campaign = {campaign_root.name}",
                f"n_eval = {len(quantiles)}",
                f"ndim = {quantiles.shape[1]}",
                "Sobol scramble = yes",
                f"centered discrepancy = {distances:.6g}",
                "",
                "Interpretation:",
                "lower discrepancy => more even space-filling.",
            ]
        ),
        va="top",
        family="monospace",
    )
    fig.savefig(validation_root / "sobol_design_summary.png", dpi=180)
    plt.close(fig)


def _write_subset_command_scripts(campaign_root: Path, n_eval: int) -> None:
    subset_sizes = [size for size in [64, 128, 256, 512, 1024] if size <= n_eval]
    for subset_size in subset_sizes:
        subset_commands = campaign_root / f"commands_first_{subset_size}.txt"
        subset_script = campaign_root / f"commands_first_{subset_size}.sh"
        lines = (campaign_root / "commands.txt").read_text().splitlines()[:subset_size]
        subset_commands.write_text("\n".join(lines) + "\n")
        subset_script.write_text("#!/bin/bash\nset -euo pipefail\n" + "\n".join(lines) + "\n")
        _make_executable(subset_script)


def _prior_to_dict(prior: object) -> dict[str, object]:
    return {
        "class": type(prior).__name__,
        "parameters": {field.name: getattr(prior, field.name) for field in fields(prior)},
    }


def _parameter_spec_to_dict(spec: ParameterSpec) -> dict[str, object]:
    return {
        "short_name": spec.short_name,
        "path": spec.path,
        "prior": _prior_to_dict(spec.prior),
    }


def _write_command_log(campaign_root: Path, args: argparse.Namespace) -> None:
    command_line = "python " + shlex.join(sys.argv)
    lines = [
        "# Command Log",
        "",
        f"Campaign: `{args.campaign_name}`",
        "",
        "## Build Command",
        "",
        "```bash",
        f"cd {REPO_ROOT}",
        command_line,
        "```",
        "",
        "## Verification Commands",
        "",
        "```bash",
        f"python -m json.tool runs/campaigns/{args.campaign_name}/campaign_design.json",
        f"wc -l runs/campaigns/{args.campaign_name}/commands.txt",
        f"find runs/campaigns/{args.campaign_name}/evaluations -mindepth 1 -maxdepth 1 -type d | wc -l",
        f"find runs/campaigns/{args.campaign_name} -maxdepth 1 -name 'commands_first_*' -print",
        f"sed -n '1,40p' runs/campaigns/{args.campaign_name}/evaluations/{args.campaign_name}-eval-0000/run_eval.sh",
        "```",
        "",
        "## Metadata Notes",
        "",
        "- `model_changes_template` is the starting XML file for per-evaluation `model_changes.xml` files.",
        "- `free_parameters` snapshots the short name, Galacticus XML path, and prior used to transform Sobol quantiles.",
        "- `defaulted_template_parameters` lists template changes removed so Galacticus defaults are used.",
        "- `run_definition_changes` lists the shared Galacticus parameter files supplied to every evaluation.",
        "",
    ]
    (campaign_root / "COMMAND_LOG.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    template_changes_path = Path(args.model_changes_template)
    if not template_changes_path.is_absolute():
        template_changes_path = REPO_ROOT / template_changes_path
    if not template_changes_path.exists():
        raise FileNotFoundError(template_changes_path)

    parameter_specs = _free_parameter_specs(args.fixed_parameter)
    fixed_parameter_specs = _fixed_parameter_specs(args.fixed_parameter)
    fixed_parameter_paths = {spec.path for spec in fixed_parameter_specs}
    run_definition_changes = [
        *RUN_DEFINITION_CHANGES,
        "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/haloMasses/massTreeMax1e14.xml",
        args.sfh_parameters_file,
        "$GALACTICUS_PARAMS_PATH/fineTuningFiles/emissionLineLuminosityFunctions.xml",
        "$GALACTICUS_PARAMS_PATH/emissionLines/emissionLines.xml",
        *args.extra_run_definition_change,
    ]

    campaign_root = REPO_ROOT / "runs" / "campaigns" / args.campaign_name
    if campaign_root.exists():
        if args.overwrite_campaign:
            shutil.rmtree(campaign_root)
        elif any(campaign_root.iterdir()):
            raise FileExistsError(
                f"{campaign_root} already exists and is not empty; rerun with --overwrite-campaign to replace it."
            )
    evaluations_root = campaign_root / "evaluations"
    logs_root = campaign_root / "logs"
    evaluations_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    quantiles, samples = _sample_parameter_space(parameter_specs, args.n_eval, args.seed)

    with (campaign_root / "samples.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation_id", *[f"{spec.short_name}_quantile" for spec in parameter_specs], *[spec.short_name for spec in parameter_specs]])
        for index, (quantile_row, sample_row) in enumerate(zip(quantiles, samples, strict=True)):
            writer.writerow(
                [
                    f"{args.campaign_name}-eval-{index:04d}",
                    *[f"{value:.16g}" for value in quantile_row],
                    *[f"{value:.16g}" for value in sample_row],
                ]
            )

    commands: list[str] = []
    for index, (quantile_row, sample_row) in enumerate(zip(quantiles, samples, strict=True)):
        evaluation_id = f"{args.campaign_name}-eval-{index:04d}"
        evaluation_dir = evaluations_root / evaluation_id
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        model_changes = evaluation_dir / "model_changes.xml"
        output_changes = evaluation_dir / "output.xml"
        params_output = evaluation_dir / "params.xml"
        output_hdf5 = evaluation_dir / "galacticus.hdf5"
        run_script = evaluation_dir / "run_eval.sh"

        updates = {spec.path: float(value) for spec, value in zip(parameter_specs, sample_row, strict=True)}
        _update_change_values(template_changes_path, model_changes, updates, remove_paths=fixed_parameter_paths)
        _write_single_change(output_changes, "outputFileName", str(output_hdf5.relative_to(campaign_root)))

        command_parts = [
            args.galacticus_executable,
            *run_definition_changes,
            str(model_changes.relative_to(campaign_root)),
            str(output_changes.relative_to(campaign_root)),
            "--output-processed-parameters",
            str(params_output.relative_to(campaign_root)),
        ]
        run_lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")/../.."',
            " ".join(command_parts),
        ]

        galacticus_input_json = evaluation_dir / "galacticus_input_values.json"
        if args.enable_halpha_dust_postprocess or args.enable_emission_line_dust_postprocess:
            galacticus_input_json.write_text(
                json.dumps(
                    {
                        "evaluation_id": evaluation_id,
                        **{f"{spec.short_name}_quantile": float(value) for spec, value in zip(parameter_specs, quantile_row, strict=True)},
                        **{spec.short_name: float(value) for spec, value in zip(parameter_specs, sample_row, strict=True)},
                    },
                    indent=2,
                )
                + "\n"
            )
        if args.enable_halpha_dust_postprocess:
            run_lines.extend(
                [
                    'if [ -z "${GALACTICUS_EMU_ROOT:-}" ]; then',
                    '  echo "GALACTICUS_EMU_ROOT is not set; needed for Halpha dust post-processing." >&2',
                    "  exit 1",
                    "fi",
                    " ".join(
                        [
                            "python",
                            '"$GALACTICUS_EMU_ROOT/scripts/process_halpha_dust_evaluation.py"',
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
                    ),
                ]
            )
        if args.enable_emission_line_dust_postprocess:
            run_lines.extend(
                [
                    'if [ -z "${GALACTICUS_EMU_ROOT:-}" ]; then',
                    '  echo "GALACTICUS_EMU_ROOT is not set; needed for emission-line dust post-processing." >&2',
                    "  exit 1",
                    "fi",
                    " ".join(
                        [
                            "python",
                            '"$GALACTICUS_EMU_ROOT/scripts/process_emission_line_dust_evaluation.py"',
                            str(output_hdf5.relative_to(campaign_root)),
                            "--evaluation-id",
                            evaluation_id,
                            "--output-dir",
                            str((evaluation_dir / args.emission_line_dust_output_dir_name).relative_to(campaign_root)),
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
                    ),
                ]
            )
        run_lines.append("")
        run_script.write_text("\n".join(run_lines))
        _make_executable(run_script)
        commands.append(f"bash {run_script.relative_to(campaign_root)}")

    (campaign_root / "commands.txt").write_text("\n".join(commands) + "\n")
    (campaign_root / "commands.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
    _make_executable(campaign_root / "commands.sh")

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
    _write_validation_plots(campaign_root, quantiles, samples, parameter_specs)
    if args.write_subset_command_scripts:
        _write_subset_command_scripts(campaign_root, args.n_eval)

    campaign_summary = {
        "schema_version": 2,
        "campaign_name": args.campaign_name,
        "n_eval": args.n_eval,
        "seed": args.seed,
        "model_changes_template": str(template_changes_path),
        "parameter_spec_registry": "galacticus_emu.specs.trinity_parameter_specs",
        "free_parameter_short_names": [spec.short_name for spec in parameter_specs],
        "free_parameters": [_parameter_spec_to_dict(spec) for spec in parameter_specs],
        "defaulted_template_parameter_short_names": [spec.short_name for spec in fixed_parameter_specs],
        "defaulted_template_parameters": [_parameter_spec_to_dict(spec) for spec in fixed_parameter_specs],
        "run_definition_changes": run_definition_changes,
        "galacticus_executable": args.galacticus_executable,
        "halpha_dust_draws": args.halpha_dust_draws,
        "halpha_dust_scatter_mode": args.halpha_dust_scatter_mode,
        "halpha_dust_postprocess_enabled": args.enable_halpha_dust_postprocess,
        "emission_line_dust_postprocess_enabled": args.enable_emission_line_dust_postprocess,
        "emission_line_dust_output_dir_name": args.emission_line_dust_output_dir_name,
    }
    (campaign_root / "campaign_design.json").write_text(json.dumps(campaign_summary, indent=2) + "\n")
    _write_command_log(campaign_root, args)

    print(campaign_root)


if __name__ == "__main__":
    main()
