from __future__ import annotations

from pathlib import Path
import csv
import json

import h5py
import numpy as np
import pandas as pd


TRINITY_RUN_CONFIGS = {
    "z0": {
        "filename": "z0.hdf5",
        "stellar_mean_key": "stellarHaloMassRelationUniverseMachinez1",
        "stellar_scatter_key": "stellarHaloMassRelationScatterUniverseMachinez1",
        "black_hole_mean_key": "blackHoleHaloMassRelationTRINITYz1",
        "black_hole_scatter_key": "blackHoleHaloMassRelationScatterTRINITYz1",
        "label": "z=0",
    },
    "z1": {
        "filename": "z1.hdf5",
        "stellar_mean_key": "stellarHaloMassRelationUniverseMachinez4",
        "stellar_scatter_key": "stellarHaloMassRelationScatterUniverseMachinez4",
        "black_hole_mean_key": "blackHoleHaloMassRelationTRINITYz2",
        "black_hole_scatter_key": "blackHoleHaloMassRelationScatterTRINITYz2",
        "label": "z=1",
    },
    "z2": {
        "filename": "z2.hdf5",
        "stellar_mean_key": "stellarHaloMassRelationUniverseMachinez6",
        "stellar_scatter_key": "stellarHaloMassRelationScatterUniverseMachinez6",
        "black_hole_mean_key": "blackHoleHaloMassRelationTRINITYz3",
        "black_hole_scatter_key": "blackHoleHaloMassRelationScatterTRINITYz3",
        "label": "z=2",
    },
}

TRINITY_MEAN_OUTPUT_COLUMNS = [
    *[f"{run_name}_mass_stellar_log10_{index}" for run_name in TRINITY_RUN_CONFIGS for index in range(3)],
    *[f"{run_name}_mass_black_hole_log10_{index}" for run_name in TRINITY_RUN_CONFIGS for index in range(3)],
]

TRINITY_SCATTER_OUTPUT_COLUMNS = [
    *[f"{run_name}_mass_stellar_log10_scatter_{index}" for run_name in TRINITY_RUN_CONFIGS for index in range(3)],
    *[f"{run_name}_mass_black_hole_log10_scatter_{index}" for run_name in TRINITY_RUN_CONFIGS for index in range(3)],
]

TRINITY_TRAINABLE_OUTPUT_COLUMNS = [
    *TRINITY_MEAN_OUTPUT_COLUMNS,
    *TRINITY_SCATTER_OUTPUT_COLUMNS,
]

TRINITY_OUTPUT_LABELS = {
    **{
        f"{run_name}_mass_stellar_log10_{index}": rf"$\langle \log_{{10}} M_\star \rangle$ {cfg['label']} bin {index + 1}"
        for run_name, cfg in TRINITY_RUN_CONFIGS.items()
        for index in range(3)
    },
    **{
        f"{run_name}_mass_black_hole_log10_{index}": rf"$\langle \log_{{10}} M_{{\rm BH}} \rangle$ {cfg['label']} bin {index + 1}"
        for run_name, cfg in TRINITY_RUN_CONFIGS.items()
        for index in range(3)
    },
    **{
        f"{run_name}_mass_stellar_log10_scatter_{index}": rf"$\sigma(\log_{{10}} M_\star)$ {cfg['label']} bin {index + 1}"
        for run_name, cfg in TRINITY_RUN_CONFIGS.items()
        for index in range(3)
    },
    **{
        f"{run_name}_mass_black_hole_log10_scatter_{index}": rf"$\sigma(\log_{{10}} M_{{\rm BH}})$ {cfg['label']} bin {index + 1}"
        for run_name, cfg in TRINITY_RUN_CONFIGS.items()
        for index in range(3)
    },
}


def _diag_errors(covariance: np.ndarray) -> list[float]:
    return np.sqrt(np.diag(covariance)).tolist()


def _active_bin_mask(mean_group: h5py.Group, value_dataset: str) -> np.ndarray:
    return mean_group[value_dataset][:] != 0.0


def _flatten_relation(
    row: dict[str, float | str],
    prefix: str,
    meta_prefix: str,
    mean_group: h5py.Group,
    scatter_group: h5py.Group,
    mass_dataset: str,
    value_dataset: str,
    target_dataset: str,
    covariance_dataset: str,
    covariance_target_dataset: str,
    scatter_dataset: str,
    scatter_target_dataset: str,
    scatter_covariance_dataset: str,
    scatter_covariance_target_dataset: str,
) -> None:
    mask = _active_bin_mask(mean_group, value_dataset)
    mass_halo = mean_group[mass_dataset][:][mask].tolist()
    values = mean_group[value_dataset][:][mask].tolist()
    values_target = mean_group[target_dataset][:][mask].tolist()
    values_error = np.array(_diag_errors(mean_group[covariance_dataset][:]))[mask].tolist()
    values_target_error = np.array(_diag_errors(mean_group[covariance_target_dataset][:]))[mask].tolist()
    scatter_values = scatter_group[scatter_dataset][:][mask].tolist()
    scatter_values_target = scatter_group[scatter_target_dataset][:][mask].tolist()
    scatter_values_error = np.array(_diag_errors(scatter_group[scatter_covariance_dataset][:]))[mask].tolist()
    scatter_values_target_error = np.array(_diag_errors(scatter_group[scatter_covariance_target_dataset][:]))[mask].tolist()

    row[f"{meta_prefix}_analysis_mean_key"] = mean_group.name.split("/")[-1]
    row[f"{meta_prefix}_analysis_scatter_key"] = scatter_group.name.split("/")[-1]
    row[f"{meta_prefix}_log_likelihood_mean"] = float(mean_group.attrs["logLikelihood"])
    row[f"{meta_prefix}_log_likelihood_scatter"] = float(scatter_group.attrs["logLikelihood"])
    row[f"{meta_prefix}_log_likelihood_total"] = row[f"{meta_prefix}_log_likelihood_mean"] + row[f"{meta_prefix}_log_likelihood_scatter"]

    value_prefix = "mass_stellar_log10" if value_dataset == "massStellarLog10" else "mass_black_hole_log10"
    scatter_prefix = f"{value_prefix}_scatter"

    for index, value in enumerate(mass_halo):
        row[f"{prefix}_mass_halo_{index}"] = value
    for index, value in enumerate(values):
        row[f"{prefix}_{value_prefix}_{index}"] = value
    for index, value in enumerate(values_target):
        row[f"{prefix}_{value_prefix}_target_{index}"] = value
    for index, value in enumerate(values_error):
        row[f"{prefix}_{value_prefix}_error_{index}"] = value
    for index, value in enumerate(values_target_error):
        row[f"{prefix}_{value_prefix}_target_error_{index}"] = value
    for index, value in enumerate(scatter_values):
        row[f"{prefix}_{scatter_prefix}_{index}"] = value
    for index, value in enumerate(scatter_values_target):
        row[f"{prefix}_{scatter_prefix}_target_{index}"] = value
    for index, value in enumerate(scatter_values_error):
        row[f"{prefix}_{scatter_prefix}_error_{index}"] = value
    for index, value in enumerate(scatter_values_target_error):
        row[f"{prefix}_{scatter_prefix}_target_error_{index}"] = value


def extract_trinity_row(evaluation_dir: str | Path) -> dict[str, float | str]:
    evaluation_dir = Path(evaluation_dir)
    row: dict[str, float | str] = {"evaluation_id": evaluation_dir.name}

    for run_name, config in TRINITY_RUN_CONFIGS.items():
        with h5py.File(evaluation_dir / config["filename"], "r") as handle:
            analyses = handle["analyses"]
            _flatten_relation(
                row=row,
                prefix=run_name,
                meta_prefix=f"{run_name}_stellar",
                mean_group=analyses[config["stellar_mean_key"]],
                scatter_group=analyses[config["stellar_scatter_key"]],
                mass_dataset="massHalo",
                value_dataset="massStellarLog10",
                target_dataset="massStellarLog10Target",
                covariance_dataset="massStellarLog10Covariance",
                covariance_target_dataset="massStellarLog10CovarianceTarget",
                scatter_dataset="massStellarLog10Scatter",
                scatter_target_dataset="massStellarLog10ScatterTarget",
                scatter_covariance_dataset="massStellarLog10ScatterCovariance",
                scatter_covariance_target_dataset="massStellarLog10ScatterCovarianceTarget",
            )
            _flatten_relation(
                row=row,
                prefix=run_name,
                meta_prefix=f"{run_name}_black_hole",
                mean_group=analyses[config["black_hole_mean_key"]],
                scatter_group=analyses[config["black_hole_scatter_key"]],
                mass_dataset="massHalo",
                value_dataset="massBlackHoleLog10",
                target_dataset="massBlackHoleLog10Target",
                covariance_dataset="massBlackHoleLog10Covariance",
                covariance_target_dataset="massBlackHoleLog10CovarianceTarget",
                scatter_dataset="massBlackHoleLog10Scatter",
                scatter_target_dataset="massBlackHoleLog10ScatterTarget",
                scatter_covariance_dataset="massBlackHoleLog10ScatterCovariance",
                scatter_covariance_target_dataset="massBlackHoleLog10ScatterCovarianceTarget",
            )
    return row


def summarize_trinity_campaign(campaign_root: str | Path) -> Path:
    campaign_root = Path(campaign_root)
    evaluations_root = campaign_root / "evaluations"
    output_path = campaign_root / "summary.csv"
    jsonl_path = campaign_root / "summary.jsonl"

    rows: list[dict[str, float | str]] = []
    for evaluation_dir in sorted(evaluations_root.iterdir()):
        if not evaluation_dir.is_dir():
            continue
        required = [evaluation_dir / cfg["filename"] for cfg in TRINITY_RUN_CONFIGS.values()]
        if not all(path.exists() for path in required):
            continue
        try:
            rows.append(extract_trinity_row(evaluation_dir))
        except (OSError, RuntimeError, KeyError):
            continue

    if not rows:
        raise FileNotFoundError(f"No completed Trinity Galacticus outputs found under {evaluations_root}")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    return output_path


def load_or_build_emulator_table(campaign_root: str | Path) -> pd.DataFrame:
    campaign_root = Path(campaign_root)
    emulator_table_path = campaign_root / "emulator_table.csv"
    if emulator_table_path.exists():
        return pd.read_csv(emulator_table_path)
    samples = pd.read_csv(campaign_root / "samples.csv")
    summary = pd.read_csv(campaign_root / "summary.csv")
    merged = samples.merge(summary, on="evaluation_id", how="inner", validate="one_to_one")
    merged.to_csv(emulator_table_path, index=False)
    return merged
