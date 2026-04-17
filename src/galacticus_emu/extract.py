from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import h5py
import numpy as np


@dataclass(frozen=True)
class SHMRSummary:
    evaluation_id: str
    analysis_mean_key: str
    analysis_scatter_key: str
    mass_halo: list[float]
    mass_stellar_log10: list[float]
    mass_stellar_log10_target: list[float]
    mass_stellar_log10_error: list[float]
    mass_stellar_log10_target_error: list[float]
    mass_stellar_log10_scatter: list[float]
    mass_stellar_log10_scatter_target: list[float]
    mass_stellar_log10_scatter_error: list[float]
    mass_stellar_log10_scatter_target_error: list[float]
    log_likelihood_mean: float
    log_likelihood_scatter: float


def _diag_errors(covariance: np.ndarray) -> list[float]:
    return np.sqrt(np.diag(covariance)).tolist()


def _active_bin_mask(mean_group: h5py.Group, scatter_group: h5py.Group) -> np.ndarray:
    mean_model = mean_group["massStellarLog10"][:]
    scatter_model = scatter_group["massStellarLog10Scatter"][:]
    return np.logical_or(mean_model != 0.0, scatter_model != 0.0)


def extract_shmr_summary(
    hdf5_path: str | Path,
    evaluation_id: str,
    analysis_mean_key: str = "stellarHaloMassRelationUniverseMachinez6",
    analysis_scatter_key: str = "stellarHaloMassRelationScatterUniverseMachinez6",
    active_only: bool = True,
) -> SHMRSummary:
    hdf5_path = Path(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        mean_group = handle["analyses"][analysis_mean_key]
        scatter_group = handle["analyses"][analysis_scatter_key]
        mask = _active_bin_mask(mean_group, scatter_group)
        if not active_only:
            mask = np.ones_like(mask, dtype=bool)

        return SHMRSummary(
            evaluation_id=evaluation_id,
            analysis_mean_key=analysis_mean_key,
            analysis_scatter_key=analysis_scatter_key,
            mass_halo=mean_group["massHalo"][:][mask].tolist(),
            mass_stellar_log10=mean_group["massStellarLog10"][:][mask].tolist(),
            mass_stellar_log10_target=mean_group["massStellarLog10Target"][:][mask].tolist(),
            mass_stellar_log10_error=np.array(_diag_errors(mean_group["massStellarLog10Covariance"][:]))[mask].tolist(),
            mass_stellar_log10_target_error=np.array(_diag_errors(mean_group["massStellarLog10CovarianceTarget"][:]))[mask].tolist(),
            mass_stellar_log10_scatter=scatter_group["massStellarLog10Scatter"][:][mask].tolist(),
            mass_stellar_log10_scatter_target=scatter_group["massStellarLog10ScatterTarget"][:][mask].tolist(),
            mass_stellar_log10_scatter_error=np.array(_diag_errors(scatter_group["massStellarLog10ScatterCovariance"][:]))[mask].tolist(),
            mass_stellar_log10_scatter_target_error=np.array(_diag_errors(scatter_group["massStellarLog10ScatterCovarianceTarget"][:]))[mask].tolist(),
            log_likelihood_mean=float(mean_group.attrs["logLikelihood"]),
            log_likelihood_scatter=float(scatter_group.attrs["logLikelihood"]),
        )


def flatten_shmr_summary(summary: SHMRSummary) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "evaluation_id": summary.evaluation_id,
        "analysis_mean_key": summary.analysis_mean_key,
        "analysis_scatter_key": summary.analysis_scatter_key,
        "log_likelihood_mean": summary.log_likelihood_mean,
        "log_likelihood_scatter": summary.log_likelihood_scatter,
        "log_likelihood_total": summary.log_likelihood_mean + summary.log_likelihood_scatter,
    }
    for index, value in enumerate(summary.mass_halo):
        row[f"mass_halo_{index}"] = value
    for index, value in enumerate(summary.mass_stellar_log10):
        row[f"mass_stellar_log10_{index}"] = value
    for index, value in enumerate(summary.mass_stellar_log10_target):
        row[f"mass_stellar_log10_target_{index}"] = value
    for index, value in enumerate(summary.mass_stellar_log10_error):
        row[f"mass_stellar_log10_error_{index}"] = value
    for index, value in enumerate(summary.mass_stellar_log10_target_error):
        row[f"mass_stellar_log10_target_error_{index}"] = value
    for index, value in enumerate(summary.mass_stellar_log10_scatter):
        row[f"mass_stellar_log10_scatter_{index}"] = value
    for index, value in enumerate(summary.mass_stellar_log10_scatter_target):
        row[f"mass_stellar_log10_scatter_target_{index}"] = value
    for index, value in enumerate(summary.mass_stellar_log10_scatter_error):
        row[f"mass_stellar_log10_scatter_error_{index}"] = value
    for index, value in enumerate(summary.mass_stellar_log10_scatter_target_error):
        row[f"mass_stellar_log10_scatter_target_error_{index}"] = value
    return row


def summarize_campaign(
    campaign_root: str | Path,
    analysis_mean_key: str = "stellarHaloMassRelationUniverseMachinez6",
    analysis_scatter_key: str = "stellarHaloMassRelationScatterUniverseMachinez6",
    active_only: bool = True,
) -> Path:
    campaign_root = Path(campaign_root)
    evaluations_root = campaign_root / "evaluations"
    output_path = campaign_root / "summary.csv"
    jsonl_path = campaign_root / "summary.jsonl"

    rows: list[dict[str, float | str]] = []
    for evaluation_dir in sorted(evaluations_root.iterdir()):
        if not evaluation_dir.is_dir():
            continue
        hdf5_path = evaluation_dir / "galacticus.hdf5"
        if not hdf5_path.exists():
            continue
        summary = extract_shmr_summary(
            hdf5_path=hdf5_path,
            evaluation_id=evaluation_dir.name,
            analysis_mean_key=analysis_mean_key,
            analysis_scatter_key=analysis_scatter_key,
            active_only=active_only,
        )
        rows.append(flatten_shmr_summary(summary))

    if not rows:
        raise FileNotFoundError(f"No completed Galacticus outputs found under {evaluations_root}")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    return output_path
