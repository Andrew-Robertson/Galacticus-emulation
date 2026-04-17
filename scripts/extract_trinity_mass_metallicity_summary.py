from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import numpy as np
import pandas as pd

from galacticus_emu import trinity_parameter_specs


HALO_BIN_TARGETS = np.array([1.0e11, 1.0e12, 1.0e13], dtype=float)
HALO_BIN_EDGES = np.array(
    [0.0, np.sqrt(HALO_BIN_TARGETS[0] * HALO_BIN_TARGETS[1]), np.sqrt(HALO_BIN_TARGETS[1] * HALO_BIN_TARGETS[2]), np.inf],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract approximate z=0 mass-metallicity summaries for a Trinity campaign.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--observation-file",
        default="/Users/arobertson/Documents/Projects/GalacticusEmu/Galacticus/datasets/static/observations/abundances/massMetallicityRelationBlanc2019.hdf5",
        help="Path to the Blanc 2019 mass-metallicity HDF5 file.",
    )
    parser.add_argument(
        "--hydrogen-mass-fraction",
        type=float,
        default=0.70,
        help="Assumed hydrogen mass fraction X used when approximately converting total gas metallicity to O/H.",
    )
    parser.add_argument(
        "--oxygen-metal-mass-fraction",
        type=float,
        default=0.44,
        help="Assumed fraction of total metal mass that is oxygen, M_O / M_Z.",
    )
    return parser.parse_args()


def _approximate_oxygen_abundance_12logoh(
    metallicity_fraction: np.ndarray,
    hydrogen_mass_fraction: float,
    oxygen_metal_mass_fraction: float,
) -> np.ndarray:
    return 12.0 + np.log10((oxygen_metal_mass_fraction * metallicity_fraction) / (16.0 * hydrogen_mass_fraction))


def _load_observation_interpolator(observation_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(observation_file, "r") as handle:
        stellar_mass_log10 = np.log10(handle["massStellar"][:])
        abundance_mean = handle["abundanceOxygenMean"][:]
    return stellar_mass_log10, abundance_mean


def _extract_row(
    output_path: Path,
    observed_stellar_mass_log10: np.ndarray,
    observed_abundance_mean: np.ndarray,
    hydrogen_mass_fraction: float,
    oxygen_metal_mass_fraction: float,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"evaluation_id": output_path.parent.name}
    with h5py.File(output_path, "r") as handle:
        node_data = handle["Outputs/Output1/nodeData"]
        disk_mass_stellar = node_data["diskMassStellar"][:]
        spheroid_mass_stellar = node_data["spheroidMassStellar"][:]
        disk_mass_gas = node_data["diskMassGas"][:]
        spheroid_mass_gas = node_data["spheroidMassGas"][:]
        disk_metals_gas = node_data["diskAbundancesGasMetals"][:]
        spheroid_metals_gas = node_data["spheroidAbundancesGasMetals"][:]
        basic_mass = node_data["basicMass"][:]
        node_is_isolated = node_data["nodeIsIsolated"][:].astype(bool)

    stellar_mass = disk_mass_stellar + spheroid_mass_stellar
    gas_mass = disk_mass_gas + spheroid_mass_gas
    gas_metals = disk_metals_gas + spheroid_metals_gas
    valid = (
        node_is_isolated
        & (stellar_mass > 0.0)
        & (gas_mass > 0.0)
        & np.isfinite(stellar_mass)
        & np.isfinite(gas_mass)
        & np.isfinite(gas_metals)
        & np.isfinite(basic_mass)
    )
    metallicity = np.full_like(stellar_mass, np.nan, dtype=float)
    metallicity[valid] = gas_metals[valid] / gas_mass[valid]
    positive = valid & (metallicity > 0.0)

    halo_mass = basic_mass[positive]
    log10_stellar_mass = np.log10(stellar_mass[positive])
    approx_oxygen_abundance = _approximate_oxygen_abundance_12logoh(
        metallicity[positive],
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        oxygen_metal_mass_fraction=oxygen_metal_mass_fraction,
    )
    bin_index = np.digitize(halo_mass, HALO_BIN_EDGES[1:-1])

    row["z0_mass_metallicity_hydrogen_mass_fraction"] = hydrogen_mass_fraction
    row["z0_mass_metallicity_oxygen_metal_mass_fraction"] = oxygen_metal_mass_fraction
    for index, target_mass in enumerate(HALO_BIN_TARGETS):
        mask = bin_index == index
        row[f"z0_mass_metallicity_halo_target_{index}"] = float(target_mass)
        row[f"z0_mass_metallicity_count_{index}"] = int(np.count_nonzero(mask))
        if np.any(mask):
            mean_log10_stellar_mass = float(np.mean(log10_stellar_mass[mask]))
            mean_abundance_oxygen = float(np.mean(approx_oxygen_abundance[mask]))
            target_abundance_oxygen = float(
                np.interp(
                    mean_log10_stellar_mass,
                    observed_stellar_mass_log10,
                    observed_abundance_mean,
                    left=observed_abundance_mean[0],
                    right=observed_abundance_mean[-1],
                )
            )
        else:
            mean_log10_stellar_mass = np.nan
            mean_abundance_oxygen = np.nan
            target_abundance_oxygen = np.nan
        row[f"z0_mass_metallicity_mass_stellar_log10_{index}"] = mean_log10_stellar_mass
        row[f"z0_mass_metallicity_abundance_oxygen_12logoh_{index}"] = mean_abundance_oxygen
        row[f"z0_mass_metallicity_abundance_oxygen_12logoh_target_{index}"] = target_abundance_oxygen
        row[f"z0_mass_metallicity_abundance_oxygen_12logoh_residual_{index}"] = (
            mean_abundance_oxygen - target_abundance_oxygen
            if np.isfinite(mean_abundance_oxygen) and np.isfinite(target_abundance_oxygen)
            else np.nan
        )
    return row


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    observation_file = Path(args.observation_file).resolve()
    evaluations_root = campaign_root / "evaluations"
    summary_path = campaign_root / "mass_metallicity_summary.csv"
    jsonl_path = campaign_root / "mass_metallicity_summary.jsonl"
    merged_path = campaign_root / "mass_metallicity_emulator_table.csv"
    observed_stellar_mass_log10, observed_abundance_mean = _load_observation_interpolator(observation_file)

    rows: list[dict[str, float | str]] = []
    for evaluation_dir in sorted(evaluations_root.iterdir()):
        if not evaluation_dir.is_dir():
            continue
        output_path = evaluation_dir / "z0.hdf5"
        if not output_path.exists():
            continue
        try:
            rows.append(
                _extract_row(
                    output_path=output_path,
                    observed_stellar_mass_log10=observed_stellar_mass_log10,
                    observed_abundance_mean=observed_abundance_mean,
                    hydrogen_mass_fraction=args.hydrogen_mass_fraction,
                    oxygen_metal_mass_fraction=args.oxygen_metal_mass_fraction,
                )
            )
        except (OSError, RuntimeError, KeyError, ValueError):
            continue

    if not rows:
        raise FileNotFoundError(f"No readable z0 outputs found under {evaluations_root}")

    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    samples = pd.read_csv(campaign_root / "samples.csv")
    summary = pd.read_csv(summary_path)
    input_columns = [spec.short_name for spec in trinity_parameter_specs()]
    merged = samples[["evaluation_id", *input_columns]].merge(summary, on="evaluation_id", how="inner", validate="one_to_one")
    merged.to_csv(merged_path, index=False)

    print(summary_path)
    print(merged_path)


if __name__ == "__main__":
    main()
