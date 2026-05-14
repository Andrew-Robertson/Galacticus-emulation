from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import h5py
import numpy as np
import pandas as pd

from calculate_halpha_dust_lf_grid import _combined_node_weights, _log10_std_from_linear  # type: ignore
from process_emission_line_dust_evaluation import (  # type: ignore
    LINE_GROUPS,
    _attenuation_coefficient,
    _case_definitions,
    _dust_attenuated_luminosity,
    _lf_from_expected_scatter,
    _lf_from_luminosities,
    _line_group_luminosities,
    _stellar_mass,
    _total_luminosity,
)


OII_OBSERVABLES = {"oii_comparat", "oii_khostovan"}
DEFAULT_OII_BOOST_LOG10 = [0.0, 0.15, 0.30, 0.45, 0.60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process one Galacticus evaluation to compute [OII] luminosity "
            "functions on a grid of luminosity boost factors, reusing an existing "
            "emission_line_dust/dust_draws.csv file."
        )
    )
    parser.add_argument("galacticus_file", type=Path)
    parser.add_argument("--evaluation-id", default=None)
    parser.add_argument(
        "--dust-draws-csv",
        type=Path,
        default=None,
        help="Existing dust_draws.csv. Defaults to <evaluation_dir>/emission_line_dust/dust_draws.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <evaluation_dir>/oii_boost_grid.",
    )
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument(
        "--oii-boost-log10",
        type=float,
        action="append",
        default=None,
        help=(
            "log10 luminosity boost to apply to [OII] before binning. "
            "Can be supplied multiple times. Defaults to 0.0, 0.15, 0.30, 0.45, 0.60."
        ),
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dust_case_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "label": str(row["dust_case"]),
        "dust_model": "gb10_generalised",
        "dust_params": {
            "delta_0": float(row["delta_0"]),
            "delta_z": float(row["delta_z"]),
            "delta_M": float(row["delta_M"]),
            "delta_Mz": float(row["delta_Mz"]),
            "attenuation_scatter": float(row["attenuation_scatter"]),
            "z_pivot": float(row["z_pivot"]),
        },
        "dust_law": "calzetti",
        "random_uniform_index": None,
        "scatter_mode": str(row["scatter_mode"]),
        "dust_draw_index": int(row["dust_draw_index"]),
    }


def _compute_oii_lfs(
    galacticus_file: Path,
    dust_case: dict[str, Any],
    *,
    boost_log10: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boost_factor = 10.0**boost_log10
    with h5py.File(galacticus_file, "r") as handle:
        cases = [case for case in _case_definitions(handle) if case["observable"] in OII_OBSERVABLES]
        for case in cases:
            observable = case["observable"]
            output_group = handle[f"/Outputs/{case['output_name']}"]
            weights = _combined_node_weights(output_group)
            line_luminosities = _line_group_luminosities(output_group, observable)
            boosted_line_luminosities = {
                line_name: luminosity * boost_factor for line_name, luminosity in line_luminosities.items()
            }

            if dust_case["scatter_mode"] == "expected":
                intrinsic = _total_luminosity(boosted_line_luminosities, observable)
                lf = _lf_from_expected_scatter(
                    intrinsic,
                    _stellar_mass(output_group),
                    weights,
                    float(case["redshift"]),
                    dust_case["dust_params"],
                    np.asarray(case["log10_edges"], dtype=float),
                    _attenuation_coefficient(float(LINE_GROUPS[observable]["effective_wavelength"])),
                )
                phi_var_conservative = lf["phi_variance_conservative_mpc3_dex2"]
                phi_var_smoothed = lf["phi_variance_smoothed_mpc3_dex2"]
                dn_var_conservative = lf["dn_dlnL_variance_conservative_mpc3"]
                dn_var_smoothed = lf["dn_dlnL_variance_smoothed_mpc3"]
            else:
                rng = np.random.default_rng(
                    np.random.SeedSequence([base_seed, dust_case["dust_draw_index"], int(case["output_name"][6:])])
                )
                luminosity = _dust_attenuated_luminosity(
                    output_group,
                    float(case["redshift"]),
                    observable,
                    boosted_line_luminosities,
                    dust_case,
                    rng=rng,
                )
                lf = _lf_from_luminosities(luminosity, weights, np.asarray(case["log10_edges"], dtype=float))
                phi_var_conservative = lf["phi_variance_mpc3_dex2"]
                phi_var_smoothed = lf["phi_variance_mpc3_dex2"]
                dn_var_conservative = lf["dn_dlnL_variance_mpc3"]
                dn_var_smoothed = lf["dn_dlnL_variance_mpc3"]

            phi_std_conservative = np.sqrt(np.clip(phi_var_conservative, 0.0, None))
            phi_std_smoothed = np.sqrt(np.clip(phi_var_smoothed, 0.0, None))
            dn_std_conservative = np.sqrt(np.clip(dn_var_conservative, 0.0, None))
            dn_std_smoothed = np.sqrt(np.clip(dn_var_smoothed, 0.0, None))
            centers = np.asarray(case["log10_centers"], dtype=float)
            edges = np.asarray(case["log10_edges"], dtype=float)

            for bin_index, center in enumerate(centers):
                value_phi = float(lf["phi_mpc3_dex"][bin_index])
                value_dn = float(lf["dn_dlnL_mpc3"][bin_index])
                row = {
                    "observable": observable,
                    "observable_label": LINE_GROUPS[observable]["label"],
                    "sample_label": case["sample_label"],
                    "output_name": case["output_name"],
                    "redshift": float(case["redshift"]),
                    "bin_index": int(bin_index),
                    "log10_luminosity_min": float(edges[bin_index]),
                    "log10_luminosity_center": float(center),
                    "log10_luminosity_max": float(edges[bin_index + 1]),
                    "phi_mpc3_dex": value_phi,
                    "phi_mpc3_dex_shot_noise_std_conservative": float(phi_std_conservative[bin_index]),
                    "phi_mpc3_dex_shot_noise_std_smoothed_expectation": float(phi_std_smoothed[bin_index]),
                    "log10_phi_shot_noise_std_conservative": _log10_std_from_linear(
                        value_phi,
                        float(phi_std_conservative[bin_index]),
                    ),
                    "log10_phi_shot_noise_std_smoothed_expectation": _log10_std_from_linear(
                        value_phi,
                        float(phi_std_smoothed[bin_index]),
                    ),
                    "dn_dlnL_mpc3": value_dn,
                    "dn_dlnL_mpc3_shot_noise_std_conservative": float(dn_std_conservative[bin_index]),
                    "dn_dlnL_mpc3_shot_noise_std_smoothed_expectation": float(dn_std_smoothed[bin_index]),
                    "log10_dn_dlnL_shot_noise_std_conservative": _log10_std_from_linear(
                        value_dn,
                        float(dn_std_conservative[bin_index]),
                    ),
                    "log10_dn_dlnL_shot_noise_std_smoothed_expectation": _log10_std_from_linear(
                        value_dn,
                        float(dn_std_smoothed[bin_index]),
                    ),
                }
                if "z_min" in case:
                    row["z_min"] = case["z_min"]
                    row["z_max"] = case["z_max"]
                if "target_redshift" in case:
                    row["target_redshift"] = case["target_redshift"]
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    galacticus_file = args.galacticus_file.resolve()
    evaluation_id = args.evaluation_id or galacticus_file.parent.name
    dust_draws_csv = (
        args.dust_draws_csv.resolve()
        if args.dust_draws_csv is not None
        else galacticus_file.parent / "emission_line_dust" / "dust_draws.csv"
    )
    output_dir = args.output_dir.resolve() if args.output_dir is not None else galacticus_file.parent / "oii_boost_grid"
    output_dir.mkdir(parents=True, exist_ok=True)
    boost_grid = args.oii_boost_log10 if args.oii_boost_log10 is not None else DEFAULT_OII_BOOST_LOG10

    input_values: dict[str, Any] = {}
    if args.input_json is not None:
        input_values = json.loads(args.input_json.read_text())

    dust_draws = pd.read_csv(dust_draws_csv)
    lf_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    print(f"starting [OII] boost-grid post-processing for {evaluation_id}")
    for _, draw in dust_draws.iterrows():
        dust_case = _dust_case_from_row(draw)
        draw_metadata = {
            "evaluation_id": evaluation_id,
            "dust_draw_index": int(draw["dust_draw_index"]),
            "dust_case": str(draw["dust_case"]),
            "scatter_mode": str(draw["scatter_mode"]),
            "delta_0": float(draw["delta_0"]),
            "delta_z": float(draw["delta_z"]),
            "delta_M": float(draw["delta_M"]),
            "delta_Mz": float(draw["delta_Mz"]),
            "attenuation_scatter": float(draw["attenuation_scatter"]),
            "z_pivot": float(draw["z_pivot"]),
        }
        print(f"{evaluation_id} dust draw {draw_metadata['dust_draw_index']} on {len(boost_grid)} boost values")
        for boost_log10 in boost_grid:
            boost_log10 = float(boost_log10)
            boost_metadata = {
                **draw_metadata,
                "oii_boost_log10": boost_log10,
                "oii_boost_factor": 10.0**boost_log10,
            }
            wide_row = {
                **boost_metadata,
                **input_values,
            }
            for row in _compute_oii_lfs(
                galacticus_file,
                dust_case,
                boost_log10=boost_log10,
                base_seed=args.base_seed,
            ):
                merged = dict(boost_metadata)
                merged.update(row)
                lf_rows.append(merged)
                prefix = f"{row['observable']}_{row['sample_label'].lower()}_bin{row['bin_index']}"
                wide_row[prefix] = row["dn_dlnL_mpc3"]
                wide_row[f"{prefix}_shot_noise_std_conservative"] = row[
                    "dn_dlnL_mpc3_shot_noise_std_conservative"
                ]
                wide_row[f"{prefix}_shot_noise_std_smoothed_expectation"] = row[
                    "dn_dlnL_mpc3_shot_noise_std_smoothed_expectation"
                ]
                wide_row[f"{prefix}_shot_noise_log10_std_conservative"] = row[
                    "log10_dn_dlnL_shot_noise_std_conservative"
                ]
                wide_row[f"{prefix}_shot_noise_log10_std_smoothed_expectation"] = row[
                    "log10_dn_dlnL_shot_noise_std_smoothed_expectation"
                ]
                wide_row[f"{prefix}_phi_mpc3_dex"] = row["phi_mpc3_dex"]
                wide_row[f"{prefix}_phi_shot_noise_std_conservative"] = row[
                    "phi_mpc3_dex_shot_noise_std_conservative"
                ]
                wide_row[f"{prefix}_phi_shot_noise_std_smoothed_expectation"] = row[
                    "phi_mpc3_dex_shot_noise_std_smoothed_expectation"
                ]
            table_rows.append(wide_row)

    dust_boost_grid_path = output_dir / "oii_boost_grid.csv"
    lf_long_path = output_dir / "oii_boost_grid_lf_long.csv"
    emulator_table_path = output_dir / "oii_boost_grid_lf_emulator_table.csv"
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "manifest.csv"

    boost_rows = [
        {"oii_boost_log10": float(value), "oii_boost_factor": 10.0 ** float(value)} for value in boost_grid
    ]
    _write_csv(dust_boost_grid_path, boost_rows)
    _write_csv(lf_long_path, lf_rows)
    _write_csv(emulator_table_path, table_rows)
    metadata_path.write_text(
        json.dumps(
            {
                "evaluation_id": evaluation_id,
                "galacticus_file": str(galacticus_file),
                "dust_draws_csv": str(dust_draws_csv),
                "boost_grid_log10": [float(value) for value in boost_grid],
                "observables": sorted(OII_OBSERVABLES),
                "row_count_long": len(lf_rows),
                "row_count_table": len(table_rows),
                "outputs": {
                    "dust_boost_grid": str(dust_boost_grid_path),
                    "lf_long": str(lf_long_path),
                    "emulator_table": str(emulator_table_path),
                    "metadata": str(metadata_path),
                    "manifest": str(manifest_path),
                },
            },
            indent=2,
        )
    )
    _write_csv(
        manifest_path,
        [
            {"file": str(dust_boost_grid_path), "description": "Grid of [OII] luminosity boost factors."},
            {"file": str(lf_long_path), "description": "Long-form [OII] LF values for this boost grid."},
            {"file": str(emulator_table_path), "description": "Wide-form emulator rows for this boost grid."},
            {"file": str(metadata_path), "description": "Processing metadata for this evaluation."},
        ],
    )

    print(f"finished [OII] boost-grid post-processing for {evaluation_id}")
    print(dust_boost_grid_path)
    print(lf_long_path)
    print(emulator_table_path)
    print(metadata_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
