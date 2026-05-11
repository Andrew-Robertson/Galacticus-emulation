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

from calculate_halpha_dust_lf_grid import (  # type: ignore
    SOBRAL_CASES,
    _attenuated_halpha,
    _combined_node_weights,
    _intrinsic_halpha,
    _lf_from_expected_scatter,
    _lf_from_luminosities,
    _log10_std_from_linear,
    _sobral_bin_edges,
)
from process_halpha_dust_campaign import _load_dust_priors, _sample_dust_params  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process one Galacticus evaluation to compute dust-attenuated "
            "Halpha luminosity functions for many dust-model draws."
        )
    )
    parser.add_argument("galacticus_file", type=Path, help="Path to one Galacticus HDF5 output.")
    parser.add_argument(
        "--evaluation-id",
        default=None,
        help="Optional evaluation identifier. Defaults to the parent directory name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the compact outputs should be written. Defaults to <evaluation_dir>/halpha_dust.",
    )
    parser.add_argument("--n-dust-draws", type=int, default=32)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--evaluation-index", type=int, default=0, help="Stable evaluation index used in the RNG seed.")
    parser.add_argument("--scatter-mode", choices=["draw", "expected"], default="expected")
    parser.add_argument("--z-pivot", type=float, default=1.0)
    parser.add_argument("--random-uniform-index", type=int, default=None)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional JSON file containing the slow/Galacticus input parameters for this evaluation.",
    )
    parser.add_argument(
        "--dust-prior-json",
        action="append",
        default=[],
        help=(
            "Override one dust-parameter prior using label='{\"distribution\":\"uniform\",\"lower\":...,\"upper\":...}' "
            "or label='{\"distribution\":\"normal\",\"mean\":...,\"sigma\":...}' "
            "or label='{\"distribution\":\"truncated_normal\",\"mean\":...,\"sigma\":...,\"lower\":...}'."
        ),
    )
    return parser.parse_args()


def _compute_case_lfs(
    galacticus_file: Path,
    dust_case: dict[str, Any],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    with h5py.File(galacticus_file, "r") as handle:
        for output_name, redshift, sobral_label, analysis_name in SOBRAL_CASES:
            output_group = handle[f"/Outputs/{output_name}"]
            analysis_group = handle[f"/analyses/{analysis_name}"]
            weights = _combined_node_weights(output_group)
            nd = output_group["nodeData"]
            stellar_mass = (
                np.asarray(nd["diskMassStellar"][...], dtype=float)
                + np.asarray(nd["spheroidMassStellar"][...], dtype=float)
            )
            intrinsic_luminosity = _intrinsic_halpha(output_group)
            log10_centers = np.log10(np.asarray(analysis_group["luminosity"][...], dtype=float))
            log10_edges = _sobral_bin_edges(log10_centers)

            if dust_case["scatter_mode"] == "expected":
                lf, variance_conservative, variance_smoothed = _lf_from_expected_scatter(
                    intrinsic_luminosity,
                    stellar_mass,
                    weights,
                    redshift,
                    dust_case["dust_params"],
                    log10_edges,
                )
            else:
                luminosity = _attenuated_halpha(output_group, redshift, dust_case)
                lf, variance_conservative, variance_smoothed = _lf_from_luminosities(luminosity, weights, log10_edges)

            std_conservative = np.sqrt(np.clip(variance_conservative, 0.0, None))
            std_smoothed = np.sqrt(np.clip(variance_smoothed, 0.0, None))

            target_lf = np.asarray(analysis_group["luminosityFunctionTarget"][...], dtype=float)
            target_cov = np.asarray(analysis_group["luminosityFunctionCovarianceTarget"][...], dtype=float)
            target_std = np.sqrt(np.clip(np.diag(target_cov), 0.0, None))
            for bin_index, (center, value, sigma_conservative, sigma_smoothed, target, target_sigma) in enumerate(
                zip(log10_centers, lf, std_conservative, std_smoothed, target_lf, target_std, strict=True)
            ):
                output_rows.append(
                    {
                        "output_name": output_name,
                        "sobral_label": sobral_label,
                        "redshift": float(redshift),
                        "analysis_name": analysis_name,
                        "bin_index": int(bin_index),
                        "log10_luminosity_center": float(center),
                        "dn_dlnL_mpc3": float(value),
                        "dn_dlnL_mpc3_shot_noise_std_conservative": float(sigma_conservative),
                        "dn_dlnL_mpc3_shot_noise_std_smoothed_expectation": float(sigma_smoothed),
                        "log10_dn_dlnL_shot_noise_std_conservative": _log10_std_from_linear(
                            float(value),
                            float(sigma_conservative),
                        ),
                        "log10_dn_dlnL_shot_noise_std_smoothed_expectation": _log10_std_from_linear(
                            float(value),
                            float(sigma_smoothed),
                        ),
                        "target_dn_dlnL_mpc3": float(target),
                        "target_std_dn_dlnL_mpc3": float(target_sigma),
                    }
                )
    return output_rows


def main() -> None:
    args = parse_args()
    galacticus_file = args.galacticus_file.resolve()
    evaluation_id = args.evaluation_id or galacticus_file.parent.name
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (galacticus_file.parent / "halpha_dust")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_values: dict[str, Any] = {}
    if args.input_json is not None:
        input_values = json.loads(args.input_json.read_text())

    priors = _load_dust_priors(args)
    rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, args.evaluation_index]))

    dust_draw_rows: list[dict[str, Any]] = []
    lf_rows: list[dict[str, Any]] = []
    emulator_rows: list[dict[str, Any]] = []

    print(f"starting dust post-processing for {evaluation_id}")
    for dust_draw_index in range(args.n_dust_draws):
        print(f"{evaluation_id} dust draw {dust_draw_index + 1}/{args.n_dust_draws}")
        dust_params = _sample_dust_params(
            priors,
            z_pivot=args.z_pivot,
            rng=rng,
        )
        dust_case = {
            "label": f"dust_draw_{dust_draw_index:04d}",
            "dust_model": "gb10_generalised",
            "dust_params": dust_params,
            "dust_law": "calzetti",
            "random_uniform_index": args.random_uniform_index,
            "scatter_mode": args.scatter_mode,
        }
        draw_metadata = {
            "evaluation_id": evaluation_id,
            "dust_draw_index": int(dust_draw_index),
            "dust_case": dust_case["label"],
            "scatter_mode": args.scatter_mode,
            **dust_params,
        }
        dust_draw_rows.append(draw_metadata)

        wide_row = {
            "evaluation_id": evaluation_id,
            "dust_draw_index": int(dust_draw_index),
            "dust_case": dust_case["label"],
            "scatter_mode": args.scatter_mode,
            **input_values,
            **dust_params,
        }
        draw_lf_rows = _compute_case_lfs(galacticus_file, dust_case)
        for row in draw_lf_rows:
            merged = dict(draw_metadata)
            merged.update(row)
            lf_rows.append(merged)
            output_column = f"halpha_sobral_{row['sobral_label'].lower()}_bin{row['bin_index']}"
            wide_row[output_column] = row["dn_dlnL_mpc3"]
            wide_row[f"{output_column}_shot_noise_std_conservative"] = row[
                "dn_dlnL_mpc3_shot_noise_std_conservative"
            ]
            wide_row[f"{output_column}_shot_noise_std_smoothed_expectation"] = row[
                "dn_dlnL_mpc3_shot_noise_std_smoothed_expectation"
            ]
            wide_row[f"{output_column}_shot_noise_log10_std_conservative"] = row[
                "log10_dn_dlnL_shot_noise_std_conservative"
            ]
            wide_row[f"{output_column}_shot_noise_log10_std_smoothed_expectation"] = row[
                "log10_dn_dlnL_shot_noise_std_smoothed_expectation"
            ]
            wide_row[f"{output_column}_target"] = row["target_dn_dlnL_mpc3"]
            wide_row[f"{output_column}_target_std"] = row["target_std_dn_dlnL_mpc3"]
        emulator_rows.append(wide_row)

    dust_draws_path = output_dir / "dust_draws.csv"
    with dust_draws_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dust_draw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dust_draw_rows)

    lf_long_path = output_dir / "halpha_dust_lf_long.csv"
    with lf_long_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lf_rows[0].keys()))
        writer.writeheader()
        writer.writerows(lf_rows)

    emulator_table_path = output_dir / "halpha_dust_lf_emulator_table.csv"
    with emulator_table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(emulator_rows[0].keys()))
        writer.writeheader()
        writer.writerows(emulator_rows)

    metadata = {
        "evaluation_id": evaluation_id,
        "galacticus_file": str(galacticus_file),
        "n_dust_draws": args.n_dust_draws,
        "scatter_mode": args.scatter_mode,
        "z_pivot": args.z_pivot,
        "random_uniform_index": args.random_uniform_index,
        "dust_priors": priors,
        "input_values": input_values,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "description"])
        writer.writeheader()
        writer.writerows(
            [
                {"file": str(dust_draws_path), "description": "Dust parameter draws for this evaluation."},
                {"file": str(lf_long_path), "description": "Long-form Sobral-bin Halpha LF values for this evaluation."},
                {"file": str(emulator_table_path), "description": "Wide-form emulator table rows for this evaluation."},
                {"file": str(metadata_path), "description": "Processing metadata for this evaluation."},
            ]
        )

    print(f"finished dust post-processing for {evaluation_id}")
    print(dust_draws_path)
    print(lf_long_path)
    print(emulator_table_path)
    print(metadata_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
