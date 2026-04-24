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

from calculate_halpha_dust_lf_grid import (  # type: ignore
    SOBRAL_CASES,
    _attenuated_halpha,
    _combined_node_weights,
    _intrinsic_halpha,
    _lf_from_expected_scatter,
    _lf_from_luminosities,
    _sobral_bin_edges,
)


DEFAULT_DUST_PRIORS = {
    "delta_0": {"distribution": "normal", "mean": 0.0, "sigma": 1.0},
    "delta_z": {"distribution": "normal", "mean": 0.0, "sigma": 1.0},
    "delta_M": {"distribution": "normal", "mean": 0.0, "sigma": 1.0},
    "delta_Mz": {"distribution": "normal", "mean": 0.0, "sigma": 1.0},
    "attenuation_scatter": {"distribution": "truncated_normal", "mean": 0.25, "sigma": 0.1, "lower": 0.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a completed Galacticus campaign to compute dust-attenuated "
            "Halpha luminosity functions for many dust-model draws per evaluation."
        )
    )
    parser.add_argument("campaign_root", type=Path, help="Path to a completed campaign directory.")
    parser.add_argument(
        "--hdf5-filename",
        default="galacticus.hdf5",
        help="Name of the Galacticus output inside each evaluation directory.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="halpha_dust_campaign",
        help="Directory name inside campaign_root where compact post-processed outputs will be written.",
    )
    parser.add_argument(
        "--n-dust-draws",
        type=int,
        default=32,
        help="Number of independent dust parameter draws to evaluate for each Galacticus run.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=12345,
        help="Base seed used to generate deterministic but different dust draws for each evaluation.",
    )
    parser.add_argument(
        "--scatter-mode",
        choices=["draw", "expected"],
        default="expected",
        help="How to treat attenuation_scatter for the LF calculation.",
    )
    parser.add_argument(
        "--z-pivot",
        type=float,
        default=1.0,
        help="Pivot redshift used by the gb10_generalised dust model.",
    )
    parser.add_argument(
        "--random-uniform-index",
        type=int,
        default=None,
        help=(
            "Optional randomUniform column to use when --scatter-mode=draw and the Galacticus file "
            "contains nodeData/randomUniform."
        ),
    )
    parser.add_argument(
        "--dust-prior-json",
        action="append",
        default=[],
        help=(
            "Override one dust-parameter prior using label='{\"distribution\":\"uniform\",\"lower\":...,\"upper\":...}' "
            "or label='{\"distribution\":\"normal\",\"mean\":...,\"sigma\":...}' "
            "or label='{\"distribution\":\"truncated_normal\",\"mean\":...,\"sigma\":...,\"lower\":...}'. "
            "Supported labels: delta_0, delta_z, delta_M, delta_Mz, attenuation_scatter."
        ),
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Optional limit on the number of evaluation directories to process.",
    )
    return parser.parse_args()


def _parse_labelled_spec(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Expected 'label=value' format, got: {text}")
    label, value = text.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise ValueError(f"Expected non-empty label and value, got: {text}")
    return label, value


def _load_dust_priors(args: argparse.Namespace) -> dict[str, dict[str, float | str]]:
    priors = dict(DEFAULT_DUST_PRIORS)
    for spec in args.dust_prior_json:
        label, json_text = _parse_labelled_spec(spec)
        if label not in priors:
            raise ValueError(f"Unsupported dust parameter prior override: {label}")
        priors[label] = json.loads(json_text)
    return priors


def _sample_prior(rng: np.random.Generator, spec: dict[str, float | str]) -> float:
    distribution = str(spec["distribution"]).lower()
    if distribution == "uniform":
        return float(rng.uniform(float(spec["lower"]), float(spec["upper"])))
    if distribution == "normal":
        return float(rng.normal(float(spec["mean"]), float(spec["sigma"])))
    if distribution == "truncated_normal":
        lower = float(spec.get("lower", -np.inf))
        upper = float(spec.get("upper", np.inf))
        mean = float(spec["mean"])
        sigma = float(spec["sigma"])
        if sigma <= 0.0:
            raise ValueError("truncated_normal requires sigma > 0")
        for _ in range(10_000):
            value = float(rng.normal(mean, sigma))
            if lower <= value <= upper:
                return value
        raise RuntimeError("Failed to sample from truncated_normal prior after many attempts")
    raise ValueError(f"Unsupported prior distribution: {distribution}")


def _sample_dust_params(
    priors: dict[str, dict[str, float | str]],
    *,
    z_pivot: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    params = {name: _sample_prior(rng, spec) for name, spec in priors.items()}
    params["z_pivot"] = float(z_pivot)
    return params


def _iter_evaluation_dirs(campaign_root: Path) -> list[Path]:
    evaluations_root = campaign_root / "evaluations"
    return sorted(path for path in evaluations_root.iterdir() if path.is_dir())


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
                lf = _lf_from_expected_scatter(
                    intrinsic_luminosity,
                    stellar_mass,
                    weights,
                    redshift,
                    dust_case["dust_params"],
                    log10_edges,
                )
            else:
                luminosity = _attenuated_halpha(output_group, redshift, dust_case)
                lf = _lf_from_luminosities(luminosity, weights, log10_edges)

            target_lf = np.asarray(analysis_group["luminosityFunctionTarget"][...], dtype=float)
            target_cov = np.asarray(analysis_group["luminosityFunctionCovarianceTarget"][...], dtype=float)
            target_std = np.sqrt(np.clip(np.diag(target_cov), 0.0, None))
            for bin_index, (center, value, target, target_sigma) in enumerate(
                zip(log10_centers, lf, target_lf, target_std, strict=True)
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
                        "target_dn_dlnL_mpc3": float(target),
                        "target_std_dn_dlnL_mpc3": float(target_sigma),
                    }
                )
    return output_rows


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    output_root = campaign_root / args.output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(campaign_root / "samples.csv")
    input_columns = [column for column in samples.columns if column != "evaluation_id"]
    priors = _load_dust_priors(args)

    evaluation_dirs = _iter_evaluation_dirs(campaign_root)
    if args.max_evals is not None:
        evaluation_dirs = evaluation_dirs[: args.max_evals]

    dust_draw_rows: list[dict[str, Any]] = []
    lf_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    processed = 0
    for evaluation_index, evaluation_dir in enumerate(evaluation_dirs):
        evaluation_id = evaluation_dir.name
        galacticus_file = evaluation_dir / args.hdf5_filename
        if not galacticus_file.exists():
            continue

        sample_match = samples.loc[samples["evaluation_id"] == evaluation_id]
        if sample_match.empty:
            continue
        sample_row = sample_match.iloc[0].to_dict()

        try:
            with h5py.File(galacticus_file, "r"):
                pass
        except OSError:
            continue

        eval_rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, evaluation_index]))
        print(f"[{processed + 1}/{len(evaluation_dirs)} evals] starting {evaluation_id}")
        for dust_draw_index in range(args.n_dust_draws):
            print(
                f"[{processed + 1}/{len(evaluation_dirs)} evals] "
                f"{evaluation_id} dust draw {dust_draw_index + 1}/{args.n_dust_draws}"
            )
            dust_params = _sample_dust_params(
                priors,
                z_pivot=args.z_pivot,
                rng=eval_rng,
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

            draw_lf_rows = _compute_case_lfs(galacticus_file, dust_case)
            wide_row = {
                "evaluation_id": evaluation_id,
                "dust_draw_index": int(dust_draw_index),
                "dust_case": dust_case["label"],
                "scatter_mode": args.scatter_mode,
                **{name: sample_row[name] for name in input_columns},
                **dust_params,
            }
            for row in draw_lf_rows:
                merged_row = dict(draw_metadata)
                merged_row.update(row)
                lf_rows.append(merged_row)
                output_column = f"halpha_sobral_{row['sobral_label'].lower()}_bin{row['bin_index']}"
                target_column = f"{output_column}_target"
                target_std_column = f"{output_column}_target_std"
                wide_row[output_column] = row["dn_dlnL_mpc3"]
                wide_row[target_column] = row["target_dn_dlnL_mpc3"]
                wide_row[target_std_column] = row["target_std_dn_dlnL_mpc3"]
            table_rows.append(wide_row)

        processed += 1
        print(f"[{processed}/{len(evaluation_dirs)} evals] finished {evaluation_id}")

    if not table_rows:
        raise FileNotFoundError(f"No readable {args.hdf5_filename} files found under {campaign_root / 'evaluations'}")

    dust_draws_path = output_root / "dust_draws.csv"
    pd.DataFrame(dust_draw_rows).to_csv(dust_draws_path, index=False)

    lf_long_path = output_root / "halpha_dust_lf_long.csv"
    pd.DataFrame(lf_rows).to_csv(lf_long_path, index=False)

    emulator_table_path = output_root / "halpha_dust_lf_emulator_table.csv"
    pd.DataFrame(table_rows).to_csv(emulator_table_path, index=False)

    metadata = {
        "campaign_root": str(campaign_root),
        "hdf5_filename": args.hdf5_filename,
        "output_dir": str(output_root),
        "n_evaluations_processed": processed,
        "n_dust_draws_per_evaluation": args.n_dust_draws,
        "scatter_mode": args.scatter_mode,
        "z_pivot": args.z_pivot,
        "random_uniform_index": args.random_uniform_index,
        "dust_priors": priors,
        "sobral_cases": [
            {
                "output_name": output_name,
                "redshift": redshift,
                "label": label,
                "analysis_name": analysis_name,
            }
            for output_name, redshift, label, analysis_name in SOBRAL_CASES
        ],
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    summary_rows = [
        {
            "file": str(dust_draws_path),
            "description": "Per-evaluation dust parameter draws used for post-processing.",
        },
        {
            "file": str(lf_long_path),
            "description": "Long-form Halpha luminosity function values for every evaluation, dust draw, redshift, and Sobral bin.",
        },
        {
            "file": str(emulator_table_path),
            "description": "Wide-form emulator table with one row per evaluation/dust draw.",
        },
        {
            "file": str(metadata_path),
            "description": "Processing metadata and dust-prior definitions.",
        },
    ]
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "description"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(dust_draws_path)
    print(lf_long_path)
    print(emulator_table_path)
    print(metadata_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
