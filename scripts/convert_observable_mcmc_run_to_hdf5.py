from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from galacticus_emu.mcmc_results import thin_full_chain, split_thinned_chain, write_mcmc_results_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an existing observable-MCMC run directory into the canonical HDF5 results format. "
            "Old full *_chain.npy and *_log_prob.npy files are only deleted if explicitly requested."
        )
    )
    parser.add_argument("mcmc_dir", type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--thin",
        type=int,
        default=None,
        help=(
            "Override the thinning factor used to build the HDF5 burn_in/posterior groups. "
            "The HDF5 run summary is written with this value, as if the run had natively used it."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=["auto", "standard", "sidecar", "combined"],
        default="auto",
        help="Which MCMC script produced the run. Default infers from mcmc_inputs keys.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--delete-source-full-npy",
        action="store_true",
        help="After successful conversion, delete full unthinned *_chain.npy and *_log_prob.npy files.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _infer_kind(inputs: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    if "standard_bundle_path" in inputs and "sidecar_bundle_path" in inputs:
        return "combined"
    if "slow_count" in inputs or "parameter_names" in inputs:
        return "sidecar"
    return "standard"


def _parameter_names(inputs: dict, summary: dict) -> list[str]:
    for key in ("parameter_names", "input_columns"):
        if key in inputs:
            return list(inputs[key])
        if key in summary:
            return list(summary[key])
    best_theta = summary.get("best_theta")
    if isinstance(best_theta, dict):
        return list(best_theta.keys())
    raise ValueError("Could not infer parameter names from mcmc_inputs or run_summary")


def _chain_split(
    mcmc_dir: Path,
    output_prefix: str,
    summary: dict,
    parameter_names: list[str],
) -> tuple[dict, Path | None, Path | None, bool]:
    thin = int(summary["thin"])
    burn_in = int(summary["burn_in"])

    full_chain_path = mcmc_dir / f"{output_prefix}_chain.npy"
    full_log_prob_path = mcmc_dir / f"{output_prefix}_log_prob.npy"
    if full_chain_path.exists() and full_log_prob_path.exists():
        chain = np.load(full_chain_path, mmap_mode="r")
        log_prob = np.load(full_log_prob_path, mmap_mode="r")
        split = thin_full_chain(chain, log_prob, burn_in=burn_in, thin=thin)
        return split, full_chain_path, full_log_prob_path, True

    thinned_chain_path = mcmc_dir / f"{output_prefix}_chain_thinned.npy"
    thinned_log_prob_path = mcmc_dir / f"{output_prefix}_log_prob_thinned.npy"
    if thinned_chain_path.exists() and thinned_log_prob_path.exists():
        chain = np.load(thinned_chain_path, mmap_mode="r")
        log_prob = np.load(thinned_log_prob_path, mmap_mode="r")
        split = split_thinned_chain(chain, log_prob, burn_in=burn_in, thin=thin)
        return split, thinned_chain_path, thinned_log_prob_path, False

    posterior_path = mcmc_dir / f"{output_prefix}_posterior_samples.csv"
    posterior = pd.read_csv(posterior_path)
    missing = [name for name in parameter_names if name not in posterior.columns]
    if missing:
        raise ValueError(f"{posterior_path} is missing parameter column(s): {missing}")
    samples = posterior[parameter_names].to_numpy(dtype=float)
    log_probability = posterior["log_probability"].to_numpy(dtype=float)
    return (
        {
            "burn_in_chain": np.empty((0, 1, len(parameter_names)), dtype=float),
            "burn_in_log_probability": np.empty((0, 1), dtype=float),
            "burn_in_original_step": np.empty((0,), dtype=np.int64),
            "posterior_chain": samples[:, None, :],
            "posterior_log_probability": log_probability[:, None],
            "posterior_original_step": np.arange(samples.shape[0], dtype=np.int64),
        },
        posterior_path,
        None,
        False,
    )


def _update_summary_map_from_source(
    summary: dict,
    parameter_names: list[str],
    split: dict,
    *,
    source_chain_path: Path | None,
    source_log_prob_path: Path | None,
    source_is_full: bool,
) -> dict:
    updated = dict(summary)
    if source_is_full and source_chain_path is not None and source_log_prob_path is not None:
        burn_in = int(updated["burn_in"])
        chain = np.load(source_chain_path, mmap_mode="r")
        log_probability = np.load(source_log_prob_path, mmap_mode="r")
        post_log_probability = np.asarray(log_probability[burn_in:])
        flat_index = int(np.nanargmax(post_log_probability))
        step_after_burn, walker = np.unravel_index(flat_index, post_log_probability.shape)
        best_theta = np.asarray(chain[burn_in + step_after_burn, walker, :], dtype=float)
        best_log_probability = float(log_probability[burn_in + step_after_burn, walker])
        best_index = [int(step_after_burn), int(walker)]
    else:
        posterior_chain = np.asarray(split["posterior_chain"])
        posterior_log_probability = np.asarray(split["posterior_log_probability"])
        if posterior_chain.size == 0:
            return updated
        flat_index = int(np.nanargmax(posterior_log_probability))
        step_index, walker = np.unravel_index(flat_index, posterior_log_probability.shape)
        best_theta = np.asarray(posterior_chain[step_index, walker, :], dtype=float)
        best_log_probability = float(posterior_log_probability[step_index, walker])
        best_index = [int(step_index), int(walker)]

    updated["best_theta"] = {
        name: float(value) for name, value in zip(parameter_names, best_theta, strict=True)
    }
    updated["best_log_probability"] = best_log_probability
    updated["best_chain_index_after_burn_in"] = best_index
    updated.pop("maximum_a_posteriori_model_changes", None)
    return updated


def _prediction_paths(mcmc_dir: Path, output_prefix: str, kind: str) -> dict[str, Path]:
    if kind == "combined":
        return {
            "best_fit_standard_observables": mcmc_dir / f"{output_prefix}_best_fit_standard_observables.csv",
            "best_fit_sidecar_lfs": mcmc_dir / f"{output_prefix}_best_fit_sidecar_lfs.csv",
        }
    return {"best_fit_observables": mcmc_dir / f"{output_prefix}_best_fit_observables.csv"}


def _delete_full_npy_files(mcmc_dir: Path, output_prefix: str) -> list[Path]:
    deleted = []
    for suffix in ("chain", "log_prob", "sampler_chain", "sampler_log_prob"):
        path = mcmc_dir / f"{output_prefix}_{suffix}.npy"
        if path.exists():
            path.unlink()
            deleted.append(path)
    return deleted


def main() -> None:
    args = parse_args()
    mcmc_dir = args.mcmc_dir.expanduser().resolve()
    summary_path = mcmc_dir / f"{args.output_prefix}_run_summary.json"
    inputs_path = mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib"
    posterior_path = mcmc_dir / f"{args.output_prefix}_posterior_samples.csv"
    output_path = args.output_path or mcmc_dir / f"{args.output_prefix}_mcmc_results.hdf5"
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    summary = _load_json(summary_path)
    if args.thin is not None:
        if args.thin < 1:
            raise ValueError("--thin must be >= 1")
        summary = dict(summary)
        summary["thin"] = int(args.thin)
        summary.pop("trace_thin", None)
        summary.pop("trace_autocorrelation_time", None)
        summary.pop("trace_autocorrelation_time_burn_in_thinned", None)
    inputs = joblib.load(inputs_path)
    kind = _infer_kind(inputs, args.kind)
    parameter_names = _parameter_names(inputs, summary)
    parameter_specs = list(inputs.get("parameter_specs", [])) or None
    split, source_chain_path, source_log_prob_path, source_is_full = _chain_split(
        mcmc_dir,
        args.output_prefix,
        summary,
        parameter_names,
    )
    summary = _update_summary_map_from_source(
        summary,
        parameter_names,
        split,
        source_chain_path=source_chain_path,
        source_log_prob_path=source_log_prob_path,
        source_is_full=source_is_full,
    )

    map_changes_path = summary.get("maximum_a_posteriori_model_changes")
    map_changes = Path(map_changes_path) if map_changes_path else mcmc_dir / "maximum_a_posteriori_model_changes.xml"
    if not map_changes.is_absolute():
        map_changes = mcmc_dir / map_changes

    write_mcmc_results_hdf5(
        output_path,
        parameter_names=parameter_names,
        parameter_specs=parameter_specs,
        summary=summary,
        map_changes_path=map_changes if map_changes.exists() else None,
        prediction_paths=_prediction_paths(mcmc_dir, args.output_prefix, kind),
        posterior_samples_path=None if args.thin is not None else posterior_path if posterior_path.exists() else None,
        mcmc_inputs_path=inputs_path,
        source_chain_path=None if args.thin is not None else source_chain_path,
        source_log_prob_path=None if args.thin is not None else source_log_prob_path,
        notes=None,
        **split,
    )
    print(output_path)

    if args.delete_source_full_npy:
        if not source_is_full:
            print("No full unthinned source chain was used; leaving npy files untouched.")
            return
        deleted = _delete_full_npy_files(mcmc_dir, args.output_prefix)
        for path in deleted:
            print(f"deleted {path}")


if __name__ == "__main__":
    main()
