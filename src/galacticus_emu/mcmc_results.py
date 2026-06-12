from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .lhs import ParameterSpec


FORMAT_NAME = "galacticus_emu_mcmc_results"
FORMAT_VERSION = "1.0"
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
PRIOR_FIELDS = ("lower", "upper", "mean", "sigma", "x0")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=_json_default)


def _string_array(values: list[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def _write_string_dataset(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=str(value), dtype=STRING_DTYPE)


def _create_numeric_dataset(group: h5py.Group, name: str, values: np.ndarray) -> h5py.Dataset:
    array = np.asarray(values)
    if array.size == 0:
        return group.create_dataset(name, data=array)
    return group.create_dataset(name, data=array, compression="gzip", shuffle=True)


def split_thinned_chain(
    chain: np.ndarray,
    log_probability: np.ndarray,
    *,
    burn_in: int,
    thin: int,
) -> dict[str, np.ndarray]:
    if thin < 1:
        raise ValueError("thin must be >= 1")
    values = np.asarray(chain)
    logp = np.asarray(log_probability)
    if values.ndim != 3:
        raise ValueError("chain must have shape (n_saved_steps, n_walkers, n_parameters)")
    if logp.shape != values.shape[:2]:
        raise ValueError(f"log_probability shape {logp.shape} does not match chain shape {values.shape[:2]}")

    original_steps = np.arange(thin - 1, thin - 1 + thin * values.shape[0], thin, dtype=np.int64)
    burn_mask = original_steps < int(burn_in)
    posterior_mask = np.logical_not(burn_mask)
    return {
        "burn_in_chain": values[burn_mask],
        "burn_in_log_probability": logp[burn_mask],
        "burn_in_original_step": original_steps[burn_mask],
        "posterior_chain": values[posterior_mask],
        "posterior_log_probability": logp[posterior_mask],
        "posterior_original_step": original_steps[posterior_mask],
    }


def thin_full_chain(
    chain: np.ndarray,
    log_probability: np.ndarray,
    *,
    burn_in: int,
    thin: int,
) -> dict[str, np.ndarray]:
    if thin < 1:
        raise ValueError("thin must be >= 1")
    values = np.asarray(chain)
    logp = np.asarray(log_probability)
    if values.ndim != 3:
        raise ValueError("chain must have shape (n_steps, n_walkers, n_parameters)")
    if logp.shape != values.shape[:2]:
        raise ValueError(f"log_probability shape {logp.shape} does not match chain shape {values.shape[:2]}")

    n_steps = values.shape[0]
    burn_steps = np.arange(thin - 1, min(int(burn_in), n_steps), thin, dtype=np.int64)
    posterior_steps = np.arange(int(burn_in) + thin - 1, n_steps, thin, dtype=np.int64)
    return {
        "burn_in_chain": values[burn_steps],
        "burn_in_log_probability": logp[burn_steps],
        "burn_in_original_step": burn_steps,
        "posterior_chain": values[posterior_steps],
        "posterior_log_probability": logp[posterior_steps],
        "posterior_original_step": posterior_steps,
    }


def _write_parameters(group: h5py.Group, parameter_names: list[str], parameter_specs: list[ParameterSpec] | None) -> None:
    group.create_dataset("names", data=_string_array(parameter_names), dtype=STRING_DTYPE)
    if parameter_specs is None:
        return
    group.create_dataset("short_names", data=_string_array([spec.short_name for spec in parameter_specs]), dtype=STRING_DTYPE)
    group.create_dataset("paths", data=_string_array([spec.path for spec in parameter_specs]), dtype=STRING_DTYPE)
    group.create_dataset(
        "prior_type",
        data=_string_array([type(spec.prior).__name__ for spec in parameter_specs]),
        dtype=STRING_DTYPE,
    )
    for field in PRIOR_FIELDS:
        values = []
        for spec in parameter_specs:
            values.append(float(getattr(spec.prior, field, np.nan)))
        group.create_dataset(f"prior_{field}", data=np.asarray(values, dtype=float))


def _write_chain_group(
    group: h5py.Group,
    *,
    chain: np.ndarray,
    log_probability: np.ndarray,
    original_step: np.ndarray,
) -> None:
    _create_numeric_dataset(group, "chain", np.asarray(chain, dtype=float))
    _create_numeric_dataset(group, "log_probability", np.asarray(log_probability, dtype=float))
    group.create_dataset("original_step", data=np.asarray(original_step, dtype=np.int64))
    if np.asarray(chain).size == 0:
        n_parameters = np.asarray(chain).shape[-1] if np.asarray(chain).ndim == 3 else 0
        group.create_dataset("samples_flat", data=np.empty((0, n_parameters), dtype=float))
        group.create_dataset("log_probability_flat", data=np.empty((0,), dtype=float))
        return
    group.create_dataset(
        "samples_flat",
        data=np.asarray(chain, dtype=float).reshape(-1, np.asarray(chain).shape[-1]),
        compression="gzip",
        shuffle=True,
    )
    group.create_dataset(
        "log_probability_flat",
        data=np.asarray(log_probability, dtype=float).reshape(-1),
        compression="gzip",
        shuffle=True,
    )


def write_mcmc_results_hdf5(
    output_path: Path,
    *,
    parameter_names: list[str],
    parameter_specs: list[ParameterSpec] | None,
    burn_in_chain: np.ndarray,
    burn_in_log_probability: np.ndarray,
    burn_in_original_step: np.ndarray,
    posterior_chain: np.ndarray,
    posterior_log_probability: np.ndarray,
    posterior_original_step: np.ndarray,
    summary: dict[str, Any],
    map_changes_path: Path | None = None,
    prediction_paths: dict[str, Path] | None = None,
    posterior_samples_path: Path | None = None,
    mcmc_inputs_path: Path | None = None,
    source_chain_path: Path | None = None,
    source_log_prob_path: Path | None = None,
    notes: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_paths = prediction_paths or {}

    with h5py.File(output_path, "w") as handle:
        handle.attrs["format_name"] = FORMAT_NAME
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["n_parameters"] = len(parameter_names)

        metadata = handle.create_group("metadata")
        _write_string_dataset(metadata, "run_summary_json", _json_dumps(summary))
        if notes:
            _write_string_dataset(metadata, "notes", notes)
        if posterior_samples_path is not None:
            metadata.attrs["posterior_samples_csv"] = str(posterior_samples_path)
        if mcmc_inputs_path is not None:
            metadata.attrs["mcmc_inputs_joblib"] = str(mcmc_inputs_path)
        if source_chain_path is not None:
            metadata.attrs["source_chain"] = str(source_chain_path)
        if source_log_prob_path is not None:
            metadata.attrs["source_log_probability"] = str(source_log_prob_path)

        _write_parameters(handle.create_group("parameters"), parameter_names, parameter_specs)
        _write_chain_group(
            handle.create_group("burn_in"),
            chain=burn_in_chain,
            log_probability=burn_in_log_probability,
            original_step=burn_in_original_step,
        )
        _write_chain_group(
            handle.create_group("posterior"),
            chain=posterior_chain,
            log_probability=posterior_log_probability,
            original_step=posterior_original_step,
        )

        map_group = handle.create_group("map")
        best_theta = summary.get("best_theta", {})
        if isinstance(best_theta, dict):
            map_group.create_dataset(
                "theta",
                data=np.asarray([best_theta.get(name, np.nan) for name in parameter_names], dtype=float),
            )
        best_quantiles = summary.get("best_quantiles", {})
        if isinstance(best_quantiles, dict):
            map_group.create_dataset(
                "quantile",
                data=np.asarray([best_quantiles.get(name, np.nan) for name in parameter_names], dtype=float),
            )
        if "best_log_probability" in summary:
            map_group.attrs["log_probability"] = float(summary["best_log_probability"])
        if map_changes_path is not None:
            map_group.attrs["model_changes_xml_path"] = str(map_changes_path)
            path = Path(map_changes_path)
            if path.exists():
                _write_string_dataset(map_group, "model_changes_xml", path.read_text())

        predictions_group = handle.create_group("predictions")
        for label, path in prediction_paths.items():
            path = Path(path)
            group = predictions_group.create_group(str(label))
            group.attrs["source_path"] = str(path)
            if path.exists():
                _write_string_dataset(group, "csv", path.read_text())

    return output_path


def load_posterior_frame(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in {".hdf5", ".h5"}:
        with h5py.File(path, "r") as handle:
            samples = handle["posterior/samples_flat"][...]
            names = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in handle["parameters/names"][...]
            ]
            frame = pd.DataFrame(samples, columns=names)
            if "posterior/log_probability_flat" in handle:
                frame["log_probability"] = handle["posterior/log_probability_flat"][...]
            return frame
    return pd.read_csv(path)


def load_run_summary(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in {".hdf5", ".h5"}:
        with h5py.File(path, "r") as handle:
            return json.loads(handle["metadata/run_summary_json"][()])
    return json.loads(path.read_text())
