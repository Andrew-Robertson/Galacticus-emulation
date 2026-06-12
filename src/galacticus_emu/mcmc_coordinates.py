from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from .lhs import ParameterSpec, TruncatedLogNormalPrior


def add_sampler_coordinate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sample-transformed-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run emcee in sampler-friendly transformed coordinates. Currently this samples log(value) "
            "for parameters with TruncatedLogNormalPrior and physical value for other parameters. "
            "Saved posterior samples and best-fit outputs remain in physical coordinates."
        ),
    )


def _log_coordinate_mask(parameter_specs: Sequence[ParameterSpec], enabled: bool) -> np.ndarray:
    return np.asarray(
        [enabled and isinstance(spec.prior, TruncatedLogNormalPrior) for spec in parameter_specs],
        dtype=bool,
    )


def sampler_coordinate_names(
    parameter_specs: Sequence[ParameterSpec],
    parameter_names: Sequence[str],
    *,
    enabled: bool,
) -> list[str]:
    mask = _log_coordinate_mask(parameter_specs, enabled)
    return [f"log({name})" if use_log else str(name) for name, use_log in zip(parameter_names, mask, strict=True)]


def sampler_coordinate_summary(
    parameter_specs: Sequence[ParameterSpec],
    parameter_names: Sequence[str],
    *,
    enabled: bool,
) -> dict:
    mask = _log_coordinate_mask(parameter_specs, enabled)
    return {
        "enabled": bool(enabled),
        "log_parameters": [str(name) for name, use_log in zip(parameter_names, mask, strict=True) if use_log],
        "coordinate_names": sampler_coordinate_names(parameter_specs, parameter_names, enabled=enabled),
    }


def physical_to_sampler_coordinates(
    parameter_specs: Sequence[ParameterSpec],
    physical: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    values = np.asarray(physical, dtype=float).copy()
    mask = _log_coordinate_mask(parameter_specs, enabled)
    if np.any(mask):
        values[..., mask] = np.log(values[..., mask])
    return values


def sampler_to_physical_coordinates(
    parameter_specs: Sequence[ParameterSpec],
    sampler_values: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    values = np.asarray(sampler_values, dtype=float).copy()
    mask = _log_coordinate_mask(parameter_specs, enabled)
    if np.any(mask):
        values[..., mask] = np.exp(values[..., mask])
    return values


def log_abs_det_physical_wrt_sampler(
    parameter_specs: Sequence[ParameterSpec],
    physical: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    values = np.asarray(physical, dtype=float)
    mask = _log_coordinate_mask(parameter_specs, enabled)
    if not np.any(mask):
        return np.zeros(values.shape[:-1], dtype=float)
    return np.sum(np.log(values[..., mask]), axis=-1)


def physical_log_prob_from_sampler_log_prob(
    parameter_specs: Sequence[ParameterSpec],
    sampler_values: np.ndarray,
    sampler_log_prob: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    values = np.asarray(sampler_values, dtype=float)
    result = np.asarray(sampler_log_prob, dtype=float)
    mask = _log_coordinate_mask(parameter_specs, enabled)
    if not np.any(mask):
        return result
    return result - np.sum(values[..., mask], axis=-1)


def best_physical_sample_from_sampler_history(
    parameter_specs: Sequence[ParameterSpec],
    sampler_values: np.ndarray,
    sampler_log_prob: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, float, tuple[int, ...]]:
    physical_log_prob = physical_log_prob_from_sampler_log_prob(
        parameter_specs,
        sampler_values,
        sampler_log_prob,
        enabled=enabled,
    )
    flat_index = int(np.nanargmax(physical_log_prob))
    multi_index = np.unravel_index(flat_index, physical_log_prob.shape)
    sampler_theta = np.asarray(sampler_values[multi_index], dtype=float)
    physical_theta = sampler_to_physical_coordinates(
        parameter_specs,
        sampler_theta[None, :],
        enabled=enabled,
    )[0]
    return physical_theta, float(physical_log_prob[multi_index]), tuple(int(value) for value in multi_index)


def best_physical_sample_from_sampler(
    parameter_specs: Sequence[ParameterSpec],
    sampler,
    *,
    burn_in: int,
    enabled: bool,
    chunk_steps: int = 2048,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    start = max(int(burn_in), 0)
    if hasattr(sampler, "backend") and hasattr(sampler.backend, "chain") and hasattr(sampler.backend, "log_prob"):
        sampler_values = sampler.backend.chain
        sampler_log_prob = sampler.backend.log_prob
    else:
        sampler_values = sampler.get_chain()
        sampler_log_prob = sampler.get_log_prob()

    if sampler_values.ndim != 3:
        raise ValueError("sampler chain must have shape (n_steps, n_walkers, n_parameters)")
    if sampler_log_prob.shape != sampler_values.shape[:2]:
        raise ValueError(
            f"sampler log_prob shape {sampler_log_prob.shape} does not match chain shape {sampler_values.shape[:2]}"
        )
    if start >= sampler_values.shape[0]:
        raise ValueError(f"burn_in={burn_in} leaves no post-burn-in samples")

    mask = _log_coordinate_mask(parameter_specs, enabled)
    best_log_probability = -np.inf
    best_step = -1
    best_walker = -1
    chunk_steps = max(int(chunk_steps), 1)

    for chunk_start in range(start, sampler_values.shape[0], chunk_steps):
        chunk_stop = min(chunk_start + chunk_steps, sampler_values.shape[0])
        log_probability = np.asarray(sampler_log_prob[chunk_start:chunk_stop], dtype=float)
        if np.any(mask):
            log_probability = log_probability - np.sum(sampler_values[chunk_start:chunk_stop, :, mask], axis=-1)
        if not np.any(np.isfinite(log_probability)):
            continue
        flat_index = int(np.nanargmax(log_probability))
        local_step, walker = np.unravel_index(flat_index, log_probability.shape)
        value = float(log_probability[local_step, walker])
        if value > best_log_probability:
            best_log_probability = value
            best_step = chunk_start + int(local_step)
            best_walker = int(walker)

    if best_step < 0:
        raise ValueError("No finite post-burn-in log-probability values found")
    sampler_theta = np.asarray(sampler_values[best_step, best_walker, :], dtype=float)
    physical_theta = sampler_to_physical_coordinates(
        parameter_specs,
        sampler_theta[None, :],
        enabled=enabled,
    )[0]
    return physical_theta, best_log_probability, (best_step - start, best_walker)


class SamplerCoordinatePosterior:
    def __init__(self, posterior, parameter_specs: Sequence[ParameterSpec], *, enabled: bool):
        self.posterior = posterior
        self.parameter_specs = list(parameter_specs)
        self.enabled = bool(enabled)

    def log_probability(self, theta: np.ndarray) -> float:
        return float(self.log_probability_batch(np.asarray(theta, dtype=float)[None, :])[0])

    def log_probability_batch(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        physical = sampler_to_physical_coordinates(self.parameter_specs, theta, enabled=self.enabled)
        log_prob = np.asarray(self.posterior.log_probability_batch(physical), dtype=float)
        finite = np.isfinite(log_prob)
        if np.any(finite):
            log_prob[finite] = log_prob[finite] + log_abs_det_physical_wrt_sampler(
                self.parameter_specs,
                physical[finite],
                enabled=self.enabled,
            )
        return log_prob
