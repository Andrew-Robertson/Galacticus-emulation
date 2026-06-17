from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmceeBackendRun:
    path: Path | None
    store_thin: int
    chain_thin: int
    burn_in_discard: int
    n_backend_iterations: int | None

    @property
    def enabled(self) -> bool:
        return self.path is not None


def add_emcee_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--emcee-backend-hdf5",
        default="auto",
        help=(
            "Raw emcee HDFBackend checkpoint path. Use 'auto' for "
            "mcmc-dir/output-prefix_emcee_backend.hdf5, or 'none' to disable. "
            "When enabled, the backend is written thinned by --thin as the chain runs."
        ),
    )
    parser.add_argument(
        "--delete-emcee-backend-on-success",
        action="store_true",
        help="Delete the raw emcee backend after the canonical MCMC outputs and figures are written.",
    )


def resolve_emcee_backend_path(value: str | Path | None, *, mcmc_dir: Path, output_prefix: str) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if text.lower() in {"", "none", "no", "false", "off", "disable", "disabled"}:
        return None
    if text.lower() == "auto":
        return mcmc_dir / f"{output_prefix}_emcee_backend.hdf5"
    return Path(value)


def prepare_emcee_backend(
    emcee_module,
    backend_path: Path | None,
    *,
    n_walkers: int,
    n_dim: int,
):
    if backend_path is None:
        return None
    backend_path.parent.mkdir(parents=True, exist_ok=True)
    backend = emcee_module.backends.HDFBackend(str(backend_path))
    backend.reset(int(n_walkers), int(n_dim))
    return backend


def emcee_backend_run_settings(
    *,
    backend_path: Path | None,
    n_steps: int,
    burn_in: int,
    thin: int,
) -> EmceeBackendRun:
    if thin < 1:
        raise ValueError("--thin must be >= 1")
    if n_steps < 1:
        raise ValueError("--n-steps must be >= 1")
    if burn_in < 0:
        raise ValueError("--burn-in must be >= 0")
    if backend_path is None:
        return EmceeBackendRun(
            path=None,
            store_thin=int(thin),
            chain_thin=int(thin),
            burn_in_discard=int(burn_in),
            n_backend_iterations=None,
        )
    if n_steps % thin != 0:
        raise ValueError(
            f"Thinned emcee backend requires --n-steps to be divisible by --thin; got {n_steps} and {thin}."
        )
    return EmceeBackendRun(
        path=backend_path,
        store_thin=int(thin),
        chain_thin=1,
        burn_in_discard=int(np.ceil(float(burn_in) / float(thin))),
        n_backend_iterations=int(n_steps // thin),
    )


def run_mcmc_with_backend_settings(
    sampler,
    initial_positions,
    *,
    backend_run: EmceeBackendRun,
    n_steps: int,
    progress: bool,
    skip_initial_state_check: bool = True,
) -> None:
    if backend_run.enabled:
        sampler.run_mcmc(
            initial_positions,
            backend_run.n_backend_iterations,
            thin_by=backend_run.store_thin,
            progress=progress,
            skip_initial_state_check=skip_initial_state_check,
        )
    else:
        sampler.run_mcmc(
            initial_positions,
            n_steps,
            progress=progress,
            skip_initial_state_check=skip_initial_state_check,
        )


def original_steps_for_saved_chain(n_saved_steps: int, *, store_thin: int) -> np.ndarray:
    return np.arange(store_thin - 1, store_thin - 1 + store_thin * int(n_saved_steps), store_thin, dtype=np.int64)


def trace_chain_thin(*, requested_trace_thin: int, stored_thin: int, backend_enabled: bool) -> int:
    if requested_trace_thin < 1:
        raise ValueError("trace thinning must be >= 1")
    if not backend_enabled:
        return requested_trace_thin
    if requested_trace_thin < stored_thin:
        return 1
    if requested_trace_thin % stored_thin != 0:
        raise ValueError(
            f"With a thinned emcee backend, --trace-thin must be a multiple of --thin; "
            f"got trace_thin={requested_trace_thin} and thin={stored_thin}."
        )
    return requested_trace_thin // stored_thin
