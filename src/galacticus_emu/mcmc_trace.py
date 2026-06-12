from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .lhs import ParameterSpec, TruncatedLogNormalPrior

try:
    import emcee
except ModuleNotFoundError:  # pragma: no cover
    emcee = None


def _plot_values_and_labels(
    chain: np.ndarray,
    labels: Sequence[str],
    parameter_specs: Sequence[ParameterSpec] | None,
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(chain, dtype=float).copy()
    plot_labels = [str(label) for label in labels]
    if parameter_specs is None:
        return values, plot_labels

    for index, spec in enumerate(parameter_specs):
        if not isinstance(spec.prior, TruncatedLogNormalPrior):
            continue
        if np.any(values[:, :, index] <= 0.0):
            raise ValueError(f"Cannot log-transform non-positive samples for parameter '{plot_labels[index]}'")
        values[:, :, index] = np.log10(values[:, :, index])
        plot_labels[index] = f"log10({plot_labels[index]})"
    return values, plot_labels


def trace_autocorrelation_times(chain: np.ndarray, burn_in: int = 0) -> np.ndarray:
    values = np.asarray(chain, dtype=float)
    if values.ndim != 3:
        raise ValueError("chain must have shape (n_steps, n_walkers, n_parameters)")
    if emcee is None:
        return np.full(values.shape[2], np.nan, dtype=float)
    start = min(max(int(burn_in), 0), values.shape[0])
    post_burn = values[start:]
    if post_burn.shape[0] < 2:
        return np.full(values.shape[2], np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(emcee.autocorr.integrated_time(post_burn, quiet=True, tol=0), dtype=float)


def make_trace_plot(
    chain: np.ndarray,
    labels: Sequence[str],
    path: Path,
    *,
    parameter_specs: Sequence[ParameterSpec] | None = None,
    burn_in: int = 0,
    row_height: float = 1.65,
    color: str = "tab:blue",
    alpha: float = 0.22,
    linewidth: float = 0.5,
) -> dict[str, float]:
    values, plot_labels = _plot_values_and_labels(chain, labels, parameter_specs)
    n_steps, _, n_dim = values.shape
    tau = trace_autocorrelation_times(values, burn_in=burn_in)

    fig, axes = plt.subplots(n_dim, 1, figsize=(9.0, row_height * n_dim), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    burn_stop = min(max(int(burn_in), 0), n_steps)
    for axis, label, dim_index in zip(axes, plot_labels, range(n_dim), strict=True):
        axis.plot(values[:, :, dim_index], color=color, alpha=alpha, lw=linewidth)
        if burn_stop > 0:
            axis.axvspan(0, burn_stop, color="0.5", alpha=0.18, lw=0, zorder=0)
        axis.set_ylabel(label)
        tau_value = tau[dim_index]
        tau_text = "tau = n/a" if not np.isfinite(tau_value) else f"tau = {tau_value:.0f}"
        axis.text(
            0.985,
            0.92,
            tau_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.78, "boxstyle": "round,pad=0.18"},
        )
    axes[-1].set_xlabel("Step")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {str(label): float(value) for label, value in zip(plot_labels, tau, strict=True)}
