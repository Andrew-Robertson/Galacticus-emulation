from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu.mcmc_moves import build_emcee_moves, describe_emcee_moves


MOVE_PRESETS = {
    "stretch": ["StretchMove:1.0"],
    "de": ["DEMove:1.0"],
    "de_snooker": ["DEMove:0.9", "DESnookerMove:0.1"],
    "de_heavy": ["StretchMove:0.3", "DEMove:0.6", "DESnookerMove:0.1"],
    "de_rescue": ["StretchMove:0.65", "DEMove:0.25", "DEMove:0.05,gamma0=1.0", "DESnookerMove:0.05"],
    "stretch_de": ["StretchMove:0.7", "DEMove:0.3"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Playground benchmark for emcee move mixtures on analytic toy posteriors."
    )
    parser.add_argument("--ndim", type=int, default=25)
    parser.add_argument("--n-walkers", type=int, action="append", default=None)
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "playing" / "emcee_move_playground")
    parser.add_argument(
        "--target",
        action="append",
        choices=["gaussian", "correlated_gaussian", "two_gaussians", "funnel"],
        default=None,
        help="Target(s) to run. Repeat to choose several. Default: all.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(MOVE_PRESETS),
        default=None,
        help="Move preset(s) to run. Repeat to choose several. Default: all.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _logsumexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    vmax = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(vmax, axis=axis) + np.log(np.sum(np.exp(values - vmax), axis=axis))


def _log_prob_gaussian(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(theta)
    return -0.5 * np.sum(theta**2, axis=1)


def _log_prob_correlated_gaussian(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(theta)
    ndim = theta.shape[1]
    scales = np.geomspace(0.2, 5.0, ndim)
    y = theta / scales[None, :]
    return -0.5 * np.sum(y**2, axis=1)


def _log_prob_two_gaussians(theta: np.ndarray, separation: float = 5.0) -> np.ndarray:
    theta = np.atleast_2d(theta)
    ndim = theta.shape[1]
    offset = np.zeros(ndim)
    offset[0] = separation
    left = -0.5 * np.sum((theta + offset[None, :]) ** 2, axis=1)
    right = -0.5 * np.sum((theta - offset[None, :]) ** 2, axis=1)
    return _logsumexp(np.vstack([left, right]), axis=0) - np.log(2.0)


def _log_prob_funnel(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(theta)
    y = theta[:, 0]
    x = theta[:, 1:]
    sigma2 = np.exp(y)[:, None]
    return -0.5 * (y / 3.0) ** 2 - 0.5 * np.sum(x**2 / sigma2 + np.log(sigma2), axis=1)


def _target(name: str):
    if name == "gaussian":
        return _log_prob_gaussian
    if name == "correlated_gaussian":
        return _log_prob_correlated_gaussian
    if name == "two_gaussians":
        return _log_prob_two_gaussians
    if name == "funnel":
        return _log_prob_funnel
    raise ValueError(f"Unknown target {name!r}")


def _initial_positions(name: str, rng: np.random.Generator, n_walkers: int, ndim: int) -> np.ndarray:
    if name == "two_gaussians":
        positions = rng.normal(scale=0.2, size=(n_walkers, ndim))
        positions[: n_walkers // 2, 0] -= 5.0
        positions[n_walkers // 2 :, 0] += 5.0
        return positions
    return rng.normal(scale=0.2, size=(n_walkers, ndim))


def _autocorr_times(sampler: emcee.EnsembleSampler) -> tuple[np.ndarray, bool]:
    try:
        return sampler.get_autocorr_time(tol=0), True
    except Exception:
        return np.full(sampler.ndim, np.nan), False


def _mode_switch_count(chain: np.ndarray) -> int:
    signs = np.sign(chain[:, :, 0])
    switched = signs[1:] * signs[:-1] < 0
    return int(np.count_nonzero(switched))


def _summarize_run(
    *,
    target_name: str,
    preset_name: str,
    move_specs: list[str],
    sampler: emcee.EnsembleSampler,
    elapsed_seconds: float,
    burn_in: int,
) -> dict:
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    tau, tau_ok = _autocorr_times(sampler)
    post_chain = chain[burn_in:]
    moved = np.any(np.diff(post_chain, axis=0) != 0.0, axis=2)
    post_acceptance = moved.mean(axis=0)
    return {
        "target": target_name,
        "preset": preset_name,
        "moves": describe_emcee_moves(move_specs),
        "n_steps": int(chain.shape[0]),
        "n_walkers": int(chain.shape[1]),
        "ndim": int(chain.shape[2]),
        "elapsed_seconds": float(elapsed_seconds),
        "steps_per_second": float(chain.shape[0] / elapsed_seconds),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "post_acceptance_mean": float(np.mean(post_acceptance)),
        "post_acceptance_min": float(np.min(post_acceptance)),
        "post_acceptance_p05": float(np.quantile(post_acceptance, 0.05)),
        "post_accept_lt_001": int(np.sum(post_acceptance < 0.01)),
        "post_accept_lt_002": int(np.sum(post_acceptance < 0.02)),
        "best_log_probability": float(np.nanmax(log_prob)),
        "final_log_probability_median": float(np.nanmedian(log_prob[-1])),
        "tau_estimate_ok": bool(tau_ok),
        "tau_median": float(np.nanmedian(tau)),
        "tau_max": float(np.nanmax(tau)),
        "mode_switches_dim0": _mode_switch_count(chain[burn_in:]),
    }


def _plot_trace(chain: np.ndarray, path: Path, title: str) -> None:
    n_plot = min(4, chain.shape[2])
    fig, axes = plt.subplots(n_plot, 1, figsize=(9.0, 1.8 * n_plot), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(chain[:, :, index], alpha=0.18, lw=0.45)
        axis.set_ylabel(f"x{index}")
    axes[0].set_title(title)
    axes[-1].set_xlabel("step")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    n_walkers_values = args.n_walkers or [128]
    invalid_n_walkers = [value for value in n_walkers_values if value < 2 * args.ndim]
    if invalid_n_walkers:
        raise ValueError(f"--n-walkers should be at least 2 * --ndim; invalid values: {invalid_n_walkers}")
    targets = args.target or ["gaussian", "correlated_gaussian", "two_gaussians", "funnel"]
    presets = args.preset or list(MOVE_PRESETS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for target_name in targets:
        log_prob_fn = _target(target_name)
        for preset_name in presets:
            for n_walkers in n_walkers_values:
                move_specs = MOVE_PRESETS[preset_name]
                moves = build_emcee_moves(emcee, move_specs)
                rng = np.random.default_rng(args.seed)
                initial_positions = _initial_positions(target_name, rng, n_walkers, args.ndim)
                sampler = emcee.EnsembleSampler(
                    nwalkers=n_walkers,
                    ndim=args.ndim,
                    log_prob_fn=log_prob_fn,
                    vectorize=True,
                    moves=moves,
                )
                start = time.perf_counter()
                sampler.run_mcmc(
                    initial_positions,
                    args.n_steps,
                    progress=args.progress,
                    skip_initial_state_check=True,
                )
                elapsed = time.perf_counter() - start
                row = _summarize_run(
                    target_name=target_name,
                    preset_name=preset_name,
                    move_specs=move_specs,
                    sampler=sampler,
                    elapsed_seconds=elapsed,
                    burn_in=args.burn_in,
                )
                rows.append(row)
                print(
                    f"{target_name:20s} {preset_name:12s} "
                    f"walkers={n_walkers:4d} "
                    f"accept={row['mean_acceptance_fraction']:.3f} "
                    f"tau_med={row['tau_median']:.1f} "
                    f"switches={row['mode_switches_dim0']} "
                    f"best_logp={row['best_log_probability']:.2f}",
                    flush=True,
                )
                if args.make_plots:
                    plot_path = args.output_dir / f"{target_name}_{preset_name}_{n_walkers}walkers_trace.png"
                    _plot_trace(sampler.get_chain(), plot_path, f"{target_name} / {preset_name} / {n_walkers} walkers")

    results = pd.DataFrame(rows)
    results_path = args.output_dir / "emcee_move_playground_summary.csv"
    results.to_csv(results_path, index=False)
    (args.output_dir / "emcee_move_playground_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(results_path)


if __name__ == "__main__":
    main()
