from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import joblib
import matplotlib.pyplot as plt
import numpy as np

from galacticus_emu.interactive_observables import load_observables_bundle, predict_observables_bundle
from galacticus_emu.lhs import transform_from_prior_quantiles, transform_to_prior_quantiles
from galacticus_emu.plotting import finite_limits_from_values
from run_interactive_observables_mcmc import _selected_targets


BOTTOM_LEFT_LEGEND = {"loc": "lower left", "bbox_to_anchor": (0.02, 0.02)}
UPPER_LEFT_LEGEND = {"loc": "upper left"}

PIT_STANDARD_LEGEND_KWARGS = {
    "smf_z0": BOTTOM_LEFT_LEGEND,
    "smf_z3": BOTTOM_LEFT_LEGEND,
    "sfr_function_robotham2011": BOTTOM_LEFT_LEGEND,
    "size_mass_vdw2014_star_forming_z0": UPPER_LEFT_LEGEND,
    "size_mass_vdw2014_quiescent_z0": UPPER_LEFT_LEGEND,
    "bh_halo_mass_trinity_z1": UPPER_LEFT_LEGEND,
    "mzr_blanc2019": UPPER_LEFT_LEGEND,
}


@dataclass(frozen=True)
class FrameSample:
    checkpoint_step: int
    sample_step: int
    walker: int
    log_probability: float
    theta: np.ndarray
    label: str
    target_log_probability: float | None = None
    interpolation_fraction: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate the evolution of the standard-observable emulator best fit from a saved "
            "standard+sidecar MCMC run."
        )
    )
    parser.add_argument("mcmc_dir", type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument(
        "--selection",
        choices=["ensemble-best", "walker-best", "walker-current"],
        default="ensemble-best",
        help=(
            "How to choose burn-in samples: best so far across all walkers, best so far for "
            "one walker, or the current position of one walker."
        ),
    )
    parser.add_argument(
        "--walker",
        default="map",
        help=(
            "Walker index for walker-* selections, or 'map' to use the walker that produced "
            "the saved MAP sample."
        ),
    )
    parser.add_argument(
        "--timeline",
        choices=["burnin-plus-final", "burnin", "full-to-map", "log-probability"],
        default="burnin-plus-final",
        help=(
            "Use burn-in checkpoints plus a final saved-MAP frame, burn-in checkpoints only, "
            "checkpoints from step 0 to the saved MAP step, or single-walker checkpoints "
            "chosen from equally spaced log-probability levels."
        ),
    )
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--ylim-source",
        choices=["all-frames", "final-frame"],
        default="all-frames",
        help=(
            "Use y-limits spanning all animation frames, or y-limits computed like the static "
            "best-fit figure from the final frame only."
        ),
    )
    parser.add_argument(
        "--layout-preset",
        choices=["default", "pit-standard-stage01"],
        default="default",
        help=(
            "Use the script's original animation layout, or match the fixed legends, labels, "
            "and final-frame axis limits from the PIT standard_observables_stage01 figure."
        ),
    )
    parser.add_argument(
        "--final-interpolation-frames",
        type=int,
        default=0,
        help=(
            "Insert this many additional synthetic frames before the saved final MAP frame, "
            "linearly interpolating physical parameters from the preceding frame to the MAP."
        ),
    )
    parser.add_argument(
        "--final-interpolation-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Add this multiple of fading synthetic jitter to final interpolation frames. "
            "The jitter amplitude is estimated from recent pre-final frame-to-frame "
            "parameter changes in prior-quantile space."
        ),
    )
    parser.add_argument(
        "--final-interpolation-noise-window",
        type=int,
        default=6,
        help=(
            "Number of selected pre-final frames used to estimate the final interpolation "
            "jitter amplitude. Default 6 gives five recent frame-to-frame changes."
        ),
    )
    parser.add_argument(
        "--final-interpolation-noise-seed",
        type=int,
        default=12345,
        help="Random seed for synthetic final interpolation jitter.",
    )
    parser.add_argument(
        "--final-interpolation-noise-smooth-passes",
        type=int,
        default=3,
        help="Number of temporal smoothing passes applied to synthetic final interpolation jitter.",
    )
    parser.add_argument("--keep-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-title", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _axis_metadata(meta: dict | None, observable_key: str, fallback: dict) -> dict[str, str]:
    observable = (meta or {}).get("observables", {}).get(observable_key, {})
    return {
        "label": observable.get("label", fallback.get("label", observable_key)),
        "x_axis_label": observable.get("x_axis_label", fallback.get("x_axis_label", "x")),
        "y_axis_label": observable.get("y_axis_label", fallback.get("y_axis_label", "y")),
    }


def _load_standard_meta(standard_bundle_path: Path) -> dict | None:
    meta_path = standard_bundle_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    return _load_json(meta_path)


def _install_numpy_random_pickle_compatibility() -> None:
    """Allow older NumPy RNG pickles that stored BitGenerator classes."""
    import numpy.random._pickle as random_pickle

    original_ctor = random_pickle.__bit_generator_ctor

    def compatible_bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type) and hasattr(bit_generator_name, "__name__"):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    random_pickle.__bit_generator_ctor = compatible_bit_generator_ctor


def _resolve_existing_path(raw_path: str | None, fallback: Path) -> Path:
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = fallback.parent / path
        if path.exists():
            return path
    return fallback


def _best_theta_array(summary: dict, parameter_names: list[str]) -> np.ndarray:
    best_theta = summary.get("best_theta")
    if not isinstance(best_theta, dict):
        raise KeyError("run summary does not contain a best_theta dictionary")
    missing = [name for name in parameter_names if name not in best_theta]
    if missing:
        raise KeyError(f"best_theta is missing parameter(s): {missing}")
    return np.asarray([best_theta[name] for name in parameter_names], dtype=float)


def _find_saved_map_index(
    chain: np.ndarray,
    log_probability: np.ndarray,
    theta: np.ndarray,
    *,
    burn_in: int,
    thin: int,
    best_log_probability: float | None,
) -> tuple[int, int] | None:
    candidate_steps = np.arange(max(int(burn_in), 0), chain.shape[0], max(int(thin), 1), dtype=int)
    best_distance = np.inf
    best_index: tuple[int, int] | None = None
    for step in candidate_steps:
        block = np.asarray(chain[step], dtype=float)
        distances = np.max(np.abs(block - theta[None, :]), axis=1)
        walker = int(np.argmin(distances))
        distance = float(distances[walker])
        if distance < best_distance:
            best_distance = distance
            best_index = (int(step), walker)
        if distance <= 1.0e-10:
            return int(step), walker

    if best_log_probability is not None and np.isfinite(best_log_probability):
        post = np.asarray(log_probability[max(int(burn_in), 0) :], dtype=float)
        close = np.flatnonzero(np.isclose(post.ravel(), float(best_log_probability), rtol=0.0, atol=1.0e-10))
        if close.size:
            local_step, walker = np.unravel_index(int(close[0]), post.shape)
            return max(int(burn_in), 0) + int(local_step), int(walker)

    return best_index


def _unique_linspace(start: int, stop: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if stop < start:
        return [start] * count
    values = np.rint(np.linspace(start, stop, count)).astype(int).tolist()
    if len(values) == count:
        return values
    return values[:count]


def _frame_checkpoints(
    *,
    timeline: str,
    n_frames: int,
    burn_in: int,
    map_step: int,
) -> tuple[list[int], bool]:
    if n_frames < 1:
        raise ValueError("--n-frames must be >= 1")
    burn_stop = max(int(burn_in) - 1, 0)
    if timeline == "burnin-plus-final":
        return _unique_linspace(0, burn_stop, n_frames - 1), True
    if timeline == "burnin":
        return _unique_linspace(0, burn_stop, n_frames), False
    if timeline == "full-to-map":
        return _unique_linspace(0, max(int(map_step), 0), n_frames - 1), True
    if timeline == "log-probability":
        raise ValueError("log-probability checkpoints require a resolved walker")
    raise ValueError(f"Unknown timeline: {timeline}")


def _changed_walker_steps(chain: np.ndarray, *, walker: int, stop_step: int) -> np.ndarray:
    stop_step = min(max(int(stop_step), 0), chain.shape[0] - 1)
    values = np.asarray(chain[: stop_step + 1, walker, :], dtype=float)
    if values.shape[0] == 1:
        return np.asarray([0], dtype=int)
    changed = np.empty(values.shape[0], dtype=bool)
    changed[0] = True
    changed[1:] = np.any(np.diff(values, axis=0) != 0.0, axis=1)
    return np.flatnonzero(changed)


def _walker_best_update_steps(log_probability: np.ndarray, *, walker: int, stop_step: int) -> np.ndarray:
    stop_step = min(max(int(stop_step), 0), log_probability.shape[0] - 1)
    values = np.asarray(log_probability[: stop_step + 1, walker], dtype=float)
    finite_values = np.where(np.isfinite(values), values, -np.inf)
    running_best = np.maximum.accumulate(finite_values)
    updated = np.empty(running_best.shape, dtype=bool)
    updated[0] = True
    updated[1:] = np.diff(running_best) > 0.0
    return np.flatnonzero(updated)


def _log_probability_checkpoints_for_walker(
    *,
    chain: np.ndarray,
    log_probability: np.ndarray,
    walker: int,
    selection: str,
    n_checkpoints: int,
    map_step: int,
    map_log_probability: float,
) -> tuple[list[int], list[float]]:
    if n_checkpoints <= 0:
        return [], []
    stop_step = min(max(int(map_step), 0), log_probability.shape[0] - 1)
    walker_log_probability = np.asarray(log_probability[: stop_step + 1, walker], dtype=float)
    start_log_probability = float(walker_log_probability[0])
    if not np.isfinite(start_log_probability):
        finite = walker_log_probability[np.isfinite(walker_log_probability)]
        if finite.size == 0:
            raise ValueError(f"walker {walker} has no finite log-probability values before the MAP step")
        start_log_probability = float(finite[0])
    if not np.isfinite(map_log_probability):
        finite = walker_log_probability[np.isfinite(walker_log_probability)]
        if finite.size == 0:
            raise ValueError(f"walker {walker} has no finite log-probability values before the MAP step")
        map_log_probability = float(np.max(finite))

    if selection == "walker-best":
        available_steps = _walker_best_update_steps(log_probability, walker=walker, stop_step=stop_step)
        values = np.maximum.accumulate(np.where(np.isfinite(walker_log_probability), walker_log_probability, -np.inf))
    elif selection == "walker-current":
        available_steps = _changed_walker_steps(chain, walker=walker, stop_step=stop_step)
        values = walker_log_probability
    else:
        raise ValueError("--timeline log-probability is only defined for walker-* selections")

    if available_steps.size == 0:
        raise ValueError(f"walker {walker} has no available states before the MAP step")

    levels = np.linspace(start_log_probability, float(map_log_probability), n_checkpoints + 1)[:-1]
    checkpoints: list[int] = []
    target_log_probabilities: list[float] = []
    previous_step = -1
    for level in levels:
        remaining_steps = available_steps[available_steps > previous_step]
        if remaining_steps.size == 0:
            step = previous_step
        else:
            remaining_values = values[remaining_steps]
            above = remaining_steps[np.isfinite(remaining_values) & (remaining_values >= level)]
            if above.size:
                step = int(above[0])
            else:
                finite = np.isfinite(remaining_values)
                if not np.any(finite):
                    step = int(remaining_steps[0])
                else:
                    finite_steps = remaining_steps[finite]
                    finite_values = remaining_values[finite]
                    step = int(finite_steps[int(np.argmin(np.abs(finite_values - level)))])
        checkpoints.append(step)
        target_log_probabilities.append(float(level))
        previous_step = step
    return checkpoints, target_log_probabilities


def _best_ensemble_sample_until(
    chain: np.ndarray,
    log_probability: np.ndarray,
    checkpoint_step: int,
    target_log_probability: float | None = None,
) -> FrameSample:
    step_stop = min(max(int(checkpoint_step), 0), log_probability.shape[0] - 1)
    block = np.asarray(log_probability[: step_stop + 1], dtype=float)
    flat_index = int(np.nanargmax(block))
    sample_step, walker = np.unravel_index(flat_index, block.shape)
    theta = np.asarray(chain[sample_step, walker, :], dtype=float)
    return FrameSample(
        checkpoint_step=step_stop,
        sample_step=int(sample_step),
        walker=int(walker),
        log_probability=float(log_probability[sample_step, walker]),
        theta=theta,
        label="best so far, all walkers",
        target_log_probability=target_log_probability,
    )


def _walker_best_sample_until(
    chain: np.ndarray,
    log_probability: np.ndarray,
    checkpoint_step: int,
    walker: int,
    target_log_probability: float | None = None,
) -> FrameSample:
    step_stop = min(max(int(checkpoint_step), 0), log_probability.shape[0] - 1)
    values = np.asarray(log_probability[: step_stop + 1, walker], dtype=float)
    sample_step = int(np.nanargmax(values))
    theta = np.asarray(chain[sample_step, walker, :], dtype=float)
    return FrameSample(
        checkpoint_step=step_stop,
        sample_step=sample_step,
        walker=int(walker),
        log_probability=float(log_probability[sample_step, walker]),
        theta=theta,
        label=f"best so far, walker {walker}",
        target_log_probability=target_log_probability,
    )


def _walker_current_sample(
    chain: np.ndarray,
    log_probability: np.ndarray,
    checkpoint_step: int,
    walker: int,
    target_log_probability: float | None = None,
) -> FrameSample:
    step = min(max(int(checkpoint_step), 0), log_probability.shape[0] - 1)
    theta = np.asarray(chain[step, walker, :], dtype=float)
    return FrameSample(
        checkpoint_step=step,
        sample_step=step,
        walker=int(walker),
        log_probability=float(log_probability[step, walker]),
        theta=theta,
        label=f"current position, walker {walker}",
        target_log_probability=target_log_probability,
    )


def _select_frames(
    *,
    chain: np.ndarray,
    log_probability: np.ndarray,
    checkpoints: list[int],
    include_final_map: bool,
    selection: str,
    walker: int | None,
    map_step: int,
    map_walker: int,
    map_log_probability: float,
    map_theta: np.ndarray,
    target_log_probabilities: list[float] | None = None,
) -> list[FrameSample]:
    frames: list[FrameSample] = []
    if target_log_probabilities is None:
        target_log_probabilities = [None] * len(checkpoints)
    if len(target_log_probabilities) != len(checkpoints):
        raise ValueError("target_log_probabilities must match checkpoints")
    for checkpoint, target_log_probability in zip(checkpoints, target_log_probabilities, strict=True):
        if selection == "ensemble-best":
            frames.append(_best_ensemble_sample_until(chain, log_probability, checkpoint, target_log_probability))
        elif selection == "walker-best":
            if walker is None:
                raise ValueError("--walker must resolve to an integer for walker-best")
            frames.append(_walker_best_sample_until(chain, log_probability, checkpoint, walker, target_log_probability))
        elif selection == "walker-current":
            if walker is None:
                raise ValueError("--walker must resolve to an integer for walker-current")
            frames.append(_walker_current_sample(chain, log_probability, checkpoint, walker, target_log_probability))
        else:
            raise ValueError(f"Unknown selection: {selection}")
    if include_final_map:
        frames.append(
            FrameSample(
                checkpoint_step=int(map_step),
                sample_step=int(map_step),
                walker=int(map_walker),
                log_probability=float(map_log_probability),
                theta=np.asarray(map_theta, dtype=float),
                label="saved final MAP",
                target_log_probability=float(map_log_probability),
            )
        )
    return frames


def _smooth_frame_noise(noise: np.ndarray, passes: int) -> np.ndarray:
    smoothed = np.asarray(noise, dtype=float)
    for _ in range(max(int(passes), 0)):
        if smoothed.shape[0] < 3:
            break
        padded = np.pad(smoothed, ((1, 1), (0, 0)), mode="edge")
        smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return smoothed


def _interpolated_thetas(
    *,
    frames: list[FrameSample],
    start_frame: FrameSample,
    final_frame: FrameSample,
    fractions: np.ndarray,
    parameter_specs: list | None,
    noise_scale: float,
    noise_window: int,
    noise_seed: int,
    noise_smooth_passes: int,
) -> tuple[np.ndarray, str]:
    if noise_scale <= 0.0:
        theta = (1.0 - fractions[:, None]) * start_frame.theta[None, :] + fractions[:, None] * final_frame.theta[None, :]
        return theta, "linear interpolation to saved final MAP"

    if parameter_specs is None or len(parameter_specs) != start_frame.theta.size:
        raise ValueError(
            "--final-interpolation-noise-scale requires parameter_specs matching the chain dimensionality"
        )

    pre_final_frames = frames[:-1]
    recent_frames = pre_final_frames[-max(int(noise_window), 2) :]
    recent_theta = np.vstack([frame.theta for frame in recent_frames])
    recent_quantiles = transform_to_prior_quantiles(parameter_specs, recent_theta)
    recent_deltas = np.diff(recent_quantiles, axis=0)
    if recent_deltas.size == 0:
        scatter = np.zeros(start_frame.theta.size, dtype=float)
    else:
        scatter = np.sqrt(np.mean(recent_deltas**2, axis=0))

    start_quantile = transform_to_prior_quantiles(parameter_specs, start_frame.theta[None, :])[0]
    final_quantile = transform_to_prior_quantiles(parameter_specs, final_frame.theta[None, :])[0]
    base_quantiles = (
        (1.0 - fractions[:, None]) * start_quantile[None, :]
        + fractions[:, None] * final_quantile[None, :]
    )

    rng = np.random.default_rng(int(noise_seed))
    noise = rng.normal(size=base_quantiles.shape) * scatter[None, :]
    noise = _smooth_frame_noise(noise, int(noise_smooth_passes))
    taper = (1.0 - fractions)[:, None]
    noisy_quantiles = base_quantiles + float(noise_scale) * taper * noise
    noisy_quantiles = np.clip(noisy_quantiles, 1.0e-6, 1.0 - 1.0e-6)
    theta = transform_from_prior_quantiles(parameter_specs, noisy_quantiles)
    return theta, "linear interpolation with fading jitter to saved final MAP"


def _with_final_parameter_interpolation(
    frames: list[FrameSample],
    n_interpolation_frames: int,
    *,
    parameter_specs: list | None = None,
    noise_scale: float = 0.0,
    noise_window: int = 6,
    noise_seed: int = 12345,
    noise_smooth_passes: int = 3,
) -> list[FrameSample]:
    n_interpolation_frames = int(n_interpolation_frames)
    if n_interpolation_frames <= 0:
        return frames
    if len(frames) < 2:
        raise ValueError("--final-interpolation-frames requires at least two selected frames")

    start_frame = frames[-2]
    final_frame = frames[-1]
    if final_frame.label != "saved final MAP":
        raise ValueError("--final-interpolation-frames requires the last frame to be the saved final MAP")

    ramp = []
    fractions = np.arange(1, n_interpolation_frames + 1, dtype=float) / (n_interpolation_frames + 1)
    theta_by_frame, label = _interpolated_thetas(
        frames=frames,
        start_frame=start_frame,
        final_frame=final_frame,
        fractions=fractions,
        parameter_specs=parameter_specs,
        noise_scale=float(noise_scale),
        noise_window=int(noise_window),
        noise_seed=int(noise_seed),
        noise_smooth_passes=int(noise_smooth_passes),
    )
    for index, (fraction, theta) in enumerate(zip(fractions, theta_by_frame, strict=True), start=1):
        log_probability = (1.0 - fraction) * start_frame.log_probability + fraction * final_frame.log_probability
        synthetic_step = int(round((1.0 - fraction) * start_frame.sample_step + fraction * final_frame.sample_step))
        ramp.append(
            FrameSample(
                checkpoint_step=synthetic_step,
                sample_step=synthetic_step,
                walker=final_frame.walker,
                log_probability=float(log_probability),
                theta=np.asarray(theta, dtype=float),
                label=label,
                target_log_probability=final_frame.target_log_probability,
                interpolation_fraction=float(fraction),
            )
        )

    return frames[:-1] + ramp + [final_frame]


def _prediction_by_key(
    standard_bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    parameter_names: list[str],
    slow_count: int,
    theta: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    params = {
        name: float(value)
        for name, value in zip(parameter_names[:slow_count], theta[:slow_count], strict=True)
    }
    predictions = predict_observables_bundle(standard_bundle, params)
    result = {}
    for key in observable_keys:
        mask = masks[key]
        prediction = predictions[key]
        result[key] = (
            np.asarray(prediction["y_pred_plot"], dtype=float)[mask],
            np.asarray(prediction["y_std_plot"], dtype=float)[mask],
        )
    return result


def _fixed_y_limits(
    standard_bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    predictions_by_frame: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    include_prediction_uncertainty: bool = True,
    include_target_uncertainty: bool = True,
    margin_fraction: float = 0.08,
    min_margin: float = 0.15,
) -> dict[str, tuple[float, float]]:
    limits = {}
    for key in observable_keys:
        observable = standard_bundle["observables"][key]
        mask = masks[key]
        target = np.asarray(observable["target_plot"], dtype=float)[mask]
        values = [target]
        target_sigma = observable.get("target_sigma_plot")
        if include_target_uncertainty and target_sigma is not None:
            sigma = np.asarray(target_sigma, dtype=float)[mask]
            values.extend([target - sigma, target + sigma])
        for prediction_by_key in predictions_by_frame:
            pred, pred_std = prediction_by_key[key]
            values.append(pred)
            if include_prediction_uncertainty:
                values.extend([pred - pred_std, pred + pred_std])
        limit = finite_limits_from_values(
            *values,
            margin_fraction=margin_fraction,
            min_margin=min_margin,
        )
        if limit is not None:
            limits[key] = limit
    return limits


def _fixed_x_limits(
    standard_bundle: dict,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    *,
    margin_fraction: float = 0.04,
    min_margin: float = 0.02,
) -> dict[str, tuple[float, float]]:
    limits = {}
    for key in observable_keys:
        observable = standard_bundle["observables"][key]
        mask = masks[key]
        x = np.asarray(observable["x_plot"], dtype=float)[mask]
        limit = finite_limits_from_values(
            x,
            margin_fraction=margin_fraction,
            min_margin=min_margin,
        )
        if limit is not None:
            limits[key] = limit
    return limits


def _ordered_legend_handles(axis: plt.Axes, labels: tuple[str, ...]) -> list[object]:
    handles_by_label = {
        label: handle for handle, label in zip(*axis.get_legend_handles_labels(), strict=False)
    }
    return [handles_by_label[label] for label in labels if label in handles_by_label]


def _legend_kwargs(
    configs: dict[str, dict[str, object]] | None,
    observable_key: str,
    default_loc: str = "upper left",
) -> dict[str, object]:
    if configs is None:
        return {"loc": default_loc}
    return dict(configs.get(observable_key, {"loc": default_loc}))


def _plot_frame(
    *,
    standard_bundle: dict,
    standard_meta: dict | None,
    observable_keys: list[str],
    masks: dict[str, np.ndarray],
    prediction_by_key: dict[str, tuple[np.ndarray, np.ndarray]],
    x_limits: dict[str, tuple[float, float]],
    y_limits: dict[str, tuple[float, float]],
    line_label: str,
    legend_labels: tuple[str, ...] | None,
    legend_kwargs_by_key: dict[str, dict[str, object]] | None,
    frame: FrameSample,
    frame_index: int,
    n_frames: int,
    output_path: Path,
    dpi: int,
    title: str | None,
) -> None:
    n_panels = len(observable_keys)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 4.0 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, key in zip(axes_flat, observable_keys, strict=False):
        observable = standard_bundle["observables"][key]
        mask = masks[key]
        x = np.asarray(observable["x_plot"], dtype=float)[mask]
        target = np.asarray(observable["target_plot"], dtype=float)[mask]
        target_sigma = observable.get("target_sigma_plot")
        if target_sigma is not None:
            target_sigma = np.asarray(target_sigma, dtype=float)[mask]
        pred, pred_std = prediction_by_key[key]
        axis.errorbar(x, target, yerr=target_sigma, fmt="o", color="0.15", label="target", zorder=4)
        axis.plot(x, pred, color="tab:blue", lw=1.8, label=line_label, zorder=3)
        axis.fill_between(x, pred - pred_std, pred + pred_std, color="tab:blue", alpha=0.2, zorder=2)
        if key in x_limits:
            axis.set_xlim(*x_limits[key])
        if key in y_limits:
            axis.set_ylim(*y_limits[key])
        metadata = _axis_metadata(standard_meta, key, observable)
        axis.set_title(metadata["label"])
        axis.set_xlabel(metadata["x_axis_label"])
        axis.set_ylabel(metadata["y_axis_label"])
        axis.grid(alpha=0.22)
        handles = _ordered_legend_handles(axis, legend_labels) if legend_labels is not None else None
        if handles is None:
            axis.legend(frameon=False, fontsize=8)
        else:
            axis.legend(
                handles=handles,
                frameon=False,
                fontsize=8,
                **_legend_kwargs(legend_kwargs_by_key, key),
            )
    for axis in axes_flat[n_panels:]:
        axis.axis("off")
    if title:
        fig.suptitle(
            (
                f"{title}\n"
                f"Frame {frame_index + 1}/{n_frames}: {frame.label}\n"
                f"checkpoint={frame.checkpoint_step}; sample={frame.sample_step}; "
                f"walker={frame.walker}; log P={frame.log_probability:.3f}"
            ),
            fontsize=10,
        )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _encode_mp4(frames_dir: Path, output_path: Path, *, fps: float) -> None:
    frame_pattern = frames_dir / "frame_%03d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{float(fps):g}",
        "-i",
        str(frame_pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    _install_numpy_random_pickle_compatibility()
    mcmc_dir = args.mcmc_dir.expanduser().resolve()
    figures_dir = args.figures_dir.expanduser().resolve() if args.figures_dir else mcmc_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_json(mcmc_dir / f"{args.output_prefix}_run_summary.json")
    inputs = joblib.load(mcmc_dir / f"{args.output_prefix}_mcmc_inputs.joblib")

    parameter_names = list(inputs.get("parameter_names", summary["parameter_names"]))
    parameter_specs = list(inputs.get("parameter_specs", []))
    slow_count = int(inputs.get("slow_count", summary.get("slow_count", len(parameter_names))))
    standard_observable_keys = list(inputs.get("standard_observable_keys", summary["standard_observable_keys"]))
    standard_bundle_path = Path(inputs.get("standard_bundle_path", summary["standard_bundle_path"]))
    standard_bundle = load_observables_bundle(standard_bundle_path)
    use_pit_layout = args.layout_preset == "pit-standard-stage01"
    standard_meta = _load_standard_meta(standard_bundle_path) if use_pit_layout else None

    masks = inputs.get("masks")
    if masks is None:
        x_ranges = {
            key: tuple(value)
            for key, value in summary.get("standard_observable_x_ranges", {}).items()
        }
        _, _, masks, _ = _selected_targets(
            standard_bundle,
            standard_observable_keys,
            x_ranges,
            float(summary.get("target_sigma_floor", 1.0e-3)),
        )

    chain_path = _resolve_existing_path(
        summary.get("full_chain_path") or summary.get("chain_path"),
        mcmc_dir / f"{args.output_prefix}_chain.npy",
    )
    log_probability_path = _resolve_existing_path(
        summary.get("full_log_prob_path") or summary.get("log_prob_path"),
        mcmc_dir / f"{args.output_prefix}_log_prob.npy",
    )
    chain = np.load(chain_path, mmap_mode="r")
    log_probability = np.load(log_probability_path, mmap_mode="r")
    if chain.ndim != 3:
        raise ValueError(f"{chain_path} has shape {chain.shape}; expected (steps, walkers, parameters)")
    if log_probability.shape != chain.shape[:2]:
        raise ValueError(
            f"{log_probability_path} shape {log_probability.shape} does not match chain shape {chain.shape[:2]}"
        )

    map_theta = _best_theta_array(summary, parameter_names)
    map_index = _find_saved_map_index(
        chain,
        log_probability,
        map_theta,
        burn_in=int(summary["burn_in"]),
        thin=int(summary.get("thin", 1)),
        best_log_probability=summary.get("best_log_probability"),
    )
    map_step, map_walker = map_index if map_index is not None else (-1, -1)
    if args.walker == "map":
        walker = map_walker if map_walker >= 0 else None
    else:
        walker = int(args.walker)
    if args.selection.startswith("walker") and walker is None:
        raise ValueError("Could not infer the saved-MAP walker; pass --walker explicitly.")
    if walker is not None and not 0 <= walker < chain.shape[1]:
        raise ValueError(f"walker={walker} is outside [0, {chain.shape[1]})")

    if args.timeline == "log-probability":
        if not args.selection.startswith("walker"):
            raise ValueError("--timeline log-probability requires --selection walker-current or walker-best")
        if walker is None:
            raise ValueError("--timeline log-probability requires a resolved --walker")
        checkpoints, target_log_probabilities = _log_probability_checkpoints_for_walker(
            chain=chain,
            log_probability=log_probability,
            walker=walker,
            selection=args.selection,
            n_checkpoints=int(args.n_frames) - 1,
            map_step=map_step,
            map_log_probability=float(summary["best_log_probability"]),
        )
        include_final_map = True
    else:
        checkpoints, include_final_map = _frame_checkpoints(
            timeline=args.timeline,
            n_frames=int(args.n_frames),
            burn_in=int(summary["burn_in"]),
            map_step=map_step,
        )
        target_log_probabilities = None

    frames = _select_frames(
        chain=chain,
        log_probability=log_probability,
        checkpoints=checkpoints,
        include_final_map=include_final_map,
        selection=args.selection,
        walker=walker,
        map_step=map_step,
        map_walker=map_walker,
        map_log_probability=float(summary["best_log_probability"]),
        map_theta=map_theta,
        target_log_probabilities=target_log_probabilities,
    )
    if len(frames) != int(args.n_frames):
        raise RuntimeError(f"Built {len(frames)} frames, expected {args.n_frames}")
    frames = _with_final_parameter_interpolation(
        frames,
        int(args.final_interpolation_frames),
        parameter_specs=parameter_specs,
        noise_scale=float(args.final_interpolation_noise_scale),
        noise_window=int(args.final_interpolation_noise_window),
        noise_seed=int(args.final_interpolation_noise_seed),
        noise_smooth_passes=int(args.final_interpolation_noise_smooth_passes),
    )

    predictions_by_frame = [
        _prediction_by_key(
            standard_bundle,
            standard_observable_keys,
            masks,
            parameter_names,
            slow_count,
            frame.theta,
        )
        for frame in frames
    ]
    if use_pit_layout:
        x_limits = _fixed_x_limits(standard_bundle, standard_observable_keys, masks)
        y_limits = _fixed_y_limits(
            standard_bundle,
            standard_observable_keys,
            masks,
            [predictions_by_frame[-1]],
            include_prediction_uncertainty=True,
            include_target_uncertainty=False,
        )
        line_label = "emulator MAP"
        legend_labels = ("target", "emulator MAP")
        legend_kwargs_by_key = PIT_STANDARD_LEGEND_KWARGS
    elif args.ylim_source == "final-frame":
        x_limits = {}
        y_limits = _fixed_y_limits(
            standard_bundle,
            standard_observable_keys,
            masks,
            [predictions_by_frame[-1]],
            include_prediction_uncertainty=False,
            include_target_uncertainty=False,
        )
        line_label = "best fit"
        legend_labels = None
        legend_kwargs_by_key = None
    else:
        x_limits = {}
        y_limits = _fixed_y_limits(
            standard_bundle,
            standard_observable_keys,
            masks,
            predictions_by_frame,
            include_prediction_uncertainty=True,
            include_target_uncertainty=True,
        )
        line_label = "best fit"
        legend_labels = None
        legend_kwargs_by_key = None

    label_parts = [args.output_prefix, "standard_best_fit_evolution", args.selection, args.timeline]
    if args.selection.startswith("walker") and walker is not None:
        label_parts.append(f"walker{walker:03d}")
    if int(args.final_interpolation_frames) > 0:
        label_parts.append(f"interp{int(args.final_interpolation_frames):03d}")
    if float(args.final_interpolation_noise_scale) > 0.0:
        noise_label = f"jitter{float(args.final_interpolation_noise_scale):.2f}".replace(".", "p")
        label_parts.append(noise_label)
    if use_pit_layout:
        label_parts.append("pit_stage01_layout")
    if args.ylim_source == "final-frame":
        label_parts.append("final_layout")
    if args.no_title:
        label_parts.append("notitle")
    label = "_".join(label_parts).replace("-", "_")
    frames_dir = args.frames_dir.expanduser().resolve() if args.frames_dir else figures_dir / f"{label}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_path.expanduser().resolve() if args.output_path else figures_dir / f"{label}.mp4"
    title = None if args.no_title else args.title or "Standard-observable best-fit evolution"

    frame_rows = []
    for index, (frame, prediction_by_key) in enumerate(zip(frames, predictions_by_frame, strict=True)):
        frame_path = frames_dir / f"frame_{index:03d}.png"
        _plot_frame(
            standard_bundle=standard_bundle,
            standard_meta=standard_meta,
            observable_keys=standard_observable_keys,
            masks=masks,
            prediction_by_key=prediction_by_key,
            x_limits=x_limits,
            y_limits=y_limits,
            line_label=line_label,
            legend_labels=legend_labels,
            legend_kwargs_by_key=legend_kwargs_by_key,
            frame=frame,
            frame_index=index,
            n_frames=len(frames),
            output_path=frame_path,
            dpi=int(args.dpi),
            title=title,
        )
        frame_rows.append(
            {
                "frame": index,
                "checkpoint_step": int(frame.checkpoint_step),
                "sample_step": int(frame.sample_step),
                "walker": int(frame.walker),
                "log_probability": float(frame.log_probability),
                "target_log_probability": (
                    None if frame.target_log_probability is None else float(frame.target_log_probability)
                ),
                "interpolation_fraction": (
                    None if frame.interpolation_fraction is None else float(frame.interpolation_fraction)
                ),
                "label": frame.label,
                "path": str(frame_path),
            }
        )

    _encode_mp4(frames_dir, output_path, fps=float(args.fps))
    metadata_path = frames_dir / "frames.json"
    metadata_path.write_text(json.dumps(frame_rows, indent=2) + "\n")

    print(output_path)
    print(frames_dir)
    print(metadata_path)


if __name__ == "__main__":
    main()
