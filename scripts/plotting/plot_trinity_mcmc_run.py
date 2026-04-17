from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from galacticus_emu import trinity_parameter_specs

try:
    import corner
except ModuleNotFoundError:  # pragma: no cover
    corner = None

try:
    from getdist import MCSamples, plots as gd_plots
except ModuleNotFoundError:  # pragma: no cover
    MCSamples = None
    gd_plots = None

LOG_PARAMETER_ALIASES = {
    "blackHoleSeedMass",
    "diskSFRFrequencyNorm",
    "spheroidSFREfficiency",
    "bondiEnhancementSpheroid",
    "bondiEnhancementHotHalo",
    "bondiTemperatureSpheroid",
    "BHefficiencyWind",
    "thinDiskMaximum",
}

DISPLAY_LABELS = {
    "diskVelocityCharacteristic": "diskVelocityCharacteristic",
    "diskExponent": "diskExponent",
    "spheroidVelocityCharacteristic": "spheroidVelocityCharacteristic",
    "spheroidExponent": "spheroidExponent",
    "BHefficiencyWind": "BHefficiencyWind",
    "BHefficiencyRadioMode": "BHefficiencyRadioMode",
    "henriquesGamma": "henriquesGamma",
    "henriquesDelta1": "henriquesDelta1",
    "henriquesDelta2": "henriquesDelta2",
    "coreRadiusOverVirialRadius": "hotHaloCoreRadius",
    "diskSFRFrequencyNorm": "normalisationBlitzSFR",
    "barFractionAngularMomentumRetainedSpheroid": "barInstabilityFracAngMomRetSpheroid",
    "coolingMultiplier": "coolingRateMultiplier",
    "spheroidSFREfficiency": "spheroidStarFormationEfficiency",
    "spheroidSFRExponentVelocity": "spheroidStarFormationExponent",
    "thinDiskMaximum": "thinDiskMaxAccretion",
    "bondiEnhancementSpheroid": "bondiHoyleSpheroidEnhancement",
    "bondiEnhancementHotHalo": "bondiHoyleHotHaloEnhancement",
    "bondiTemperatureSpheroid": "bondiHoyleSpheroidTemperature",
    "blackHoleSeedMass": "blackHoleSeedMass",
    "massRatioMajorMerger": "majorMergerMassRatio",
    "energyOrbital": "mergerRemnantOrbitalEnergyFactor",
    "barStabilityThresholdGaseous": "barInstabilityThresholdGaseous",
    "barStabilityThresholdStellar": "barInstabilityThresholdStellar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Trinity MCMC trace and corner plots using log10(parameter) for logarithmic parameters."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--mcmc-dir",
        default="emulator_mcmc_four_mean_families",
        help="Name of the MCMC output directory inside the campaign root.",
    )
    parser.add_argument(
        "--trace-name",
        default="emulator_mcmc_four_mean_families_trace.png",
        help="Filename for the trace plot in the figures directory.",
    )
    parser.add_argument(
        "--corner-name",
        default="emulator_mcmc_four_mean_families_corner.png",
        help="Filename for the corner plot in the figures directory.",
    )
    parser.add_argument(
        "--backend",
        choices=["corner", "getdist"],
        default="corner",
        help="Plotting backend to use for the corner plot.",
    )
    parser.add_argument("--max-steps-plot", type=int, default=4000)
    parser.add_argument("--max-walkers-plot", type=int, default=64)
    return parser.parse_args()


def _transform_samples(df: pd.DataFrame, input_columns: list[str]) -> tuple[np.ndarray, list[str]]:
    transformed_columns = []
    labels = []
    for name in input_columns:
        values = df[name].to_numpy(dtype=float)
        if name in LOG_PARAMETER_ALIASES:
            values = np.log10(values)
            labels.append(f"log10({DISPLAY_LABELS.get(name, name)})")
        else:
            labels.append(DISPLAY_LABELS.get(name, name))
        transformed_columns.append(values)
    return np.column_stack(transformed_columns), labels


def _parameter_ranges(input_columns: list[str]) -> dict[str, list[float]]:
    spec_map = {spec.short_name: spec for spec in trinity_parameter_specs()}
    ranges: dict[str, list[float]] = {}
    for name in input_columns:
        spec = spec_map[name]
        lower = float(spec.prior.lower)
        upper = float(spec.prior.upper)
        if name in LOG_PARAMETER_ALIASES:
            ranges[name] = [float(np.log10(lower)), float(np.log10(upper))]
        else:
            ranges[name] = [lower, upper]
    return ranges


def _make_trace_plot(
    samples: np.ndarray,
    labels: list[str],
    input_columns: list[str],
    output_path: Path,
    max_steps_plot: int,
    max_walkers_plot: int,
) -> None:
    if samples.shape[0] > max_steps_plot:
        step_stride = int(np.ceil(samples.shape[0] / max_steps_plot))
        samples = samples[::step_stride]
    if samples.shape[1] > max_walkers_plot:
        walker_stride = int(np.ceil(samples.shape[1] / max_walkers_plot))
        samples = samples[:, ::walker_stride, :]

    n_steps, _, n_dim = samples.shape
    fig, axes = plt.subplots(n_dim, 1, figsize=(9, 1.8 * n_dim), sharex=True, constrained_layout=True)
    if n_dim == 1:
        axes = [axes]
    for axis, label, name, dim_index in zip(axes, labels, input_columns, range(n_dim), strict=True):
        values = samples[:, :, dim_index]
        if name in LOG_PARAMETER_ALIASES:
            values = np.log10(values)
        axis.plot(values, alpha=0.18, linewidth=0.5)
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Step")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _make_corner_plot(samples: np.ndarray, labels: list[str], truths: np.ndarray, output_path: Path) -> None:
    ranges = []
    for dim in range(samples.shape[1]):
        lower = float(np.min(samples[:, dim]))
        upper = float(np.max(samples[:, dim]))
        margin = 0.05 * (upper - lower if upper > lower else 1.0)
        ranges.append((lower - margin, upper + margin))
    corner_fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        range=ranges,
    )
    corner_fig.savefig(output_path, dpi=180)
    plt.close(corner_fig)


def _make_getdist_plot(
    samples: np.ndarray,
    labels: list[str],
    input_columns: list[str],
    output_path: Path,
) -> None:
    ranges = _parameter_ranges(input_columns)
    root = MCSamples(
        samples=samples,
        names=input_columns,
        labels=labels,
        ranges=ranges,
        settings={"smooth_scale_1D": 0.7, "smooth_scale_2D": 0.7},
    )
    plotter = gd_plots.get_subplot_plotter(width_inch=11)
    plotter.settings.figure_legend_frame = False
    plotter.settings.linewidth = 1.5
    plotter.settings.linewidth_contour = 1.6
    plotter.settings.num_plot_contours = 2
    plotter.settings.axes_labelsize = 11
    plotter.settings.lab_fontsize = 11
    plotter.triangle_plot([root], params=input_columns, filled=False)
    plotter.export(str(output_path))


def main() -> None:
    args = parse_args()
    if args.backend == "corner" and corner is None:
        raise ModuleNotFoundError("The 'corner' package is required for plot_trinity_mcmc_run.py with --backend corner.")
    if args.backend == "getdist" and (MCSamples is None or gd_plots is None):
        raise ModuleNotFoundError("The 'getdist' package is required for plot_trinity_mcmc_run.py with --backend getdist.")

    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / "figures"
    mcmc_root = campaign_root / args.mcmc_dir
    figures_root.mkdir(parents=True, exist_ok=True)

    chain = np.load(mcmc_root / "chain.npy")
    posterior_df = pd.read_csv(mcmc_root / "posterior_samples.csv")
    input_columns = [column for column in posterior_df.columns if column != "log_probability"]

    transformed_samples, labels = _transform_samples(posterior_df, input_columns)
    best_index = int(np.argmax(posterior_df["log_probability"].to_numpy(dtype=float)))
    truths = transformed_samples[best_index]

    trace_path = figures_root / args.trace_name
    _make_trace_plot(chain, labels, input_columns, trace_path, args.max_steps_plot, args.max_walkers_plot)

    corner_path = figures_root / args.corner_name
    if args.backend == "corner":
        _make_corner_plot(transformed_samples, labels, truths, corner_path)
    else:
        _make_getdist_plot(transformed_samples, labels, input_columns, corner_path)

    print(trace_path)
    print(corner_path)


if __name__ == "__main__":
    main()
