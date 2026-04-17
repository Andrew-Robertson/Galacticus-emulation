from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from galacticus_emu import trinity_parameter_specs

CASE_NAMES = ["z0_mstar_mean", "z2_mstar_mean", "z0_mbh_mean", "z2_mbh_mean"]

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
        description="Regenerate Trinity ablation trace plots using log10(parameter) for logarithmic parameters."
    )
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    return parser.parse_args()


def _make_trace_plot(
    samples: np.ndarray,
    labels: list[str],
    output_path: Path,
    max_steps_plot: int = 4000,
    max_walkers_plot: int = 64,
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
    for axis, label, dim_index in zip(axes, labels, range(n_dim), strict=True):
        values = samples[:, :, dim_index]
        if label in LOG_PARAMETER_ALIASES:
            values = np.log10(values)
            axis.set_ylabel(f"log10({DISPLAY_LABELS.get(label, label)})")
        else:
            axis.set_ylabel(DISPLAY_LABELS.get(label, label))
        axis.plot(values, alpha=0.18, linewidth=0.5)
    axes[-1].set_xlabel("Step")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    ablation_root = campaign_root / "emulator_mcmc_ablations"
    figures_root = campaign_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    for case_name in CASE_NAMES:
        case_root = ablation_root / case_name
        chain = np.load(case_root / "chain.npy")
        input_columns = [spec.short_name for spec in trinity_parameter_specs()]
        output_path = figures_root / f"emulator_mcmc_ablation_trace_{case_name}.png"
        _make_trace_plot(chain, input_columns, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
