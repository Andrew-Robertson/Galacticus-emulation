from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu import (
    fit_scaled_gp,
    load_emulator_bundle,
    predict_scaled_gp,
    transform_to_prior_quantiles,
    trinity_parameter_specs,
)


DEFAULT_CASE_SPECS = [
    ("mstar_only", "Combined Mstar", "#1b9e77", "emulator_mcmc_mstar_z0_z2/run_summary.json"),
    ("mbh_only", "Combined Mbh", "#7570b3", "emulator_mcmc_mbh_z0_z2_no1e11/run_summary.json"),
    ("mzr_only", "z=0 MZR", "#d95f02", "emulator_mcmc_mzr_z0_upper/run_summary.json"),
    ("total", "Total Mstar + Mbh + MZR", "#000000", "emulator_mcmc_four_mean_families_no1e11_mbh_plus_mzr_z0_upper_w5/run_summary.json"),
]

TRINITY_STELLAR_OUTPUTS = [
    *[f"z0_mass_stellar_log10_{index}" for index in range(3)],
    *[f"z2_mass_stellar_log10_{index}" for index in range(3)],
]

TRINITY_BLACK_HOLE_OUTPUTS = [
    *[f"z0_mass_black_hole_log10_{index}" for index in range(3)],
    *[f"z2_mass_black_hole_log10_{index}" for index in range(3)],
]

MZR_MEAN_OUTPUTS = [
    *[f"z0_mass_metallicity_mass_stellar_log10_{index}" for index in range(3)],
    *[f"z0_mass_metallicity_abundance_oxygen_12logoh_{index}" for index in range(3)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare emulator-predicted SHMR and MZR observables at three MAP points.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    parser.add_argument(
        "--models-summary",
        default="emulator/gp_fit_summary_all_outputs_all36_opt_gw.json",
        help="Summary JSON for the serialized Trinity stellar/BH emulators, relative to campaign_root unless absolute.",
    )
    parser.add_argument(
        "--observation-file",
        default="/Users/arobertson/Documents/Projects/GalacticusEmu/Galacticus/datasets/static/observations/abundances/massMetallicityRelationBlanc2019.hdf5",
        help="Path to the Blanc 2019 mass-metallicity HDF5 file.",
    )
    parser.add_argument(
        "--figures-dir-name",
        default="figures_z0_z2_Trinity_MZR",
        help="Figures directory inside the campaign root.",
    )
    parser.add_argument(
        "--output-name",
        default="mstar_mbh_mzr_map_comparison.png",
        help="Output figure name.",
    )
    parser.add_argument(
        "--mstar-summary",
        default="emulator_mcmc_mstar_z0_z2/run_summary.json",
        help="Run-summary path for the Mstar-only case, relative to campaign_root unless absolute.",
    )
    parser.add_argument(
        "--mbh-summary",
        default="emulator_mcmc_mbh_z0_z2_no1e11/run_summary.json",
        help="Run-summary path for the Mbh-only case, relative to campaign_root unless absolute.",
    )
    parser.add_argument(
        "--mzr-summary",
        default="emulator_mcmc_mzr_z0_upper/run_summary.json",
        help="Run-summary path for the MZR-only case, relative to campaign_root unless absolute.",
    )
    parser.add_argument(
        "--mstar-label",
        default="Combined Mstar",
        help="Legend label for the Mstar-only case.",
    )
    parser.add_argument(
        "--mbh-label",
        default="Combined Mbh",
        help="Legend label for the Mbh-only case.",
    )
    parser.add_argument(
        "--mzr-label",
        default="z=0 MZR",
        help="Legend label for the MZR-only case.",
    )
    parser.add_argument(
        "--total-summary",
        default="emulator_mcmc_four_mean_families_no1e11_mbh_plus_mzr_z0_upper_w5/run_summary.json",
        help="Run-summary path for the total case, relative to campaign_root unless absolute.",
    )
    parser.add_argument(
        "--total-label",
        default="Total Mstar + Mbh + MZR",
        help="Legend label for the total case.",
    )
    return parser.parse_args()


def _resolve_path(campaign_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = campaign_root / path
    return path


def _load_map_thetas(
    campaign_root: Path,
    mstar_summary: Path,
    mstar_label: str,
    mbh_summary: Path,
    mbh_label: str,
    mzr_summary: Path,
    mzr_label: str,
    total_summary: Path,
    total_label: str,
) -> dict[str, dict]:
    case_specs = [
        ("mstar_only", mstar_label, "#1b9e77", str(mstar_summary)),
        ("mbh_only", mbh_label, "#7570b3", str(mbh_summary)),
        ("mzr_only", mzr_label, "#d95f02", str(mzr_summary)),
        ("total", total_label, "#000000", str(total_summary)),
    ]
    thetas: dict[str, dict] = {}
    for key, label, color, relative_summary in case_specs:
        summary_path = _resolve_path(campaign_root, relative_summary)
        summary = json.loads(summary_path.read_text())
        thetas[key] = {
            "label": label,
            "color": color,
            "theta": summary["best_theta"],
        }
    return thetas


def _load_stellar_models(summary_path: Path) -> dict[str, dict]:
    summary = json.loads(summary_path.read_text())
    bundles: dict[str, dict] = {}
    for model_item in summary["models"]:
        bundle = load_emulator_bundle(model_item["model_path"])
        if bundle["output_column"] in TRINITY_STELLAR_OUTPUTS or bundle["output_column"] in TRINITY_BLACK_HOLE_OUTPUTS:
            bundles[bundle["output_column"]] = bundle
    missing = [name for name in [*TRINITY_STELLAR_OUTPUTS, *TRINITY_BLACK_HOLE_OUTPUTS] if name not in bundles]
    if missing:
        raise FileNotFoundError(f"Missing serialized emulator bundles for: {missing}")
    return bundles


def _fit_mzr_mean_models(campaign_root: Path) -> dict[str, tuple]:
    table = pd.read_csv(campaign_root / "mass_metallicity_emulator_table.csv")
    parameter_specs = trinity_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    x = transform_to_prior_quantiles(parameter_specs, table[input_columns].to_numpy(dtype=float))
    models: dict[str, tuple] = {}
    for output_name in MZR_MEAN_OUTPUTS:
        y = table[output_name].to_numpy(dtype=float)
        models[output_name] = fit_scaled_gp(
            x=x,
            y=y,
            n_restarts_optimizer=1,
            optimize_hyperparameters=True,
        )
    return models


def _predict_bundle(bundle: dict, theta: np.ndarray, parameter_specs) -> float:
    x = transform_to_prior_quantiles(parameter_specs, theta[None, :])
    pred, _ = predict_scaled_gp(bundle["model"], bundle["y_mean"], bundle["y_std"], x)
    return float(pred[0])


def _predict_gp_model(model_tuple: tuple, theta: np.ndarray, parameter_specs) -> float:
    model, y_mean, y_std = model_tuple
    x = transform_to_prior_quantiles(parameter_specs, theta[None, :])
    pred, _ = predict_scaled_gp(model, y_mean, y_std, x)
    return float(pred[0])


def _load_relation_targets(campaign_root: Path) -> dict[str, dict]:
    evaluations_root = campaign_root / "evaluations"
    eval_dir = next(path for path in sorted(evaluations_root.iterdir()) if path.is_dir())
    results: dict[str, dict] = {}
    config_map = {
        "z0_stellar": ("z0.hdf5", "stellarHaloMassRelationUniverseMachinez1", "massStellarLog10"),
        "z2_stellar": ("z2.hdf5", "stellarHaloMassRelationUniverseMachinez6", "massStellarLog10"),
        "z0_black_hole": ("z0.hdf5", "blackHoleHaloMassRelationTRINITYz1", "massBlackHoleLog10"),
        "z2_black_hole": ("z2.hdf5", "blackHoleHaloMassRelationTRINITYz3", "massBlackHoleLog10"),
    }
    for run_name, (filename, analysis_key, value_name) in config_map.items():
        with h5py.File(eval_dir / filename, "r") as handle:
            group = handle["analyses"][analysis_key]
            values = group[value_name][:]
            mask = values != 0.0
            results[run_name] = {
                "mass_halo_log10": np.log10(group["massHalo"][:][mask]),
                "target": group[f"{value_name}Target"][:][mask],
                "target_err": np.sqrt(np.diag(group[f"{value_name}CovarianceTarget"][:]))[mask],
            }
    return results


def _load_blanc(observation_file: Path) -> dict[str, np.ndarray]:
    with h5py.File(observation_file, "r") as handle:
        mass = np.log10(handle["massStellar"][:])
        mean = handle["abundanceOxygenMean"][:]
        lo = handle["abundanceOxygen16PercentCI"][:]
        hi = handle["abundanceOxygen84PercentCI"][:]
    return {"mass_log10": mass, "mean": mean, "lo": lo, "hi": hi}


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    output_path = figures_root / args.output_name

    models_summary_path = _resolve_path(campaign_root, args.models_summary)
    mstar_summary_path = _resolve_path(campaign_root, args.mstar_summary)
    mbh_summary_path = _resolve_path(campaign_root, args.mbh_summary)
    mzr_summary_path = _resolve_path(campaign_root, args.mzr_summary)
    total_summary_path = _resolve_path(campaign_root, args.total_summary)

    parameter_specs = trinity_parameter_specs()
    input_columns = [spec.short_name for spec in parameter_specs]
    shmr_bundles = _load_stellar_models(models_summary_path)
    mzr_models = _fit_mzr_mean_models(campaign_root)
    map_thetas = _load_map_thetas(
        campaign_root=campaign_root,
        mstar_summary=mstar_summary_path,
        mstar_label=args.mstar_label,
        mbh_summary=mbh_summary_path,
        mbh_label=args.mbh_label,
        mzr_summary=mzr_summary_path,
        mzr_label=args.mzr_label,
        total_summary=total_summary_path,
        total_label=args.total_label,
    )
    relation_targets = _load_relation_targets(campaign_root)
    blanc = _load_blanc(Path(args.observation_file).resolve())

    fig, axes = plt.subplots(1, 5, figsize=(26, 5.2), constrained_layout=True)

    for axis, run_name in zip(axes[:2], ["z0_stellar", "z2_stellar"], strict=True):
        target = relation_targets[run_name]
        run_prefix = run_name.split("_", 1)[0]
        axis.errorbar(
            target["mass_halo_log10"],
            target["target"],
            yerr=target["target_err"],
            fmt="o",
            color="#444444",
            markersize=5,
            linewidth=1.2,
            capsize=2,
            label="Target",
        )
        for key, spec in map_thetas.items():
            theta = np.array([spec["theta"][name] for name in input_columns], dtype=float)
            pred = np.array(
                [
                    _predict_bundle(shmr_bundles[f"{run_prefix}_mass_stellar_log10_{index}"], theta, parameter_specs)
                    for index in range(3)
                ],
                dtype=float,
            )
            axis.plot(
                target["mass_halo_log10"],
                pred,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=spec["color"],
                label=spec["label"],
            )
        axis.set_xlabel(r"$\log_{10}(M_{\rm halo}/M_\odot)$")
        axis.set_ylabel(r"$\langle \log_{10} M_\star \rangle$")
        axis.set_title(f"{run_prefix} SHMR")
        axis.grid(alpha=0.2, linewidth=0.5)

    for axis, run_name in zip(axes[2:4], ["z0_black_hole", "z2_black_hole"], strict=True):
        target = relation_targets[run_name]
        run_prefix = run_name.split("_", 1)[0]
        axis.errorbar(
            target["mass_halo_log10"],
            target["target"],
            yerr=target["target_err"],
            fmt="o",
            color="#444444",
            markersize=5,
            linewidth=1.2,
            capsize=2,
            label="Target",
        )
        for key, spec in map_thetas.items():
            theta = np.array([spec["theta"][name] for name in input_columns], dtype=float)
            pred = np.array(
                [
                    _predict_bundle(shmr_bundles[f"{run_prefix}_mass_black_hole_log10_{index}"], theta, parameter_specs)
                    for index in range(3)
                ],
                dtype=float,
            )
            axis.plot(
                target["mass_halo_log10"],
                pred,
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=spec["color"],
                label=spec["label"],
            )
        axis.set_xlabel(r"$\log_{10}(M_{\rm halo}/M_\odot)$")
        axis.set_ylabel(r"$\langle \log_{10} M_{\rm BH} \rangle$")
        axis.set_title(f"{run_prefix} BHHM")
        axis.grid(alpha=0.2, linewidth=0.5)

    mzr_axis = axes[4]
    mzr_axis.errorbar(
        blanc["mass_log10"],
        blanc["mean"],
        yerr=np.vstack([blanc["lo"], blanc["hi"]]),
        fmt="o",
        color="#555555",
        markersize=3,
        linewidth=0.8,
        alpha=0.7,
        label="Blanc 2019",
    )
    mzr_axis.plot(blanc["mass_log10"], blanc["mean"], color="#999999", linewidth=1.2, alpha=0.8)
    for key, spec in map_thetas.items():
        theta = np.array([spec["theta"][name] for name in input_columns], dtype=float)
        mass_pred = np.array(
            [
                _predict_gp_model(mzr_models[f"z0_mass_metallicity_mass_stellar_log10_{index}"], theta, parameter_specs)
                for index in (1, 2)
            ],
            dtype=float,
        )
        oxygen_pred = np.array(
            [
                _predict_gp_model(mzr_models[f"z0_mass_metallicity_abundance_oxygen_12logoh_{index}"], theta, parameter_specs)
                for index in (1, 2)
            ],
            dtype=float,
        )
        mzr_axis.plot(
            mass_pred,
            oxygen_pred,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=spec["color"],
            label=spec["label"],
        )
    mzr_axis.set_xlabel(r"$\langle \log_{10} M_\star \rangle$")
    mzr_axis.set_ylabel(r"$12 + \log_{10}(\mathrm{O/H})$")
    mzr_axis.set_title("z=0 Mass-Metallicity")
    mzr_axis.grid(alpha=0.2, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
