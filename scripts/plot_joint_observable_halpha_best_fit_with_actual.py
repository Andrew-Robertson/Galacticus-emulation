from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import h5py
import joblib
import numpy as np
import pandas as pd

from galacticus_emu.observable_plot_metadata import DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET
from galacticus_emu.lhs import transform_to_prior_quantiles
from run_joint_observable_halpha_emulator_mcmc import (
    ALL_PARAMETER_SPECS,
    HALPHA_ORDER,
    SLOW_DIM,
    _plot_best_fit_halpha,
    _plot_joint_best_fit_observables,
)
from run_mass_function_observable_emulator_mcmc import (
    LIKELIHOOD_CASES,
    _load_analysis_bundle,
    _log10_normal_to_linear_moments,
    _pca_component_count,
    _parse_pca_component_overrides,
    _predict_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay actual Galacticus best-fit results on the joint observable+Halpha emulator best-fit figures."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--actual-hdf5", type=Path, required=True)
    parser.add_argument("--actual-halpha-long-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", default="joint_observable_halpha_draws2")
    parser.add_argument("--figures-dir-name", default="figures_joint_observable_halpha_mcmc_draws2")
    return parser.parse_args()


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_actual_observables(actual_hdf5: Path, observable_bundles: list[dict[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    by_analysis = {bundle["analysis_key"]: bundle["analysis"] for bundle in observable_bundles}
    results: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(actual_hdf5, "r") as handle:
        for bundle in observable_bundles:
            analysis = bundle["analysis"]
            group = handle[f"/analyses/{analysis}"]
            attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
            x_values = np.asarray(group[attrs["xDataset"]][...], dtype=float)
            y_values = np.asarray(group[attrs["yDataset"]][...], dtype=float)
            if bool(attrs.get("xAxisIsLog", False)):
                x_values = np.log10(x_values)
            results[bundle["analysis_key"]] = {
                "x_plot": x_values,
                "y_value": y_values,
            }
    return results


def _load_actual_halpha(long_csv: Path) -> dict[str, dict[str, np.ndarray]]:
    table = pd.read_csv(long_csv)
    results: dict[str, dict[str, np.ndarray]] = {}
    for label in HALPHA_ORDER:
        sub = table[table["sobral_label"] == label].sort_values("bin_index")
        if sub.empty:
            continue
        results[label] = {
            "x_plot": sub["log10_luminosity_center"].to_numpy(dtype=float),
            "y_plot": (
                np.log10(np.maximum(sub["dn_dlnL_mpc3"].to_numpy(dtype=float), 1.0e-30))
                + DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET
            ),
        }
    return results


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    summary = json.loads(args.run_summary.expanduser().resolve().read_text())
    fitted_bundle_path = args.run_summary.expanduser().resolve().with_name(
        args.run_summary.expanduser().resolve().name.replace("_run_summary.json", "_fitted_emulators.joblib")
    )
    fitted_bundle = joblib.load(fitted_bundle_path) if fitted_bundle_path.exists() else None

    best_theta = np.array([summary["best_theta"][spec.short_name] for spec in ALL_PARAMETER_SPECS], dtype=float)
    best_theta_quantiles = transform_to_prior_quantiles(ALL_PARAMETER_SPECS, best_theta[None, :])[0]

    if fitted_bundle is not None:
        observable_bundles = fitted_bundle["observable_bundles"]
        halpha_bundles = fitted_bundle["halpha_bundles"]
    else:
        pca_component_overrides = _parse_pca_component_overrides(
            [f"{key}={value}" for key, value in summary.get("pca_component_overrides", {}).items()]
        )
        observable_keys = LIKELIHOOD_CASES[summary["observable_likelihood_case"]]
        observable_bundles = [
            _load_analysis_bundle(
                campaign_root,
                key,
                "galacticus_reduced.hdf5",
                n_restarts_optimizer=summary["n_restarts_optimizer"],
                emulator_mode=summary["observable_emulator_mode"],
                pca_scaling=summary["pca_scaling"],
                pca_components=_pca_component_count(
                    key,
                    default_pca_components=summary["default_pca_components"],
                    overrides=pca_component_overrides,
                ),
            )
            for key in observable_keys
        ]

        halpha_bundles = []
        for label in HALPHA_ORDER:
            path_text = summary["halpha_bundles"].get(label)
            if path_text is None:
                continue
            halpha_bundles.append(joblib.load(Path(path_text)))

    prediction_mean_by_key = {}
    prediction_std_by_key = {}
    for bundle in observable_bundles:
        pred, pred_std = _predict_bundle(bundle, best_theta_quantiles[:SLOW_DIM][None, :])
        if bundle["y_is_log"]:
            mean_linear, var_linear = _log10_normal_to_linear_moments(pred, pred_std**2)
        else:
            mean_linear, var_linear = pred, pred_std**2
        prediction_mean_by_key[bundle["analysis_key"]] = mean_linear[0]
        prediction_std_by_key[bundle["analysis_key"]] = np.sqrt(np.maximum(var_linear[0], 0.0))

    actual_observables = _load_actual_observables(args.actual_hdf5.expanduser().resolve(), observable_bundles)
    actual_halpha = _load_actual_halpha(args.actual_halpha_long_csv.expanduser().resolve())

    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    observable_path = figures_root / f"{args.output_prefix}_best_fit_observables_with_actual.png"
    _plot_joint_best_fit_observables(
        observable_bundles,
        prediction_mean_by_key,
        prediction_std_by_key,
        best_theta_quantiles[:SLOW_DIM],
        actual_observables,
        observable_path,
    )

    halpha_path = figures_root / f"{args.output_prefix}_best_fit_halpha_with_actual.png"
    _plot_best_fit_halpha(
        halpha_bundles,
        best_theta_quantiles,
        actual_halpha,
        halpha_path,
    )

    print(observable_path)
    print(halpha_path)


if __name__ == "__main__":
    main()
