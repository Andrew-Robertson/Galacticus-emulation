from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .gp import fit_scaled_gp, predict_scaled_gp
from .persistence import load_emulator_bundle


DEFAULT_ANALYSIS = "massFunctionStellarTomczak2014ZFOURGEz0"
DEFAULT_INPUT_COLUMN = "diskVelocityCharacteristic"
DEFAULT_HDF5_FILENAME = "galacticus_reduced.hdf5"


def load_smf_campaign(
    campaign_root: str | Path,
    *,
    analysis: str = DEFAULT_ANALYSIS,
    input_columns: list[str] | None = None,
    hdf5_filename: str = DEFAULT_HDF5_FILENAME,
) -> dict[str, np.ndarray | pd.DataFrame]:
    campaign_root = Path(campaign_root)
    if input_columns is None:
        input_columns = [DEFAULT_INPUT_COLUMN]
    samples = pd.read_csv(campaign_root / "samples.csv")
    x = samples[input_columns].to_numpy(dtype=float)
    mass_bins: np.ndarray | None = None
    target_linear: np.ndarray | None = None
    target_covariance: np.ndarray | None = None
    rows: list[np.ndarray] = []
    for evaluation_id in samples["evaluation_id"]:
        path = campaign_root / "evaluations" / evaluation_id / hdf5_filename
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            group = handle[f"/analyses/{analysis}"]
            current_mass_bins = np.asarray(group["massStellar"][...], dtype=float)
            current_target = np.asarray(group["massFunctionTarget"][...], dtype=float)
            current_target_cov = np.asarray(group["massFunctionCovarianceTarget"][...], dtype=float)
            current_mass_function = np.asarray(group["massFunction"][...], dtype=float)
        if mass_bins is None:
            mass_bins = current_mass_bins
            target_linear = current_target
            target_covariance = current_target_cov
        else:
            if not np.allclose(current_mass_bins, mass_bins):
                raise ValueError(f"Mass bins differ for {evaluation_id}")
            if not np.allclose(current_target, target_linear):
                raise ValueError(f"Target SMF differs for {evaluation_id}")
            if not np.allclose(current_target_cov, target_covariance):
                raise ValueError(f"Target SMF covariance differs for {evaluation_id}")
        rows.append(current_mass_function)
    assert mass_bins is not None
    assert target_linear is not None
    assert target_covariance is not None
    y_linear = np.vstack(rows)
    if np.any(y_linear <= 0.0):
        raise ValueError("SMF contains non-positive values; choose a floor before taking log10.")
    if np.any(target_linear <= 0.0):
        raise ValueError("Target SMF contains non-positive values; choose a floor before taking log10.")
    target_sigma_linear = np.sqrt(np.maximum(np.diag(target_covariance), 0.0))
    target_sigma_log10 = np.maximum(target_sigma_linear / (target_linear * np.log(10.0)), 1.0e-6)
    return {
        "samples": samples,
        "x": x,
        "mass_bins": mass_bins,
        "mass_bins_log10": np.log10(mass_bins),
        "y_linear": y_linear,
        "y_log10": np.log10(y_linear),
        "target_linear": target_linear,
        "target_log10": np.log10(target_linear),
        "target_sigma_linear": target_sigma_linear,
        "target_sigma_log10": target_sigma_log10,
    }


def train_smf_bundle(
    campaign_root: str | Path,
    *,
    analysis: str = DEFAULT_ANALYSIS,
    input_columns: list[str] | None = None,
    hdf5_filename: str = DEFAULT_HDF5_FILENAME,
    n_restarts_optimizer: int = 5,
) -> dict:
    loaded = load_smf_campaign(
        campaign_root,
        analysis=analysis,
        input_columns=input_columns,
        hdf5_filename=hdf5_filename,
    )
    samples: pd.DataFrame = loaded["samples"]  # type: ignore[assignment]
    x: np.ndarray = loaded["x"]  # type: ignore[assignment]
    y_log10: np.ndarray = loaded["y_log10"]  # type: ignore[assignment]

    models = []
    y_means = []
    y_stds = []
    kernels = []
    for bin_index in range(y_log10.shape[1]):
        model, y_mean, y_std = fit_scaled_gp(
            x,
            y_log10[:, bin_index],
            n_restarts_optimizer=n_restarts_optimizer,
        )
        models.append(model)
        y_means.append(y_mean)
        y_stds.append(y_std)
        kernels.append(str(model.kernel_))

    input_columns_actual = input_columns or [DEFAULT_INPUT_COLUMN]
    input_ranges = {
        column: {
            "min": float(samples[column].min()),
            "max": float(samples[column].max()),
            "default": float(samples[column].median()),
        }
        for column in input_columns_actual
    }
    y_all = np.concatenate(
        [
            loaded["y_log10"].ravel(),  # type: ignore[index]
            loaded["target_log10"].ravel(),  # type: ignore[index]
        ]
    )
    bundle = {
        "bundle_type": "interactive_smf_bin_by_bin_gp",
        "campaign_root": str(Path(campaign_root).resolve()),
        "analysis": analysis,
        "input_columns": input_columns_actual,
        "input_ranges": input_ranges,
        "evaluation_ids": samples["evaluation_id"].tolist(),
        "x_train": x,
        "mass_bins": loaded["mass_bins"],
        "mass_bins_log10": loaded["mass_bins_log10"],
        "y_train_linear": loaded["y_linear"],
        "y_train_log10": loaded["y_log10"],
        "target_linear": loaded["target_linear"],
        "target_log10": loaded["target_log10"],
        "target_sigma_linear": loaded["target_sigma_linear"],
        "target_sigma_log10": loaded["target_sigma_log10"],
        "models": models,
        "y_means": np.asarray(y_means, dtype=float),
        "y_stds": np.asarray(y_stds, dtype=float),
        "kernels": kernels,
        "n_restarts_optimizer": int(n_restarts_optimizer),
        "x_axis_label": r"$\log_{10}(M_\star/M_\odot)$",
        "y_axis_label": r"$\log_{10}(\Phi/\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$",
        "observable_label": r"Tomczak+14 SMF $0.20<z<0.50$",
        "y_plot_min": float(np.floor(np.min(y_all) - 0.25)),
        "y_plot_max": float(np.ceil(np.max(y_all) + 0.25)),
    }
    return bundle


def predict_smf_bundle(bundle: dict, params: dict[str, float]) -> dict[str, np.ndarray]:
    x = np.asarray([[float(params[column]) for column in bundle["input_columns"]]], dtype=float)
    y_pred = np.zeros(len(bundle["models"]), dtype=float)
    y_std = np.zeros_like(y_pred)
    for index, model in enumerate(bundle["models"]):
        pred, pred_std = predict_scaled_gp(
            model,
            float(bundle["y_means"][index]),
            float(bundle["y_stds"][index]),
            x,
        )
        y_pred[index] = float(pred[0])
        y_std[index] = float(pred_std[0])
    return {
        "y_pred_log10": y_pred,
        "y_std_log10": y_std,
        "y_pred_linear": 10.0 ** y_pred,
        "y_lower_linear": 10.0 ** (y_pred - y_std),
        "y_upper_linear": 10.0 ** (y_pred + y_std),
    }


def bundle_meta(bundle: dict) -> dict:
    return {
        "bundle_type": bundle["bundle_type"],
        "analysis": bundle["analysis"],
        "input_columns": bundle["input_columns"],
        "input_ranges": bundle["input_ranges"],
        "mass_bins_log10": np.asarray(bundle["mass_bins_log10"], dtype=float).tolist(),
        "y_train_log10": np.asarray(bundle["y_train_log10"], dtype=float).tolist(),
        "target_log10": np.asarray(bundle["target_log10"], dtype=float).tolist(),
        "target_sigma_log10": np.asarray(bundle["target_sigma_log10"], dtype=float).tolist(),
        "x_axis_label": bundle["x_axis_label"],
        "y_axis_label": bundle["y_axis_label"],
        "observable_label": bundle["observable_label"],
        "y_plot_min": float(bundle["y_plot_min"]),
        "y_plot_max": float(bundle["y_plot_max"]),
    }


def load_smf_bundle(path: str | Path) -> dict:
    bundle = load_emulator_bundle(path)
    if bundle.get("bundle_type") != "interactive_smf_bin_by_bin_gp":
        raise ValueError(f"Unexpected bundle_type: {bundle.get('bundle_type')}")
    return bundle
