from __future__ import annotations

import numpy as np


DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET = float(np.log10(np.log(10.0)))
HALPHA_LF_Y_DISPLAY_OFFSET = 0.0
HALPHA_LF_X_AXIS_LABEL = r"$\log_{10}(L_{\mathrm{H}\alpha}/\mathrm{erg\,s}^{-1})$"
HALPHA_LF_Y_AXIS_LABEL = r"$\log_{10}(\Phi/\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$"

SOBRAL_HALPHA_PLOT_METADATA = {
    "Z1": {
        "label": r"H$\alpha$ luminosity function, $z \approx 0.40$",
        "target_label": "Sobral et al. (2013)",
    },
    "Z2": {
        "label": r"H$\alpha$ luminosity function, $z \approx 0.84$",
        "target_label": "Sobral et al. (2013)",
    },
    "Z3": {
        "label": r"H$\alpha$ luminosity function, $z \approx 1.47$",
        "target_label": "Sobral et al. (2013)",
    },
    "Z4": {
        "label": r"H$\alpha$ luminosity function, $z \approx 2.23$",
        "target_label": "Sobral et al. (2013)",
    },
}

# Paper-facing display metadata for the standard-observable panels. These values
# intentionally override the raw Galacticus/HDF5 axis strings when plotting.
STANDARD_OBSERVABLE_PLOT_METADATA = {
    "smf_z0": {
        "label": r"Stellar mass function, $0.20<z<0.50$",
        "target_label": "Tomczak et al. (2014)",
        "x_axis_label": r"$\log_{10}(M_\star/M_\odot)$",
        "y_axis_label": r"$\log_{10}[(\mathrm{d}n/\mathrm{d}\log_{10} M_\star)/\mathrm{Mpc}^{-3}]$",
        "y_display_offset": DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET,
    },
    "smf_z3": {
        "label": r"Stellar mass function, $1.00<z<1.25$",
        "target_label": "Tomczak et al. (2014)",
        "x_axis_label": r"$\log_{10}(M_\star/M_\odot)$",
        "y_axis_label": r"$\log_{10}[(\mathrm{d}n/\mathrm{d}\log_{10} M_\star)/\mathrm{Mpc}^{-3}]$",
        "y_display_offset": DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET,
    },
    "sfr_function_robotham2011": {
        "label": r"Star formation rate function, $0.013<z<0.1$",
        "target_label": "Robotham et al. (2011)",
        "x_axis_label": r"$\log_{10}(\dot{M}_\star/(M_\odot\,\mathrm{Gyr}^{-1}))$",
        "y_axis_label": r"$\log_{10}[(\mathrm{d}n/\mathrm{d}\log_{10}\dot{M}_\star)/\mathrm{Mpc}^{-3}]$",
        "y_display_offset": DENSITY_PER_NATURAL_LOG_TO_PER_DEX_OFFSET,
    },
    "size_mass_vdw2014_star_forming_z0": {
        "label": r"Star-forming galaxy sizes, $0<z<0.5$",
        "target_label": "van der Wel et al. (2014)",
        "x_axis_label": r"$\log_{10}(M_\star/M_\odot)$",
        "y_axis_label": r"$\log_{10}(R_{\mathrm{eff}}/\mathrm{kpc})$",
        "y_display_offset": 3.0,
    },
    "size_mass_vdw2014_quiescent_z0": {
        "label": r"Quiescent galaxy sizes, $0<z<0.5$",
        "target_label": "van der Wel et al. (2014)",
        "x_axis_label": r"$\log_{10}(M_\star/M_\odot)$",
        "y_axis_label": r"$\log_{10}(R_{\mathrm{eff}}/\mathrm{kpc})$",
        "y_display_offset": 3.0,
    },
    "bh_velocity_dispersion": {
        "label": "Black hole mass–velocity dispersion",
        "target_label": "McConnell & Ma (2013)",
        "x_axis_label": r"$\log_{10}(\sigma_{\star,\mathrm{spheroid}}/\mathrm{km\,s}^{-1})$",
        "y_axis_label": r"$\log_{10}(M_{\mathrm{BH}}/M_\odot)$",
    },
    "mzr_blanc2019": {
        "label": "Mass–metallicity relation",
        "target_label": "Blanc et al. (2019)",
        "x_axis_label": r"$\log_{10}(M_\star/M_\odot)$",
        "y_axis_label": r"$12+\log_{10}(\mathrm{O/H})$",
    },
}


def y_display_offset(observable: dict) -> float:
    return float(observable.get("y_display_offset", 0.0))


def standard_observable_y_display_offset(observable_key: str) -> float:
    return float(STANDARD_OBSERVABLE_PLOT_METADATA.get(str(observable_key), {}).get("y_display_offset", 0.0))


def apply_y_display_offset(observable: dict, values):
    offset = y_display_offset(observable)
    array = np.asarray(values, dtype=float)
    if offset == 0.0:
        return array
    return array + offset


def normalized_sobral_label(label: str | None) -> str | None:
    if label is None:
        return None
    value = str(label).strip().upper()
    return value if value in SOBRAL_HALPHA_PLOT_METADATA else None


def is_halpha_sobral_observable(observable: str | dict | None) -> bool:
    if isinstance(observable, dict):
        observable = observable.get("observable") or observable.get("observable_key")
    return str(observable or "").startswith("halpha_sobral")


def sidecar_halpha_sobral_plot_metadata(sample_label: str | None) -> dict:
    label = normalized_sobral_label(sample_label)
    if label is None:
        return {}
    return SOBRAL_HALPHA_PLOT_METADATA[label]


def sidecar_lf_axis_label(observable: str | dict | None, sample_label: str | None, fallback: str) -> str:
    if is_halpha_sobral_observable(observable):
        return sidecar_halpha_sobral_plot_metadata(sample_label).get("label", fallback)
    if isinstance(observable, dict):
        return str(observable.get("label") or fallback)
    return fallback


def sidecar_lf_target_label(observable: str | dict | None, sample_label: str | None, fallback: str = "target") -> str:
    if is_halpha_sobral_observable(observable):
        return sidecar_halpha_sobral_plot_metadata(sample_label).get("target_label", fallback)
    return fallback


def sidecar_lf_y_axis_label(observable: str | dict | None) -> str:
    if is_halpha_sobral_observable(observable):
        return HALPHA_LF_Y_AXIS_LABEL
    return r"$\log_{10}\Phi$"


def sidecar_lf_y_display_offset(observable: str | dict | None) -> float:
    return HALPHA_LF_Y_DISPLAY_OFFSET if is_halpha_sobral_observable(observable) else 0.0


def apply_sidecar_lf_y_display_offset(observable: str | dict | None, values):
    array = np.asarray(values, dtype=float)
    offset = sidecar_lf_y_display_offset(observable)
    if offset == 0.0:
        return array
    return array + offset


def apply_sidecar_lf_plot_metadata_overrides(bundle: dict) -> dict:
    for observable_key, observable in bundle.get("observables", {}).items():
        if not is_halpha_sobral_observable(observable):
            continue
        sample_label = str(observable.get("sample_label", "")).upper()
        observable["label"] = sidecar_lf_axis_label(observable, sample_label, observable_key)
        observable["target_label"] = sidecar_lf_target_label(observable, sample_label)
        observable["x_axis_label"] = HALPHA_LF_X_AXIS_LABEL
        observable["y_axis_label"] = HALPHA_LF_Y_AXIS_LABEL
        observable["y_display_offset"] = HALPHA_LF_Y_DISPLAY_OFFSET
    return bundle


def apply_observable_plot_metadata_overrides(bundle: dict) -> dict:
    for observable_key, metadata in STANDARD_OBSERVABLE_PLOT_METADATA.items():
        observable = bundle.get("observables", {}).get(observable_key)
        if observable is not None:
            observable.update(metadata)
            desired_offset = y_display_offset(observable)
            applied_offset = float(observable.get("_y_display_offset_applied", 0.0))
            delta_offset = desired_offset - applied_offset
            if delta_offset != 0.0:
                for field in ("target_plot", "y_train_preview"):
                    if observable.get(field) is not None:
                        observable[field] = np.asarray(observable[field], dtype=float) + delta_offset
                for field in ("y_plot_min", "y_plot_max"):
                    if field in observable and np.isfinite(float(observable[field])):
                        observable[field] = float(observable[field]) + delta_offset
                observable["_y_display_offset_applied"] = desired_offset
        observable_config = bundle.get("observable_configs", {}).get(observable_key)
        if observable_config is not None and "label" in metadata:
            observable_config["label"] = metadata["label"]
    return bundle
