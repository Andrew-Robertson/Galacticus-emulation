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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galacticus_emu.observable_plot_metadata import (
    HALPHA_LF_X_AXIS_LABEL,
    apply_observable_plot_metadata_overrides,
    sidecar_lf_axis_label,
    sidecar_lf_target_label,
    sidecar_lf_y_axis_label,
    standard_observable_y_display_offset,
)
from galacticus_emu.plotting import set_ylim_from_values
from plot_interactive_observables_mcmc_best_fit_overlay import (
    _axis_metadata,
    _read_galacticus_predictions,
)
from process_halpha_dust_evaluation import _compute_case_lfs as _compute_halpha_case_lfs


DUST_PARAMETER_NAMES = ["delta_0", "delta_z", "delta_M", "delta_Mz", "attenuation_scatter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay an actual Galacticus MAP run on standard-observable and Halpha-sidecar "
            "best-fit emulator figures from a joint MCMC."
        )
    )
    parser.add_argument("--standard-bundle-meta", type=Path, required=True)
    parser.add_argument("--standard-predictions", type=Path, required=True)
    parser.add_argument("--sidecar-predictions", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--actual-hdf5", type=Path, required=True)
    parser.add_argument(
        "--extra-actual-hdf5",
        action="append",
        default=[],
        help="Additional actual Galacticus overlay for standard panels, as LABEL=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--actual-sidecar-dir",
        type=Path,
        default=None,
        help=(
            "Optional emission_line_dust sidecar directory for the actual MAP run. "
            "When supplied, sidecar LF overlays are read from this directory instead "
            "of being recomputed from --actual-hdf5."
        ),
    )
    parser.add_argument(
        "--actual-sidecar-case",
        default=None,
        help="Optional dust_case label to select from --actual-sidecar-dir.",
    )
    parser.add_argument(
        "--extra-actual-sidecar-dir",
        action="append",
        default=[],
        help="Additional actual sidecar overlay for LF panels, as LABEL=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--posterior-draw-runs-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing posterior_draw_XX actual Galacticus runs. Thin low-alpha "
            "overlays are drawn from each romanEPS_massFunction_reduced.hdf5 and "
            "emission_line_dust sidecar without one legend entry per draw."
        ),
    )
    parser.add_argument("--standard-output-path", type=Path, required=True)
    parser.add_argument("--sidecar-output-path", type=Path, required=True)
    parser.add_argument("--actual-label", default="Galacticus MAP")
    parser.add_argument("--min-log10-lf", type=float, default=-8.0)
    parser.add_argument("--z-pivot", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Expected non-empty LABEL in {value!r}")
    return label, Path(path).expanduser().resolve()


def _read_standard_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "observable_key",
        "x_plot",
        "target_plot",
        "target_sigma_plot",
        "prediction_plot",
        "prediction_sigma_plot",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing standard prediction column(s): {missing}")
    return frame


def _read_sidecar_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "observable_key",
        "x_plot",
        "target_log10_phi",
        "target_log10_phi_std",
        "prediction_log10_phi",
        "prediction_log10_phi_std",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing sidecar prediction column(s): {missing}")
    return frame


def _actual_halpha_predictions(
    actual_hdf5: Path,
    run_summary: dict,
    *,
    min_log10_lf: float,
    z_pivot: float,
) -> dict[str, pd.DataFrame]:
    best_theta = run_summary["best_theta"]
    missing = [name for name in DUST_PARAMETER_NAMES if name not in best_theta]
    if missing:
        raise ValueError(f"{run_summary} does not provide MAP dust parameter(s): {missing}")
    dust_params = {name: float(best_theta[name]) for name in DUST_PARAMETER_NAMES}
    dust_params["z_pivot"] = float(z_pivot)
    dust_case = {
        "label": "actual_map",
        "dust_model": "gb10_generalised",
        "dust_params": dust_params,
        "dust_law": "calzetti",
        "random_uniform_index": None,
        "scatter_mode": "expected",
    }
    rows = pd.DataFrame(_compute_halpha_case_lfs(actual_hdf5, dust_case))
    predictions = {}
    for sobral_label, group in rows.groupby("sobral_label"):
        ordered = group.sort_values("bin_index")
        phi = ordered["dn_dlnL_mpc3"].to_numpy(dtype=float) * np.log(10.0)
        predictions[f"halpha_sobral_{str(sobral_label).lower()}"] = pd.DataFrame(
            {
                "x_plot": ordered["log10_luminosity_center"].to_numpy(dtype=float),
                "prediction_log10_phi": np.log10(np.maximum(phi, 10.0**min_log10_lf)),
            }
        )
    return predictions


def _observable_key(observable: str, sample_label: str) -> str:
    return f"{observable}_{sample_label.lower().replace('.', 'p')}"


def _actual_sidecar_predictions(
    sidecar_dir: Path,
    *,
    min_log10_lf: float,
    dust_case_label: str | None,
) -> dict[str, pd.DataFrame]:
    path = sidecar_dir / "emission_line_dust_lf_long.csv"
    frame = pd.read_csv(path)
    required = {"observable", "sample_label", "log10_luminosity_center"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing sidecar LF column(s): {missing}")
    if "dn_dlnL_mpc3" not in frame.columns and "phi_mpc3_dex" not in frame.columns:
        raise ValueError(f"{path} must contain dn_dlnL_mpc3 or phi_mpc3_dex")

    if "dust_case" in frame.columns:
        cases = sorted(frame["dust_case"].astype(str).unique())
        if dust_case_label is None:
            if len(cases) != 1:
                raise ValueError(
                    f"{path} contains multiple dust_case values {cases}; pass --actual-sidecar-case"
                )
            dust_case_label = cases[0]
        frame = frame.loc[frame["dust_case"].astype(str) == str(dust_case_label)].copy()
        if frame.empty:
            raise ValueError(f"{path} contains no rows for dust_case={dust_case_label!r}")

    predictions = {}
    for (observable, sample_label), group in frame.groupby(["observable", "sample_label"]):
        ordered = group.sort_values("bin_index" if "bin_index" in group.columns else "log10_luminosity_center")
        if "phi_mpc3_dex" in ordered:
            phi = ordered["phi_mpc3_dex"].to_numpy(dtype=float)
        else:
            phi = ordered["dn_dlnL_mpc3"].to_numpy(dtype=float) * np.log(10.0)
        predictions[_observable_key(str(observable), str(sample_label))] = pd.DataFrame(
            {
                "x_plot": ordered["log10_luminosity_center"].to_numpy(dtype=float),
                "prediction_log10_phi": np.log10(np.maximum(phi, 10.0**min_log10_lf)),
            }
        )
    return predictions


def _posterior_draw_run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.expanduser().resolve().glob("posterior_draw_[0-9][0-9]") if path.is_dir())


def _sidecar_axis_label(rows: pd.DataFrame, observable_key: str) -> str:
    observable = str(rows["observable"].iloc[0]) if "observable" in rows else "sidecar LF"
    sample_label = str(rows["sample_label"].iloc[0]) if "sample_label" in rows else observable_key
    return sidecar_lf_axis_label(observable, sample_label, f"{observable} {sample_label}")


def _plot_standard(
    meta: dict,
    predictions: pd.DataFrame,
    actual_overlays: list[tuple[str | None, dict[str, pd.DataFrame], str, str, float, float, int]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    observable_keys = list(dict.fromkeys(predictions["observable_key"].astype(str)))
    ncols = 2 if len(observable_keys) > 1 else 1
    nrows = int(np.ceil(len(observable_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, observable_key in zip(axes_flat, observable_keys, strict=False):
        rows = predictions[predictions["observable_key"] == observable_key].copy().sort_values("x_plot")
        y_offset = standard_observable_y_display_offset(observable_key)
        x = rows["x_plot"].to_numpy(dtype=float)
        target = rows["target_plot"].to_numpy(dtype=float) + y_offset
        target_sigma = rows["target_sigma_plot"].to_numpy(dtype=float)
        pred = rows["prediction_plot"].to_numpy(dtype=float) + y_offset
        pred_sigma = rows["prediction_sigma_plot"].to_numpy(dtype=float)
        metadata = _axis_metadata(meta, observable_key)
        axis.errorbar(
            x,
            target,
            yerr=target_sigma,
            fmt="o",
            color="0.15",
            label=metadata.get("target_label") or "target",
            zorder=4,
        )
        axis.plot(x, pred, color="tab:blue", lw=1.8, label="emulator MAP")
        axis.fill_between(x, pred - pred_sigma, pred + pred_sigma, color="tab:blue", alpha=0.2)
        actual_values = []
        for label, actual_predictions, linestyle, color, linewidth, alpha, zorder in actual_overlays:
            if observable_key not in actual_predictions:
                continue
            actual = actual_predictions[observable_key].sort_values("x_plot")
            actual_x = actual["x_plot"].to_numpy(dtype=float)
            actual_y = actual["prediction_plot"].to_numpy(dtype=float) + y_offset
            finite = np.isfinite(actual_x) & np.isfinite(actual_y)
            actual_values.append(actual_y)
            axis.plot(
                actual_x[finite],
                actual_y[finite],
                color=color,
                lw=linewidth,
                ls=linestyle,
                alpha=alpha,
                label=label,
                zorder=zorder,
            )
        set_ylim_from_values(axis, target, pred, *actual_values)
        axis.set_title(metadata["label"])
        axis.set_xlabel(metadata["x_axis_label"])
        axis.set_ylabel(metadata["y_axis_label"])
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[len(observable_keys) :]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_sidecar(
    predictions: pd.DataFrame,
    actual_overlays: list[tuple[str | None, dict[str, pd.DataFrame], str, str, float, float, int]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    observable_keys = list(dict.fromkeys(predictions["observable_key"].astype(str)))
    ncols = 2 if len(observable_keys) > 1 else 1
    nrows = int(np.ceil(len(observable_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, observable_key in zip(axes_flat, observable_keys, strict=False):
        rows = predictions[predictions["observable_key"] == observable_key].copy().sort_values("x_plot")
        observable = str(rows["observable"].iloc[0]) if "observable" in rows else observable_key
        sample_label = str(rows["sample_label"].iloc[0]) if "sample_label" in rows else observable_key
        x = rows["x_plot"].to_numpy(dtype=float)
        target = rows["target_log10_phi"].to_numpy(dtype=float)
        target_sigma = rows["target_log10_phi_std"].to_numpy(dtype=float)
        pred = rows["prediction_log10_phi"].to_numpy(dtype=float)
        pred_sigma = rows["prediction_log10_phi_std"].to_numpy(dtype=float)
        axis.errorbar(
            x,
            target,
            yerr=target_sigma,
            fmt="o",
            color="0.15",
            label=sidecar_lf_target_label(observable, sample_label),
            zorder=4,
        )
        axis.plot(x, pred, color="tab:blue", lw=1.8, label="emulator MAP")
        axis.fill_between(x, pred - pred_sigma, pred + pred_sigma, color="tab:blue", alpha=0.2)
        actual_values = []
        for label, actual_predictions, linestyle, color, linewidth, alpha, zorder in actual_overlays:
            if observable_key not in actual_predictions:
                continue
            actual = actual_predictions[observable_key].sort_values("x_plot")
            actual_x = actual["x_plot"].to_numpy(dtype=float)
            actual_y = actual["prediction_log10_phi"].to_numpy(dtype=float)
            finite = np.isfinite(actual_x) & np.isfinite(actual_y)
            actual_values.append(actual_y)
            axis.plot(
                actual_x[finite],
                actual_y[finite],
                color=color,
                lw=linewidth,
                ls=linestyle,
                alpha=alpha,
                label=label,
                zorder=zorder,
            )
        set_ylim_from_values(axis, target, pred, *actual_values)
        axis.set_title(_sidecar_axis_label(rows, observable_key))
        axis.set_xlabel(HALPHA_LF_X_AXIS_LABEL if observable == "halpha_sobral" else r"$\log_{10}(L/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(sidecar_lf_y_axis_label(observable))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes_flat[len(observable_keys) :]:
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    meta = apply_observable_plot_metadata_overrides(
        json.loads(args.standard_bundle_meta.expanduser().resolve().read_text())
    )
    standard_predictions = _read_standard_predictions(args.standard_predictions.expanduser().resolve())
    sidecar_predictions = _read_sidecar_predictions(args.sidecar_predictions.expanduser().resolve())
    run_summary = json.loads(args.run_summary.expanduser().resolve().read_text())
    actual_hdf5 = args.actual_hdf5.expanduser().resolve()

    standard_keys = list(dict.fromkeys(standard_predictions["observable_key"].astype(str)))
    actual_standard = _read_galacticus_predictions(
        actual_hdf5,
        meta,
        standard_keys,
        standard_predictions,
        min_log10_y=args.min_log10_lf,
    )
    standard_overlays = [(args.actual_label, actual_standard, "--", "black", 2.0, 1.0, 6)]
    for extra in args.extra_actual_hdf5:
        label, extra_hdf5 = _parse_label_path(extra)
        extra_standard = _read_galacticus_predictions(
            extra_hdf5,
            meta,
            standard_keys,
            standard_predictions,
            min_log10_y=args.min_log10_lf,
        )
        standard_overlays.append((label, extra_standard, ":", "black", 2.0, 1.0, 6))

    if args.posterior_draw_runs_dir is not None:
        first_label = "actual posterior draws"
        for run_dir in _posterior_draw_run_dirs(args.posterior_draw_runs_dir):
            hdf5_path = run_dir / "romanEPS_massFunction_reduced.hdf5"
            if not hdf5_path.exists():
                continue
            draw_standard = _read_galacticus_predictions(
                hdf5_path,
                meta,
                standard_keys,
                standard_predictions,
                min_log10_y=args.min_log10_lf,
            )
            standard_overlays.append((first_label, draw_standard, "-", "0.35", 0.8, 0.28, 2))
            first_label = None

    if args.actual_sidecar_dir is None:
        actual_halpha = _actual_halpha_predictions(
            actual_hdf5,
            run_summary,
            min_log10_lf=args.min_log10_lf,
            z_pivot=args.z_pivot,
        )
    else:
        actual_halpha = _actual_sidecar_predictions(
            args.actual_sidecar_dir.expanduser().resolve(),
            min_log10_lf=args.min_log10_lf,
            dust_case_label=args.actual_sidecar_case,
        )
    sidecar_overlays = [(args.actual_label, actual_halpha, "--", "black", 2.0, 1.0, 6)]
    for extra in args.extra_actual_sidecar_dir:
        label, extra_sidecar_dir = _parse_label_path(extra)
        sidecar_overlays.append(
            (
                label,
                _actual_sidecar_predictions(
                    extra_sidecar_dir,
                    min_log10_lf=args.min_log10_lf,
                    dust_case_label=args.actual_sidecar_case,
                ),
                ":",
                "black",
                2.0,
                1.0,
                6,
            )
        )

    if args.posterior_draw_runs_dir is not None:
        first_label = "actual posterior draws"
        for run_dir in _posterior_draw_run_dirs(args.posterior_draw_runs_dir):
            sidecar_dir = run_dir / "emission_line_dust"
            if not sidecar_dir.exists():
                continue
            sidecar_overlays.append(
                (
                    first_label,
                    _actual_sidecar_predictions(
                        sidecar_dir,
                        min_log10_lf=args.min_log10_lf,
                        dust_case_label=args.actual_sidecar_case,
                    ),
                    "-",
                    "0.35",
                    0.8,
                    0.28,
                    2,
                )
            )
            first_label = None

    _plot_standard(
        meta,
        standard_predictions,
        standard_overlays,
        args.standard_output_path.expanduser().resolve(),
        dpi=args.dpi,
    )
    _plot_sidecar(
        sidecar_predictions,
        sidecar_overlays,
        args.sidecar_output_path.expanduser().resolve(),
        dpi=args.dpi,
    )
    print(args.standard_output_path)
    print(args.sidecar_output_path)


if __name__ == "__main__":
    main()
