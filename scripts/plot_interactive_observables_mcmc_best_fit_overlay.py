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
import h5py

from galacticus_emu.observable_plot_metadata import (
    apply_observable_plot_metadata_overrides,
    standard_observable_y_display_offset,
)
from galacticus_emu.plotting import set_ylim_from_values
from plot_observable_mcmc_overlay_getdist import _style_for_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay subset-MCMC best-fit observable predictions on the combined-fit panels."
    )
    parser.add_argument("--bundle-meta", type=Path, required=True)
    parser.add_argument(
        "--combined",
        required=True,
        help="Combined fit in label=best_fit_observables.csv form.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
        help="Subset fit in label=best_fit_observables.csv form. Repeat for multiple.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--galacticus-hdf5",
        type=Path,
        default=None,
        help="Optional Galacticus HDF5 file whose outputAnalysis results should be overplotted.",
    )
    parser.add_argument("--galacticus-label", default="Galacticus MAP")
    parser.add_argument("--galacticus-min-log10-y", type=float, default=-8.0)
    return parser.parse_args()


def _parse_labeled_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected label=path, got {text!r}")
    label, path_text = text.split("=", 1)
    return label.strip(), Path(path_text.strip()).expanduser().resolve()


def _read_bundle_meta(path: Path) -> dict:
    path = path.expanduser().resolve()
    candidate_paths = [path]
    if path.suffix != ".json":
        candidate_paths.append(path.with_suffix(".meta.json"))

    errors = []
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            errors.append(f"{candidate_path}: does not exist")
            continue
        try:
            return apply_observable_plot_metadata_overrides(json.loads(candidate_path.read_text()))
        except UnicodeDecodeError as error:
            errors.append(
                f"{candidate_path}: is not a UTF-8 JSON file; if this is a .joblib bundle, "
                "pass the adjacent .meta.json file or keep this path and ensure it exists"
            )
        except json.JSONDecodeError as error:
            errors.append(f"{candidate_path}: invalid JSON ({error})")

    raise ValueError("Could not read bundle metadata:\n  " + "\n  ".join(errors))


def _read_predictions(path: Path) -> pd.DataFrame:
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
        raise ValueError(f"{path} is missing required column(s): {missing}")
    return frame


def _axis_metadata(meta: dict, observable_key: str) -> dict[str, str]:
    observable = meta["observables"].get(observable_key, {})
    return {
        "label": observable.get("label", observable_key),
        "x_axis_label": observable.get("x_axis_label", "x"),
        "y_axis_label": observable.get("y_axis_label", "y"),
        "target_label": observable.get("target_label", "target"),
    }


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _transform_x(values: np.ndarray, *, is_log: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not is_log:
        return values
    result = np.full(values.shape, np.nan, dtype=float)
    mask = values > 0.0
    result[mask] = np.log10(values[mask])
    return result


def _transform_y(values: np.ndarray, *, is_log: bool, min_log10_y: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not is_log:
        return values
    result = np.full(values.shape, np.nan, dtype=float)
    mask = values > 0.0
    result[mask] = np.maximum(np.log10(values[mask]), min_log10_y)
    return result


def _align_to_reference(x_actual: np.ndarray, y_actual: np.ndarray, x_reference: np.ndarray) -> np.ndarray:
    x_actual = np.asarray(x_actual, dtype=float)
    y_actual = np.asarray(y_actual, dtype=float)
    x_reference = np.asarray(x_reference, dtype=float)
    if x_actual.shape == x_reference.shape and np.allclose(x_actual, x_reference, rtol=1.0e-7, atol=1.0e-10, equal_nan=True):
        return y_actual

    aligned = np.full(x_reference.shape, np.nan, dtype=float)
    finite_x = np.isfinite(x_actual)
    for index, x_value in enumerate(x_reference):
        if not np.isfinite(x_value):
            continue
        matches = np.where(finite_x & np.isclose(x_actual, x_value, rtol=1.0e-7, atol=1.0e-10))[0]
        if matches.size:
            aligned[index] = y_actual[matches[0]]

    missing = ~np.isfinite(aligned) & np.isfinite(x_reference)
    finite_for_interp = np.isfinite(x_actual) & np.isfinite(y_actual)
    if np.any(missing) and np.count_nonzero(finite_for_interp) >= 2:
        order = np.argsort(x_actual[finite_for_interp])
        x_sorted = x_actual[finite_for_interp][order]
        y_sorted = y_actual[finite_for_interp][order]
        in_range = missing & (x_reference >= x_sorted[0]) & (x_reference <= x_sorted[-1])
        aligned[in_range] = np.interp(x_reference[in_range], x_sorted, y_sorted)
    return aligned


def _read_galacticus_predictions(
    hdf5_path: Path,
    meta: dict,
    observable_keys: list[str],
    combined: pd.DataFrame,
    *,
    min_log10_y: float,
    apply_display_offsets: bool = False,
) -> dict[str, pd.DataFrame]:
    predictions = {}
    with h5py.File(hdf5_path, "r") as handle:
        for observable_key in observable_keys:
            observable = meta["observables"].get(observable_key, {})
            analysis = observable.get("analysis")
            if not analysis or f"/analyses/{analysis}" not in handle:
                continue
            group = handle[f"/analyses/{analysis}"]
            attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
            x_dataset = attrs["xDataset"]
            y_dataset = attrs["yDataset"]
            x_actual = _transform_x(group[x_dataset][...], is_log=bool(attrs.get("xAxisIsLog", False)))
            y_actual = _transform_y(
                group[y_dataset][...],
                is_log=bool(attrs.get("yAxisIsLog", False)),
                min_log10_y=min_log10_y,
            )
            if apply_display_offsets:
                y_actual = y_actual + standard_observable_y_display_offset(observable_key)
            rows = combined.loc[combined["observable_key"] == observable_key].copy().sort_values("x_plot")
            x_reference = rows["x_plot"].to_numpy(dtype=float)
            predictions[observable_key] = pd.DataFrame(
                {
                    "observable_key": observable_key,
                    "x_plot": x_reference,
                    "prediction_plot": _align_to_reference(x_actual, y_actual, x_reference),
                }
            )
    return predictions


def _plot_case(axis, frame: pd.DataFrame, observable_key: str, *, label: str, color: str, linewidth: float) -> None:
    rows = frame.loc[frame["observable_key"] == observable_key].copy()
    if rows.empty:
        return
    rows = rows.sort_values("x_plot")
    x = rows["x_plot"].to_numpy(dtype=float)
    y = rows["prediction_plot"].to_numpy(dtype=float)
    y_sigma = rows["prediction_sigma_plot"].to_numpy(dtype=float)
    axis.plot(x, y, color=color, lw=linewidth, label=label)
    axis.fill_between(x, y - y_sigma, y + y_sigma, color=color, alpha=0.12)


def _plot_galacticus_case(axis, frame: pd.DataFrame, observable_key: str, *, label: str) -> None:
    rows = frame.loc[frame["observable_key"] == observable_key].copy()
    if rows.empty:
        return
    rows = rows.sort_values("x_plot")
    x = rows["x_plot"].to_numpy(dtype=float)
    y = rows["prediction_plot"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    axis.plot(x[finite], y[finite], color="black", lw=2.0, ls="--", label=label, zorder=6)


def main() -> None:
    args = parse_args()
    meta = _read_bundle_meta(args.bundle_meta)
    combined_label, combined_path = _parse_labeled_path(args.combined)
    combined = _read_predictions(combined_path)
    subsets = [
        (label, _read_predictions(path))
        for label, path in (_parse_labeled_path(value) for value in args.subset)
    ]

    observable_keys = list(dict.fromkeys(combined["observable_key"].astype(str)))
    galacticus_predictions = None
    if args.galacticus_hdf5 is not None:
        galacticus_predictions = _read_galacticus_predictions(
            args.galacticus_hdf5.expanduser().resolve(),
            meta,
            observable_keys,
            combined,
            min_log10_y=args.galacticus_min_log10_y,
            apply_display_offsets=True,
        )
    ncols = max(1, int(args.ncols))
    nrows = int(np.ceil(len(observable_keys) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 4.0 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    combined_style = {
        **_style_for_case(combined_label, len(subsets)),
        "color": "black",
        "linewidth": 2.4,
    }
    for axis, observable_key in zip(axes_flat, observable_keys, strict=False):
        metadata = _axis_metadata(meta, observable_key)
        rows = combined.loc[combined["observable_key"] == observable_key].copy().sort_values("x_plot")
        x = rows["x_plot"].to_numpy(dtype=float)
        target = rows["target_plot"].to_numpy(dtype=float)
        target_sigma = rows["target_sigma_plot"].to_numpy(dtype=float)
        target_sigma = np.where(np.isfinite(target_sigma), target_sigma, np.nan)
        axis.errorbar(
            x,
            target,
            yerr=target_sigma,
            fmt="o",
            color="0.15",
            ms=4.0,
            label=metadata.get("target_label") or "target",
            zorder=4,
        )

        for case_index, (subset_label, subset_frame) in enumerate(subsets):
            style = _style_for_case(subset_label, case_index)
            _plot_case(
                axis,
                subset_frame,
                observable_key,
                label=str(style.get("label", subset_label)),
                color=str(style.get("color")),
                linewidth=float(style.get("linewidth", 1.8)),
            )

        _plot_case(
            axis,
            combined,
            observable_key,
            label=str(combined_style.get("label", combined_label)),
            color=str(combined_style.get("color")),
            linewidth=float(combined_style.get("linewidth", 2.2)),
        )
        if galacticus_predictions is not None and observable_key in galacticus_predictions:
            _plot_galacticus_case(
                axis,
                galacticus_predictions[observable_key],
                observable_key,
                label=args.galacticus_label,
            )
            galacticus_y = galacticus_predictions[observable_key]["prediction_plot"].to_numpy(dtype=float)
        else:
            galacticus_y = None

        central_values = [target]
        for _, subset_frame in subsets:
            subset_rows = subset_frame.loc[subset_frame["observable_key"] == observable_key]
            central_values.append(subset_rows["prediction_plot"].to_numpy(dtype=float))
        combined_rows = combined.loc[combined["observable_key"] == observable_key]
        central_values.append(combined_rows["prediction_plot"].to_numpy(dtype=float))
        central_values.append(galacticus_y)
        set_ylim_from_values(axis, *central_values)

        axis.set_title(metadata["label"])
        axis.set_xlabel(metadata["x_axis_label"])
        axis.set_ylabel(metadata["y_axis_label"])
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)

    for axis in axes_flat[len(observable_keys) :]:
        axis.axis("off")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi)
    plt.close(fig)
    print(args.output_path)


if __name__ == "__main__":
    main()
