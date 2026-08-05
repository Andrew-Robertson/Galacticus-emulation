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

from galacticus_emu.observable_plot_metadata import apply_observable_plot_metadata_overrides
from galacticus_emu.plotting import set_ylim_from_values
from plot_interactive_observables_mcmc_best_fit_overlay import _axis_metadata
from plot_observable_mcmc_overlay_getdist import DEFAULT_STYLE, FALLBACK_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay standard-observable posterior emulator-mean summary curves from multiple MCMCs."
    )
    parser.add_argument("--bundle-meta", type=Path, required=True)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Posterior prediction summary in LABEL=CSV form. Repeat for multiple cases.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Expected non-empty LABEL in {value!r}")
    return label, Path(path).expanduser().resolve()


def _read_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "observable_key",
        "x_plot",
        "target_plot",
        "target_sigma_plot",
        "posterior_prediction_lower",
        "posterior_prediction_median",
        "posterior_prediction_upper",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing column(s): {missing}")
    return frame


def _style(label: str, index: int) -> dict[str, object]:
    style = dict(DEFAULT_STYLE.get(label, {}))
    style.setdefault("color", FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
    style.setdefault("label", label)
    style.setdefault("linewidth", 1.8)
    return style


def _plot(meta: dict, summaries: list[tuple[str, pd.DataFrame]], output_path: Path, *, dpi: int) -> None:
    observable_keys = list(dict.fromkeys(summaries[0][1]["observable_key"].astype(str)))
    ncols = 2 if len(observable_keys) > 1 else 1
    nrows = int(np.ceil(len(observable_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for axis, observable_key in zip(axes_flat, observable_keys, strict=False):
        reference_rows = summaries[0][1].loc[
            summaries[0][1]["observable_key"].astype(str) == observable_key
        ].copy().sort_values("x_plot")
        x = reference_rows["x_plot"].to_numpy(dtype=float)
        target = reference_rows["target_plot"].to_numpy(dtype=float)
        target_sigma = reference_rows["target_sigma_plot"].to_numpy(dtype=float)
        metadata = _axis_metadata(meta, observable_key)
        axis.errorbar(
            x,
            target,
            yerr=target_sigma,
            fmt="o",
            color="0.15",
            label=metadata.get("target_label") or "target",
            zorder=10,
        )

        y_for_limits = [target]
        for index, (label, frame) in enumerate(summaries):
            rows = frame.loc[frame["observable_key"].astype(str) == observable_key].copy().sort_values("x_plot")
            if rows.empty:
                continue
            style = _style(label, index)
            case_x = rows["x_plot"].to_numpy(dtype=float)
            lower = rows["posterior_prediction_lower"].to_numpy(dtype=float)
            median = rows["posterior_prediction_median"].to_numpy(dtype=float)
            upper = rows["posterior_prediction_upper"].to_numpy(dtype=float)
            color = str(style["color"])
            axis.fill_between(case_x, lower, upper, color=color, alpha=0.12, linewidth=0.0)
            axis.plot(
                case_x,
                median,
                color=color,
                lw=float(style.get("linewidth", 1.8)),
                label=str(style.get("label", label)),
                zorder=4 if label != "Combined" else 5,
            )
            y_for_limits.extend([lower, median, upper])

        set_ylim_from_values(axis, *y_for_limits)
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


def main() -> None:
    args = parse_args()
    meta = apply_observable_plot_metadata_overrides(json.loads(args.bundle_meta.expanduser().resolve().read_text()))
    summaries = [(label, _read_summary(path)) for label, path in (_parse_labeled_path(value) for value in args.summary)]
    _plot(meta, summaries, args.output_path.expanduser().resolve(), dpi=args.dpi)
    print(args.output_path)


if __name__ == "__main__":
    main()
