from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the emulator best-fit Halpha LF to a one-off re-dusted Galacticus run "
            "evaluated at the nearest slow model."
        )
    )
    parser.add_argument("best_fit_csv", type=Path)
    parser.add_argument("redusted_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="halpha_best_fit_vs_redusted")
    return parser.parse_args()


def _load_and_merge(best_fit_csv: Path, redusted_csv: Path) -> pd.DataFrame:
    best = pd.read_csv(best_fit_csv).copy()
    redusted = pd.read_csv(redusted_csv).copy()
    redusted = redusted.rename(columns={"dn_dlnL_mpc3": "redusted_phi"})
    merged = best.merge(
        redusted[["sobral_label", "log10_luminosity_center", "redusted_phi"]],
        on=["sobral_label", "log10_luminosity_center"],
        how="inner",
        validate="one_to_one",
    )
    merged["redusted_log10_lf"] = np.log10(np.maximum(merged["redusted_phi"].to_numpy(dtype=float), 1.0e-300))
    merged["delta_log10_lf"] = merged["prediction_log10_lf"] - merged["redusted_log10_lf"]
    merged["delta_phi_frac"] = (
        merged["prediction_phi"].to_numpy(dtype=float) - merged["redusted_phi"].to_numpy(dtype=float)
    ) / merged["redusted_phi"].to_numpy(dtype=float)
    return merged


def _metrics_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sobral_label, sub in merged.groupby("sobral_label", sort=False):
        diff = sub["delta_log10_lf"].to_numpy(dtype=float)
        frac = sub["delta_phi_frac"].to_numpy(dtype=float)
        rows.append(
            {
                "sobral_label": sobral_label,
                "n_bins": int(len(sub)),
                "mean_abs_delta_log10_lf": float(np.mean(np.abs(diff))),
                "rmse_delta_log10_lf": float(np.sqrt(np.mean(diff**2))),
                "max_abs_delta_log10_lf": float(np.max(np.abs(diff))),
                "mean_abs_frac_delta_phi": float(np.mean(np.abs(frac))),
            }
        )
    all_diff = merged["delta_log10_lf"].to_numpy(dtype=float)
    all_frac = merged["delta_phi_frac"].to_numpy(dtype=float)
    rows.append(
        {
            "sobral_label": "all",
            "n_bins": int(len(merged)),
            "mean_abs_delta_log10_lf": float(np.mean(np.abs(all_diff))),
            "rmse_delta_log10_lf": float(np.sqrt(np.mean(all_diff**2))),
            "max_abs_delta_log10_lf": float(np.max(np.abs(all_diff))),
            "mean_abs_frac_delta_phi": float(np.mean(np.abs(all_frac))),
        }
    )
    return pd.DataFrame(rows)


def _make_plot(merged: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    axes_flat = axes.ravel()
    titles = {"Z1": "z = 0.40", "Z2": "z = 0.84", "Z3": "z = 1.47", "Z4": "z = 2.23"}
    for axis, sobral_label in zip(axes_flat, ["Z1", "Z2", "Z3", "Z4"], strict=True):
        sub = merged.loc[merged["sobral_label"] == sobral_label].sort_values("log10_luminosity_center")
        x = sub["log10_luminosity_center"].to_numpy(dtype=float)
        y_target = sub["target_log10_lf"].to_numpy(dtype=float)
        y_target_linear = sub["target_phi"].to_numpy(dtype=float)
        y_target_linear_std = sub["target_phi_std"].to_numpy(dtype=float)
        lower_linear = np.maximum(y_target_linear - y_target_linear_std, 1.0e-300)
        upper_linear = y_target_linear + y_target_linear_std
        yerr_lower = y_target - np.log10(lower_linear)
        yerr_upper = np.log10(upper_linear) - y_target
        y_emu = sub["prediction_log10_lf"].to_numpy(dtype=float)
        y_emu_std = sub["prediction_log10_lf_std"].to_numpy(dtype=float)
        y_red = sub["redusted_log10_lf"].to_numpy(dtype=float)
        axis.errorbar(
            x,
            y_target,
            yerr=np.vstack([yerr_lower, yerr_upper]),
            fmt="o",
            color="0.15",
            label="Sobral target",
        )
        axis.plot(x, y_emu, color="tab:blue", lw=1.8, label="Emulator best fit")
        axis.fill_between(x, y_emu - y_emu_std, y_emu + y_emu_std, color="tab:blue", alpha=0.18)
        axis.plot(x, y_red, color="tab:orange", lw=1.8, ls="--", label="Nearest slow model + exact dust")
        axis.set_title(titles.get(sobral_label, sobral_label))
        axis.set_xlabel(r"$\log_{10}(L_{\mathrm{H}\alpha}/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(r"$\log_{10}\Phi$")
        axis.set_ylim(-6.0, -1.0)
        axis.grid(alpha=0.22)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = _load_and_merge(args.best_fit_csv.resolve(), args.redusted_csv.resolve())
    metrics = _metrics_table(merged)

    merged_path = output_dir / f"{args.output_prefix}_merged.csv"
    metrics_path = output_dir / f"{args.output_prefix}_metrics.csv"
    figure_path = output_dir / f"{args.output_prefix}.png"

    merged.to_csv(merged_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    _make_plot(merged, figure_path)

    print(merged_path)
    print(metrics_path)
    print(figure_path)


if __name__ == "__main__":
    main()
