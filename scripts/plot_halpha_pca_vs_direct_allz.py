from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLOR_MAP = {
    "Z1": "#1f77b4",
    "Z2": "#ff7f0e",
    "Z3": "#2ca02c",
    "Z4": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay direct bin-by-bin and PCA Halpha-emulator metrics versus luminosity."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument(
        "--direct-metrics",
        default="emulator_halpha_allz_cv/gp_allz_cv_metrics.csv",
        help="Path relative to campaign_root for direct-emulation metrics CSV.",
    )
    parser.add_argument(
        "--pca-metrics",
        action="append",
        default=[],
        help=(
            "Repeatable mapping of the form Z4=emulator_halpha_pca_z4/gp_cv_metrics_halpha_z4_unscaled_pca5.csv. "
            "Only supplied Sobral labels will have PCA curves overlaid."
        ),
    )
    parser.add_argument("--figures-dir-name", default="figures_halpha_pca_compare")
    parser.add_argument("--output-name", default="pca_vs_direct_allz_rmse_r2.png")
    return parser.parse_args()


def _parse_mapping(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Expected LABEL=path form, got: {text}")
    label, path = text.split("=", 1)
    label = label.strip().upper()
    return label, path.strip()


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)

    direct = pd.read_csv(campaign_root / args.direct_metrics)
    direct = direct[direct["sobral_label"].isin(["Z1", "Z2", "Z3", "Z4"])].copy()

    pca_tables: dict[str, pd.DataFrame] = {}
    for item in args.pca_metrics:
        label, rel_path = _parse_mapping(item)
        table = pd.read_csv(campaign_root / rel_path).copy()
        table = table.loc[table["bin_index"] != "all"].copy()
        table["bin_index"] = table["bin_index"].astype(int)
        pca_tables[label] = table

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)

    for sobral_label in ["Z1", "Z2", "Z3", "Z4"]:
        color = COLOR_MAP[sobral_label]
        sub = direct.loc[direct["sobral_label"] == sobral_label].sort_values("log10_luminosity_center")
        if len(sub) == 0:
            continue
        axes[0].plot(
            sub["log10_luminosity_center"],
            sub["rmse_dex"],
            color=color,
            marker="o",
            lw=1.8,
            label=f"{sobral_label} direct",
        )
        axes[1].plot(
            sub["log10_luminosity_center"],
            sub["r2"],
            color=color,
            marker="o",
            lw=1.8,
            label=f"{sobral_label} direct",
        )

        if sobral_label in pca_tables:
            pca = pca_tables[sobral_label].sort_values("log10_luminosity_center")
            axes[0].plot(
                pca["log10_luminosity_center"],
                pca["rmse_dex"],
                color=color,
                marker="s",
                lw=1.5,
                ls="--",
                label=f"{sobral_label} PCA",
            )
            axes[1].plot(
                pca["log10_luminosity_center"],
                pca["r2"],
                color=color,
                marker="s",
                lw=1.5,
                ls="--",
                label=f"{sobral_label} PCA",
            )

    axes[0].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    axes[0].set_ylabel("RMSE [dex]")
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, ncol=2)

    output_path = figures_root / args.output_name
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
