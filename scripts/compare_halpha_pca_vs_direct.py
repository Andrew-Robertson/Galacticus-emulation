from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PCA-based and direct bin-by-bin Halpha LF emulator metrics."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--sobral-label", default="Z4", choices=["Z1", "Z2", "Z3", "Z4"])
    parser.add_argument("--direct-metrics", default="emulator_halpha_allz_cv/gp_allz_cv_metrics.csv")
    parser.add_argument("--pca-metrics", required=True, help="Path relative to campaign_root for PCA metrics CSV.")
    parser.add_argument("--figures-dir-name", default="figures_halpha_pca_compare")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_pca_compare")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    output_prefix = args.output_prefix or f"halpha_{args.sobral_label.lower()}"

    direct = pd.read_csv(campaign_root / args.direct_metrics)
    direct = direct.loc[direct["sobral_label"] == args.sobral_label].copy()
    pca = pd.read_csv(campaign_root / args.pca_metrics).copy()
    pca = pca.loc[pca["bin_index"] != "all"].copy()
    pca["bin_index"] = pca["bin_index"].astype(int)

    merged = direct.merge(
        pca,
        on=["bin_index", "log10_luminosity_center"],
        suffixes=("_direct", "_pca"),
        validate="one_to_one",
    )
    merged["rmse_ratio_pca_over_direct"] = merged["rmse_dex_pca"] / merged["rmse_dex_direct"]
    merged["r2_delta_pca_minus_direct"] = merged["r2_pca"] - merged["r2_direct"]

    compare_path = emulator_root / f"pca_vs_direct_metrics_{output_prefix}.csv"
    merged.to_csv(compare_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    axes[0].plot(merged["log10_luminosity_center"], merged["rmse_dex_direct"], "o-", label="Direct bin-by-bin")
    axes[0].plot(merged["log10_luminosity_center"], merged["rmse_dex_pca"], "o-", label="PCA")
    axes[0].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    axes[0].set_ylabel("RMSE [dex]")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(merged["log10_luminosity_center"], merged["r2_direct"], "o-", label="Direct bin-by-bin")
    axes[1].plot(merged["log10_luminosity_center"], merged["r2_pca"], "o-", label="PCA")
    axes[1].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    plot_path = figures_root / f"pca_vs_direct_{output_prefix}.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    delta_fig, delta_axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    delta_axes[0].axhline(1.0, color="0.3", lw=1.0, ls="--")
    delta_axes[0].plot(merged["log10_luminosity_center"], merged["rmse_ratio_pca_over_direct"], "o-")
    delta_axes[0].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    delta_axes[0].set_ylabel("RMSE(PCA) / RMSE(direct)")
    delta_axes[0].grid(alpha=0.2)

    delta_axes[1].axhline(0.0, color="0.3", lw=1.0, ls="--")
    delta_axes[1].plot(merged["log10_luminosity_center"], merged["r2_delta_pca_minus_direct"], "o-")
    delta_axes[1].set_xlabel(r"$\log_{10}L_{\mathrm{H}\alpha}$")
    delta_axes[1].set_ylabel(r"$R^2_{\mathrm{PCA}} - R^2_{\mathrm{direct}}$")
    delta_axes[1].grid(alpha=0.2)

    delta_path = figures_root / f"pca_vs_direct_delta_{output_prefix}.png"
    delta_fig.savefig(delta_path, dpi=220)
    plt.close(delta_fig)

    print(compare_path)
    print(plot_path)
    print(delta_path)


if __name__ == "__main__":
    main()
