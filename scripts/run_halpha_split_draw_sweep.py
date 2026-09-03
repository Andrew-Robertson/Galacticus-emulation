from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COMBOS = [
    ("Z1", 7),
    ("Z1", 16),
    ("Z4", 1),
    ("Z4", 12),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sweep over numbers of dust draws per Galacticus evaluation for the "
            "clean Halpha train/test split setup, then aggregate RMSE and R2."
        )
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--train-draws", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--n-train-evals", type=int, default=16)
    parser.add_argument("--n-restarts-optimizer", type=int, default=3)
    parser.add_argument("--noise-model", default="white", choices=["white"])
    parser.add_argument("--noise-floor-dex", type=float, default=1.0e-4)
    parser.add_argument("--min-log10-lf", type=float, default=-7.0)
    parser.add_argument("--max-abs-delta", type=float, default=2.0)
    parser.add_argument("--max-attenuation-scatter", type=float, default=0.45)
    parser.add_argument("--figures-dir-name", default="figures_halpha_dust_draw_sweep")
    parser.add_argument("--emulator-dir-name", default="emulator_halpha_dust_draw_sweep")
    return parser.parse_args()


def _run_one(
    script_path: Path,
    campaign_root: Path,
    sobral_label: str,
    bin_index: int,
    train_draws: int,
    args: argparse.Namespace,
) -> Path:
    tag = f"sweep_{sobral_label.lower()}_bin{bin_index}_draws{train_draws}"
    cmd = [
        sys.executable,
        str(script_path),
        str(campaign_root),
        "--sobral-label",
        sobral_label,
        "--bin-indices",
        str(bin_index),
        "--n-train-evals",
        str(args.n_train_evals),
        "--train-draws-per-eval",
        str(train_draws),
        "--n-restarts-optimizer",
        str(args.n_restarts_optimizer),
        "--noise-model",
        args.noise_model,
        "--noise-floor-dex",
        str(args.noise_floor_dex),
        "--min-log10-lf",
        str(args.min_log10_lf),
        "--max-abs-delta",
        str(args.max_abs_delta),
        "--max-attenuation-scatter",
        str(args.max_attenuation_scatter),
        "--figures-dir-name",
        args.figures_dir_name,
        "--emulator-dir-name",
        args.emulator_dir_name,
        "--output-tag",
        tag,
    ]
    subprocess.run(cmd, check=True)
    suffix = f"_split_{sobral_label.lower()}_trainevals{args.n_train_evals}_traindraws{train_draws}_{tag}"
    return campaign_root / args.emulator_dir_name / f"gp_split_metrics{suffix}.csv"


def _combo_label(sobral_label: str, bin_index: int) -> str:
    return f"{sobral_label} bin {bin_index}"


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    script_path = Path(__file__).resolve().with_name("fit_halpha_train_test_split.py")
    figures_root = campaign_root / args.figures_dir_name
    emulator_root = campaign_root / args.emulator_dir_name
    figures_root.mkdir(parents=True, exist_ok=True)
    emulator_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for sobral_label, bin_index in DEFAULT_COMBOS:
        for train_draws in args.train_draws:
            print(f"running {sobral_label} bin {bin_index} with {train_draws} draws/eval", flush=True)
            metrics_path = _run_one(script_path, campaign_root, sobral_label, bin_index, train_draws, args)
            metrics = pd.read_csv(metrics_path)
            row = metrics.loc[metrics["output"] == "all"].iloc[0].to_dict()
            row["sobral_label"] = sobral_label
            row["bin_index"] = bin_index
            row["train_draws_per_eval"] = train_draws
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = emulator_root / "gp_split_draw_sweep_summary.csv"
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    for sobral_label, bin_index in DEFAULT_COMBOS:
        sub = summary.loc[(summary["sobral_label"] == sobral_label) & (summary["bin_index"] == bin_index)].sort_values("train_draws_per_eval")
        label = _combo_label(sobral_label, bin_index)
        axes[0].plot(sub["train_draws_per_eval"], sub["rmse_dex"], marker="o", lw=1.8, label=label)
        axes[1].plot(sub["train_draws_per_eval"], sub["r2"], marker="o", lw=1.8, label=label)
    axes[0].set_xlabel("Dust draws per Galacticus evaluation")
    axes[0].set_ylabel("RMSE [dex]")
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel("Dust draws per Galacticus evaluation")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    plot_path = figures_root / "gp_split_draw_sweep_rmse_r2.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()
