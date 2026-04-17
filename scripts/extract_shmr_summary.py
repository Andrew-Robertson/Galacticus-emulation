from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu.extract import summarize_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SHMR mean/scatter summaries from a completed campaign."
    )
    parser.add_argument("campaign_root", help="Path to a campaign directory under runs/campaigns/")
    parser.add_argument(
        "--analysis-mean-key",
        default="stellarHaloMassRelationUniverseMachinez6",
    )
    parser.add_argument(
        "--analysis-scatter-key",
        default="stellarHaloMassRelationScatterUniverseMachinez6",
    )
    parser.add_argument(
        "--all-bins",
        action="store_true",
        help="Keep all Galacticus analysis bins instead of only the active nonzero target bins.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = summarize_campaign(
        campaign_root=args.campaign_root,
        analysis_mean_key=args.analysis_mean_key,
        analysis_scatter_key=args.analysis_scatter_key,
        active_only=not args.all_bins,
    )
    print(output_path)


if __name__ == "__main__":
    main()
