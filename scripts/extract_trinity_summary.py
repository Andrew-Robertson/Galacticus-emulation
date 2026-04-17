from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu import summarize_trinity_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Trinity z0/z1/z2 summary vectors from completed campaign outputs.")
    parser.add_argument("campaign_root", help="Path to a completed Trinity campaign directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = summarize_trinity_campaign(args.campaign_root)
    print(output_path)


if __name__ == "__main__":
    main()
