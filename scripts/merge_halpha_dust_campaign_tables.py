from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-evaluation Halpha dust post-processing tables into campaign-level CSV files."
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--dust-dir-name", default="halpha_dust")
    parser.add_argument("--wide-filename", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--long-filename", default="halpha_dust_lf_long.csv")
    parser.add_argument("--output-wide", default="halpha_dust_lf_emulator_table.csv")
    parser.add_argument("--output-long", default="halpha_dust_lf_long.csv")
    parser.add_argument(
        "--exclude-evaluation-id",
        action="append",
        default=[],
        help=(
            "Evaluation ID to skip when merging. Can be supplied multiple times. "
            "Accepts full IDs, eval-0271-style suffixes, or bare numeric indices."
        ),
    )
    parser.add_argument(
        "--exclude-evaluation-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional text file of evaluation IDs to skip, one per line. Blank lines and "
            "lines starting with # are ignored. If omitted, excluded_evaluations.txt in "
            "the campaign root is used when present."
        ),
    )
    return parser.parse_args()


def _normalize_evaluation_id(campaign_root: Path, text: str) -> str:
    value = text.strip()
    if not value:
        raise ValueError("Empty evaluation ID")
    if value.isdigit():
        return f"{campaign_root.name}-eval-{int(value):04d}"
    if value.startswith("eval-"):
        return f"{campaign_root.name}-{value}"
    return value


def _read_exclusions(args: argparse.Namespace, campaign_root: Path) -> set[str]:
    raw_values = list(args.exclude_evaluation_id)
    exclude_file = args.exclude_evaluation_ids_file
    if exclude_file is None:
        default_file = campaign_root / "excluded_evaluations.txt"
        exclude_file = default_file if default_file.exists() else None
    elif not exclude_file.is_absolute():
        exclude_file = campaign_root / exclude_file

    if exclude_file is not None:
        for line in exclude_file.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                raw_values.append(stripped)

    return {_normalize_evaluation_id(campaign_root, value) for value in raw_values}


def _collect_tables(
    campaign_root: Path,
    dust_dir_name: str,
    filename: str,
    excluded_evaluation_ids: set[str],
) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    evaluations_root = campaign_root / "evaluations"
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    missing: list[str] = []
    for evaluation_dir in sorted(path for path in evaluations_root.iterdir() if path.is_dir()):
        if evaluation_dir.name in excluded_evaluation_ids:
            skipped.append(evaluation_dir.name)
            continue
        path = evaluation_dir / dust_dir_name / filename
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            missing.append(evaluation_dir.name)
    if not frames:
        raise FileNotFoundError(f"No {filename} files found under {evaluations_root}/*/{dust_dir_name}")
    return frames, skipped, missing


def main() -> None:
    args = parse_args()
    campaign_root = args.campaign_root.resolve()
    excluded_evaluation_ids = _read_exclusions(args, campaign_root)

    wide_frames, wide_skipped, wide_missing = _collect_tables(
        campaign_root,
        args.dust_dir_name,
        args.wide_filename,
        excluded_evaluation_ids,
    )
    long_frames, long_skipped, long_missing = _collect_tables(
        campaign_root,
        args.dust_dir_name,
        args.long_filename,
        excluded_evaluation_ids,
    )
    wide = pd.concat(wide_frames, ignore_index=True)
    long = pd.concat(long_frames, ignore_index=True)

    wide_path = campaign_root / args.output_wide
    long_path = campaign_root / args.output_long
    wide.to_csv(wide_path, index=False)
    long.to_csv(long_path, index=False)

    metadata = {
        "campaign_root": str(campaign_root),
        "dust_dir_name": args.dust_dir_name,
        "wide_filename": args.wide_filename,
        "long_filename": args.long_filename,
        "output_wide": str(wide_path),
        "output_long": str(long_path),
        "excluded_evaluation_ids": sorted(excluded_evaluation_ids),
        "skipped_wide_evaluation_ids": wide_skipped,
        "skipped_long_evaluation_ids": long_skipped,
        "missing_wide_evaluation_ids": wide_missing,
        "missing_long_evaluation_ids": long_missing,
        "n_wide_rows": int(len(wide)),
        "n_long_rows": int(len(long)),
        "n_wide_evaluations": int(wide["evaluation_id"].nunique()) if "evaluation_id" in wide.columns else None,
        "n_long_evaluations": int(long["evaluation_id"].nunique()) if "evaluation_id" in long.columns else None,
    }
    metadata_path = campaign_root / "halpha_dust_lf_merge_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(wide_path)
    print(long_path)
    print(f"{len(wide)} wide rows")
    print(f"{len(long)} long rows")
    if excluded_evaluation_ids:
        print(f"Excluded {len(excluded_evaluation_ids)} requested evaluation(s): {', '.join(sorted(excluded_evaluation_ids))}")
    if wide_missing or long_missing:
        missing = sorted(set(wide_missing).union(long_missing))
        print(f"Missing per-evaluation table(s) for {len(missing)} non-excluded evaluation(s).")
    print(metadata_path)


if __name__ == "__main__":
    main()
