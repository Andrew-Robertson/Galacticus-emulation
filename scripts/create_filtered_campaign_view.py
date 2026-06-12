from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight campaign view that excludes failed evaluations. "
            "Evaluation directories are symlinked by default, while root-level CSV files "
            "with an evaluation_id column are filtered."
        )
    )
    parser.add_argument("source_campaign", type=Path)
    parser.add_argument("output_campaign", type=Path)
    parser.add_argument(
        "--exclude-evaluation-id",
        action="append",
        default=[],
        help=(
            "Evaluation ID to exclude. Can be supplied multiple times. Accepts full IDs, "
            "eval-0271-style suffixes, or bare numeric indices."
        ),
    )
    parser.add_argument(
        "--exclude-evaluation-ids-file",
        type=Path,
        default=None,
        help="Optional text file of evaluation IDs to exclude, one per line.",
    )
    parser.add_argument(
        "--copy-evaluations",
        action="store_true",
        help="Copy evaluation directories instead of symlinking them.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default=None,
        help="Evaluation storage mode. Equivalent to default symlinks or --copy-evaluations.",
    )
    parser.add_argument(
        "--include-sidecars",
        action="store_true",
        help="Accepted for clarity; sidecars are included automatically because whole evaluation directories are linked/copied.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Keep only the first N evaluation directories, after sorting by evaluation id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace OUTPUT_CAMPAIGN if it already exists.",
    )
    parser.add_argument(
        "--skip-derived-directories",
        action="store_true",
        help=(
            "Do not symlink derived root-level directories such as emulator_* and figures_*. "
            "Useful when the filtered view will be retrained independently."
        ),
    )
    return parser.parse_args()


def _evaluation_id_prefix(source_campaign: Path) -> str:
    source_evaluations = source_campaign / "evaluations"
    for source_eval in sorted(path for path in source_evaluations.iterdir() if path.is_dir()):
        if "-eval-" in source_eval.name:
            return source_eval.name.rsplit("-eval-", 1)[0]
    return source_campaign.name


def _normalize_evaluation_id(evaluation_id_prefix: str, text: str) -> str:
    value = text.strip()
    if not value:
        raise ValueError("Empty evaluation ID")
    if value.isdigit():
        return f"{evaluation_id_prefix}-eval-{int(value):04d}"
    if value.startswith("eval-"):
        return f"{evaluation_id_prefix}-{value}"
    return value


def _read_exclusions(args: argparse.Namespace, source_campaign: Path) -> set[str]:
    raw_values = list(args.exclude_evaluation_id)
    if args.exclude_evaluation_ids_file is not None:
        exclude_file = args.exclude_evaluation_ids_file
        if not exclude_file.is_absolute():
            exclude_file = source_campaign / exclude_file
        for line in exclude_file.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                raw_values.append(stripped)
    evaluation_id_prefix = _evaluation_id_prefix(source_campaign)
    exclusions = {_normalize_evaluation_id(evaluation_id_prefix, value) for value in raw_values}
    if args.max_evaluations is not None:
        if args.max_evaluations < 1:
            raise ValueError("--max-evaluations must be at least 1")
        source_evaluations = source_campaign / "evaluations"
        evaluation_ids = sorted(path.name for path in source_evaluations.iterdir() if path.is_dir())
        exclusions.update(evaluation_ids[args.max_evaluations :])
    return exclusions


def _copy_or_filter_root_file(source: Path, destination: Path, excluded_evaluation_ids: set[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
        if "evaluation_id" in frame.columns:
            frame = frame.loc[~frame["evaluation_id"].isin(excluded_evaluation_ids)].reset_index(drop=True)
        frame.to_csv(destination, index=False)
        return
    if source.name == "commands.txt":
        lines = [
            line
            for line in source.read_text().splitlines()
            if not any(evaluation_id in line for evaluation_id in excluded_evaluation_ids)
        ]
        destination.write_text("\n".join(lines) + "\n")
        return
    if source.name == "commands.sh":
        lines = [
            line
            for line in source.read_text().splitlines()
            if not any(evaluation_id in line for evaluation_id in excluded_evaluation_ids)
        ]
        destination.write_text("\n".join(lines) + "\n")
        destination.chmod(source.stat().st_mode)
        return
    if source.name == "campaign_design.json":
        data = json.loads(source.read_text())
        data["filtered_campaign_view"] = {
            "source_campaign": str(source.parent.resolve()),
            "excluded_evaluation_ids": sorted(excluded_evaluation_ids),
        }
        destination.write_text(json.dumps(data, indent=2) + "\n")
        return
    shutil.copy2(source, destination)


def _link_or_copy_evaluations(
    source_campaign: Path,
    output_campaign: Path,
    excluded_evaluation_ids: set[str],
    *,
    copy_evaluations: bool,
) -> int:
    source_evaluations = source_campaign / "evaluations"
    output_evaluations = output_campaign / "evaluations"
    output_evaluations.mkdir(parents=True, exist_ok=True)
    kept = 0
    for source_eval in sorted(path for path in source_evaluations.iterdir() if path.is_dir()):
        if source_eval.name in excluded_evaluation_ids:
            continue
        destination_eval = output_evaluations / source_eval.name
        if copy_evaluations:
            shutil.copytree(source_eval, destination_eval)
        else:
            destination_eval.symlink_to(source_eval.resolve(), target_is_directory=True)
        kept += 1
    return kept


def _is_derived_directory(path: Path) -> bool:
    return path.name.startswith(("emulator_", "figures_"))


def main() -> None:
    args = parse_args()
    source_campaign = args.source_campaign.resolve()
    output_campaign = args.output_campaign.resolve()
    if not source_campaign.is_dir():
        raise NotADirectoryError(source_campaign)
    if not (source_campaign / "evaluations").is_dir():
        raise NotADirectoryError(source_campaign / "evaluations")

    excluded_evaluation_ids = _read_exclusions(args, source_campaign)
    if not excluded_evaluation_ids:
        raise ValueError("No excluded evaluations were supplied.")
    copy_evaluations = bool(args.copy_evaluations or args.link_mode == "copy")

    if output_campaign.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_campaign} exists; use --overwrite to replace it.")
        shutil.rmtree(output_campaign)
    output_campaign.mkdir(parents=True)

    for source in sorted(path for path in source_campaign.iterdir() if path.is_file()):
        _copy_or_filter_root_file(source, output_campaign / source.name, excluded_evaluation_ids)

    for source in sorted(path for path in source_campaign.iterdir() if path.is_dir()):
        if source.name in {"evaluations", "logs"}:
            continue
        if args.skip_derived_directories and _is_derived_directory(source):
            continue
        destination = output_campaign / source.name
        destination.symlink_to(source.resolve(), target_is_directory=True)

    kept = _link_or_copy_evaluations(
        source_campaign,
        output_campaign,
        excluded_evaluation_ids,
        copy_evaluations=copy_evaluations,
    )
    (output_campaign / "excluded_evaluations.txt").write_text(
        "\n".join(sorted(excluded_evaluation_ids)) + "\n"
    )
    metadata = {
        "source_campaign": str(source_campaign),
        "output_campaign": str(output_campaign),
        "excluded_evaluation_ids": sorted(excluded_evaluation_ids),
        "n_evaluations_kept": kept,
        "max_evaluations": args.max_evaluations,
        "evaluation_storage": "copy" if copy_evaluations else "symlink",
        "include_sidecars": True,
        "skip_derived_directories": bool(args.skip_derived_directories),
    }
    (output_campaign / "filtered_campaign_view.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(output_campaign)
    print(f"Kept {kept} evaluation directories")
    print(f"Excluded {len(excluded_evaluation_ids)} evaluation(s): {', '.join(sorted(excluded_evaluation_ids))}")


if __name__ == "__main__":
    main()
