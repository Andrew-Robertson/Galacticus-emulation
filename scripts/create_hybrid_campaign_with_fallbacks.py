from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import pandas as pd


SKIP_ROOT_FILES = {
    "excluded_evaluations.txt",
    "filtered_campaign_view.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hybrid campaign by taking most evaluation directories from a primary "
            "campaign and selected fallback evaluation directories from a second campaign."
        )
    )
    parser.add_argument("primary_campaign", type=Path)
    parser.add_argument("fallback_campaign", type=Path)
    parser.add_argument("output_campaign", type=Path)
    parser.add_argument(
        "--fallback-index",
        type=int,
        action="append",
        required=True,
        help="Evaluation index to source from fallback_campaign. Repeat for multiple.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to place evaluation directories in output_campaign. Default: copy.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _eval_prefix(campaign: Path) -> str:
    evaluations = campaign / "evaluations"
    for path in sorted(evaluations.iterdir()):
        if path.is_dir() and "-eval-" in path.name:
            return path.name.rsplit("-eval-", 1)[0]
    return campaign.name


def _eval_id(prefix: str, index: int) -> str:
    return f"{prefix}-eval-{index:04d}"


def _eval_index(evaluation_id: str) -> int:
    match = re.search(r"-eval-(\d+)$", evaluation_id)
    if match is None:
        raise ValueError(f"Could not parse evaluation index from {evaluation_id!r}")
    return int(match.group(1))


def _copy_or_link(source: Path, destination: Path, *, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _copy_root_files(primary: Path, output: Path) -> None:
    for source in sorted(primary.iterdir()):
        if not source.is_file() or source.name in SKIP_ROOT_FILES or source.name == "samples.csv":
            continue
        shutil.copy2(source, output / source.name)


def _load_samples(path: Path) -> pd.DataFrame:
    samples_path = path / "samples.csv"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)
    if "evaluation_id" not in pd.read_csv(samples_path, nrows=0).columns:
        raise ValueError(f"{samples_path} has no evaluation_id column")
    return pd.read_csv(samples_path)


def _write_hybrid_samples(
    primary: Path,
    fallback: Path,
    output: Path,
    *,
    primary_prefix: str,
    fallback_prefix: str,
    fallback_indices: set[int],
) -> None:
    primary_samples = _load_samples(primary)
    fallback_samples = _load_samples(fallback)
    rows = []
    for row in primary_samples.to_dict("records"):
        index = _eval_index(str(row["evaluation_id"]))
        if index not in fallback_indices:
            row["evaluation_id"] = _eval_id(primary_prefix, index)
            rows.append(row)
    for index in sorted(fallback_indices):
        fallback_id = _eval_id(fallback_prefix, index)
        match = fallback_samples.loc[fallback_samples["evaluation_id"] == fallback_id]
        if match.empty:
            raise ValueError(f"Fallback samples.csv has no row for {fallback_id}")
        row = match.iloc[0].to_dict()
        row["evaluation_id"] = _eval_id(primary_prefix, index)
        rows.append(row)
    hybrid = pd.DataFrame(rows)
    hybrid["__eval_index"] = hybrid["evaluation_id"].map(_eval_index)
    hybrid = hybrid.sort_values("__eval_index").drop(columns=["__eval_index"]).reset_index(drop=True)
    if hybrid["evaluation_id"].duplicated().any():
        duplicated = sorted(hybrid.loc[hybrid["evaluation_id"].duplicated(), "evaluation_id"].unique())
        raise ValueError(f"Duplicate evaluation_id rows in hybrid samples: {duplicated}")
    hybrid.to_csv(output / "samples.csv", index=False)


def _rewrite_evaluation_ids_in_csvs(directory: Path, old_id: str, new_id: str) -> int:
    rewritten = 0
    for path in sorted(directory.rglob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if "evaluation_id" not in header.columns:
            continue
        frame = pd.read_csv(path)
        frame["evaluation_id"] = frame["evaluation_id"].replace(old_id, new_id)
        frame.to_csv(path, index=False)
        rewritten += 1
    return rewritten


def main() -> None:
    args = parse_args()
    primary = args.primary_campaign.expanduser().resolve()
    fallback = args.fallback_campaign.expanduser().resolve()
    output = args.output_campaign.expanduser().resolve()
    fallback_indices = set(args.fallback_index)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; use --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "evaluations").mkdir()

    primary_prefix = _eval_prefix(primary)
    fallback_prefix = _eval_prefix(fallback)
    _copy_root_files(primary, output)
    _write_hybrid_samples(
        primary,
        fallback,
        output,
        primary_prefix=primary_prefix,
        fallback_prefix=fallback_prefix,
        fallback_indices=fallback_indices,
    )

    provenance_rows = []
    primary_evaluations = {path.name: path for path in (primary / "evaluations").iterdir() if path.is_dir()}
    for primary_id, source in sorted(primary_evaluations.items(), key=lambda item: _eval_index(item[0])):
        index = _eval_index(primary_id)
        if index in fallback_indices:
            continue
        destination = output / "evaluations" / primary_id
        _copy_or_link(source, destination, link_mode=args.link_mode)
        provenance_rows.append(
            {
                "evaluation_index": index,
                "evaluation_id": primary_id,
                "source": "primary",
                "source_campaign": str(primary),
                "source_evaluation_id": primary_id,
                "link_mode": args.link_mode,
                "rewritten_sidecar_csvs": 0,
            }
        )

    for index in sorted(fallback_indices):
        primary_id = _eval_id(primary_prefix, index)
        fallback_id = _eval_id(fallback_prefix, index)
        source = fallback / "evaluations" / fallback_id
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = output / "evaluations" / primary_id
        _copy_or_link(source, destination, link_mode=args.link_mode)
        rewritten = 0
        if args.link_mode == "copy":
            rewritten = _rewrite_evaluation_ids_in_csvs(destination, fallback_id, primary_id)
        provenance_rows.append(
            {
                "evaluation_index": index,
                "evaluation_id": primary_id,
                "source": "fallback",
                "source_campaign": str(fallback),
                "source_evaluation_id": fallback_id,
                "link_mode": args.link_mode,
                "rewritten_sidecar_csvs": rewritten,
            }
        )

    provenance = pd.DataFrame(provenance_rows).sort_values("evaluation_index").reset_index(drop=True)
    provenance.to_csv(output / "hybrid_source_manifest.csv", index=False)
    (output / "fallback_evaluations.txt").write_text(
        "\n".join(_eval_id(primary_prefix, index) for index in sorted(fallback_indices)) + "\n"
    )
    metadata = {
        "primary_campaign": str(primary),
        "fallback_campaign": str(fallback),
        "primary_prefix": primary_prefix,
        "fallback_prefix": fallback_prefix,
        "fallback_indices": sorted(fallback_indices),
        "link_mode": args.link_mode,
        "n_evaluations": int(len(provenance)),
    }
    (output / "hybrid_campaign.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)
    print(output / "samples.csv")
    print(output / "hybrid_source_manifest.csv")


if __name__ == "__main__":
    main()
