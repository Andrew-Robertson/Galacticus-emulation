#!/usr/bin/env python3
"""Extract Blanton-style AGN fractions from full Galacticus outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu.blanton_agn_campaign import (  # noqa: E402
    DEFAULT_LAMBDA_THRESHOLDS,
    DEFAULT_MODES,
    DEFAULT_REDSHIFTS,
    DEFAULT_SSFR_CUTS,
    DEFAULT_WEIGHT_TOLERANCE,
    aggregate_campaign_shards,
    default_output_root,
    evaluation_ids_from_samples,
    extract_campaign_evaluation,
    extract_evaluation,
    extraction_config,
    write_evaluation_shard,
)
from galacticus_emu.agn_demographics import PAPER_II_MASS_EDGES  # noqa: E402


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--redshifts", type=float, nargs="+", default=DEFAULT_REDSHIFTS)
    parser.add_argument("--redshift-tolerance", type=float, default=0.02)
    parser.add_argument("--mass-edges", type=float, nargs="+", default=PAPER_II_MASS_EDGES)
    parser.add_argument("--ssfr-cuts", type=float, nargs="+", default=DEFAULT_SSFR_CUTS)
    parser.add_argument(
        "--lambda-thresholds", type=float, nargs="+", default=DEFAULT_LAMBDA_THRESHOLDS
    )
    parser.add_argument("--modes", choices=DEFAULT_MODES, nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--weight-tolerance", type=float, default=DEFAULT_WEIGHT_TOLERANCE)


def _config(args: argparse.Namespace) -> dict[str, object]:
    return extraction_config(
        redshifts=args.redshifts,
        redshift_tolerance=args.redshift_tolerance,
        mass_edges=args.mass_edges,
        ssfr_cuts=args.ssfr_cuts,
        lambda_thresholds=args.lambda_thresholds,
        modes=args.modes,
        weight_tolerance=args.weight_tolerance,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract unweighted Blanton-style AGN and quiescent fractions after checking "
            "that all combined Galacticus node weights agree within the requested tolerance."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluation = subparsers.add_parser(
        "evaluation", help="Extract one evaluation from a campaign (the SLURM-array entry point)."
    )
    evaluation.add_argument("campaign_root", type=Path)
    selector = evaluation.add_mutually_exclusive_group(required=True)
    selector.add_argument("--evaluation-index", type=int)
    selector.add_argument("--evaluation-id")
    evaluation.add_argument("--output-root", type=Path)
    evaluation.add_argument("--overwrite", action="store_true")
    _add_config_arguments(evaluation)

    direct = subparsers.add_parser(
        "file", help="Extract one standalone full Galacticus HDF5 file (useful for a MAP run)."
    )
    direct.add_argument("galacticus_file", type=Path)
    direct.add_argument("--evaluation-id", required=True)
    direct.add_argument("--params-xml", type=Path)
    direct.add_argument("--output-shard", type=Path, required=True)
    direct.add_argument("--overwrite", action="store_true")
    _add_config_arguments(direct)

    aggregate = subparsers.add_parser(
        "aggregate", help="Combine completed shards and join their fractions to samples.csv."
    )
    aggregate.add_argument("campaign_root", type=Path)
    aggregate.add_argument("--output-root", type=Path)
    aggregate.add_argument(
        "--allow-missing",
        action="store_true",
        help="Aggregate available shards even if some samples.csv evaluations are absent.",
    )
    aggregate.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "evaluation":
        evaluation_ids = evaluation_ids_from_samples(args.campaign_root)
        if args.evaluation_index is not None:
            try:
                evaluation_id = evaluation_ids[args.evaluation_index]
            except IndexError as error:
                raise IndexError(
                    f"evaluation index {args.evaluation_index} is outside 0..{len(evaluation_ids) - 1}"
                ) from error
        else:
            evaluation_id = args.evaluation_id
            if evaluation_id not in evaluation_ids:
                raise ValueError(f"{evaluation_id!r} is not present in samples.csv")
        path = extract_campaign_evaluation(
            args.campaign_root,
            evaluation_id,
            output_root=args.output_root,
            config=_config(args),
            overwrite=args.overwrite,
        )
        print(path)
        return

    if args.command == "file":
        fraction_rows, quiescent_rows, metadata = extract_evaluation(
            args.galacticus_file,
            evaluation_id=args.evaluation_id,
            params_xml=args.params_xml,
            config=_config(args),
        )
        path = write_evaluation_shard(
            args.output_shard,
            fraction_rows,
            quiescent_rows,
            metadata,
            overwrite=args.overwrite,
        )
        print(path)
        return

    output_root = args.output_root or default_output_root(args.campaign_root)
    outputs = aggregate_campaign_shards(
        args.campaign_root,
        output_root=output_root,
        require_complete=not args.allow_missing,
        overwrite=args.overwrite,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
