from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu.specs import trinity_parameter_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample rows from an MCMC posterior CSV and write changes.xml-style "
            "files that can be run through Galacticus."
        )
    )
    parser.add_argument("--posterior-samples", type=Path, required=True)
    parser.add_argument(
        "--run-summary",
        type=Path,
        required=True,
        help=(
            "MCMC run summary JSON. Used for parameter order, slow_count, and "
            "sidecar LF nuisance parameter names."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-draws", type=int, default=25)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Allow repeated posterior rows if --n-draws exceeds the number of posterior samples.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First integer used in posterior_draw_XX.xml filenames.",
    )
    parser.add_argument(
        "--filename-template",
        default="posterior_draw_{index:02d}.xml",
        help="Output filename template. Available fields: index, row_index.",
    )
    parser.add_argument(
        "--include-map",
        action="store_true",
        help="Also write maximum_a_posteriori_model_changes.xml from run_summary['best_theta'].",
    )
    return parser.parse_args()


def _parameter_specs_by_name():
    return {spec.short_name: spec for spec in trinity_parameter_specs()}


def _write_model_changes(
    *,
    theta: dict[str, float],
    parameter_names: list[str],
    slow_count: int,
    output_path: Path,
    comments: list[str],
) -> Path:
    specs_by_name = _parameter_specs_by_name()
    slow_parameter_names = parameter_names[:slow_count]
    missing = [name for name in slow_parameter_names if name not in specs_by_name]
    if missing:
        raise ValueError(f"No Galacticus parameter specification found for slow parameter(s): {missing}")

    root = ET.Element("changes")
    for comment in comments:
        root.append(ET.Comment(f" {comment} "))
    for name in slow_parameter_names:
        spec = specs_by_name[name]
        ET.SubElement(
            root,
            "change",
            type="update",
            path=spec.path,
            value=f"{float(theta[name]):.17g}",
        )

    sidecar_names = parameter_names[slow_count:]
    if sidecar_names:
        missing_sidecar = [name for name in sidecar_names if name not in theta]
        if missing_sidecar:
            raise ValueError(f"Posterior row is missing sidecar parameter(s): {missing_sidecar}")
        root.append(
            ET.Comment(
                " Sidecar LF nuisance parameters: "
                + ", ".join(f"{name}={float(theta[name]):.6g}" for name in sidecar_names)
                + " "
            )
        )

    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="UTF-8", xml_declaration=True)
    return output_path


def _theta_from_row(row: pd.Series, parameter_names: list[str]) -> dict[str, float]:
    missing = [name for name in parameter_names if name not in row.index]
    if missing:
        raise ValueError(f"Posterior samples are missing parameter column(s): {missing}")
    return {name: float(row[name]) for name in parameter_names}


def _theta_from_summary(summary: dict, parameter_names: list[str]) -> dict[str, float]:
    best_theta = summary.get("best_theta")
    if not isinstance(best_theta, dict):
        raise ValueError("run summary does not contain a best_theta object")
    missing = [name for name in parameter_names if name not in best_theta]
    if missing:
        raise ValueError(f"run summary best_theta is missing parameter(s): {missing}")
    return {name: float(best_theta[name]) for name in parameter_names}


def main() -> None:
    args = parse_args()
    if args.n_draws < 1:
        raise ValueError("--n-draws must be >= 1")

    summary = json.loads(args.run_summary.expanduser().resolve().read_text())
    posterior = pd.read_csv(args.posterior_samples.expanduser().resolve())
    parameter_names = summary.get("parameter_names")
    if not isinstance(parameter_names, list) or not all(isinstance(name, str) for name in parameter_names):
        parameter_names = [name for name in posterior.columns if name != "log_probability"]
    slow_count = int(summary.get("slow_count", len(parameter_names)))
    if not (0 <= slow_count <= len(parameter_names)):
        raise ValueError(f"Invalid slow_count={slow_count} for {len(parameter_names)} parameter(s)")
    if args.n_draws > len(posterior) and not args.replace:
        raise ValueError(
            f"Requested {args.n_draws} draws but posterior has only {len(posterior)} rows. "
            "Pass --replace to allow repeated rows."
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    row_indices = rng.choice(len(posterior), size=args.n_draws, replace=args.replace)
    manifest_rows = []
    for draw_number, row_index in enumerate(row_indices, start=args.start_index):
        row = posterior.iloc[int(row_index)]
        theta = _theta_from_row(row, parameter_names)
        filename = args.filename_template.format(index=draw_number, row_index=int(row_index))
        output_path = output_dir / filename
        comments = [
            f"Generated from posterior row {int(row_index)} of {args.posterior_samples}",
        ]
        if "log_probability" in row.index:
            comments.append(f"log_probability={float(row['log_probability']):.17g}")
        _write_model_changes(
            theta=theta,
            parameter_names=parameter_names,
            slow_count=slow_count,
            output_path=output_path,
            comments=comments,
        )
        manifest_row = {
            "draw_index": draw_number,
            "posterior_row_index": int(row_index),
            "xml_path": str(output_path),
        }
        if "log_probability" in row.index:
            manifest_row["log_probability"] = float(row["log_probability"])
        manifest_row.update(theta)
        manifest_rows.append(manifest_row)

    if args.include_map:
        map_path = output_dir / "maximum_a_posteriori_model_changes.xml"
        map_theta = _theta_from_summary(summary, parameter_names)
        comments = [f"Generated from best_theta in {args.run_summary}"]
        if "best_log_probability" in summary:
            comments.append(f"best_log_probability={float(summary['best_log_probability']):.17g}")
        _write_model_changes(
            theta=map_theta,
            parameter_names=parameter_names,
            slow_count=slow_count,
            output_path=map_path,
            comments=comments,
        )

    manifest_path = output_dir / "posterior_draw_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in manifest_rows for key in row)))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(output_dir)
    print(manifest_path)


if __name__ == "__main__":
    main()
