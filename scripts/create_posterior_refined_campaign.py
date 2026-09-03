from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from galacticus_emu.interactive_observables import _parameter_specs_for_columns
from galacticus_emu.lhs import transform_to_prior_quantiles


DEFAULT_SOURCE_CAMPAIGN = (
    REPO_ROOT
    / "runs/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_freeYield_20p_moreHalos_512_hybrid_5fallback"
)
DEFAULT_MCMC_DIR = (
    DEFAULT_SOURCE_CAMPAIGN
    / "pipeline/MCMCs/standard_observables_plus_emission_line_lfs"
    / "all_standard_observables_plus_halpha_sobral_1dustdraw_pca99"
)
DEFAULT_OUTPUT_CAMPAIGN = (
    REPO_ROOT
    / "runs/campaigns"
    / "sobol_mass_function_emissionlines_dust_simpleSizes_freeYield_20p_moreHalos_512_hybrid_5fallback_plus_25posterior_draws"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a symlinked campaign that augments a base campaign with actual posterior-draw runs."
    )
    parser.add_argument("--source-campaign", type=Path, default=DEFAULT_SOURCE_CAMPAIGN)
    parser.add_argument("--mcmc-dir", type=Path, default=DEFAULT_MCMC_DIR)
    parser.add_argument("--posterior-runs-dir", type=Path, default=None)
    parser.add_argument("--output-campaign", type=Path, default=DEFAULT_OUTPUT_CAMPAIGN)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _relative_symlink_target(source: Path, link_path: Path) -> str:
    return os.path.relpath(source.resolve(), start=link_path.parent.resolve())


def _symlink(source: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(_relative_symlink_target(source, link_path), target_is_directory=source.is_dir())


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _posterior_samples(
    base_samples: pd.DataFrame,
    posterior_manifest: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(base_samples.columns)
    physical_columns = [column for column in columns if column in posterior_manifest.columns]
    quantile_columns = [column for column in columns if column.endswith("_quantile")]
    input_columns = [column.removesuffix("_quantile") for column in quantile_columns]
    specs = _parameter_specs_for_columns(input_columns)

    physical = posterior_manifest[input_columns].to_numpy(dtype=float)
    quantiles = transform_to_prior_quantiles(specs, physical)

    rows = []
    for row_index, manifest_row in posterior_manifest.iterrows():
        row = {column: np.nan for column in columns}
        draw_index = int(manifest_row["draw_index"])
        row["evaluation_id"] = f"posterior_draw_{draw_index:02d}"
        for column in physical_columns:
            row[column] = manifest_row[column]
        for column_index, column in enumerate(quantile_columns):
            row[column] = quantiles[row_index, column_index]
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _patch_sidecar_tables(run_dir: Path, output_sidecar_dir: Path, sample_row: pd.Series) -> list[str]:
    output_sidecar_dir.mkdir(parents=True, exist_ok=True)
    patched = []
    quantile_values = {
        column: sample_row[column]
        for column in sample_row.index
        if str(column).endswith("_quantile")
    }
    for filename in [
        "emission_line_dust_lf_emulator_table.csv",
        "emission_line_dust_lf_long.csv",
        "dust_draws.csv",
        "manifest.csv",
    ]:
        source = run_dir / "emission_line_dust" / filename
        if not source.exists():
            continue
        if filename.endswith(".csv"):
            frame = pd.read_csv(source)
            for column, value in quantile_values.items():
                if column not in frame.columns:
                    frame[column] = float(value)
            if "evaluation_id" in frame.columns:
                frame["evaluation_id"] = str(sample_row["evaluation_id"])
            frame.to_csv(output_sidecar_dir / filename, index=False)
        else:
            shutil.copy2(source, output_sidecar_dir / filename)
        patched.append(filename)
    for filename in ["metadata.json"]:
        _copy_if_exists(run_dir / "emission_line_dust" / filename, output_sidecar_dir / filename)
    return patched


def main() -> None:
    args = parse_args()
    source_campaign = args.source_campaign.expanduser().resolve()
    mcmc_dir = args.mcmc_dir.expanduser().resolve()
    posterior_runs_dir = (
        args.posterior_runs_dir.expanduser().resolve()
        if args.posterior_runs_dir is not None
        else mcmc_dir / "posterior_draw_actual_runs"
    )
    output_campaign = args.output_campaign.expanduser().resolve()

    if output_campaign.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_campaign} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_campaign)

    source_samples = pd.read_csv(source_campaign / "samples.csv")
    posterior_manifest = pd.read_csv(posterior_runs_dir / "posterior_draw_manifest.csv")
    posterior_samples = _posterior_samples(source_samples, posterior_manifest)
    all_samples = pd.concat([source_samples, posterior_samples], ignore_index=True)

    output_campaign.mkdir(parents=True)
    (output_campaign / "evaluations").mkdir()
    all_samples.to_csv(output_campaign / "samples.csv", index=False)
    posterior_samples.to_csv(output_campaign / "posterior_refinement_samples.csv", index=False)

    for filename in [
        "campaign_design.json",
        "hybrid_campaign.json",
        "hybrid_source_manifest.csv",
        "fallback_evaluations.txt",
        "COMMAND_LOG.md",
        "commands.sh",
        "commands.txt",
    ]:
        _copy_if_exists(source_campaign / filename, output_campaign / filename)

    _symlink(source_campaign, output_campaign / "source_campaign")
    _symlink(mcmc_dir, output_campaign / "source_mcmc")
    _symlink(posterior_runs_dir, output_campaign / "source_posterior_draw_actual_runs")
    source_pipeline_yaml = source_campaign / "pipeline" / "pipeline.yaml"
    if source_pipeline_yaml.exists():
        pipeline_dir = output_campaign / "pipeline"
        pipeline_dir.mkdir()
        pipeline_yaml = pipeline_dir / "pipeline.yaml"
        source_root_text = str(source_campaign.relative_to(REPO_ROOT))
        output_root_text = str(output_campaign.relative_to(REPO_ROOT))
        pipeline_yaml.write_text(source_pipeline_yaml.read_text().replace(source_root_text, output_root_text))
        (pipeline_dir / "README.md").write_text(
            "# Clean Pipeline Scaffold\n\n"
            "This pipeline directory is intentionally empty apart from this README and `pipeline.yaml`.\n\n"
            "Any emulators, MCMCs, figures, or validation products in this directory should be "
            "generated fresh from the posterior-refined campaign data in the parent directory.\n"
        )

    source_eval_rows = []
    for evaluation_id in source_samples["evaluation_id"].astype(str):
        source_eval_dir = source_campaign / "evaluations" / evaluation_id
        if not source_eval_dir.exists():
            raise FileNotFoundError(f"Missing source evaluation directory: {source_eval_dir}")
        link_path = output_campaign / "evaluations" / evaluation_id
        _symlink(source_eval_dir, link_path)
        source_eval_rows.append(
            {
                "evaluation_id": evaluation_id,
                "source_type": "base_campaign",
                "source_path": str(source_eval_dir),
                "link_path": str(link_path),
            }
        )

    posterior_eval_rows = []
    for sample_row in posterior_samples.itertuples(index=False):
        evaluation_id = str(sample_row.evaluation_id)
        run_dir = posterior_runs_dir / evaluation_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing posterior run directory: {run_dir}")
        output_eval_dir = output_campaign / "evaluations" / evaluation_id
        output_eval_dir.mkdir()

        reduced_hdf5 = run_dir / "romanEPS_massFunction_reduced.hdf5"
        if not reduced_hdf5.exists():
            raise FileNotFoundError(f"Missing reduced HDF5 for {evaluation_id}: {reduced_hdf5}")
        for link_name in ["romanEPS_massFunction_reduced.hdf5", "galacticus.hdf5", "galacticus_reduced.hdf5"]:
            _symlink(reduced_hdf5, output_eval_dir / link_name)

        for filename in ["model_changes.xml", "params.xml"]:
            _symlink(run_dir / filename, output_eval_dir / filename)

        row_series = posterior_samples.loc[posterior_samples["evaluation_id"] == evaluation_id].iloc[0]
        patched_files = _patch_sidecar_tables(run_dir, output_eval_dir / "emission_line_dust", row_series)
        posterior_eval_rows.append(
            {
                "evaluation_id": evaluation_id,
                "source_type": "posterior_refinement",
                "source_path": str(run_dir),
                "link_path": str(output_eval_dir),
                "patched_sidecar_files": ";".join(patched_files),
            }
        )

    provenance = {
        "campaign_type": "posterior_refined_symlink_campaign",
        "source_campaign": str(source_campaign),
        "source_mcmc": str(mcmc_dir),
        "posterior_runs_dir": str(posterior_runs_dir),
        "output_campaign": str(output_campaign),
        "base_evaluation_count": int(len(source_samples)),
        "posterior_refinement_evaluation_count": int(len(posterior_samples)),
        "total_evaluation_count": int(len(all_samples)),
        "notes": [
            "Base campaign evaluation directories are symlinks.",
            "Posterior refinement evaluation directories symlink reduced HDF5/XML files.",
            "Posterior refinement emission_line_dust CSVs are copied and augmented with slow-parameter quantile columns.",
            "This derived campaign is intended for local emulator refinement, not independent validation.",
        ],
    }
    (output_campaign / "posterior_refined_campaign_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    pd.DataFrame(source_eval_rows + posterior_eval_rows).to_csv(
        output_campaign / "evaluation_source_manifest.csv",
        index=False,
    )

    readme = f"""# Posterior-Refined Campaign

This derived campaign augments the base 512-run hybrid campaign with 25 actual
Galacticus posterior-draw runs from the joint standard-observable plus Halpha MCMC.

- Base campaign: `{source_campaign}`
- MCMC source: `{mcmc_dir}`
- Posterior actual runs: `{posterior_runs_dir}`
- Total evaluations in `samples.csv`: {len(all_samples)}

Bookkeeping:

- `evaluations/<base eval>` are symlinks to the base campaign evaluations.
- `evaluations/posterior_draw_XX` are lightweight directories:
  - `galacticus.hdf5`, `galacticus_reduced.hdf5`, and `romanEPS_massFunction_reduced.hdf5`
    all symlink to the copied posterior reduced HDF5.
  - `model_changes.xml` and `params.xml` are symlinks.
  - `emission_line_dust/*.csv` are copied and augmented with slow-parameter
    quantile columns so sidecar LF emulator training can consume them.

Use this campaign for emulator refinement around the fitted posterior region.
Keep the original campaign and the 25 posterior runs as separate validation
objects when assessing bias or claiming predictive performance.

The clean `pipeline/pipeline.yaml` scaffold has its `ROOT` variable rewritten to
this derived campaign, so config-driven emulator/MCMC outputs will go below this
campaign's `pipeline/` directory when generated fresh.
"""
    (output_campaign / "README.md").write_text(readme)

    print(output_campaign)
    print(output_campaign / "samples.csv")
    print(output_campaign / "evaluation_source_manifest.csv")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
