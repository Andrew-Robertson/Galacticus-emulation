from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu import EvaluationManifest, ParameterPoint
from galacticus_emu import build_galacticus_commands, load_paths_config


CONFIG_PATH = REPO_ROOT / "config" / "paths.toml"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "z0_validation_mcmc_bridge.json"


def starter_manifest() -> EvaluationManifest:
    return EvaluationManifest(
        evaluation_id="z0-validation-mcmc-bridge",
        base_parameters="$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/romanEPS_local.xml",
        run_definition_changes=[
            "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/universeMachine/z0_Likelihood.xml",
        ],
        parameter_points=[
            ParameterPoint("coolingRate/multiplier", 1.0),
            ParameterPoint("blackHoleWind/efficiencyWind", 0.01),
            ParameterPoint("hotHaloOutflowReincorporation/gamma", 0.5),
        ],
        mode="low_cost_likelihood",
        notes="Validation case bridging the old direct-MCMC workflow to the emulator workflow using the existing UniverseMachine z=0 likelihood setup.",
    )


def create_manifest() -> None:
    manifest = starter_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(MANIFEST_PATH)
    print(MANIFEST_PATH)


def show_commands() -> None:
    config = load_paths_config(CONFIG_PATH)
    manifest = EvaluationManifest.from_json(MANIFEST_PATH)
    for command in build_galacticus_commands(manifest, config):
        print(" ".join(command))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create-manifest", "show-commands"])
    args = parser.parse_args()

    if args.action == "create-manifest":
        create_manifest()
    elif args.action == "show-commands":
        show_commands()


if __name__ == "__main__":
    main()
