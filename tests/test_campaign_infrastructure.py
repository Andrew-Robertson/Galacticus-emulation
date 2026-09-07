from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from galacticus_emu import (
    CampaignDefinition,
    ParameterSpec,
    PathsConfig,
    SlurmArrayDefinition,
    UniformPrior,
    write_campaign,
    write_changes_file,
)


def _paths_config(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        galacticus_binary_expr="$GALACTICUS_EXEC_PATH/Galacticus.exe",
        galacticus_parameter_files_expr="$GALACTICUS_PARAMETER_FILES",
        run_root_expr=str(tmp_path / "runs"),
        galacticus_binary=Path("/opt/galacticus/Galacticus.exe"),
        galacticus_parameter_files=Path("/opt/galacticus-parameters"),
        run_root=tmp_path / "runs",
        base_parameters="$GALACTICUS_PARAMETER_FILES/base.xml",
        mass_function_parameters="$GALACTICUS_PARAMETER_FILES/mass-function.xml",
        threads=4,
        environment={},
    )


def test_write_changes_file(tmp_path: Path) -> None:
    output_path = write_changes_file(
        [("coolingRate/multiplier", 0.75), ("outputFileName", "output.hdf5")],
        tmp_path / "nested" / "changes.xml",
        random_seed=42,
    )

    root = ET.parse(output_path).getroot()
    changes = root.findall("change")
    assert [(change.get("type"), change.get("path"), change.get("value")) for change in changes[:2]] == [
        ("update", "coolingRate/multiplier", "0.75"),
        ("update", "outputFileName", "output.hdf5"),
    ]
    assert changes[2].get("type") == "replaceOrAppend"
    assert changes[2].find("randomNumberGenerator/seed").get("value") == "42"


def test_write_campaign_with_native_slurm_configuration(tmp_path: Path) -> None:
    config = _paths_config(tmp_path)
    campaign = CampaignDefinition(
        campaign_name="example",
        n_eval=2,
        seed=42,
        base_parameters="$GALACTICUS_PARAMETER_FILES/base.xml",
        run_definition_changes=["$GALACTICUS_PARAMETER_FILES/observables.xml"],
        run_groups=None,
        mode="emulator_training",
        notes="Test campaign",
    )
    slurm = SlurmArrayDefinition(
        cpus_per_task=8,
        time_limit="24:00:00",
        partition="science",
        account="galaxies",
        conda_env="galacticemu",
        memory_per_cpu="8G",
        qos="normal",
        module_commands=("module load conda",),
        email="researcher@example.org",
        mail_type="FAIL",
    )

    root = write_campaign(
        config=config,
        campaign=campaign,
        parameter_specs=[
            ParameterSpec(
                path="coolingRate/multiplier",
                short_name="coolingMultiplier",
                prior=UniformPrior(lower=0.5, upper=1.0),
            )
        ],
        slurm_array=slurm,
    )

    commands = (root / "commands.txt").read_text().splitlines()
    assert len(commands) == 2
    assert "$GALACTICUS_EXEC_PATH/Galacticus.exe" in commands[0]
    assert "$GALACTICUS_PARAMETER_FILES/base.xml" in commands[0]
    assert "evaluations/example-eval-0000/theta.xml" in commands[0]

    definition = json.loads((root / "campaign.json").read_text())
    assert definition["slurm_array"]["module_commands"] == ["module load conda"]

    script = (root / "submit_slurm_array.sh").read_text()
    assert "#SBATCH --array=0-1" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem-per-cpu=8G" in script
    assert "#SBATCH --partition=science" in script
    assert "#SBATCH --account=galaxies" in script
    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --mail-user=researcher@example.org" in script
    assert "#SBATCH --mail-type=FAIL" in script
    assert "module load conda" in script
    assert "conda activate galacticemu" in script
