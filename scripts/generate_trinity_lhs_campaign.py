from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from galacticus_emu import CampaignDefinition, RunGroup, SlurmArrayDefinition, load_paths_config, write_campaign
from galacticus_emu.lhs import ParameterSpec, TruncatedLogNormalPrior, UniformPrior


CONFIG_PATH = REPO_ROOT / "config" / "paths.toml"


def trinity_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic",
            short_name="diskVelocityCharacteristic",
            prior=TruncatedLogNormalPrior(lower=25.0, upper=300.0, x0=150.0, sigma=0.5),
        ),
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent",
            short_name="diskExponent",
            prior=UniformPrior(lower=0.0, upper=4.0),
        ),
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='stellarFeedbackSpheroids']/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic",
            short_name="spheroidVelocityCharacteristic",
            prior=TruncatedLogNormalPrior(lower=10.0, upper=150.0, x0=50.0, sigma=1.0),
        ),
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='stellarFeedbackSpheroids']/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent",
            short_name="spheroidExponent",
            prior=UniformPrior(lower=0.0, upper=4.0),
        ),
        ParameterSpec(
            path="coolingRate/multiplier",
            short_name="coolingMultiplier",
            prior=TruncatedLogNormalPrior(lower=0.05, upper=1.2, x0=0.5, sigma=1.0),
        ),
        ParameterSpec(
            path="blackHoleCGMHeating/efficiencyRadioMode",
            short_name="BHefficiencyRadioMode",
            prior=TruncatedLogNormalPrior(lower=0.03, upper=1.0, x0=0.3, sigma=0.5),
        ),
        ParameterSpec(
            path="blackHoleWind/efficiencyWind",
            short_name="BHefficiencyWind",
            prior=TruncatedLogNormalPrior(lower=0.00024, upper=0.024, x0=0.0024, sigma=1.0),
        ),
        ParameterSpec(
            path="accretionDisks/accretionRateThinDiskMaximum",
            short_name="thinDiskMaximum",
            prior=TruncatedLogNormalPrior(lower=0.03, upper=30.0, x0=3.0, sigma=2.0),
        ),
        ParameterSpec(
            path="blackHoleAccretionRate/bondiHoyleAccretionEnhancementSpheroid",
            short_name="bondiEnhancementSpheroid",
            prior=TruncatedLogNormalPrior(lower=0.05, upper=100.0, x0=5.0, sigma=2.0),
        ),
        ParameterSpec(
            path="blackHoleAccretionRate/bondiHoyleAccretionEnhancementHotHalo",
            short_name="bondiEnhancementHotHalo",
            prior=TruncatedLogNormalPrior(lower=0.05, upper=100.0, x0=6.0, sigma=2.0),
        ),
        ParameterSpec(
            path="blackHoleAccretionRate/bondiHoyleAccretionTemperatureSpheroid",
            short_name="bondiTemperatureSpheroid",
            prior=TruncatedLogNormalPrior(lower=10.0, upper=1000.0, x0=100.0, sigma=2.0),
        ),
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='blackHolesSeed']/blackHoleSeeds/mass",
            short_name="blackHoleSeedMass",
            prior=TruncatedLogNormalPrior(lower=100.0, upper=1.0e5, x0=3.0e3, sigma=4.0),
        ),
        ParameterSpec(
            path="starFormationRateSpheroids/starFormationTimescale/efficiency",
            short_name="spheroidSFREfficiency",
            prior=TruncatedLogNormalPrior(lower=1.0e-4, upper=1.0, x0=0.01, sigma=4.0),
        ),
        ParameterSpec(
            path="starFormationRateSpheroids/starFormationTimescale/exponentVelocity",
            short_name="spheroidSFRExponentVelocity",
            prior=UniformPrior(lower=-3.5, upper=2.0),
        ),
        ParameterSpec(
            path="starFormationRateSurfaceDensityDisks/starFormationFrequencyNormalization",
            short_name="diskSFRFrequencyNorm",
            prior=TruncatedLogNormalPrior(lower=1.0e-10, upper=1.0e-8, x0=1.0e-9, sigma=2.0),
        ),
        ParameterSpec(
            path="hotHaloMassDistributionCoreRadius/coreRadiusOverVirialRadius",
            short_name="coreRadiusOverVirialRadius",
            prior=TruncatedLogNormalPrior(lower=0.03, upper=1.0, x0=0.3, sigma=1.0),
        ),
        ParameterSpec(
            path="hotHaloOutflowReincorporation/gamma",
            short_name="henriquesGamma",
            prior=TruncatedLogNormalPrior(lower=1.0, upper=25.0, x0=5.0, sigma=0.5),
        ),
        ParameterSpec(
            path="hotHaloOutflowReincorporation/delta1",
            short_name="henriquesDelta1",
            prior=UniformPrior(lower=-2.0, upper=4.0),
        ),
        ParameterSpec(
            path="hotHaloOutflowReincorporation/delta2",
            short_name="henriquesDelta2",
            prior=UniformPrior(lower=-1.0, upper=5.0),
        ),
        ParameterSpec(
            path="mergerMassMovements/massRatioMajorMerger",
            short_name="massRatioMajorMerger",
            prior=TruncatedLogNormalPrior(lower=0.1, upper=0.5, x0=0.25, sigma=0.5),
        ),
        ParameterSpec(
            path="mergerRemnantSize/energyOrbital",
            short_name="energyOrbital",
            prior=TruncatedLogNormalPrior(lower=0.25, upper=4.0, x0=1.0, sigma=0.5),
        ),
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='barInstability']/galacticDynamicsBarInstability/fractionAngularMomentumRetainedSpheroid",
            short_name="barFractionAngularMomentumRetainedSpheroid",
            prior=TruncatedLogNormalPrior(lower=0.05, upper=0.95, x0=0.2, sigma=0.5),
        ),
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='barInstability']/galacticDynamicsBarInstability/stabilityThresholdGaseous",
            short_name="barStabilityThresholdGaseous",
            prior=TruncatedLogNormalPrior(lower=0.5, upper=0.9, x0=0.7, sigma=0.5),
        ),
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='barInstability']/galacticDynamicsBarInstability/stabilityThresholdStellar",
            short_name="barStabilityThresholdStellar",
            prior=TruncatedLogNormalPrior(lower=0.8, upper=1.4, x0=1.1, sigma=0.5),
        ),
    ]


def default_run_groups() -> list[RunGroup]:
    common_prefix = [
        "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/handCalibratedChanges.xml",
        "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/odeAcc1e-4.xml",
    ]
    seed_file = "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/seeds/seed1234.xml"
    return [
        RunGroup(
            name="z0",
            run_definition_changes=[
                *common_prefix,
                "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/universeMachine/Trinity/z0_Likelihood_TrinityScatter.xml",
                seed_file,
            ],
        ),
        RunGroup(
            name="z1",
            run_definition_changes=[
                *common_prefix,
                "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/universeMachine/Trinity/z1_Likelihood_TrinityScatter.xml",
                seed_file,
            ],
        ),
        RunGroup(
            name="z2",
            run_definition_changes=[
                *common_prefix,
                "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/universeMachine/Trinity/z2_Likelihood_TrinityScatter.xml",
                seed_file,
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Trinity z0/z1/z2 Latin hypercube Galacticus campaign."
    )
    parser.add_argument("--campaign-name", default="lhs_trinity_moreparams_512")
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-parameters",
        default="$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/romanEPS.xml",
    )
    parser.add_argument(
        "--mode",
        default="low_cost_likelihood",
        choices=["low_cost_likelihood", "emulator_training"],
    )
    parser.add_argument(
        "--notes",
        default=(
            "Dry-run Trinity z0/z1/z2 Latin hypercube campaign over the 24-parameter PSO-inspired space. "
            "Each theta is evaluated through three Galacticus runs sharing the same parameter changes file."
        ),
    )
    parser.add_argument("--write-slurm-array", action="store_true")
    parser.add_argument("--platform-config", default=None)
    parser.add_argument("--platform-config-file", default=None)
    parser.add_argument("--slurm-cpus-per-task", type=int, default=None)
    parser.add_argument("--slurm-time-limit", default=None)
    parser.add_argument("--slurm-partition", default=None)
    parser.add_argument("--slurm-account", default=None)
    parser.add_argument("--slurm-conda-env", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paths_config(CONFIG_PATH)
    campaign = CampaignDefinition(
        campaign_name=args.campaign_name,
        n_eval=args.n_eval,
        seed=args.seed,
        base_parameters=args.base_parameters,
        run_definition_changes=[],
        run_groups=default_run_groups(),
        mode=args.mode,
        notes=args.notes,
    )
    slurm_array = None
    if args.write_slurm_array:
        slurm_array = SlurmArrayDefinition(
            cpus_per_task=args.slurm_cpus_per_task,
            time_limit=args.slurm_time_limit,
            partition=args.slurm_partition,
            account=args.slurm_account,
            conda_env=args.slurm_conda_env,
            platform_config_name=args.platform_config,
            platform_config_file=args.platform_config_file,
        )
    output_root = write_campaign(
        config=config,
        campaign=campaign,
        parameter_specs=trinity_parameter_specs(),
        slurm_array=slurm_array,
    )
    print(output_root)


if __name__ == "__main__":
    main()
