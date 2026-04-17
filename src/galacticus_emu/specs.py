from __future__ import annotations

from .lhs import ParameterSpec, TruncatedLogNormalPrior, UniformPrior


def default_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedbackOutflows/stellarFeedbackOutflows/velocityCharacteristic",
            short_name="diskVelocityCharacteristic",
            prior=TruncatedLogNormalPrior(lower=25.0, upper=300.0, x0=150.0, sigma=0.5),
        ),
        ParameterSpec(
            path="nodeOperator/nodeOperator[@value='stellarFeedbackDisks']/stellarFeedbackOutflows/stellarFeedbackOutflows/exponent",
            short_name="diskExponent",
            prior=UniformPrior(lower=0.0, upper=5.0),
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
            path="blackHoleCGMHeating/efficiencyRadioMode",
            short_name="BHefficiencyRadioMode",
            prior=TruncatedLogNormalPrior(lower=0.03, upper=1.0, x0=0.3, sigma=0.5),
        ),
        ParameterSpec(
            path="componentSpheroid/ratioAngularMomentumScaleRadius",
            short_name="spheroidRatioAngularMomentumScaleRadius",
            prior=TruncatedLogNormalPrior(lower=0.1, upper=0.5, x0=0.2, sigma=0.5),
        ),
    ]


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
            prior=TruncatedLogNormalPrior(lower=0.0001, upper=1.0, x0=0.01, sigma=4.0),
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
