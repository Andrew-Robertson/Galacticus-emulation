"""GalacticusEmu package."""

from .config import PathsConfig, load_paths_config
from .campaign import CampaignDefinition, SlurmArrayDefinition, write_campaign
from .extract import summarize_campaign
from .gp import compute_metrics, fit_cv_predictions, fit_scaled_gp, predict_scaled_gp
from .lhs import (
    ParameterSpec,
    TruncatedLogNormalPrior,
    UniformPrior,
    log_prior_density,
    transform_from_prior_quantiles,
    transform_to_prior_quantiles,
)
from .manifest import EvaluationManifest, ParameterPoint, RunGroup
from .persistence import load_emulator_bundle, sanitize_output_name, save_emulator_bundle
from .runner import build_galacticus_commands
from .specs import default_parameter_specs, trinity_parameter_specs
from .trinity import (
    TRINITY_MEAN_OUTPUT_COLUMNS,
    TRINITY_OUTPUT_LABELS,
    TRINITY_RUN_CONFIGS,
    TRINITY_SCATTER_OUTPUT_COLUMNS,
    TRINITY_TRAINABLE_OUTPUT_COLUMNS,
    load_or_build_emulator_table,
    summarize_trinity_campaign,
)

__all__ = [
    "CampaignDefinition",
    "compute_metrics",
    "EvaluationManifest",
    "fit_cv_predictions",
    "fit_scaled_gp",
    "log_prior_density",
    "ParameterSpec",
    "ParameterPoint",
    "RunGroup",
    "PathsConfig",
    "predict_scaled_gp",
    "SlurmArrayDefinition",
    "TruncatedLogNormalPrior",
    "UniformPrior",
    "build_galacticus_commands",
    "default_parameter_specs",
    "load_paths_config",
    "load_or_build_emulator_table",
    "summarize_campaign",
    "summarize_trinity_campaign",
    "save_emulator_bundle",
    "sanitize_output_name",
    "load_emulator_bundle",
    "transform_from_prior_quantiles",
    "transform_to_prior_quantiles",
    "trinity_parameter_specs",
    "TRINITY_MEAN_OUTPUT_COLUMNS",
    "TRINITY_OUTPUT_LABELS",
    "TRINITY_RUN_CONFIGS",
    "TRINITY_SCATTER_OUTPUT_COLUMNS",
    "TRINITY_TRAINABLE_OUTPUT_COLUMNS",
    "write_campaign",
]
