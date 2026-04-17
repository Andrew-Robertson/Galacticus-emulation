from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


def expand_env_vars(value: str, env: dict[str, str]) -> str:
    """Expand shell-style environment variables in a string."""
    expanded = value
    merged_env = os.environ.copy()
    merged_env.update(env)
    for key, env_value in merged_env.items():
        expanded = expanded.replace(f"${key}", env_value)
        expanded = expanded.replace(f"${{{key}}}", env_value)
    return os.path.expanduser(expanded)


@dataclass(frozen=True)
class PathsConfig:
    galacticus_binary_expr: str
    galacticus_parameter_files_expr: str
    galacticus_dust_modelling_expr: str
    run_root_expr: str
    galacticus_binary: Path
    galacticus_parameter_files: Path
    galacticus_dust_modelling: Path
    run_root: Path
    base_parameters: str
    mass_function_parameters: str
    threads: int
    environment: dict[str, str]

    def resolve_path(self, value: str) -> Path:
        return Path(expand_env_vars(value, self.environment))

    def expanded_base_parameters(self) -> Path:
        return self.resolve_path(self.base_parameters)

    def expanded_mass_function_parameters(self) -> Path:
        return self.resolve_path(self.mass_function_parameters)


def load_paths_config(path: str | Path) -> PathsConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    paths = raw["paths"]
    defaults = raw.get("defaults", {})
    env = raw.get("environment", {}).copy()

    if "galacticus_parameter_files" in paths:
        env.setdefault(
            "GALACTICUS_PARAMETER_FILES",
            expand_env_vars(paths["galacticus_parameter_files"], env),
        )
    if "galacticus_dust_modelling" in paths:
        env.setdefault(
            "GALACTICUS_DUST_ROOT",
            expand_env_vars(paths["galacticus_dust_modelling"], env),
        )
    if "galacticus_data_path" in raw.get("environment", {}):
        env.setdefault("GALACTICUS_DATA_PATH", raw["environment"]["galacticus_data_path"])
    if "GALACTICUS_EXEC_PATH" in raw.get("environment", {}):
        env.setdefault("GALACTICUS_EXEC_PATH", raw["environment"]["GALACTICUS_EXEC_PATH"])

    return PathsConfig(
        galacticus_binary_expr=paths["galacticus_binary"],
        galacticus_parameter_files_expr=paths["galacticus_parameter_files"],
        galacticus_dust_modelling_expr=paths["galacticus_dust_modelling"],
        run_root_expr=paths["run_root"],
        galacticus_binary=Path(expand_env_vars(paths["galacticus_binary"], env)),
        galacticus_parameter_files=Path(expand_env_vars(paths["galacticus_parameter_files"], env)),
        galacticus_dust_modelling=Path(expand_env_vars(paths["galacticus_dust_modelling"], env)),
        run_root=Path(expand_env_vars(paths["run_root"], env)),
        base_parameters=defaults["base_parameters"],
        mass_function_parameters=defaults["mass_function_parameters"],
        threads=int(defaults.get("threads", 1)),
        environment=env,
    )
