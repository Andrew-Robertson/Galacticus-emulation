from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

try:
    import yaml
except ImportError as error:  # pragma: no cover - exercised only in missing dependency environments.
    raise SystemExit(
        "PyYAML is required to read pipeline YAML files. Install pyyaml or run in the galacticus workspace env."
    ) from error


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
WHOLE_VARIABLE_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or print a config-driven campaign post-processing pipeline. "
            "Commands are stored as argv lists in YAML and are dry-run by default."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", action="append", default=None, help="Stage name to run/print. Repeat to run several.")
    parser.add_argument("--all", action="store_true", help="Run/print every non-manual stage in config order.")
    parser.add_argument("--profile", action="append", default=[], help="Apply a named config profile before --set overrides.")
    parser.add_argument(
        "--no-default-profiles",
        action="store_true",
        help="Ignore any top-level default_profiles listed in the config.",
    )
    parser.add_argument("--list", action="store_true", help="List available stages.")
    parser.add_argument("--show-vars", action="store_true", help="Print resolved variables.")
    parser.add_argument("--execute", action="store_true", help="Actually execute commands. Default is dry-run.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a scalar variable.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--write-log", type=Path, default=None, help="Write the rendered commands to a markdown file.")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return dict(config)


def _parse_overrides(values: Sequence[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --set KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty override key in {value!r}")
        overrides[key] = raw
    return overrides


def _expand_scalar(text: str, variables: Mapping[str, Any], stack: tuple[str, ...] = ()) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in stack:
            cycle = " -> ".join((*stack, name))
            raise ValueError(f"Variable expansion cycle: {cycle}")
        if name not in variables:
            raise KeyError(f"Unknown variable {name!r}")
        value = _resolve_value(variables[name], variables, (*stack, name))
        if isinstance(value, list):
            raise TypeError(f"Variable {name!r} is a list and cannot be embedded in a scalar")
        return str(value)

    return VARIABLE_RE.sub(replace, text)


def _resolve_value(value: Any, variables: Mapping[str, Any], stack: tuple[str, ...] = ()) -> Any:
    if isinstance(value, str):
        return _expand_scalar(value, variables, stack)
    if isinstance(value, list):
        return [_resolve_value(item, variables, stack) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item, variables, stack) for key, item in value.items()}
    return value


def _profile_variables(config: Mapping[str, Any], profile_names: Sequence[str]) -> dict[str, Any]:
    profiles = config.get("profiles", {}) or {}
    if not isinstance(profiles, Mapping):
        raise ValueError("config['profiles'] must be a mapping when present")
    merged: dict[str, Any] = {}
    for profile_name in profile_names:
        if profile_name not in profiles:
            raise ValueError(f"Unknown profile {profile_name!r}")
        profile = profiles[profile_name]
        if not isinstance(profile, Mapping):
            raise ValueError(f"Profile {profile_name!r} must be a mapping")
        variables = profile.get("variables", {})
        if not isinstance(variables, Mapping):
            raise ValueError(f"Profile {profile_name!r} variables must be a mapping")
        merged.update(variables)
    return merged


def _default_profiles(config: Mapping[str, Any]) -> list[str]:
    profiles = config.get("default_profiles", [])
    if profiles is None:
        return []
    if isinstance(profiles, str):
        return [profiles]
    if not isinstance(profiles, list):
        raise ValueError("config['default_profiles'] must be a string or list when present")
    return [str(profile) for profile in profiles]


def _resolve_variables(raw_variables: Mapping[str, Any], profile_variables: Mapping[str, Any], overrides: Mapping[str, str]) -> dict[str, Any]:
    variables = dict(raw_variables)
    variables.update(profile_variables)
    variables.update(overrides)
    return {key: _resolve_value(value, variables, (key,)) for key, value in variables.items()}


def _expand_argv(argv: Sequence[Any], variables: Mapping[str, Any]) -> list[str]:
    expanded: list[str] = []
    for item in argv:
        if not isinstance(item, str):
            expanded.append(str(item))
            continue
        whole_match = WHOLE_VARIABLE_RE.match(item)
        if whole_match is not None:
            value = variables.get(whole_match.group(1))
            if isinstance(value, list):
                expanded.extend(str(entry) for entry in value)
                continue
            if value is None:
                raise KeyError(f"Unknown variable {whole_match.group(1)!r}")
            expanded.append(str(value))
            continue
        expanded.append(_expand_scalar(item, variables))
    return expanded


def _stage_by_name(stages: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for stage in stages:
        name = str(stage.get("name", ""))
        if not name:
            raise ValueError("Every stage must have a non-empty name")
        if name in result:
            raise ValueError(f"Duplicate stage name: {name}")
        result[name] = stage
    return result


def _selected_stages(config: Mapping[str, Any], args: argparse.Namespace) -> list[Mapping[str, Any]]:
    stages = config.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("config['stages'] must be a list")
    if args.list or args.show_vars:
        return []
    if args.all:
        return [stage for stage in stages if not bool(stage.get("manual", False))]
    if not args.stage:
        raise ValueError("Pass --stage NAME, --all, --list, or --show-vars")
    by_name = _stage_by_name(stages)
    missing = [name for name in args.stage if name not in by_name]
    if missing:
        raise ValueError(f"Unknown stage(s): {', '.join(missing)}")
    return [by_name[name] for name in args.stage]


def _command_entries(stage: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    commands = stage.get("commands", [])
    if commands is None:
        return []
    if not isinstance(commands, list):
        raise ValueError(f"Stage {stage.get('name')} commands must be a list")
    return commands


def _render_stage(stage: Mapping[str, Any], variables: Mapping[str, Any]) -> list[tuple[str, list[str], Path]]:
    rendered = []
    for command_index, command in enumerate(_command_entries(stage), start=1):
        if "argv" not in command:
            raise ValueError(f"Stage {stage.get('name')} command {command_index} has no argv")
        argv = _expand_argv(command["argv"], variables)
        cwd = Path(_expand_scalar(str(command.get("cwd", str(REPO_ROOT))), variables)).expanduser().resolve()
        label = str(command.get("name", f"command {command_index}"))
        rendered.append((label, argv, cwd))
    return rendered


def _format_markdown(stages: Sequence[Mapping[str, Any]], variables: Mapping[str, Any]) -> str:
    lines = ["# Rendered Pipeline Commands", ""]
    for stage in stages:
        lines.extend([f"## {stage['name']}", ""])
        if stage.get("description"):
            lines.extend([str(stage["description"]), ""])
        if stage.get("manual"):
            lines.extend(["Manual stage; no commands were rendered.", ""])
            if stage.get("notes"):
                lines.extend([_expand_scalar(str(stage["notes"]), variables), ""])
            continue
        commands = _render_stage(stage, variables)
        if not commands:
            lines.extend(["No commands.", ""])
            continue
        for label, argv, cwd in commands:
            lines.extend([f"### {label}", "", "```bash", f"cd {shlex.quote(str(cwd))}", shlex.join(argv), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _print_stage(stage: Mapping[str, Any], variables: Mapping[str, Any]) -> None:
    print(f"\n# {stage['name']}")
    if stage.get("description"):
        print(str(stage["description"]))
    if stage.get("manual"):
        print("Manual stage; skipping command execution.")
        if stage.get("notes"):
            print(_expand_scalar(str(stage["notes"]), variables))
        return
    for label, argv, cwd in _render_stage(stage, variables):
        print(f"\n## {label}")
        print(f"cd {shlex.quote(str(cwd))}")
        print(shlex.join(argv))


def _execute_stage(stage: Mapping[str, Any], variables: Mapping[str, Any], *, continue_on_error: bool) -> None:
    if stage.get("manual"):
        raise SystemExit(f"Refusing to execute manual stage {stage['name']!r}")
    for label, argv, cwd in _render_stage(stage, variables):
        print(f"\n[{stage['name']}] {label}", flush=True)
        print(f"cd {cwd}", flush=True)
        print(shlex.join(argv), flush=True)
        completed = subprocess.run(argv, cwd=cwd)
        if completed.returncode != 0:
            message = f"Command failed with exit code {completed.returncode}: {label}"
            if continue_on_error:
                print(message, file=sys.stderr)
            else:
                raise SystemExit(message)


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    profile_names = [] if args.no_default_profiles else _default_profiles(config)
    profile_names.extend(args.profile)
    variables = _resolve_variables(config.get("variables", {}), _profile_variables(config, profile_names), _parse_overrides(args.set))
    stages = config.get("stages", [])

    if args.list:
        for stage in stages:
            manual = " manual" if stage.get("manual") else ""
            description = str(stage.get("description", "")).strip().splitlines()
            suffix = f" - {description[0]}" if description else ""
            print(f"{stage['name']}{manual}{suffix}")

    if args.show_vars:
        for key in sorted(variables):
            value = variables[key]
            if isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}={value}")

    selected = _selected_stages(config, args)
    if args.write_log is not None:
        args.write_log.expanduser().resolve().write_text(_format_markdown(selected, variables))

    for stage in selected:
        _print_stage(stage, variables)
        if args.execute:
            _execute_stage(stage, variables, continue_on_error=args.continue_on_error)
        elif not stage.get("manual"):
            print("\n(dry-run; pass --execute to run)")


if __name__ == "__main__":
    main()
