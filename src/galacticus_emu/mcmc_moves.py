from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any


MOVE_ALIASES = {
    "de": "DEMove",
    "demove": "DEMove",
    "desnooker": "DESnookerMove",
    "desnookermove": "DESnookerMove",
    "gaussian": "GaussianMove",
    "gaussianmove": "GaussianMove",
    "kde": "KDEMove",
    "kdemove": "KDEMove",
    "stretch": "StretchMove",
    "stretchmove": "StretchMove",
    "walk": "WalkMove",
    "walkmove": "WalkMove",
}


def add_emcee_move_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--move",
        action="append",
        default=[],
        metavar="NAME[:WEIGHT][,KEY=VALUE...]",
        help=(
            "emcee proposal move to use. Repeat for a weighted mixture, e.g. "
            "--move StretchMove:0.3 --move DEMove:0.6 --move DESnookerMove:0.1. "
            "Optional constructor kwargs may be supplied after commas."
        ),
    )


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_move_spec(spec: str) -> tuple[str, float, dict[str, Any]]:
    parts = [part.strip() for part in spec.split(",")]
    if not parts or not parts[0]:
        raise ValueError("--move entries must start with a move name")

    name_part = parts[0]
    weight = 1.0
    if ":" in name_part:
        name, raw_weight = name_part.split(":", 1)
        if not raw_weight:
            raise ValueError(f"Missing weight in --move {spec!r}")
        weight = float(raw_weight)
    else:
        name = name_part

    kwargs: dict[str, Any] = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --move option {part!r}; expected KEY=VALUE")
        key, raw_value = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty --move keyword in {spec!r}")
        if key == "weight":
            weight = float(raw_value)
        else:
            kwargs[key] = _parse_scalar(raw_value.strip())

    if weight <= 0.0:
        raise ValueError(f"Move weight must be positive in --move {spec!r}")

    canonical = MOVE_ALIASES.get(name.replace("_", "").replace("-", "").lower(), name)
    return canonical, weight, kwargs


def build_emcee_moves(emcee_module: Any, specs: Sequence[str]) -> list[tuple[Any, float]] | None:
    if not specs:
        return None
    moves = []
    for spec in specs:
        class_name, weight, kwargs = _parse_move_spec(spec)
        move_class = getattr(emcee_module.moves, class_name, None)
        if move_class is None:
            available = sorted(name for name in dir(emcee_module.moves) if name.endswith("Move"))
            raise ValueError(f"Unknown emcee move {class_name!r}. Available moves include: {', '.join(available)}")
        moves.append((move_class(**kwargs), weight))
    return moves


def describe_emcee_moves(specs: Sequence[str]) -> list[dict[str, Any]] | str:
    if not specs:
        return "emcee_default"
    description = []
    for spec in specs:
        class_name, weight, kwargs = _parse_move_spec(spec)
        description.append({"name": class_name, "weight": float(weight), "kwargs": kwargs})
    return description
