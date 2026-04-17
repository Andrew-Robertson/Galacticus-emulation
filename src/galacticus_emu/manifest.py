from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ParameterPoint:
    name: str
    value: float


@dataclass(frozen=True)
class RunGroup:
    name: str
    run_definition_changes: list[str]


@dataclass(frozen=True)
class EvaluationManifest:
    evaluation_id: str
    base_parameters: str
    run_definition_changes: list[str]
    parameter_points: list[ParameterPoint]
    run_groups: list[RunGroup] | None = None
    mode: str = "low_cost_likelihood"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return output_path

    @classmethod
    def from_json(cls, path: str | Path) -> "EvaluationManifest":
        raw = json.loads(Path(path).read_text())
        return cls(
            evaluation_id=raw["evaluation_id"],
            base_parameters=raw["base_parameters"],
            run_definition_changes=list(raw["run_definition_changes"]),
            parameter_points=[ParameterPoint(**item) for item in raw["parameter_points"]],
            run_groups=[RunGroup(**item) for item in raw.get("run_groups", [])] or None,
            mode=raw.get("mode", "low_cost_likelihood"),
            notes=raw.get("notes", ""),
        )
