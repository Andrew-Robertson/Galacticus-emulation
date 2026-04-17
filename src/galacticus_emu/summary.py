from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class SummaryStatistic:
    name: str
    value: float
    uncertainty: float | None = None


@dataclass(frozen=True)
class EvaluationSummary:
    evaluation_id: str
    statistics: list[SummaryStatistic]
    source_files: list[str]

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return output_path
