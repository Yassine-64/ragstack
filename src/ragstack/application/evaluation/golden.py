from __future__ import annotations

import json
from pathlib import Path

from ragstack.domain.models import EvalCase


def load_golden(path: Path) -> list[EvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            cases.append(EvalCase.model_validate(json.loads(line)))
    return cases
