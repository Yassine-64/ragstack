from __future__ import annotations

import math

from ragstack.domain.models import EvalRun

_COLUMNS = [
    "recall@k",
    "mrr",
    "ndcg@k",
    "faithfulness",
    "citation_precision",
    "correctness",
    "refusal_correct",
    "leak_rate",
    "latency_p50_ms",
    "cost_usd_per_query",
]


def _fmt(v: float | None, key: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if key.endswith("_ms"):
        return f"{v:.0f}"
    if key.startswith("cost"):
        return f"${v:.4f}"
    return f"{v:.3f}"


def comparison_table(runs: list[EvalRun]) -> str:
    cols = [c for c in _COLUMNS if any(c in r.metrics for r in runs)]
    lines = ["| config | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in runs:
        lines.append(f"| {r.name} | " + " | ".join(_fmt(r.metrics.get(c), c) for c in cols) + " |")
    return "\n".join(lines) + "\n"
