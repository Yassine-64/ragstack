"""Rank metrics over ids. `retrieved` is ordered; `relevant` is a set. All in [0,1]."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, rid in enumerate(retrieved[:k], 1) if rid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
