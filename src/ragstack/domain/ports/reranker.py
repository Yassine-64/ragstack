from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, same order. Higher is better."""
        ...
