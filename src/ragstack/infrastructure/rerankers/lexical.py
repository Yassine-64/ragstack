"""Offline reranker for tests: token overlap. Deterministic, dependency-free."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")


class LexicalReranker:
    name = "lexical"

    async def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(_TOKEN.findall(query.lower()))
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(_TOKEN.findall(t.lower()))) / len(q) for t in texts]
