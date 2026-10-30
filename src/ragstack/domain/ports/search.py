from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ragstack.domain.models import Chunk, Principal


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    source_name: str
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None


@runtime_checkable
class SearchIndex(Protocol):
    """Hybrid candidate generation. MUST apply the principal's ACL inside the query."""

    async def search(
        self,
        principal: Principal,
        query_text: str,
        query_vector: list[float] | None,
        dense_k: int,
        sparse_k: int,
    ) -> list[SearchHit]: ...
