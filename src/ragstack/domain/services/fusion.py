"""Reciprocal Rank Fusion (Cormack et al. 2009): score = Σ 1/(k + rank). k=60 is the usual default."""

from __future__ import annotations

from ragstack.domain.models import RetrievedChunk, Scores
from ragstack.domain.ports.search import SearchHit


def reciprocal_rank_fusion(
    hits_per_query: list[tuple[str, list[SearchHit]]], k: int = 60
) -> list[RetrievedChunk]:
    """Fuses dense and sparse ranks across one or more (sub)queries. One entry per chunk id."""
    fused: dict[str, RetrievedChunk] = {}
    for query, hits in hits_per_query:
        for hit in hits:
            contribution = 0.0
            for rank in (hit.dense_rank, hit.sparse_rank):
                if rank is not None:
                    contribution += 1.0 / (k + rank)
            existing = fused.get(hit.chunk.id)
            if existing is None:
                fused[hit.chunk.id] = RetrievedChunk(
                    chunk=hit.chunk,
                    source_name=hit.source_name,
                    query=query,
                    scores=Scores(
                        dense=hit.dense_score,
                        sparse=hit.sparse_score,
                        dense_rank=hit.dense_rank,
                        sparse_rank=hit.sparse_rank,
                        rrf=contribution,
                    ),
                )
            else:
                existing.scores.rrf = (existing.scores.rrf or 0.0) + contribution
                if hit.dense_rank is not None and (
                    existing.scores.dense_rank is None or hit.dense_rank < existing.scores.dense_rank
                ):
                    existing.scores.dense_rank, existing.scores.dense = (
                        hit.dense_rank,
                        hit.dense_score,
                    )
                if hit.sparse_rank is not None and (
                    existing.scores.sparse_rank is None or hit.sparse_rank < existing.scores.sparse_rank
                ):
                    existing.scores.sparse_rank, existing.scores.sparse = (
                        hit.sparse_rank,
                        hit.sparse_score,
                    )
    return sorted(fused.values(), key=lambda r: r.scores.rrf or 0.0, reverse=True)
