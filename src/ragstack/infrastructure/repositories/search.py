"""The hybrid search statement. The ACL is a CTE every branch joins: unauthorized chunks are never
ranked, never returned, never logged (VaultRAG's argument). Dense and sparse top-N are computed
separately and returned with their ranks; fusion happens in the domain so it stays testable."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ragstack.domain.models import Principal
from ragstack.domain.ports.search import SearchHit

from .postgres import _chunk

_SQL = """
WITH visible AS (
    SELECT d.id, d.source_name
    FROM documents d
    WHERE d.tenant_id = %(tenant)s AND d.deleted_at IS NULL
      AND EXISTS (
          SELECT 1 FROM access_grants g
          WHERE g.document_id = d.id
            AND (   g.principal_type = 'everyone'
                 OR (g.principal_type = 'user'  AND g.principal_id = %(user)s)
                 OR (g.principal_type = 'group' AND g.principal_id = ANY(%(groups)s)))
      )
),
dense AS (
    SELECT c.id, 1 - (c.embedding <=> %(qvec)s::vector) AS score,
           row_number() OVER (ORDER BY c.embedding <=> %(qvec)s::vector) AS rank
    FROM chunks c JOIN visible v ON v.id = c.document_id
    WHERE %(dense_k)s > 0
    ORDER BY c.embedding <=> %(qvec)s::vector
    LIMIT %(dense_k)s
),
-- BM25 over context + text (index chunks_bm25_idx). `|||` tokenizes the raw question and matches any
-- term, so user text needs no escaping. ponytail: IDF comes from the whole table, not one tenant;
-- partition chunks by tenant if corpus statistics must not cross tenants.
sparse AS (
    SELECT c.id, pdb.score(c.id) AS score,
           row_number() OVER (ORDER BY pdb.score(c.id) DESC) AS rank
    FROM chunks c JOIN visible v ON v.id = c.document_id
    WHERE %(sparse_k)s > 0 AND (c.context || ' ' || c.text) ||| %(qtext)s
    ORDER BY pdb.score(c.id) DESC
    LIMIT %(sparse_k)s
),
candidates AS (
    SELECT id FROM dense UNION SELECT id FROM sparse
)
SELECT c.*, v.source_name,
       d.score AS dense_score, d.rank AS dense_rank, s.score AS sparse_score, s.rank AS sparse_rank
FROM candidates x
JOIN chunks c ON c.id = x.id
JOIN visible v ON v.id = c.document_id
LEFT JOIN dense d ON d.id = x.id
LEFT JOIN sparse s ON s.id = x.id
"""


class PostgresHybridSearch:
    def __init__(self, pool: AsyncConnectionPool, dimension: int) -> None:
        self.pool = pool
        self.dimension = dimension

    async def search(
        self,
        principal: Principal,
        query_text: str,
        query_vector: list[float] | None,
        dense_k: int,
        sparse_k: int,
    ) -> list[SearchHit]:
        if query_vector is None:
            query_vector = [0.0] * self.dimension
            dense_k = 0
        params: dict[str, Any] = {
            "tenant": principal.tenant_id,
            "user": principal.user_id,
            "groups": list(principal.group_ids),
            "qvec": query_vector,
            "qtext": query_text,
            "dense_k": dense_k,
            "sparse_k": sparse_k,
        }
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(_SQL, params)
            rows = await cur.fetchall()
        return [
            SearchHit(
                chunk=_chunk(r),
                source_name=r["source_name"],
                dense_rank=r["dense_rank"],
                sparse_rank=r["sparse_rank"],
                dense_score=float(r["dense_score"]) if r["dense_score"] is not None else None,
                sparse_score=float(r["sparse_score"]) if r["sparse_score"] is not None else None,
            )
            for r in rows
        ]
