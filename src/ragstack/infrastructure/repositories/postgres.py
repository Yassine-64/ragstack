"""All Postgres repositories. One file: they share the pool and the row→model helpers."""

from __future__ import annotations

import json
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ragstack.domain.models import (
    AccessGrant,
    Chunk,
    Document,
    EvalRun,
    Principal,
    PrincipalType,
    Trace,
)


def _chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        index=row["idx"],
        section_path=list(row["section_path"] or []),
        page=row["page"],
        bbox=row["bbox"],
        context=row["context"],
        text=row["text"],
        token_count=row["token_count"],
        metadata=row["metadata"] or {},
    )


class PostgresDocuments:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def add(self, document: Document, grants: list[AccessGrant]) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO documents (id, tenant_id, source_name, media_type, metadata, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    document.id,
                    document.tenant_id,
                    document.source_name,
                    document.media_type,
                    json.dumps(document.metadata),
                    document.created_at,
                ),
            )
            for g in grants:
                await conn.execute(
                    "INSERT INTO access_grants (document_id, principal_type, principal_id) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    (document.id, g.principal_type.value, g.principal_id),
                )

    async def get(self, tenant_id: str, document_id: str) -> Document | None:
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM documents WHERE tenant_id=%s AND id=%s", (tenant_id, document_id)
            )
            row = await cur.fetchone()
        return Document(**row) if row else None

    async def list(self, tenant_id: str, limit: int = 100) -> list[Document]:
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM documents WHERE tenant_id=%s AND deleted_at IS NULL ORDER BY created_at DESC LIMIT %s",
                (tenant_id, limit),
            )
            return [Document(**r) for r in await cur.fetchall()]

    async def soft_delete(self, tenant_id: str, document_id: str) -> bool:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE documents SET deleted_at=now() WHERE tenant_id=%s AND id=%s AND deleted_at IS NULL",
                (tenant_id, document_id),
            )
            return cur.rowcount == 1


class PostgresChunks:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def add_many(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        rows = [
            (
                c.id,
                c.document_id,
                c.index,
                c.section_path,
                c.page,
                json.dumps(c.bbox) if c.bbox else None,
                c.context,
                c.text,
                c.token_count,
                json.dumps(c.metadata),
                e,
            )
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO chunks (id, document_id, idx, section_path, page, bbox, context, text, token_count, metadata, embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                rows,
            )

    async def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT * FROM chunks WHERE id = ANY(%s)", (chunk_ids,))
            return [_chunk(r) for r in await cur.fetchall()]

    async def siblings(self, chunk: Chunk, window: int) -> list[Chunk]:
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM chunks WHERE document_id=%s AND idx BETWEEN %s AND %s AND id<>%s ORDER BY idx",
                (chunk.document_id, chunk.index - window, chunk.index + window, chunk.id),
            )
            return [_chunk(r) for r in await cur.fetchall()]

    async def count(self, tenant_id: str) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.tenant_id=%s AND d.deleted_at IS NULL",
                (tenant_id,),
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0


class PostgresAccess:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def resolve(self, tenant_id: str, user_id: str) -> Principal:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT group_id FROM group_members WHERE tenant_id=%s AND user_id=%s",
                (tenant_id, user_id),
            )
            groups = frozenset(r[0] for r in await cur.fetchall())
        return Principal(tenant_id=tenant_id, user_id=user_id, group_ids=groups)

    async def ensure_user(self, tenant_id: str, user_id: str, group_ids: list[str]) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO users (tenant_id, id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (tenant_id, user_id),
            )
            for g in group_ids:
                await conn.execute(
                    "INSERT INTO group_members (tenant_id, group_id, user_id) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    (tenant_id, g, user_id),
                )

    async def grant(self, grant: AccessGrant) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO access_grants (document_id, principal_type, principal_id) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                (grant.document_id, grant.principal_type.value, grant.principal_id),
            )

    async def revoke(self, grant: AccessGrant) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "DELETE FROM access_grants WHERE document_id=%s AND principal_type=%s AND principal_id=%s",
                (grant.document_id, grant.principal_type.value, grant.principal_id),
            )


class PostgresAudit:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def record(self, principal: Principal, trace: Trace, shown_chunk_ids: list[str]) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO queries (id, tenant_id, user_id, question, trace) VALUES (%s,%s,%s,%s,%s)",
                (
                    trace.query_id,
                    principal.tenant_id,
                    principal.user_id,
                    trace.question,
                    trace.model_dump_json(),
                ),
            )
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO retrievals (query_id, chunk_id, rank) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    [(trace.query_id, cid, i) for i, cid in enumerate(shown_chunk_ids, 1)],
                )

    async def exposures(self, tenant_id: str, document_id: str) -> list[dict[str, object]]:
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT q.user_id, q.question, q.created_at, count(*) AS chunks FROM retrievals r "
                "JOIN chunks c ON c.id=r.chunk_id JOIN queries q ON q.id=r.query_id "
                "WHERE q.tenant_id=%s AND c.document_id=%s GROUP BY q.id ORDER BY q.created_at DESC LIMIT 200",
                (tenant_id, document_id),
            )
            return [dict(r) for r in await cur.fetchall()]


class PostgresEvals:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def save(self, run: EvalRun) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO eval_runs (id, name, config, metrics, per_case, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    run.id,
                    run.name,
                    json.dumps(run.config),
                    json.dumps(run.metrics),
                    json.dumps(run.per_case, default=str),
                    run.created_at,
                ),
            )

    async def list(self, limit: int = 20) -> list[EvalRun]:
        async with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT %s", (limit,))
            return [EvalRun(**r) for r in await cur.fetchall()]


__all__ = [
    "PostgresAccess",
    "PostgresAudit",
    "PostgresChunks",
    "PostgresDocuments",
    "PostgresEvals",
    "PrincipalType",
]
