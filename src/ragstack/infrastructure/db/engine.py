"""Connection pool + migration runner. Plain SQL files, applied in order, recorded in schema_migrations."""

from __future__ import annotations

from pathlib import Path

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

MIGRATIONS = Path(__file__).parent / "migrations"


async def _configure(conn: AsyncConnection) -> None:
    """Runs on every new pooled connection. The extension must exist before the type can be registered,
    and the first connection arrives before any migration has run, so create it here (idempotent).
    pg_search warns on every join it cannot push down (our ACL join), so its planner warnings are off."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("SET paradedb.planner_warnings = 'Off'")
    await conn.commit()
    await register_vector_async(conn)


def make_pool(database_url: str, max_size: int = 10) -> AsyncConnectionPool:
    return AsyncConnectionPool(database_url, min_size=1, max_size=max_size, open=False, configure=_configure)


async def migrate(pool: AsyncConnectionPool, dimension: int) -> list[str]:
    applied: list[str] = []
    async with pool.connection() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name text PRIMARY KEY, applied_at timestamptz DEFAULT now())"
        )
        done = {r[0] for r in await (await conn.execute("SELECT name FROM schema_migrations")).fetchall()}
        for sql_file in sorted(MIGRATIONS.glob("*.sql")):
            if sql_file.name in done:
                continue
            await conn.execute(sql_file.read_text().replace("{DIM}", str(dimension)))  # type: ignore[arg-type,unused-ignore]
            await conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (sql_file.name,))
            applied.append(sql_file.name)
        await conn.commit()
    return applied
