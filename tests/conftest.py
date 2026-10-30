"""Integration tests need Postgres+pgvector: RAGSTACK_TEST_DATABASE_URL (CI) or testcontainers (local)."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from ragstack.application.config import RetrievalConfig
from ragstack.infrastructure.config.settings import Settings
from ragstack.infrastructure.di.container import Container


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    url = os.environ.get("RAGSTACK_TEST_DATABASE_URL")
    if url:
        yield url
        return
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover
        pytest.skip("no RAGSTACK_TEST_DATABASE_URL and testcontainers not installed")
    with PostgresContainer(
        "pgvector/pgvector:pg16", username="ragstack", password="ragstack", dbname="ragstack"
    ) as pg:
        yield pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


def offline_settings(database_url: str, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        database_url=database_url,
        embedder="hashing",
        reranker="lexical",
        llm="scripted",
        **overrides,
    )  # type: ignore[arg-type]


@pytest.fixture
async def container(database_url: str) -> AsyncIterator[Container]:
    c = Container(offline_settings(database_url), RetrievalConfig(search={"final_k": 10, "rerank_top_n": 50}))  # type: ignore[arg-type]
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


@pytest.fixture
def tenant() -> str:
    """Every test gets its own tenant, which is also what isolates its data."""
    return f"t-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def corpus_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "evals" / "corpus"


@pytest.fixture
async def seeded(container: Container, tenant: str, corpus_dir: Path) -> dict[str, str]:
    """alice∈engineering, bob∈finance, carol∈hr, dave∈∅; public docs for everyone. Returns source→doc id."""
    for user, groups in {"alice": ["engineering"], "bob": ["finance"], "carol": ["hr"], "dave": []}.items():
        await container.access.ensure_user(tenant, user, groups)
    ids: dict[str, str] = {}
    for folder, grant in {
        "public": {"everyone": True},
        "engineering": {"groups": ["engineering"]},
        "finance": {"groups": ["finance"]},
        "hr": {"groups": ["hr"]},
    }.items():
        for path in sorted((corpus_dir / folder).glob("*.md")):
            doc, _ = await container.ingest_document(path, tenant, **grant)  # type: ignore[arg-type]
            ids[path.name] = doc.id
    return ids
