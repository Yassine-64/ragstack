"""Composition root: Settings + RetrievalConfig → adapters → use cases."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml
from psycopg_pool import AsyncConnectionPool

from ragstack.application.agent import AgenticRag
from ragstack.application.config import RetrievalConfig
from ragstack.application.evaluation import EvalRunner
from ragstack.application.evaluation.judge import Judge
from ragstack.application.generation import EvidenceGrader, GroundedAnswerer
from ragstack.application.ingestion import Contextualizer, IngestionPipeline
from ragstack.application.retrieval import QueryPlanner, RetrievalPipeline
from ragstack.application.use_cases import Ask, IngestDocument, Search
from ragstack.domain.ports import LLM, DocumentParser, Embedder, Reranker
from ragstack.infrastructure.config.settings import Settings
from ragstack.infrastructure.db.engine import make_pool, migrate
from ragstack.infrastructure.repositories.postgres import (
    PostgresAccess,
    PostgresAudit,
    PostgresChunks,
    PostgresDocuments,
    PostgresEvals,
)
from ragstack.infrastructure.repositories.search import PostgresHybridSearch

log = logging.getLogger(__name__)


def load_config(path: str | Path | None) -> RetrievalConfig:
    if not path:
        return RetrievalConfig()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return RetrievalConfig.model_validate(data)


@dataclass
class Container:
    settings: Settings
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    pool: AsyncConnectionPool | None = None

    async def start(self) -> None:
        if self.pool is None:
            self.pool = make_pool(self.settings.database_url)
        await self.pool.open()
        applied = await migrate(self.pool, self.embedder.dimension)
        if applied:
            log.info("applied migrations: %s", applied)

    async def stop(self) -> None:
        if self.pool is not None:
            await self.pool.close()

    def _pool(self) -> AsyncConnectionPool:
        assert self.pool is not None, "call Container.start() first"
        return self.pool

    # ---- engines -------------------------------------------------------------------------
    @cached_property
    def embedder(self) -> Embedder:
        s = self.settings
        if s.embedder == "openai" and s.openai_api_key:
            from ragstack.infrastructure.embeddings.openai import OpenAiEmbedder

            return OpenAiEmbedder(s.openai_api_key, s.openai_embedding_model, s.embedding_dimension)
        if s.embedder == "fastembed":
            from ragstack.infrastructure.embeddings.fastembed import FastembedEmbedder

            return FastembedEmbedder(s.fastembed_model)
        from ragstack.infrastructure.embeddings.hashing import HashingEmbedder

        return HashingEmbedder(s.embedding_dimension)

    @cached_property
    def reranker(self) -> Reranker | None:
        s = self.settings
        if s.reranker == "cohere" and s.cohere_api_key:
            from ragstack.infrastructure.rerankers.cohere import CohereReranker

            return CohereReranker(s.cohere_api_key)
        if s.reranker == "fastembed":
            from ragstack.infrastructure.rerankers.fastembed import FastembedReranker

            return FastembedReranker(s.fastembed_reranker_model)
        if s.reranker == "lexical":
            from ragstack.infrastructure.rerankers.lexical import LexicalReranker

            return LexicalReranker()
        return None

    @cached_property
    def llm(self) -> LLM:
        s = self.settings
        if s.llm == "anthropic" and s.anthropic_api_key:
            from ragstack.infrastructure.llm.anthropic import AnthropicLLM

            return AnthropicLLM(s.anthropic_api_key, s.anthropic_model)
        if s.llm == "openai" and s.openai_api_key:
            from ragstack.infrastructure.llm.openai import OpenAiLLM

            return OpenAiLLM(s.openai_api_key, s.openai_chat_model)
        if s.llm not in {"scripted", "anthropic", "openai"}:
            log.warning("unknown llm %r, using scripted", s.llm)
        elif s.llm != "scripted":
            log.warning("no API key for llm=%s, using the scripted offline LLM", s.llm)
        from ragstack.infrastructure.llm.scripted import ScriptedLLM

        return ScriptedLLM()

    @cached_property
    def judge_llm(self) -> LLM:
        s = self.settings
        if s.judge_model and s.anthropic_api_key:
            from ragstack.infrastructure.llm.anthropic import AnthropicLLM

            return AnthropicLLM(s.anthropic_api_key, s.judge_model)
        return self.llm

    @cached_property
    def parsers(self) -> list[DocumentParser]:
        from ragstack.infrastructure.parsers.text import TextParser

        parsers: list[DocumentParser] = [TextParser(max_chars=self.config.chunking.max_tokens * 4)]
        try:
            from docparse.infrastructure.config.settings import Settings as DocparseSettings
            from docparse.infrastructure.di.container import build_container as build_docparse

            from ragstack.infrastructure.parsers.docparse_adapter import DocparseParser

            parsers.append(
                DocparseParser(
                    build_docparse(DocparseSettings(_env_file=None)),
                    max_chars=self.config.chunking.max_tokens * 4,
                    overlap_blocks=self.config.chunking.overlap_blocks,
                )
            )
        except ImportError:
            log.warning("docparse not installed: only .txt/.md ingestion available")
        return parsers

    # ---- repositories --------------------------------------------------------------------
    @cached_property
    def documents(self) -> PostgresDocuments:
        return PostgresDocuments(self._pool())

    @cached_property
    def chunks(self) -> PostgresChunks:
        return PostgresChunks(self._pool())

    @cached_property
    def access(self) -> PostgresAccess:
        return PostgresAccess(self._pool())

    @cached_property
    def audit(self) -> PostgresAudit:
        return PostgresAudit(self._pool())

    @cached_property
    def evals(self) -> PostgresEvals:
        return PostgresEvals(self._pool())

    @cached_property
    def index(self) -> PostgresHybridSearch:
        return PostgresHybridSearch(self._pool(), self.embedder.dimension)

    # ---- pipelines & use cases -----------------------------------------------------------
    @cached_property
    def ingestion(self) -> IngestionPipeline:
        ctx = Contextualizer(self.llm) if self.config.chunking.contextual else None
        return IngestionPipeline(
            self.parsers, self.embedder, self.documents, self.chunks, ctx, self.config.chunking
        )

    @cached_property
    def retrieval(self) -> RetrievalPipeline:
        return RetrievalPipeline(self.index, self.embedder, self.config.search, self.reranker, self.chunks)

    @cached_property
    def agent(self) -> AgenticRag:
        return AgenticRag(
            planner=QueryPlanner(self.llm, self.config.query),
            retrieval=self.retrieval,
            answerer=GroundedAnswerer(self.llm, self.config.generation),
            grader=EvidenceGrader(self.llm),
            config=self.config.agent,
            llm=self.llm,
        )

    @cached_property
    def ingest_document(self) -> IngestDocument:
        return IngestDocument(self.ingestion)

    @cached_property
    def ask(self) -> Ask:
        return Ask(self.access, self.agent, self.audit)

    @cached_property
    def search(self) -> Search:
        return Search(self.access, self.retrieval)

    def eval_runner(self) -> EvalRunner:
        return EvalRunner(lambda: self.ask, Judge(self.judge_llm), self.evals, k=self.config.search.final_k)

    def with_config(self, config: RetrievalConfig) -> Container:
        """A sibling container sharing pool and engines but with another retrieval config (for evals)."""
        c = Container(self.settings, config, self.pool)
        for name in ("embedder", "reranker", "llm", "judge_llm", "parsers"):
            if name in self.__dict__:
                c.__dict__[name] = self.__dict__[name]
        return c


def build_container(settings: Settings | None = None, config: RetrievalConfig | None = None) -> Container:
    s = settings or Settings()
    return Container(s, config or load_config(s.config_path))
