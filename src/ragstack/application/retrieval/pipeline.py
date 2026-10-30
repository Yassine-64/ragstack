"""plan → hybrid candidates per query (ACL inside) → RRF → rerank → top-k (+ sibling expansion)."""

from __future__ import annotations

import asyncio
import time

from ragstack.application.config import SearchConfig
from ragstack.domain.models import Principal, RetrievedChunk, Trace
from ragstack.domain.ports import ChunkRepository, Embedder, Reranker, SearchIndex
from ragstack.domain.services import reciprocal_rank_fusion

from .query_planner import QueryPlan


class RetrievalPipeline:
    def __init__(
        self,
        index: SearchIndex,
        embedder: Embedder,
        config: SearchConfig,
        reranker: Reranker | None = None,
        chunks: ChunkRepository | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.config = config
        self.reranker = reranker if config.rerank else None
        self.chunks = chunks

    async def retrieve(
        self, principal: Principal, plan: QueryPlan, trace: Trace | None = None
    ) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        vectors = await asyncio.gather(*(self.embedder.embed_query(q) for q in plan.queries))
        hyde_vec = await self.embedder.embed_query(plan.hyde_text) if plan.hyde_text else None
        t1 = time.perf_counter()

        searches = [
            self.index.search(principal, q, v, self.config.dense_k, self.config.sparse_k)
            for q, v in zip(plan.queries, vectors, strict=True)
        ]
        if hyde_vec is not None:
            searches.append(self.index.search(principal, plan.queries[0], hyde_vec, self.config.dense_k, 0))
        results = await asyncio.gather(*searches)
        labels = list(plan.queries) + (["hyde"] if hyde_vec is not None else [])
        fused = reciprocal_rank_fusion(list(zip(labels, results, strict=True)), k=self.config.rrf_k)
        t2 = time.perf_counter()

        if self.reranker is not None and fused:
            head = fused[: self.config.rerank_top_n]
            scores = await self.reranker.score(plan.queries[0], [rc.chunk.indexed_text for rc in head])
            for rc, s in zip(head, scores, strict=True):
                rc.scores.rerank = s
            head.sort(key=lambda rc: rc.scores.rerank or 0.0, reverse=True)
            fused = head
        final = fused[: self.config.final_k]
        t3 = time.perf_counter()

        if self.config.expand_siblings and self.chunks is not None:
            final = await self._expand(final)

        if trace is not None:
            trace.add("embed", (t1 - t0) * 1000, queries=len(plan.queries), hyde=hyde_vec is not None)
            trace.add(
                "hybrid_search",
                (t2 - t1) * 1000,
                candidates=len(fused) if self.reranker is None else len(results and sum(results, [])),
            )
            trace.add("rerank", (t3 - t2) * 1000, enabled=self.reranker is not None, kept=len(final))
        return final

    async def _expand(self, final: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Small-to-big: pull neighbouring chunks of each hit so the model sees whole passages."""
        assert self.chunks is not None
        seen = {rc.chunk.id for rc in final}
        out = list(final)
        for rc in final:
            for sib in await self.chunks.siblings(rc.chunk, self.config.expand_siblings):
                if sib.id not in seen:
                    seen.add(sib.id)
                    out.append(
                        RetrievedChunk(
                            chunk=sib,
                            scores=rc.scores.model_copy(),
                            source_name=rc.source_name,
                            query=rc.query,
                        )
                    )
        return out
