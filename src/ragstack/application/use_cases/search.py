from __future__ import annotations

from ragstack.application.retrieval import QueryPlan, RetrievalPipeline
from ragstack.domain.models import RetrievedChunk, Trace
from ragstack.domain.ports import AccessRepository


class Search:
    """Retrieval only, no generation. Groups are resolved server-side."""

    def __init__(self, access: AccessRepository, retrieval: RetrievalPipeline) -> None:
        self.access = access
        self.retrieval = retrieval

    async def __call__(self, tenant_id: str, user_id: str, query: str) -> tuple[list[RetrievedChunk], Trace]:
        principal = await self.access.resolve(tenant_id, user_id)
        trace = Trace(question=query)
        hits = await self.retrieval.retrieve(principal, QueryPlan(original=query, queries=[query]), trace)
        return hits, trace
