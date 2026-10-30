from __future__ import annotations

from fastapi import APIRouter

from ragstack.domain.models import RetrievedChunk

from ..dependencies import AuthDep, CallerDep, ContainerDep
from ..schemas import AskRequest, AskResponse, EvidenceItem, SearchResponse

router = APIRouter(tags=["ask"], dependencies=[AuthDep])


def _evidence(rc: RetrievedChunk) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=rc.chunk.id,
        document_id=rc.chunk.document_id,
        source_name=rc.source_name,
        page=rc.chunk.page,
        section_path=rc.chunk.section_path,
        text=rc.chunk.text,
        scores=rc.scores,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, c: ContainerDep, who: CallerDep) -> AskResponse:
    tenant, user = who
    r = await c.ask(tenant, user, body.question)
    return AskResponse(answer=r.answer, evidence=[_evidence(e) for e in r.evidence], trace=r.trace)


@router.get("/search", response_model=SearchResponse)
async def search(q: str, c: ContainerDep, who: CallerDep) -> SearchResponse:
    tenant, user = who
    hits, trace = await c.search(tenant, user, q)
    return SearchResponse(hits=[_evidence(h) for h in hits], trace=trace)
