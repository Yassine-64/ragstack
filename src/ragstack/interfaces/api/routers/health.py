from __future__ import annotations

from fastapi import APIRouter

from ragstack import __version__

from ..dependencies import ContainerDep
from ..schemas import Health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
async def health(c: ContainerDep) -> Health:
    return Health(
        version=__version__,
        embedder=c.embedder.name,
        reranker=c.reranker.name if c.reranker else None,
        llm=c.llm.name,
        config=c.config.name,
    )
