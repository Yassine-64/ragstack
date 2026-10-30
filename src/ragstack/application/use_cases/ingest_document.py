from __future__ import annotations

from pathlib import Path

from ragstack.application.ingestion import IngestionPipeline
from ragstack.domain.models import AccessGrant, Document, PrincipalType


class IngestDocument:
    def __init__(self, pipeline: IngestionPipeline) -> None:
        self.pipeline = pipeline

    async def __call__(
        self,
        path: Path,
        tenant_id: str,
        users: list[str] | None = None,
        groups: list[str] | None = None,
        everyone: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> tuple[Document, int]:
        grants = [
            AccessGrant(document_id="", principal_type=PrincipalType.USER, principal_id=u)
            for u in users or []
        ]
        grants += [
            AccessGrant(document_id="", principal_type=PrincipalType.GROUP, principal_id=g)
            for g in groups or []
        ]
        if everyone:
            grants.append(AccessGrant(document_id="", principal_type=PrincipalType.EVERYONE))
        return await self.pipeline.ingest(path, tenant_id, grants, metadata)
