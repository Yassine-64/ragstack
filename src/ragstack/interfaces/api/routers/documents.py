from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from ragstack.domain.exceptions import DocumentNotFound

from ..dependencies import AuthDep, CallerDep, ContainerDep
from ..schemas import DocumentResponse, ExposureItem, IngestResponse

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[AuthDep])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    c: ContainerDep,
    who: CallerDep,
    file: UploadFile = File(...),
    users: list[str] = Query(default=[]),
    groups: list[str] = Query(default=[]),
    everyone: bool = False,
) -> IngestResponse:
    tenant, _ = who
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in c.ingestion.supported_suffixes:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported format {suffix!r}")
    with tempfile.TemporaryDirectory(prefix="ragstack-") as tmp:
        path = Path(tmp) / Path(file.filename or "upload").name
        path.write_bytes(await file.read())
        doc, n = await c.ingest_document(path, tenant, users=users, groups=groups, everyone=everyone)
    return IngestResponse(document_id=doc.id, source_name=doc.source_name, chunks=n)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(c: ContainerDep, who: CallerDep) -> list[DocumentResponse]:
    tenant, _ = who
    return [DocumentResponse(**d.model_dump()) for d in await c.documents.list(tenant)]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, c: ContainerDep, who: CallerDep) -> None:
    tenant, _ = who
    if not await c.documents.soft_delete(tenant, document_id):
        raise DocumentNotFound(document_id)


@router.get("/{document_id}/exposures", response_model=list[ExposureItem])
async def exposures(document_id: str, c: ContainerDep, who: CallerDep) -> list[ExposureItem]:
    """Audit: who was shown chunks of this document, and for which question."""
    tenant, _ = who
    return [ExposureItem(**row) for row in await c.audit.exposures(tenant, document_id)]  # type: ignore[arg-type]
