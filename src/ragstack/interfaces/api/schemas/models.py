from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ragstack.domain.models import Answer, Scores, Trace


class Health(BaseModel):
    status: str = "ok"
    version: str
    embedder: str
    reranker: str | None
    llm: str
    config: str


class IngestResponse(BaseModel):
    document_id: str
    source_name: str
    chunks: int


class DocumentResponse(BaseModel):
    id: str
    source_name: str
    media_type: str
    metadata: dict[str, Any]
    created_at: datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    page: int | None
    section_path: list[str]
    text: str
    scores: Scores


class AskResponse(BaseModel):
    answer: Answer
    evidence: list[EvidenceItem]
    trace: Trace


class SearchResponse(BaseModel):
    hits: list[EvidenceItem]
    trace: Trace


class ExposureItem(BaseModel):
    user_id: str
    question: str
    created_at: datetime
    chunks: int
