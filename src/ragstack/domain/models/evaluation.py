from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    question: str
    user_id: str
    tenant_id: str = "default"
    expected_answer: str | None = None
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    forbidden_document_ids: list[str] = Field(
        default_factory=list, description="must never be retrieved for this user"
    )
    should_refuse: bool = False
    tags: list[str] = Field(default_factory=list)


class EvalRun(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    config: dict[str, Any]
    metrics: dict[str, float] = Field(default_factory=dict)
    per_case: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
