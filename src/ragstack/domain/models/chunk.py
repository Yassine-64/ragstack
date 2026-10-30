from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ChunkDraft(BaseModel):
    """A chunk before it is stored: text plus provenance, no id, no embedding yet."""

    index: int
    text: str
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    bbox: dict[str, float] | None = None  # normalized x0,y0,x1,y1 from docparse
    context: str = Field(default="", description="LLM-written situating sentence(s), prepended at index time")

    @property
    def indexed_text(self) -> str:
        return f"{self.context}\n\n{self.text}".strip() if self.context else self.text


class Chunk(ChunkDraft):
    id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
