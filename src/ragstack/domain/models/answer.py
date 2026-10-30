from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    index: int = Field(ge=1, description="the [n] marker in the answer text")
    chunk_id: str
    document_id: str
    source_name: str
    page: int | None = None
    bbox: dict[str, float] | None = None
    quote: str = Field(default="", description="first characters of the cited chunk, for display")


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    evidence_score: float = Field(
        default=0.0, ge=0, le=1, description="grader's estimate that the evidence suffices"
    )
    evidence: list[str] = Field(default_factory=list, description="chunk ids shown to the model")
