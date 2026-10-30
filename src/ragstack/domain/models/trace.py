from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Step(BaseModel):
    name: str
    ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    query_id: str = Field(default_factory=lambda: uuid4().hex)
    question: str
    rewritten: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.steps)

    def add(self, name: str, ms: float, **detail: Any) -> None:
        self.steps.append(Step(name=name, ms=ms, detail=detail))
