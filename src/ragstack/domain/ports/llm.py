from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    extra: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class LLM(Protocol):
    name: str

    async def complete(
        self, messages: list[ChatMessage], max_tokens: int = 1024, temperature: float = 0.0
    ) -> Completion: ...
