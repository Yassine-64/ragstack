from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ragstack.domain.models import ChunkDraft


@dataclass
class ParsedDocument:
    source_name: str
    media_type: str
    chunks: list[ChunkDraft]
    full_text: str  # for contextualization (whole-document prompt)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocumentParser(Protocol):
    """File → chunk drafts with provenance. The docparse adapter is the real one."""

    suffixes: frozenset[str]

    async def parse(self, path: Path) -> ParsedDocument: ...
