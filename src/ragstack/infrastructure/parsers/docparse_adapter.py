"""File → chunks with page + bbox provenance, through docparse (project 1)."""

from __future__ import annotations

from pathlib import Path

from docparse.application.use_cases import ChunkDocument
from docparse.infrastructure.di.container import Container as DocparseContainer

from ragstack.domain.models import ChunkDraft
from ragstack.domain.ports.parser import ParsedDocument


class DocparseParser:
    def __init__(self, container: DocparseContainer, max_chars: int = 1600, overlap_blocks: int = 1) -> None:
        self.container = container
        self.chunker = ChunkDocument(max_chars=max_chars, overlap_blocks=overlap_blocks)
        self.suffixes = container.parse_document.supported_suffixes

    async def parse(self, path: Path) -> ParsedDocument:
        result = await self.container.parse_document(path)
        doc = result.document
        drafts = []
        for chunk in self.chunker(doc):
            first = chunk.provenance[0] if chunk.provenance else None
            drafts.append(
                ChunkDraft(
                    index=chunk.index,
                    text=chunk.text,
                    section_path=chunk.section_path,
                    page=first.page if first else None,
                    bbox=first.bbox.model_dump() if first and first.bbox else None,
                )
            )
        return ParsedDocument(
            source_name=doc.source_name,
            media_type=doc.media_type,
            chunks=drafts,
            full_text=doc.to_markdown(page_markers=False),
            metadata={"engines": doc.metadata.get("engines", [])},
        )
