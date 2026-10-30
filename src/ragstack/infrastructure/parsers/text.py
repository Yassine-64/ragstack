"""Plain text / Markdown without docparse: split on headings, pack paragraphs up to max_chars."""

from __future__ import annotations

import re
from pathlib import Path

from ragstack.domain.models import ChunkDraft
from ragstack.domain.ports.parser import ParsedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class TextParser:
    suffixes = frozenset({".txt", ".md", ".markdown"})

    def __init__(self, max_chars: int = 1600) -> None:
        self.max_chars = max_chars

    async def parse(self, path: Path) -> ParsedDocument:
        text = path.read_text(encoding="utf-8")
        chunks: list[ChunkDraft] = []
        section: list[tuple[int, str]] = []
        buffer: list[str] = []

        def flush() -> None:
            body = "\n\n".join(buffer).strip()
            if body:
                chunks.append(ChunkDraft(index=len(chunks), text=body, section_path=[t for _, t in section]))
            buffer.clear()

        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            if m := _HEADING.match(para.splitlines()[0]):
                flush()
                level = len(m.group(1))
                section = [(lv, t) for lv, t in section if lv < level] + [(level, m.group(2).strip())]
                rest = "\n".join(para.splitlines()[1:]).strip()
                if rest:
                    buffer.append(rest)
                continue
            if sum(len(b) for b in buffer) + len(para) > self.max_chars and buffer:
                flush()
            buffer.append(para)
        flush()
        media = "text/markdown" if path.suffix != ".txt" else "text/plain"
        return ParsedDocument(source_name=path.name, media_type=media, chunks=chunks, full_text=text)
