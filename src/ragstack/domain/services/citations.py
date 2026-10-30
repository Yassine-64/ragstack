"""Citations are [n] markers. The model may only cite evidence it was shown; anything else is dropped."""

from __future__ import annotations

import re

from ragstack.domain.models import Citation, RetrievedChunk

_MARKER = re.compile(r"\[(\d{1,3})\]")


def extract_citation_indexes(text: str) -> list[int]:
    seen: list[int] = []
    for m in _MARKER.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def verify_citations(
    text: str, evidence: list[RetrievedChunk], quote_chars: int = 160
) -> tuple[str, list[Citation]]:
    """Keeps [n] markers that point at shown evidence (1-based), strips the rest.

    Returns the cleaned text and the citations in order of first appearance."""
    valid: list[Citation] = []
    by_index: dict[int, Citation] = {}
    for n in extract_citation_indexes(text):
        if 1 <= n <= len(evidence):
            rc = evidence[n - 1]
            cit = Citation(
                index=n,
                chunk_id=rc.chunk.id,
                document_id=rc.chunk.document_id,
                source_name=rc.source_name,
                page=rc.chunk.page,
                bbox=rc.chunk.bbox,
                quote=rc.chunk.text[:quote_chars],
            )
            by_index[n] = cit
            valid.append(cit)

    def keep(m: re.Match[str]) -> str:
        return m.group(0) if int(m.group(1)) in by_index else ""

    cleaned = re.sub(r"\s+([.,;:])", r"\1", _MARKER.sub(keep, text))
    return cleaned.strip(), valid
