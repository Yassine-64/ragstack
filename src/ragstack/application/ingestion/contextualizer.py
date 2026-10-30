"""Contextual retrieval: a short LLM-written sentence situating each chunk in its document.

Anthropic measured −49% retrieval failures (−67% with reranking) on top-20 recall. Cost is bounded
by prompt caching the document; here we simply send the (truncated) document each time.
"""

from __future__ import annotations

import asyncio

from ragstack.domain.models import ChunkDraft
from ragstack.domain.ports import LLM, ChatMessage

_PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Give a short succinct context (1-2 sentences) to situate this chunk within the overall document for the
purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""


class Contextualizer:
    def __init__(self, llm: LLM, max_document_chars: int = 60_000, concurrency: int = 4) -> None:
        self.llm = llm
        self.max_document_chars = max_document_chars
        self._sem = asyncio.Semaphore(concurrency)

    async def run(self, document_text: str, chunks: list[ChunkDraft]) -> list[ChunkDraft]:
        doc = document_text[: self.max_document_chars]
        await asyncio.gather(*(self._one(doc, c) for c in chunks))
        return chunks

    async def _one(self, document: str, chunk: ChunkDraft) -> None:
        async with self._sem:
            out = await self.llm.complete(
                [ChatMessage("user", _PROMPT.format(document=document, chunk=chunk.text))],
                max_tokens=150,
            )
        chunk.context = out.text.strip()
