"""Listwise reranker on the LLM port: the model sees a batch of passages and scores each 0-10.

A cross-encoder trained on web search lost nDCG on scientific and financial text; a general model
reads "purchases of property and equipment" as capital expenditure. Batches run concurrently, so
latency is one round trip per batch, not per passage.
"""

from __future__ import annotations

import asyncio
import json
import re

from ragstack.domain.ports import LLM, ChatMessage

_SYSTEM = (
    "You rank passages for a search engine. Score how useful each passage is for answering the "
    "question: 10 = contains the answer or the exact figures needed, 5 = related context, "
    "0 = unrelated. Reply with a JSON object mapping passage id to score, nothing else."
)
_PROMPT = "Question: {q}\n\n{passages}"
_JSON = re.compile(r"\{.*\}", re.S)


class LlmReranker:
    def __init__(self, llm: LLM, batch_size: int = 20, max_chars: int = 700) -> None:
        self.llm = llm
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.name = f"llm-rerank:{llm.name}"

    async def score(self, query: str, texts: list[str]) -> list[float]:
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        scored = await asyncio.gather(*(self._batch(query, b) for b in batches))
        return [s for batch in scored for s in batch]

    async def _batch(self, query: str, texts: list[str]) -> list[float]:
        passages = "\n\n".join(
            f'<passage id="{i}">\n{t[: self.max_chars]}\n</passage>' for i, t in enumerate(texts)
        )
        out = await self.llm.complete(
            [ChatMessage("system", _SYSTEM), ChatMessage("user", _PROMPT.format(q=query, passages=passages))],
            max_tokens=8 * len(texts) + 20,
        )
        m = _JSON.search(out.text)
        try:
            raw = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            raw = {}
        # an unparseable reply keeps the fused order (all zeros) instead of failing the query
        return [float(raw.get(str(i), 0.0)) / 10.0 for i in range(len(texts))]
