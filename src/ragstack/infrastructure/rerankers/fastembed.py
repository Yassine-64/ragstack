"""Cross-encoder reranker on CPU through fastembed (Xenova/ms-marco-MiniLM-L-6-v2 by default)."""

from __future__ import annotations

import asyncio


class FastembedReranker:
    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model)
        self.name = f"fastembed:{model}"

    async def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return await asyncio.to_thread(lambda: [float(s) for s in self._model.rerank(query, texts)])
