"""Local ONNX embeddings via fastembed (BAAI/bge-small-en-v1.5 = 384 dims, ~130 MB, CPU friendly)."""

from __future__ import annotations

import asyncio

_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class FastembedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", query_prefix: str = "") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed:{model}"
        self.dimension = _DIMS.get(model) or len(next(iter(self._model.embed(["x"]))))
        self.query_prefix = query_prefix

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(lambda: [v.tolist() for v in self._model.embed(texts)])

    async def embed_query(self, text: str) -> list[float]:
        vecs = await asyncio.to_thread(lambda: list(self._model.query_embed([self.query_prefix + text])))
        return [float(x) for x in vecs[0]]
