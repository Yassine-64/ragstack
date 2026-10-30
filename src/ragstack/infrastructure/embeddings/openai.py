from __future__ import annotations

_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAiEmbedder:
    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dimension: int | None = None
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"
        self.dimension = dimension or _DIMS.get(model, 1536)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self.model, input=texts, dimensions=self.dimension)
        return [d.embedding for d in resp.data]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
