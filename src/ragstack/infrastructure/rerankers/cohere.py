from __future__ import annotations


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-v3.5") -> None:
        import cohere

        self._client = cohere.AsyncClientV2(api_key=api_key)
        self.model = model
        self.name = f"cohere:{model}"

    async def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        resp = await self._client.rerank(model=self.model, query=query, documents=texts, top_n=len(texts))
        scores = [0.0] * len(texts)
        for r in resp.results:
            scores[r.index] = float(r.relevance_score)
        return scores
