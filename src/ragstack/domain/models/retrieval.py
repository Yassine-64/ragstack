from __future__ import annotations

from pydantic import BaseModel

from .chunk import Chunk


class Scores(BaseModel):
    dense: float | None = None  # cosine similarity
    sparse: float | None = None  # BM25 (pg_search)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf: float | None = None
    rerank: float | None = None

    @property
    def final(self) -> float:
        """The score the pipeline sorted by last: rerank when present, else RRF, else whichever exists."""
        for value in (self.rerank, self.rrf, self.dense, self.sparse):
            if value is not None:
                return value
        return 0.0


class RetrievedChunk(BaseModel):
    chunk: Chunk
    scores: Scores
    source_name: str = ""
    query: str = ""  # the (sub)query that retrieved it
