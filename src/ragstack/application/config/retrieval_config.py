"""Every knob the evals vary lives here, so a config file fully describes an experiment."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    max_tokens: int = 400
    overlap_blocks: int = 1
    contextual: bool = Field(
        default=False, description="LLM-written context prepended to each chunk (Anthropic 2024)"
    )


class SearchConfig(BaseModel):
    dense_k: int = 100
    sparse_k: int = 100
    rrf_k: int = 60
    rerank: bool = True
    rerank_top_n: int = 150
    final_k: int = 20
    expand_siblings: int = Field(
        default=0, ge=0, description="neighbouring chunks to pull in around each hit"
    )


class QueryConfig(BaseModel):
    rewrite: bool = False
    multi_query: int = Field(default=0, ge=0, le=5, description="extra paraphrases to search with")
    hyde: bool = Field(default=False, description="hypothetical answer embedding")


class AgentConfig(BaseModel):
    grade: bool = True
    max_retries: int = Field(default=1, ge=0, le=5)
    min_evidence_score: float = Field(default=0.35, ge=0, le=1)


class GenerationConfig(BaseModel):
    refuse_below: float = Field(default=0.35, ge=0, le=1)
    max_evidence_chars: int = 1200
    max_tokens: int = 800


class RetrievalConfig(BaseModel):
    name: str = "default"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
