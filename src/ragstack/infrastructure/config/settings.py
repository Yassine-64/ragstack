from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGSTACK_", env_file=".env", env_nested_delimiter="__", extra="ignore"
    )

    database_url: str = "postgresql://ragstack:ragstack@localhost:5433/ragstack"
    embedding_dimension: int = Field(
        default=384, description="must match the embedder; the schema is created with it"
    )

    embedder: str = Field(default="fastembed", description="fastembed | openai | hashing")
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    reranker: str = Field(default="fastembed", description="fastembed | cohere | none")
    fastembed_reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cohere_api_key: str | None = None

    llm: str = Field(default="anthropic", description="anthropic | openai | scripted")
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    judge_model: str | None = Field(default=None, description="defaults to the main LLM")

    config_path: str | None = Field(default=None, description="YAML RetrievalConfig; defaults when unset")
    api_key: str | None = Field(default=None, description="Bearer token for the HTTP API when set")
    log_level: str = "INFO"
