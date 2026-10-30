from __future__ import annotations

import logging
from pathlib import Path

from ragstack.application.config import ChunkingConfig
from ragstack.domain.models import AccessGrant, Chunk, Document
from ragstack.domain.ports import ChunkRepository, DocumentParser, DocumentRepository, Embedder
from ragstack.domain.services import estimate_tokens

from .contextualizer import Contextualizer

log = logging.getLogger(__name__)


class IngestionPipeline:
    """parse → (contextualize) → embed → store, as one unit per document."""

    def __init__(
        self,
        parsers: list[DocumentParser],
        embedder: Embedder,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        contextualizer: Contextualizer | None = None,
        config: ChunkingConfig | None = None,
        batch_size: int = 64,
    ) -> None:
        self._by_suffix = {s: p for p in parsers for s in p.suffixes}
        self.embedder = embedder
        self.documents = documents
        self.chunks = chunks
        self.contextualizer = contextualizer
        self.config = config or ChunkingConfig()
        self.batch_size = batch_size

    @property
    def supported_suffixes(self) -> frozenset[str]:
        return frozenset(self._by_suffix)

    async def ingest(
        self,
        path: Path,
        tenant_id: str,
        grants: list[AccessGrant],
        metadata: dict[str, object] | None = None,
    ) -> tuple[Document, int]:
        parser = self._by_suffix.get(path.suffix.lower())
        if parser is None:
            raise ValueError(f"no parser for {path.suffix!r}")
        parsed = await parser.parse(path)
        drafts = parsed.chunks
        if self.config.contextual and self.contextualizer is not None:
            drafts = await self.contextualizer.run(parsed.full_text, drafts)

        document = Document(
            tenant_id=tenant_id,
            source_name=parsed.source_name,
            media_type=parsed.media_type,
            metadata={**parsed.metadata, **(metadata or {})},
        )
        grants = [
            AccessGrant(
                document_id=document.id,
                principal_type=g.principal_type,
                principal_id=g.principal_id,
            )
            for g in grants
        ]
        if not grants:
            log.warning(
                "%s ingested with an empty ACL: nobody can retrieve it (deny by default)",
                parsed.source_name,
            )
        await self.documents.add(document, grants)

        chunks = [
            Chunk(
                **d.model_dump(),
                document_id=document.id,
                token_count=estimate_tokens(d.indexed_text),
            )
            for d in drafts
        ]
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = await self.embedder.embed_documents([c.indexed_text for c in batch])
            await self.chunks.add_many(batch, vectors)
        return document, len(chunks)
