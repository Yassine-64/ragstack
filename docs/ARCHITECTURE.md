# ragstack architecture

Permission-aware hybrid RAG whose every design choice is tied to a measurement. This document is
the contract; the README is the tour.

## 1. Goals

1. **Correct under access control.** A user never sees, and the LLM never reads, a chunk the user
   is not entitled to. Enforced inside the retrieval SQL, not by filtering afterwards.
2. **Measured.** Every retrieval and generation knob (hybrid weights, reranker, contextual chunks,
   query rewriting, agentic retry) is a config value, and the eval harness reports what it buys.
3. **Grounded.** Answers cite chunk ids that were actually shown to the model; citations resolve
   to document, page and bounding box (provenance comes from docparse). Weak evidence → refusal.
4. **Boring infrastructure.** One Postgres with pgvector holds chunks, vectors, full-text index,
   ACLs, audit log and eval runs. No second datastore until a measurement says so.

## 2. Pipeline

```
ingest:  file ─docparse─▶ blocks+provenance ─chunker─▶ chunks ─contextualizer(LLM, optional)─▶
         contextual chunks ─embedder─▶ vectors ─▶ Postgres (chunks + tsvector + vector + ACL)

ask:     question ─query planner─▶ {rewritten, sub-queries, hyde?}
         ─hybrid search (per query; SQL CTE: ACL ∩ (dense top-N ∪ sparse top-N))─▶ candidates
         ─RRF (k=60)─▶ fused ─reranker (top 150 → 20)─▶ evidence
         ─evidence grader─▶ enough? ──no──▶ rewrite & retry (≤ N) ──┐
                             │yes                                   │
         ─grounded answerer (citations [n])─▶ verify citations ⊂ evidence ─▶ Answer | Refusal
         ─audit (who saw which chunk) ─trace (timings, tokens, cost)
```

## 3. Layers (dependency arrows point inward)

| layer | contents | may import |
|---|---|---|
| `domain` | models (Document, Chunk, AccessGrant, Principal, RetrievedChunk, Answer, Citation, Trace, eval cases), ports (Embedder, Reranker, LLM, DocumentParser, repositories, SearchIndex), pure services (RRF fusion, citation parsing/verification, text normalisation) | stdlib, pydantic |
| `application` | use cases (IngestDocument, Ask, Search, RunEval), pipeline stages (ingestion, retrieval, generation, agent loop), evaluation metrics and runner, `RetrievalConfig` | domain |
| `infrastructure` | Postgres repositories + the hybrid search CTE, migrations, embedders (fastembed, OpenAI, hashing fake), rerankers (fastembed cross-encoder, Cohere, none), LLMs (Anthropic, OpenAI, scripted fake), docparse parser adapter, settings, DI container | application, domain |
| `interfaces` | FastAPI routers + schemas + dependencies, Typer commands | application via container |

## 4. Data model (Postgres)

```
tenants(id)
users(id, tenant_id)                         groups(id, tenant_id)        group_members(group_id, user_id)
documents(id, tenant_id, source_name, media_type, metadata jsonb, created_at, deleted_at)
access_grants(document_id, principal_type user|group|everyone, principal_id)   -- deny by default
chunks(id, document_id, idx, section_path text[], page, bbox jsonb, text, context, token_count,
       embedding vector(D), tsv tsvector GENERATED (context || text), parent_id)
queries(id, tenant_id, user_id, question, config jsonb, trace jsonb, created_at)
retrievals(query_id, chunk_id, rank, scores jsonb)        -- audit: who was shown what
eval_runs(id, name, config jsonb, metrics jsonb, created_at)
```

Hybrid search is one statement: resolve the caller's principals → `visible` CTE of document ids →
dense top-N and sparse top-N restricted to `visible` → returned together with both ranks. RRF
happens in the application layer so the fusion rule is unit-testable and swappable.

## 5. Retrieval config (what the evals vary)

```yaml
chunking:        {max_tokens: 400, overlap_tokens: 40, contextual: true}
embedding:       {model: BAAI/bge-small-en-v1.5}
retrieval:       {dense_k: 100, sparse_k: 100, rrf_k: 60, rerank: true, rerank_top_n: 150, final_k: 20}
query:           {rewrite: true, multi_query: 2, hyde: false}
agent:           {grade: true, max_retries: 2, min_evidence_score: 0.35}
generation:      {model: claude-sonnet-5-5, refuse_below: 0.35, max_citations: 8}
```

## 6. Evaluation

`evals/golden/*.jsonl`: `{id, question, principal, expected_answer?, relevant_chunk_ids?, relevant_document_ids?, should_refuse?, forbidden_document_ids?}`.

| metric | needs | how |
|---|---|---|
| recall@k, MRR, nDCG@k | relevant ids | deterministic |
| leak rate | forbidden ids | deterministic, any leak fails CI |
| citation precision | nothing | each citation must be in evidence and judged to support its sentence |
| faithfulness | nothing | LLM judge: every claim supported by cited chunks (NLI-style prompt) |
| answer correctness | expected answer | LLM judge, 0–1, calibrated on a labelled subset (Cohen's κ reported) |
| refusal correctness | should_refuse | deterministic |
| latency p50/p95, tokens, cost | nothing | from traces |

`ragstack eval --config evals/configs/*.yaml` runs every config over the golden set and writes a
markdown comparison table; the README table is generated from it, never typed by hand.

## 7. Non-goals (v0.1)

Streaming answers, UI, multi-vector/ColPali, knowledge-graph retrieval, Redis cache, OpenTelemetry
export. Each has a port or a config slot ready; none is built until a measurement asks for it.
