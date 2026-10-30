-- ragstack schema. {DIM} is replaced by the configured embedding dimension at apply time.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    tenant_id text NOT NULL,
    id        text NOT NULL,
    PRIMARY KEY (tenant_id, id)
);
CREATE TABLE IF NOT EXISTS group_members (
    tenant_id text NOT NULL,
    group_id  text NOT NULL,
    user_id   text NOT NULL,
    PRIMARY KEY (tenant_id, group_id, user_id)
);

CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,
    tenant_id   text NOT NULL,
    source_name text NOT NULL,
    media_type  text NOT NULL DEFAULT 'text/plain',
    metadata    jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    deleted_at  timestamptz
);
CREATE INDEX IF NOT EXISTS documents_tenant_idx ON documents (tenant_id) WHERE deleted_at IS NULL;

-- deny by default: a document with no grant row is visible to nobody
CREATE TABLE IF NOT EXISTS access_grants (
    document_id    text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    principal_type text NOT NULL CHECK (principal_type IN ('user', 'group', 'everyone')),
    principal_id   text NOT NULL DEFAULT '',
    PRIMARY KEY (document_id, principal_type, principal_id)
);
CREATE INDEX IF NOT EXISTS access_grants_principal_idx ON access_grants (principal_type, principal_id);

CREATE TABLE IF NOT EXISTS chunks (
    id           text PRIMARY KEY,
    document_id  text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    idx          int  NOT NULL,
    section_path text[] NOT NULL DEFAULT '{}',
    page         int,
    bbox         jsonb,
    context      text NOT NULL DEFAULT '',
    text         text NOT NULL,
    token_count  int  NOT NULL DEFAULT 0,
    metadata     jsonb NOT NULL DEFAULT '{}',
    embedding    vector({DIM}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(context, '') || ' ' || text)) STORED,
    UNIQUE (document_id, idx)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- audit: every question, and which chunks were shown to whom
CREATE TABLE IF NOT EXISTS queries (
    id         text PRIMARY KEY,
    tenant_id  text NOT NULL,
    user_id    text NOT NULL,
    question   text NOT NULL,
    trace      jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS retrievals (
    query_id text NOT NULL REFERENCES queries (id) ON DELETE CASCADE,
    chunk_id text NOT NULL,
    rank     int  NOT NULL,
    PRIMARY KEY (query_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS retrievals_chunk_idx ON retrievals (chunk_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id         text PRIMARY KEY,
    name       text NOT NULL,
    config     jsonb NOT NULL,
    metrics    jsonb NOT NULL,
    per_case   jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
