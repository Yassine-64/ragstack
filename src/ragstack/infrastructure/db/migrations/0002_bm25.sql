-- Sparse branch: real BM25 (ParadeDB pg_search, Tantivy) instead of ts_rank_cd, which has no IDF:
-- on SciFact a common word counted as much as a rare one and hybrid scored below dense alone.
-- English stemming + stopwords; context and text are indexed as one field, as the tsvector was.
CREATE EXTENSION IF NOT EXISTS pg_search;
DROP INDEX IF EXISTS chunks_tsv_idx;
ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;
CREATE INDEX IF NOT EXISTS chunks_bm25_idx ON chunks USING bm25 (
    id,
    ((context || ' ' || text)::pdb.simple('alias=content', 'stemmer=english', 'stopwords_language=english'))
);
