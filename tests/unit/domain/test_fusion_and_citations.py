from ragstack.domain.models import Chunk, RetrievedChunk, Scores
from ragstack.domain.ports.search import SearchHit
from ragstack.domain.services import extract_citation_indexes, reciprocal_rank_fusion, verify_citations


def chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="d", index=i, text=f"text {i}")


def hit(i: int, dense: int | None, sparse: int | None) -> SearchHit:
    return SearchHit(
        chunk=chunk(i),
        source_name="s",
        dense_rank=dense,
        sparse_rank=sparse,
        dense_score=0.9 if dense else None,
        sparse_score=0.5 if sparse else None,
    )


def test_rrf_prefers_chunks_ranked_by_both_signals():
    fused = reciprocal_rank_fusion([("q", [hit(1, 1, None), hit(2, 2, 1), hit(3, None, 2)])], k=60)
    assert [r.chunk.id for r in fused] == ["c2", "c1", "c3"]
    assert fused[0].scores.rrf == 1 / 62 + 1 / 61
    assert fused[0].scores.dense_rank == 2 and fused[0].scores.sparse_rank == 1


def test_rrf_accumulates_across_sub_queries_and_keeps_best_ranks():
    fused = reciprocal_rank_fusion([("q1", [hit(1, 3, None)]), ("q2", [hit(1, 1, None), hit(2, 2, None)])])
    assert fused[0].chunk.id == "c1" and fused[0].scores.dense_rank == 1
    assert fused[0].scores.rrf == 1 / 63 + 1 / 61


def test_citations_only_keep_markers_that_point_at_shown_evidence():
    evidence = [
        RetrievedChunk(chunk=chunk(1), scores=Scores(), source_name="a.md"),
        RetrievedChunk(chunk=chunk(2), scores=Scores(), source_name="b.md"),
    ]
    text, cits = verify_citations("Fact one [1]. Fact two [7]. Fact three [2][1].", evidence)
    assert text == "Fact one [1]. Fact two. Fact three [2][1]."
    assert [(c.index, c.chunk_id, c.source_name) for c in cits] == [(1, "c1", "a.md"), (2, "c2", "b.md")]
    assert extract_citation_indexes("[3] and [1] and [3]") == [3, 1]
