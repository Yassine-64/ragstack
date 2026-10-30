"""Control flow of the agentic loop with fakes: retry on weak evidence, rewrite, refusal, citations."""

from ragstack.application.agent import AgenticRag
from ragstack.application.config import AgentConfig, GenerationConfig, QueryConfig, SearchConfig
from ragstack.application.generation import EvidenceGrader, GroundedAnswerer
from ragstack.application.retrieval import QueryPlanner, RetrievalPipeline
from ragstack.domain.models import Chunk, Principal
from ragstack.domain.ports.search import SearchHit
from ragstack.infrastructure.embeddings.hashing import HashingEmbedder
from ragstack.infrastructure.llm.scripted import ScriptedLLM

PRINCIPAL = Principal(tenant_id="t", user_id="u")


class FakeIndex:
    """Returns different hits depending on the query text, to prove rewrites change retrieval."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, principal, query_text, query_vector, dense_k, sparse_k):
        self.queries.append(query_text)
        if "rollback" in query_text.lower():
            text = "Rollback is performed with deployctl rollback. The previous two releases are kept warm."
            return [
                SearchHit(
                    Chunk(id="good", document_id="d1", index=0, text=text), "runbook.md", 1, 1, 0.9, 0.5
                )
            ]
        text = "Lunch is one hour. Core hours are 10:00 to 16:00."
        return [
            SearchHit(
                Chunk(id="bad", document_id="d2", index=0, text=text), "handbook.md", 1, None, 0.2, None
            )
        ]


class RewritingLLM(ScriptedLLM):
    """Scripted LLM whose 'rewrite after failure' answer introduces the keyword the index needs."""

    def _route(self, prompt, system):
        if "returned passages that did not answer" in prompt:
            return "deployctl rollback command"
        return super()._route(prompt, system)


def build(llm, retries: int):
    index = FakeIndex()
    retrieval = RetrievalPipeline(index, HashingEmbedder(16), SearchConfig(rerank=False, final_k=5))
    agent = AgenticRag(
        planner=QueryPlanner(None, QueryConfig()),
        retrieval=retrieval,
        answerer=GroundedAnswerer(llm, GenerationConfig(refuse_below=0.35)),
        grader=EvidenceGrader(llm),
        config=AgentConfig(grade=True, max_retries=retries, min_evidence_score=0.5),
        llm=llm,
    )
    return agent, index


async def test_retry_rewrites_query_and_recovers():
    agent, index = build(RewritingLLM(), retries=2)
    answer, evidence, trace = await agent.run(PRINCIPAL, "How do I revert a release?")
    assert trace.attempts == 2 and len(index.queries) == 2
    assert not answer.refused and answer.citations[0].chunk_id == "good"
    assert "[1]" in answer.text and trace.rewritten[-1] == "deployctl rollback command"
    assert [s.name for s in trace.steps][:2] == ["plan", "embed"]


async def test_no_retry_budget_means_refusal_with_reason():
    agent, index = build(ScriptedLLM(), retries=0)
    answer, _, trace = await agent.run(PRINCIPAL, "How do I revert a release?")
    assert answer.refused and "evidence score" in (answer.refusal_reason or "")
    assert trace.attempts == 1 and len(index.queries) == 1


async def test_grading_disabled_answers_directly():
    llm = ScriptedLLM()
    index = FakeIndex()
    retrieval = RetrievalPipeline(index, HashingEmbedder(16), SearchConfig(rerank=False))
    agent = AgenticRag(
        QueryPlanner(None, QueryConfig()),
        retrieval,
        GroundedAnswerer(llm, GenerationConfig()),
        EvidenceGrader(llm),
        AgentConfig(grade=False),
        llm,
    )
    answer, _, _ = await agent.run(PRINCIPAL, "What is the rollback command?")
    assert not answer.refused and answer.evidence_score == 1.0
