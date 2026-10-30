"""retrieve → grade → (rewrite → retrieve)* → answer. A small explicit state machine, no framework."""

from __future__ import annotations

import time

from ragstack.application.config import AgentConfig
from ragstack.application.generation import EvidenceGrader, GroundedAnswerer
from ragstack.application.retrieval import QueryPlan, QueryPlanner, RetrievalPipeline
from ragstack.domain.models import Answer, Principal, RetrievedChunk, Trace
from ragstack.domain.ports import LLM, ChatMessage

_REWRITE_AFTER_FAILURE = (
    "The search query below returned passages that did not answer the question.\n"
    "Question: {q}\nPrevious query: {prev}\nGrader said: {why}\n"
    "Write one different search query that targets the missing information. Query only."
)


class AgenticRag:
    def __init__(
        self,
        planner: QueryPlanner,
        retrieval: RetrievalPipeline,
        answerer: GroundedAnswerer,
        grader: EvidenceGrader | None,
        config: AgentConfig,
        llm: LLM | None = None,
    ) -> None:
        self.planner = planner
        self.retrieval = retrieval
        self.answerer = answerer
        self.grader = grader if config.grade else None
        self.config = config
        self.llm = llm

    async def run(self, principal: Principal, question: str) -> tuple[Answer, list[RetrievedChunk], Trace]:
        trace = Trace(question=question)
        t0 = time.perf_counter()
        plan = await self.planner.plan(question)
        trace.add(
            "plan",
            (time.perf_counter() - t0) * 1000,
            queries=plan.queries,
            hyde=bool(plan.hyde_text),
        )
        trace.rewritten = list(plan.queries)
        trace.input_tokens += plan.input_tokens
        trace.output_tokens += plan.output_tokens

        best: tuple[float, list[RetrievedChunk], str] = (0.0, [], "")
        for attempt in range(self.config.max_retries + 1):
            trace.attempts = attempt + 1
            evidence = await self.retrieval.retrieve(principal, plan, trace)
            if self.grader is None:
                best = (1.0 if evidence else 0.0, evidence, "grading disabled")
                break
            t1 = time.perf_counter()
            score, why, in_tok, out_tok = await self.grader.grade(question, evidence)
            trace.add(
                "grade",
                (time.perf_counter() - t1) * 1000,
                score=score,
                reason=why,
                attempt=attempt + 1,
            )
            trace.input_tokens += in_tok
            trace.output_tokens += out_tok
            if score > best[0]:
                best = (score, evidence, why)
            if (
                score >= self.config.min_evidence_score
                or attempt == self.config.max_retries
                or self.llm is None
            ):
                break
            plan = await self._rewrite(plan, question, why, trace)

        score, evidence, _ = best
        t2 = time.perf_counter()
        answer, in_tok, out_tok = await self.answerer.answer(question, evidence, score)
        trace.add(
            "generate",
            (time.perf_counter() - t2) * 1000,
            refused=answer.refused,
            citations=len(answer.citations),
        )
        trace.input_tokens += in_tok
        trace.output_tokens += out_tok
        return answer, evidence, trace

    async def _rewrite(self, plan: QueryPlan, question: str, why: str, trace: Trace) -> QueryPlan:
        assert self.llm is not None
        out = await self.llm.complete(
            [ChatMessage("user", _REWRITE_AFTER_FAILURE.format(q=question, prev=plan.queries[0], why=why))],
            max_tokens=100,
        )
        new_query = out.text.strip().splitlines()[0] if out.text.strip() else plan.queries[0]
        trace.input_tokens += out.input_tokens
        trace.output_tokens += out.output_tokens
        trace.rewritten.append(new_query)
        return QueryPlan(original=question, queries=[new_query])
