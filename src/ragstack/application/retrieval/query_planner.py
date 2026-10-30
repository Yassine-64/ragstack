"""Query rewriting, multi-query paraphrases and HyDE, all optional, all through the LLM port."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragstack.application.config import QueryConfig
from ragstack.domain.ports import LLM, ChatMessage

_REWRITE = (
    "Rewrite the user question as a precise, self-contained search query. Keep names, numbers and "
    "dates. Answer with the query only.\n\nQuestion: {q}"
)
_MULTI = (
    "Write {n} alternative phrasings of this search query that a different person might use, one per "
    "line, no numbering, no commentary.\n\nQuery: {q}"
)
_HYDE = (
    "Write a short factual passage (3-4 sentences) that would answer this question, as it might "
    "appear in an internal document. Do not say you are guessing.\n\nQuestion: {q}"
)


@dataclass
class QueryPlan:
    original: str
    queries: list[str] = field(default_factory=list)  # strings searched with (sparse + dense)
    hyde_text: str | None = None  # embedded for dense only
    input_tokens: int = 0
    output_tokens: int = 0


class QueryPlanner:
    def __init__(self, llm: LLM | None, config: QueryConfig) -> None:
        self.llm = llm
        self.config = config

    async def plan(self, question: str) -> QueryPlan:
        plan = QueryPlan(original=question, queries=[question])
        if self.llm is None or not (self.config.rewrite or self.config.multi_query or self.config.hyde):
            return plan
        if self.config.rewrite:
            out = await self.llm.complete([ChatMessage("user", _REWRITE.format(q=question))], max_tokens=120)
            # an empty or refused rewrite must never replace the question with nothing
            first = next((ln.strip() for ln in out.text.splitlines() if ln.strip()), "")
            plan.queries = [first or question]
            plan.input_tokens, plan.output_tokens = (
                plan.input_tokens + out.input_tokens,
                plan.output_tokens + out.output_tokens,
            )
        if self.config.multi_query:
            base = plan.queries[0]
            out = await self.llm.complete(
                [ChatMessage("user", _MULTI.format(n=self.config.multi_query, q=base))],
                max_tokens=300,
            )
            extra = [ln.strip("-• ").strip() for ln in out.text.splitlines() if ln.strip()]
            plan.queries += [e for e in extra[: self.config.multi_query] if e and e not in plan.queries]
            plan.input_tokens, plan.output_tokens = (
                plan.input_tokens + out.input_tokens,
                plan.output_tokens + out.output_tokens,
            )
        if self.config.hyde:
            out = await self.llm.complete([ChatMessage("user", _HYDE.format(q=question))], max_tokens=250)
            plan.hyde_text = out.text.strip() or None
            plan.input_tokens, plan.output_tokens = (
                plan.input_tokens + out.input_tokens,
                plan.output_tokens + out.output_tokens,
            )
        return plan
