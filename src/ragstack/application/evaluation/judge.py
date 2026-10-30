"""LLM-as-judge for faithfulness, answer correctness and citation support. Prompts return a bare number."""

from __future__ import annotations

import re

from ragstack.domain.models import Answer, RetrievedChunk
from ragstack.domain.ports import LLM, ChatMessage

_NUM = re.compile(r"([01](?:\.\d+)?)")

_FAITHFUL = """Passages:
{passages}

Answer:
{answer}

What fraction of the factual claims in the answer are supported by the passages?
Reply with a single number between 0 and 1."""

_CORRECT = """Question: {q}
Reference answer: {ref}
Candidate answer: {cand}

How correct is the candidate compared with the reference? 1 = same facts, 0 = wrong or missing.
Reply with a single number between 0 and 1."""

_CITATION = """Passage [{n}]:
{passage}

Sentence from an answer that cites [{n}]:
{sentence}

Does the passage support the sentence? Reply 1 for yes, 0 for no."""


class Judge:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def _score(self, prompt: str) -> float:
        out = await self.llm.complete([ChatMessage("user", prompt)], max_tokens=10)
        m = _NUM.search(out.text)
        return min(1.0, max(0.0, float(m.group(1)))) if m else 0.0

    async def faithfulness(self, answer: Answer, evidence: list[RetrievedChunk]) -> float:
        if answer.refused:
            return float("nan")
        shown = {c.chunk_id for c in answer.citations}
        passages = "\n\n".join(
            f"[{i}] {rc.chunk.text[:800]}"
            for i, rc in enumerate(evidence, 1)
            if rc.chunk.id in shown or not shown
        )
        return await self._score(_FAITHFUL.format(passages=passages, answer=answer.text))

    async def correctness(self, question: str, reference: str, answer: Answer) -> float:
        if answer.refused:
            return 0.0
        return await self._score(_CORRECT.format(q=question, ref=reference, cand=answer.text))

    async def citation_precision(self, answer: Answer, evidence: list[RetrievedChunk]) -> float:
        """For each [n] in the answer: is the sentence carrying it supported by passage n?"""
        if answer.refused or not answer.citations:
            return float("nan")
        by_index = {i: rc for i, rc in enumerate(evidence, 1)}
        sentences = re.split(r"(?<=[.!?])\s+(?!\[)", answer.text)  # keep trailing [n] with its sentence
        checks: list[tuple[int, str]] = []
        for s in sentences:
            for n in {int(x) for x in re.findall(r"\[(\d{1,3})\]", s)}:
                if n in by_index:
                    checks.append((n, s))
        if not checks:
            return float("nan")
        scores = [
            await self._score(_CITATION.format(n=n, passage=by_index[n].chunk.text[:800], sentence=s))
            for n, s in checks
        ]
        return sum(scores) / len(scores)
