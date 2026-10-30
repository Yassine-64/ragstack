"""Does the evidence answer the question? One cheap LLM call, a number in [0,1], a reason."""

from __future__ import annotations

import re

from ragstack.domain.models import RetrievedChunk
from ragstack.domain.ports import LLM, ChatMessage

_PROMPT = """You grade whether retrieved passages contain enough information to answer a question.

Question: {q}

Passages:
{passages}

Reply with exactly two lines:
score: <number between 0 and 1, where 1 = fully answerable from the passages>
reason: <one sentence>"""

_SCORE = re.compile(r"score:\s*([01](?:\.\d+)?)", re.I)


class EvidenceGrader:
    def __init__(self, llm: LLM, max_chars_per_passage: int = 600) -> None:
        self.llm = llm
        self.max_chars = max_chars_per_passage

    async def grade(self, question: str, evidence: list[RetrievedChunk]) -> tuple[float, str, int, int]:
        if not evidence:
            return 0.0, "no passages retrieved", 0, 0
        passages = "\n\n".join(f"[{i}] {rc.chunk.text[: self.max_chars]}" for i, rc in enumerate(evidence, 1))
        out = await self.llm.complete(
            [ChatMessage("user", _PROMPT.format(q=question, passages=passages))], max_tokens=80
        )
        m = _SCORE.search(out.text)
        score = min(1.0, max(0.0, float(m.group(1)))) if m else 0.0
        reason = (
            out.text.split("reason:", 1)[-1].strip() if "reason:" in out.text.lower() else out.text.strip()
        )
        return score, reason, out.input_tokens, out.output_tokens
