"""Answer only from the evidence, cite with [n], refuse when the evidence is weak."""

from __future__ import annotations

from ragstack.application.config import GenerationConfig
from ragstack.domain.models import Answer, RetrievedChunk
from ragstack.domain.ports import LLM, ChatMessage
from ragstack.domain.services import verify_citations

_SYSTEM = """You answer questions strictly from the provided passages.

Rules:
- Use only the passages. If they do not contain the answer, reply exactly: NOT_IN_DOCS
- Cite every factual sentence with the passage number in square brackets, like [2]. Several are fine: [1][3].
- Never cite a number that is not in the list. Never invent details.
- If passages conflict, say so and cite both.
- Be concise. Plain text, no preamble."""

_USER = """Passages:
{passages}

Question: {question}"""

REFUSAL_TEXT = "I can't answer that from the documents I have access to."


class GroundedAnswerer:
    def __init__(self, llm: LLM, config: GenerationConfig) -> None:
        self.llm = llm
        self.config = config

    def format_passages(self, evidence: list[RetrievedChunk]) -> str:
        lines = []
        for i, rc in enumerate(evidence, 1):
            where = f"{rc.source_name}" + (f", p.{rc.chunk.page}" if rc.chunk.page else "")
            lines.append(f"[{i}] ({where}) {rc.chunk.text[: self.config.max_evidence_chars]}")
        return "\n\n".join(lines)

    async def answer(
        self, question: str, evidence: list[RetrievedChunk], evidence_score: float
    ) -> tuple[Answer, int, int]:
        shown = [rc.chunk.id for rc in evidence]
        if not evidence or evidence_score < self.config.refuse_below:
            return (
                Answer(
                    text=REFUSAL_TEXT,
                    refused=True,
                    evidence_score=evidence_score,
                    evidence=shown,
                    refusal_reason=f"evidence score {evidence_score:.2f} < {self.config.refuse_below}",
                ),
                0,
                0,
            )
        out = await self.llm.complete(
            [
                ChatMessage("system", _SYSTEM),
                ChatMessage("user", _USER.format(passages=self.format_passages(evidence), question=question)),
            ],
            max_tokens=self.config.max_tokens,
        )
        raw = out.text.strip()
        if not raw or raw.upper().startswith("NOT_IN_DOCS"):
            return (
                Answer(
                    text=REFUSAL_TEXT,
                    refused=True,
                    evidence_score=evidence_score,
                    evidence=shown,
                    refusal_reason="model found no answer in the passages",
                ),
                out.input_tokens,
                out.output_tokens,
            )
        text, citations = verify_citations(raw, evidence)
        if not citations:
            return (
                Answer(
                    text=REFUSAL_TEXT,
                    refused=True,
                    evidence_score=evidence_score,
                    evidence=shown,
                    refusal_reason="answer carried no valid citation",
                ),
                out.input_tokens,
                out.output_tokens,
            )
        return (
            Answer(text=text, citations=citations, evidence_score=evidence_score, evidence=shown),
            out.input_tokens,
            out.output_tokens,
        )
