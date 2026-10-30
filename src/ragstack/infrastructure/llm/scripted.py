"""Offline LLM for tests and CI. Answers by rule, so the pipeline's control flow is fully testable:

- grading prompts → score from stemmed lexical overlap between question and passages
- answer prompts   → the passage sentences that overlap the question, cited [n]; NOT_IN_DOCS when none
- judge prompts    → 1.0 / 0.0 by the same overlap rule
- everything else  → echoes the last line of the prompt (query rewrites come back unchanged)
"""

from __future__ import annotations

import re

from ragstack.domain.ports.llm import ChatMessage, Completion

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "does",
    "do",
    "did",
    "for",
    "on",
    "and",
    "or",
    "by",
    "with",
    "our",
    "we",
    "it",
    "its",
    "can",
    "i",
    "my",
    "when",
    "where",
    "must",
    "should",
    "may",
    "that",
    "this",
    "there",
    "their",
    "they",
    "you",
    "your",
    "per",
    "at",
    "from",
    "into",
    "any",
    "all",
    "not",
    "no",
    "yes",
    "than",
    "then",
    "will",
    "would",
    "has",
    "have",
    "had",
}
_PASSAGE = re.compile(r"\[(\d+)\]\s*\([^)]*\)\s*(.*?)(?=\n\s*\[\d+\]\s*\(|\Z)", re.S)
_PASSAGE_PLAIN = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\s*\[\d+\]|\Z)", re.S)


def _toks(text: str) -> set[str]:
    """Lower-cased content words, crudely stemmed to 5 chars so rollback/rolls and release/releases meet."""
    return {t[:5] for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2}


def _overlap(a: str, b: str) -> float:
    ta = _toks(a)
    return len(ta & _toks(b)) / len(ta) if ta else 0.0


class ScriptedLLM:
    name = "scripted"

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    async def complete(
        self, messages: list[ChatMessage], max_tokens: int = 1024, temperature: float = 0.0
    ) -> Completion:
        prompt = messages[-1].content
        system = "\n".join(m.content for m in messages if m.role == "system")
        text = self._route(prompt, system)
        return Completion(
            text=text, input_tokens=len(prompt) // 4, output_tokens=len(text) // 4, model=self.name
        )

    def _route(self, prompt: str, system: str) -> str:
        if "score:" in prompt and "Passages:" in prompt:
            return self._grade(prompt)
        if "strictly from the provided passages" in system:
            return self._answer(prompt)
        if "What fraction of the factual claims" in prompt:
            answer = prompt.split("Answer:", 1)[-1]
            return "1" if _overlap(answer, prompt.split("Answer:", 1)[0]) >= self.threshold else "0"
        if "Does the passage support" in prompt:
            sentence = prompt.split("Sentence from an answer", 1)[-1].split(":", 1)[-1]
            passage = prompt.split("Passage [", 1)[-1].split("Sentence from an answer", 1)[0]
            return "1" if _overlap(sentence, passage) >= self.threshold else "0"
        if "How correct is the candidate" in prompt:
            ref = prompt.split("Reference answer:")[-1].split("Candidate answer:")[0]
            cand = prompt.split("Candidate answer:")[-1]
            return f"{_overlap(ref, cand):.2f}"
        lines = [ln for ln in prompt.strip().splitlines() if ln.strip()]
        return lines[-1].split(":", 1)[-1].strip() if lines else ""

    @staticmethod
    def _passages(prompt: str) -> list[tuple[int, str]]:
        body = prompt.split("Passages:", 1)[-1].split("Question:", 1)[0]
        found = _PASSAGE.findall(body) or _PASSAGE_PLAIN.findall(body)
        return [(int(n), t.strip()) for n, t in found]

    def _grade(self, prompt: str) -> str:
        question = prompt.split("Question:", 1)[-1].split("Passages:", 1)[0]
        best = max((_overlap(question, p) for _, p in self._passages(prompt)), default=0.0)
        return f"score: {best:.2f}\nreason: lexical overlap {best:.2f}"

    def _answer(self, prompt: str) -> str:
        question = prompt.split("Question:", 1)[-1]
        out = [
            f"{sentence.strip()} [{n}]"
            for n, passage in self._passages(prompt)
            for sentence in re.split(r"(?<=[.!?])\s+", passage)
            if _overlap(question, sentence) >= self.threshold
        ]
        return " ".join(out) if out else "NOT_IN_DOCS"
