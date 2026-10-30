"""Listwise LLM reranker: scores come back in passage order, bad JSON degrades to the fused order."""

from ragstack.domain.ports import Completion
from ragstack.infrastructure.rerankers.llm import LlmReranker


class JsonLLM:
    name = "json"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, messages, max_tokens=1024, temperature=0.0):
        return Completion(text=self.reply, input_tokens=1, output_tokens=1, model="json")


async def test_scores_follow_passage_order_and_scale():
    r = LlmReranker(JsonLLM('Sure: {"0": 2, "1": 10, "2": 0}'), batch_size=20)
    assert await r.score("q", ["a", "b", "c"]) == [0.2, 1.0, 0.0]


async def test_batches_concatenate_and_garbage_is_neutral():
    r = LlmReranker(JsonLLM("no json here"), batch_size=2)
    assert await r.score("q", ["a", "b", "c"]) == [0.0, 0.0, 0.0]
