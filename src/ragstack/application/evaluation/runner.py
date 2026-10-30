"""Runs the golden set through the Ask use case and aggregates metrics into an EvalRun."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ragstack.application.use_cases import Ask
from ragstack.domain.models import EvalCase, EvalRun
from ragstack.domain.ports import EvalRepository

from .judge import Judge
from .metrics import leak_rate, mrr, ndcg_at_k, recall_at_k, refusal_correct

# ponytail: flat prices for the models we actually use; replace with a table when models multiply
_USD_PER_MTOK = {"in": 3.0, "out": 15.0}


def _doc_key(rc: Any, wanted: set[str]) -> str:
    return rc.source_name if rc.source_name in wanted else str(rc.chunk.document_id)


def _clean(value: float) -> float | None:
    """NaN means 'not applicable'; it is stored as null (JSON has no NaN) and skipped by the mean."""
    return None if math.isnan(value) else value


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


class EvalRunner:
    def __init__(
        self,
        ask_factory: Callable[[], Ask],
        judge: Judge | None = None,
        repo: EvalRepository | None = None,
        k: int = 10,
    ) -> None:
        self.ask_factory = ask_factory
        self.judge = judge
        self.repo = repo
        self.k = k

    async def run(self, name: str, config: dict[str, Any], cases: list[EvalCase]) -> EvalRun:
        ask = self.ask_factory()
        rows: list[dict[str, Any]] = []
        for case in cases:
            result = await ask(case.tenant_id, case.user_id, case.question)
            chunk_ids = [rc.chunk.id for rc in result.evidence]
            relevant_chunks = set(case.relevant_chunk_ids)
            relevant_docs = set(case.relevant_document_ids)
            forbidden_docs = set(case.forbidden_document_ids)
            # golden sets may name documents by id or by source_name; match on whichever was given
            doc_ids = [_doc_key(rc, relevant_docs | forbidden_docs) for rc in result.evidence]
            row: dict[str, Any] = {
                "id": case.id,
                "refused": result.answer.refused,
                "attempts": result.trace.attempts,
                "latency_ms": result.trace.total_ms,
                "input_tokens": result.trace.input_tokens,
                "output_tokens": result.trace.output_tokens,
                "leak_rate": leak_rate(doc_ids, forbidden_docs) if forbidden_docs else None,
                "refusal_correct": refusal_correct(result.answer.refused, case.should_refuse),
            }
            if relevant_chunks:
                row |= {
                    "recall@k": recall_at_k(chunk_ids, relevant_chunks, self.k),
                    "mrr": mrr(chunk_ids, relevant_chunks),
                    "ndcg@k": ndcg_at_k(chunk_ids, relevant_chunks, self.k),
                }
            elif relevant_docs:
                uniq_docs = list(dict.fromkeys(doc_ids))
                row |= {
                    "recall@k": recall_at_k(uniq_docs, relevant_docs, self.k),
                    "mrr": mrr(uniq_docs, relevant_docs),
                    "ndcg@k": ndcg_at_k(uniq_docs, relevant_docs, self.k),
                }
            if self.judge is not None and not case.should_refuse:
                row["faithfulness"] = _clean(await self.judge.faithfulness(result.answer, result.evidence))
                row["citation_precision"] = _clean(
                    await self.judge.citation_precision(result.answer, result.evidence)
                )
                if case.expected_answer:
                    row["correctness"] = _clean(
                        await self.judge.correctness(case.question, case.expected_answer, result.answer)
                    )
            rows.append(row)

        metrics: dict[str, float] = {}
        for key in (
            "recall@k",
            "mrr",
            "ndcg@k",
            "faithfulness",
            "citation_precision",
            "correctness",
            "refusal_correct",
            "leak_rate",
        ):
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            if vals:
                metrics[key] = _mean(vals)
        lat = sorted(float(r["latency_ms"]) for r in rows)
        if lat:
            metrics["latency_p50_ms"] = lat[len(lat) // 2]
            metrics["latency_p95_ms"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        tokens_in = sum(int(r["input_tokens"]) for r in rows)
        tokens_out = sum(int(r["output_tokens"]) for r in rows)
        metrics["cost_usd_per_query"] = (
            ((tokens_in * _USD_PER_MTOK["in"] + tokens_out * _USD_PER_MTOK["out"]) / 1e6 / len(rows))
            if rows
            else 0.0
        )
        metrics["cases"] = float(len(rows))
        run = EvalRun(name=name, config=config, metrics=metrics, per_case=rows)
        if self.repo is not None:
            await self.repo.save(run)
        return run
