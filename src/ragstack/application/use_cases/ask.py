from __future__ import annotations

from dataclasses import dataclass

from ragstack.application.agent import AgenticRag
from ragstack.domain.models import Answer, RetrievedChunk, Trace
from ragstack.domain.ports import AccessRepository, AuditRepository


@dataclass
class AskResult:
    answer: Answer
    evidence: list[RetrievedChunk]
    trace: Trace


class Ask:
    def __init__(
        self, access: AccessRepository, agent: AgenticRag, audit: AuditRepository | None = None
    ) -> None:
        self.access = access
        self.agent = agent
        self.audit = audit

    async def __call__(self, tenant_id: str, user_id: str, question: str) -> AskResult:
        principal = await self.access.resolve(tenant_id, user_id)
        answer, evidence, trace = await self.agent.run(principal, question)
        if self.audit is not None:
            await self.audit.record(principal, trace, [rc.chunk.id for rc in evidence])
        return AskResult(answer=answer, evidence=evidence, trace=trace)
