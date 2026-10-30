import pytest

pytestmark = pytest.mark.integration


async def test_ask_cites_shown_chunks_and_records_audit(container, tenant, seeded):
    r = await container.ask(tenant, "alice", "What command rolls back a service deployment?")
    assert not r.answer.refused and "deployctl" in r.answer.text
    cit = r.answer.citations[0]
    assert cit.source_name == "deployment-runbook.md" and cit.chunk_id in r.answer.evidence
    exposures = await container.audit.exposures(tenant, seeded["deployment-runbook.md"])
    assert exposures and exposures[0]["user_id"] == "alice"


async def test_forbidden_question_is_refused_and_never_exposed(container, tenant, seeded):
    r = await container.ask(tenant, "bob", "What command rolls back a service deployment?")
    assert r.answer.refused
    assert all(rc.source_name != "deployment-runbook.md" for rc in r.evidence)
    assert await container.audit.exposures(tenant, seeded["deployment-runbook.md"]) == []


async def test_eval_runner_reports_zero_leaks_on_golden_set(container, tenant, seeded, tmp_path):
    from pathlib import Path

    from ragstack.application.evaluation import load_golden

    cases = load_golden(Path(__file__).resolve().parents[2] / "evals" / "golden" / "handbook.jsonl")
    for c in cases:
        c.tenant_id = tenant
    run = await container.eval_runner().run("offline", container.config.model_dump(), cases)
    assert run.metrics["leak_rate"] == 0.0
    assert run.metrics["refusal_correct"] >= 0.8
    assert run.metrics["recall@k"] >= 0.8
