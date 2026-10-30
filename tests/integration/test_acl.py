"""The ACL lives inside the retrieval SQL. These tests run against real pgvector, not a mock."""

import pytest

from ragstack.domain.models import AccessGrant, PrincipalType

pytestmark = pytest.mark.integration


async def sources(container, tenant, user, query):
    hits, _ = await container.search(tenant, user, query)
    return {h.source_name for h in hits}


async def test_group_isolation_and_everyone(container, tenant, seeded):
    assert "expense-approval-matrix.md" in await sources(container, tenant, "bob", "salary band level 2")
    assert "expense-approval-matrix.md" not in await sources(
        container, tenant, "alice", "salary band level 2"
    )
    assert "deployment-runbook.md" not in await sources(container, tenant, "bob", "rollback deployment")
    assert "employee-handbook.md" in await sources(container, tenant, "dave", "core hours reachable")


async def test_deny_by_default_and_revocation(container, tenant, seeded, corpus_dir):
    doc, _ = await container.ingest_document(
        corpus_dir / "hr" / "performance-review-process.md", tenant
    )  # no grants
    assert "performance-review-process.md" not in await sources(
        container, tenant, "dave", "formal warning valid"
    )
    grant = AccessGrant(document_id=doc.id, principal_type=PrincipalType.USER, principal_id="dave")
    await container.access.grant(grant)
    assert "performance-review-process.md" in await sources(container, tenant, "dave", "formal warning valid")
    await container.access.revoke(grant)
    assert "performance-review-process.md" not in await sources(
        container, tenant, "dave", "formal warning valid"
    )


async def test_soft_delete_hides_immediately_and_tenants_are_isolated(container, tenant, seeded):
    assert await container.documents.soft_delete(tenant, seeded["it-security-policy.md"])
    assert "it-security-policy.md" not in await sources(container, tenant, "dave", "password rotated days")
    other_tenant = tenant + "-other"
    await container.access.ensure_user(other_tenant, "alice", ["engineering"])
    assert await sources(container, other_tenant, "alice", "rollback deployment") == set()


async def test_hybrid_hits_carry_both_ranks(container, tenant, seeded):
    hits, trace = await container.search(tenant, "alice", "deployctl rollback")
    top = hits[0]
    assert top.source_name == "deployment-runbook.md"
    assert top.scores.sparse_rank is not None and top.scores.dense_rank is not None and top.scores.rrf
    assert [s.name for s in trace.steps] == ["embed", "hybrid_search", "rerank"]
