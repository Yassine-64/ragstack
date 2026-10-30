"""PDF → docparse → chunks with page provenance → hybrid search → citation with a page number."""

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

pytestmark = pytest.mark.integration


@pytest.fixture
def policy_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "travel-policy.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, h - 72, "Travel policy")
    c.setFont("Helvetica", 11)
    c.drawString(72, h - 110, "Flights longer than six hours may be booked in premium economy.")
    c.drawString(72, h - 126, "Hotel nights are capped at 180 EUR in European capitals.")
    c.showPage()
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, h - 72, "Per diem")
    c.setFont("Helvetica", 11)
    c.drawString(72, h - 110, "The daily allowance abroad is 62 EUR and it covers meals and local transport.")
    c.showPage()
    c.save()
    return path


async def test_pdf_chunks_keep_pages_and_citations_point_at_them(container, tenant, policy_pdf):
    await container.access.ensure_user(tenant, "erin", [])
    doc, n = await container.ingest_document(policy_pdf, tenant, users=["erin"])
    assert doc.media_type == "application/pdf" and n >= 2
    assert "engines" in doc.metadata

    hits, _ = await container.search(tenant, "erin", "daily allowance abroad per diem")
    assert hits[0].source_name == "travel-policy.pdf" and hits[0].chunk.page == 2
    assert hits[0].chunk.bbox is not None and 0 <= hits[0].chunk.bbox["y0"] <= 1

    r = await container.ask(tenant, "erin", "What is the daily allowance abroad?")
    assert not r.answer.refused
    assert any(c.page == 2 and c.source_name == "travel-policy.pdf" for c in r.answer.citations)
