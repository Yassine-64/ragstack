"""FinanceBench (Patronus AI, CC-BY-NC) → company PDFs + golden set with page-level evidence.

    python evals/financebench/prepare.py            # -> .data/financebench/{pdfs/*.pdf, golden.jsonl, seed.sh}

Each filing is readable only by that company's analyst group, so the golden set also holds ACL
cases: the same question asked by another company's analyst must be refused and must never
retrieve the filing. That is the permission story on real 10-Ks, not on a toy handbook.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
PDF = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc}.pdf"
DEFAULT_DOCS = [
    "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "BOEING_2022_10K",
    "PEPSICO_2022_10K",
    "AMCOR_2023_10K",
    "3M_2022_10K",
]
TENANT = "financebench"


def group_of(doc: str) -> str:
    return doc.split("_")[0].lower()


def main(docs: list[str], out: Path = Path(".data/financebench")) -> None:
    pdf_dir = out / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in urllib.request.urlopen(RAW, timeout=60).read().decode().splitlines()]
    rows = [r for r in rows if r["doc_name"] in docs]
    for doc in docs:
        target = pdf_dir / f"{doc}.pdf"
        if not target.exists():
            target.write_bytes(urllib.request.urlopen(PDF.format(doc=doc), timeout=120).read())

    cases = []
    for i, r in enumerate(rows):
        doc, grp = r["doc_name"], group_of(r["doc_name"])
        pages = sorted({int(e["evidence_page_num"]) + 1 for e in r["evidence"]})  # 0-based in the source
        cases.append(
            {
                "id": r["financebench_id"],
                "question": r["question"],
                "user_id": f"{grp}-analyst",
                "tenant_id": TENANT,
                "expected_answer": r["answer"],
                "relevant_document_ids": [f"{doc}.pdf"],
                "relevant_pages": {f"{doc}.pdf": pages},
                "tags": ["financebench", r["question_type"]],
            }
        )
        if i % 5 == 0:  # every fifth question doubles as an ACL case for an analyst of another company
            other = next(g for g in map(group_of, docs) if g != grp)
            cases.append(
                {
                    "id": f"{r['financebench_id']}-acl",
                    "question": r["question"],
                    "user_id": f"{other}-analyst",
                    "tenant_id": TENANT,
                    "forbidden_document_ids": [f"{doc}.pdf"],
                    "should_refuse": True,
                    "tags": ["financebench", "acl"],
                }
            )
    (out / "golden.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")

    seed = ["#!/usr/bin/env bash", "set -euo pipefail", 'R="uv run ragstack"']
    for doc in docs:
        g = group_of(doc)
        seed.append(f"$R user {g}-analyst --tenant {TENANT} --group {g}")
        seed.append(f"$R ingest {pdf_dir}/{doc}.pdf --tenant {TENANT} --group {g}")
    (out / "seed.sh").write_text("\n".join(seed) + "\n", encoding="utf-8")
    acl = sum(c.get("should_refuse", False) for c in cases)
    print(f"{len(docs)} filings, {len(cases)} cases ({acl} ACL) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_DOCS)
