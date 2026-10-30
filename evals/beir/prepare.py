"""Turn a BEIR dataset into a ragstack corpus + golden set (stdlib only).

    python evals/beir/prepare.py scifact          # -> .data/beir/scifact/{corpus/*.md, golden.jsonl}

Every document becomes `<doc-id>.md` so the eval runner matches qrels on source_name.
All documents go to one tenant, readable by one user: this measures ranking, not ACLs.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
TENANT, USER = "beir", "reader"


def main(name: str, out_root: Path = Path(".data/beir")) -> None:
    out = out_root / name
    corpus_dir = out / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(URL.format(name=name), timeout=120).read()  # noqa: S310 fixed host
    zf = zipfile.ZipFile(io.BytesIO(raw))
    read = lambda member: zf.read(f"{name}/{member}").decode("utf-8")  # noqa: E731

    qrels: dict[str, list[str]] = defaultdict(list)
    for row in csv.DictReader(io.StringIO(read("qrels/test.tsv")), delimiter="\t"):
        if int(row["score"]) > 0:
            qrels[row["query-id"]].append(row["corpus-id"])

    n_docs = 0
    for line in read("corpus.jsonl").splitlines():
        doc = json.loads(line)
        # doc ids are digits or slugs; keep them filename-safe
        doc_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc["_id"])
        title = doc.get("title") or doc_id
        (corpus_dir / f"{doc_id}.md").write_text(f"# {title}\n\n{doc['text']}\n", encoding="utf-8")
        n_docs += 1

    cases = []
    for line in read("queries.jsonl").splitlines():
        q = json.loads(line)
        if q["_id"] in qrels:
            cases.append(
                {
                    "id": f"{name}-{q['_id']}",
                    "question": q["text"],
                    "user_id": USER,
                    "tenant_id": TENANT,
                    "relevant_document_ids": [f"{d}.md" for d in qrels[q["_id"]]],
                    "tags": ["beir", name],
                }
            )
    (out / "golden.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    print(f"{name}: {n_docs} documents, {len(cases)} test queries -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scifact")
