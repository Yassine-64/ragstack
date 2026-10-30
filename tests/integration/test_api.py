import pytest
from fastapi.testclient import TestClient

from ragstack.interfaces.api.app import create_app

pytestmark = pytest.mark.integration


def test_api_flow(container, tenant, corpus_dir):
    container.settings.api_key = "secret"
    app = create_app(container.settings, container)
    auth = {"Authorization": "Bearer secret", "X-Tenant-Id": tenant}
    with TestClient(app) as client:
        assert client.get("/health").json()["llm"] == "scripted"
        assert (
            client.post("/v1/ask", json={"question": "x"}, headers={"X-User-Id": "alice"}).status_code == 401
        )

        path = corpus_dir / "public" / "employee-handbook.md"
        r = client.post(
            "/v1/documents",
            params={"everyone": "true"},
            headers={**auth, "X-User-Id": "admin"},
            files={"file": (path.name, path.read_bytes(), "text/markdown")},
        )
        assert r.status_code == 201 and r.json()["chunks"] == 4
        doc_id = r.json()["document_id"]

        r = client.post(
            "/v1/ask",
            json={"question": "How many days per week can I work remotely?"},
            headers={**auth, "X-User-Id": "dave"},
        )
        body = r.json()
        assert r.status_code == 200 and not body["answer"]["refused"] and body["answer"]["citations"]
        assert body["evidence"][0]["source_name"] == "employee-handbook.md"

        assert (
            client.get(f"/v1/documents/{doc_id}/exposures", headers={**auth, "X-User-Id": "admin"}).json()[0][
                "user_id"
            ]
            == "dave"
        )
        assert (
            client.delete(f"/v1/documents/{doc_id}", headers={**auth, "X-User-Id": "admin"}).status_code
            == 204
        )
        assert (
            client.delete(f"/v1/documents/{doc_id}", headers={**auth, "X-User-Id": "admin"}).status_code
            == 404
        )
