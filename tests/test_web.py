"""Web API contract: validation, methods, auth, and path confinement."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(no_embeddings, monkeypatch):
    from amnis.server import web

    monkeypatch.setattr(web, "config", no_embeddings, raising=False)
    # Skip the embedding warm-up; the vector layer is stubbed out. RagError
    # (not a bare exception) so the API translates it into a 4xx/503 rather
    # than letting it surface as a 500.
    from amnis.rag.engine import RagError

    def _no_rag():
        raise RagError("vector layer disabled in tests")

    monkeypatch.setattr(web, "_rag", _no_rag)
    with TestClient(web.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_status_reports_degraded_instead_of_500(client):
    body = client.get("/api/status").json()
    assert body["status"] == "degraded"
    assert "rag" in body["errors"]
    assert body["memory"]["total_memories"] == 0


def test_create_validates_the_payload(client):
    assert client.post("/api/memories", json={}).status_code == 422
    assert client.post("/api/memories", json={"fact": "x", "importance": 99}).status_code == 422
    assert client.post("/api/memories", json={"fact": "x", "category": "bogus"}).status_code == 422


def test_memory_lifecycle(client):
    created = client.post(
        "/api/memories",
        json={
            "fact": "A memory created through the web API",
            "category": "fact",
            "importance": 7,
        },
    )
    assert created.status_code == 201
    memory_id = created.json()["id"]

    patched = client.patch(f"/api/memories/{memory_id}", json={"importance": 9})
    assert patched.status_code == 200
    assert patched.json()["importance"] == 9
    assert patched.json()["id"] == memory_id

    assert client.delete(f"/api/memories/{memory_id}").status_code == 200
    assert client.get(f"/api/memories/{memory_id}").status_code == 404


def test_patch_with_no_fields_is_a_400(client):
    created = client.post("/api/memories", json={"fact": "something to patch"}).json()
    assert client.patch(f"/api/memories/{created['id']}", json={}).status_code == 400


def test_consolidate_is_not_a_get(client):
    # A GET that mutates could be triggered by any prefetch or crawler.
    assert client.get("/api/consolidate").status_code == 405


def test_index_file_is_confined_to_the_data_directories(client, no_embeddings):
    blocked = client.post("/api/index-file", json={"path": "/etc/passwd"})
    assert blocked.status_code == 403
    assert "outside" in blocked.json()["detail"]

    note = no_embeddings.notes_dir / "ok.md"
    note.write_text("# fine\n")
    # Allowed through the guard; 400 only because the RAG layer is stubbed off.
    allowed = client.post("/api/index-file", json={"path": str(note)})
    assert allowed.status_code == 400
    assert "disabled in tests" in allowed.json()["detail"]


def test_traversal_out_of_the_notes_dir_is_blocked(client, no_embeddings):
    escape = str(no_embeddings.notes_dir / ".." / ".." / "etc" / "passwd")
    assert client.post("/api/index-file", json={"path": escape}).status_code == 403


def test_token_guard(client, no_embeddings, monkeypatch):
    from amnis.server import web

    monkeypatch.setattr(no_embeddings, "api_token", "s3cret")
    monkeypatch.setattr(web.config, "api_token", "s3cret", raising=False)

    assert client.post("/api/memories", json={"fact": "no token supplied"}).status_code == 401
    ok = client.post(
        "/api/memories", json={"fact": "with the right token"}, headers={"X-Amnis-Token": "s3cret"}
    )
    assert ok.status_code == 201
    # Reads stay open so the dashboard still renders.
    assert client.get("/api/memories").status_code == 200


def test_ui_is_self_contained():
    from pathlib import Path

    import amnis.server.web as web

    html = (Path(web.__file__).parent / "ui.html").read_text(encoding="utf-8")
    for host in ("unpkg.com", "cdn.jsdelivr", "fonts.googleapis.com", "fonts.gstatic.com"):
        assert host not in html, f"{host} must not be referenced"
    assert ".innerHTML =" not in html
    assert ".innerHTML=" not in html
