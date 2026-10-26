"""HTTP API: health, tool listing, auth on POST, file access stays inside the work dir."""

import pytest
from fastapi.testclient import TestClient

from glitchhunter.api.app import create_app
from glitchhunter.config import load_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_API_TOKEN", "secret")
    settings = load_settings(work_dir=tmp_path, approval_mode="web")
    (tmp_path / "hunts").mkdir()
    (tmp_path / "note.txt").write_text("hi")
    return TestClient(create_app(settings))


def test_health_reports_opencv5(client):
    body = client.get("/api/health").json()
    assert body["opencv"]["major"] >= 5 and body["auth"] is True


def test_tool_listing(client):
    body = client.get("/api/tools").json()
    assert "bisect" in body["tools"] and any(t["name"] == "zoom_in" for t in body["agent_tools"])


def test_post_requires_token(client):
    assert client.post("/api/tools/info", json={}).status_code == 401
    ok = client.post("/api/tools/info", json={}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200 and "detectors" in ok.json()["result"]


def test_files_cannot_escape_the_work_dir(client):
    assert client.get("/files/note.txt").status_code == 200
    assert client.get("/files/../../etc/passwd").status_code == 404


def test_unknown_game_revision_is_404(client):
    assert client.get("/game/no-such-rev/").status_code == 404


def test_index_cache_busts_assets(client):
    html = client.get("/").text
    assert "/static/app.js?v=" in html and "/static/style.css?v=" in html


def test_unknown_approval_is_404(client):
    r = client.post("/api/approvals/nope", json={"choice": "approve"}, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 404
