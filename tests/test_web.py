"""Integration tests for the FastAPI web application routes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.utils.monitor_store import ensure_runtime_schema
from src.utils.sqlite_store import ensure_schema
from src.web.app import create_app


def _make_config(sqlite_path: str, json_path: str) -> dict[str, Any]:
    """Build a minimal config dict pointing to temporary storage paths."""
    return {
        "output": {
            "sqlite_database_path": sqlite_path,
            "database_path": json_path,
        },
        "web": {
            "page_size": 50,
            "proxy": {
                "mode": "dual",
                "prefer_proxy_button": True,
                "rules": [],
            },
        },
        "general": {},
        "apis": {},
    }


def _insert_test_paper(sqlite_path: str) -> str:
    """Insert a test paper into the database and return its id."""
    paper_id = "test-paper-1"
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            """
            INSERT INTO papers (
                id, title, normalized_title, year, doi, abstract,
                relevance_score, relevance_level, source_api, source_trusted,
                status, date_found, authors_json, categories_json,
                keywords_matched_json, raw_json, citation_count, row_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper_id,
                "Test Paper Title",
                "test paper title",
                2024,
                "10.1234/test",
                "Abstract about hybrid RAG systems for testing.",
                75,
                "ALTA",
                "semantic_scholar",
                1,
                "new",
                "2024-01-15",
                json.dumps(["Author A", "Author B"]),
                json.dumps(["Test Category"]),
                json.dumps(["hybrid rag", "retrieval"]),
                json.dumps(
                    {
                        "title": "Test Paper Title",
                        "authors": ["Author A", "Author B"],
                        "year": 2024,
                        "doi": "10.1234/test",
                        "abstract": "Abstract about hybrid RAG systems for testing.",
                        "relevance_score": 75,
                        "relevance_level": "ALTA",
                        "source_api": "semantic_scholar",
                        "source_trusted": True,
                        "status": "new",
                        "date_found": "2024-01-15",
                        "categories": ["Test Category"],
                        "keywords_matched": ["hybrid rag", "retrieval"],
                        "citation_count": 10,
                    }
                ),
                10,
                "abc123",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return paper_id


@pytest.fixture
def web_client(tmp_path: Path):
    """Create a TestClient with a temporary SQLite database and mocked dependencies."""
    from starlette.testclient import TestClient

    sqlite_path = str(tmp_path / "test.sqlite")
    json_path = str(tmp_path / "test.json")
    config = _make_config(sqlite_path, json_path)

    # Initialize database schemas and insert test data
    ensure_schema(sqlite_path)
    ensure_runtime_schema(sqlite_path)
    paper_id = _insert_test_paper(sqlite_path)

    # Mock the API runtime tracker
    mock_tracker = MagicMock()
    mock_tracker.get_provider.return_value = {
        "last_status": "ok",
        "disabled_until": None,
        "last_run_finished_at": "2024-01-15T10:00:00",
        "last_queries_submitted": 5,
        "last_results_returned": 3,
        "last_error": "",
    }

    patches = [
        patch("src.web.app._load_app_config", return_value=config),
        patch("src.web.app.THESIS_CATEGORIES", ["Test Category"]),
        patch("src.web.app.get_api_runtime_tracker", return_value=mock_tracker),
        patch("src.web.app.get_enabled_search_apis", return_value=["semantic_scholar"]),
        patch("src.web.app.load_database", return_value=[]),
        patch("src.web.app.analyze_gap_coverage", return_value=[]),
        patch("src.web.app.load_pending_gaps", return_value=[]),
    ]

    for p in patches:
        p.start()

    try:
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        yield client, paper_id
    finally:
        for p in patches:
            p.stop()


class TestHealthEndpoint:
    def test_health_returns_ok(self, web_client):
        client, _ = web_client
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["locked"] is False


class TestDashboard:
    def test_dashboard_returns_200(self, web_client):
        client, _ = web_client
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Dashboard" in resp.text


class TestPapersList:
    def test_papers_list_returns_200(self, web_client):
        client, _ = web_client
        resp = client.get("/papers")
        assert resp.status_code == 200
        assert "Papers" in resp.text

    def test_papers_htmx_partial(self, web_client):
        client, _ = web_client
        resp = client.get("/papers", headers={"HX-Request": "true"})
        assert resp.status_code == 200
        # HTMX partial should NOT contain a full HTML document
        assert "<html" not in resp.text.lower()


class TestPaperDetail:
    def test_paper_detail_found(self, web_client):
        client, paper_id = web_client
        resp = client.get(f"/papers/{paper_id}")
        assert resp.status_code == 200
        assert "Test Paper Title" in resp.text

    def test_paper_detail_not_found(self, web_client):
        client, _ = web_client
        resp = client.get("/papers/nonexistent-id")
        assert resp.status_code == 404


class TestPaperActions:
    def test_update_status(self, web_client):
        client, paper_id = web_client
        resp = client.post(
            f"/papers/{paper_id}/status",
            data={"status": "reviewed"},
            follow_redirects=False,
        )
        assert resp.status_code == 303

    def test_update_notes(self, web_client):
        client, paper_id = web_client
        resp = client.post(
            f"/papers/{paper_id}/notes",
            data={"notes": "This is a test note."},
            follow_redirects=False,
        )
        assert resp.status_code == 303

    def test_batch_accept(self, web_client):
        client, paper_id = web_client
        resp = client.post(
            "/papers/batch",
            data={"paper_ids": [paper_id], "action": "accepted"},
            follow_redirects=False,
        )
        assert resp.status_code == 303

    def test_batch_invalid_action(self, web_client):
        client, paper_id = web_client
        resp = client.post(
            "/papers/batch",
            data={"paper_ids": [paper_id], "action": "invalid"},
            follow_redirects=False,
        )
        # Invalid action still redirects (no-op)
        assert resp.status_code == 303


class TestJobsPage:
    def test_jobs_returns_200(self, web_client):
        client, _ = web_client
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert "Jobs" in resp.text
