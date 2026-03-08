"""Tests for SQLite storage operations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.models.paper import Paper
from src.utils.sqlite_store import (
    ensure_schema,
    get_sqlite_status,
    load_papers_from_sqlite,
    rebuild_fts_index,
    search_papers_fts,
    sync_papers_to_sqlite,
)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_papers.sqlite")


class TestEnsureSchema:
    def test_creates_tables(self, db_path: str):
        ensure_schema(db_path)
        conn = sqlite3.connect(db_path)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "papers" in tables
        assert "sync_metadata" in tables

    def test_creates_indices(self, db_path: str):
        ensure_schema(db_path)
        conn = sqlite3.connect(db_path)
        indices = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        conn.close()
        assert "idx_papers_doi" in indices
        assert "idx_papers_status" in indices


class TestSyncPapers:
    def test_insert_papers(self, db_path: str, paper_list: list[Paper]):
        result = sync_papers_to_sqlite(paper_list, db_path)
        assert result["paper_count"] == 3
        assert result["changed"] is True
        assert result["upserted_count"] == 3

    def test_no_change_on_resync(self, db_path: str, paper_list: list[Paper]):
        sync_papers_to_sqlite(paper_list, db_path)
        result = sync_papers_to_sqlite(paper_list, db_path)
        assert result["changed"] is False
        assert result["upserted_count"] == 0

    def test_detects_modification(self, db_path: str, sample_paper: Paper):
        sync_papers_to_sqlite([sample_paper], db_path)
        sample_paper.notes = "Updated note"
        result = sync_papers_to_sqlite([sample_paper], db_path)
        assert result["changed"] is True
        assert result["upserted_count"] == 1

    def test_deletes_removed_papers(self, db_path: str, paper_list: list[Paper]):
        sync_papers_to_sqlite(paper_list, db_path)
        result = sync_papers_to_sqlite(paper_list[:1], db_path)
        assert result["changed"] is True
        assert result["deleted_count"] == 2


class TestLoadPapers:
    def test_round_trip(self, db_path: str, paper_list: list[Paper]):
        sync_papers_to_sqlite(paper_list, db_path)
        loaded = load_papers_from_sqlite(db_path)
        assert len(loaded) == 3
        titles = {p.title for p in loaded}
        assert paper_list[0].title in titles

    def test_empty_db(self, db_path: str):
        ensure_schema(db_path)
        loaded = load_papers_from_sqlite(db_path)
        assert loaded == []

    def test_nonexistent_db(self, tmp_path: Path):
        loaded = load_papers_from_sqlite(str(tmp_path / "nope.sqlite"))
        assert loaded == []


class TestGetStatus:
    def test_existing_db(self, db_path: str, paper_list: list[Paper]):
        sync_papers_to_sqlite(paper_list, db_path)
        status = get_sqlite_status(db_path)
        assert status["exists"] is True
        assert status["paper_count"] == 3
        assert status["index_count"] > 0
        assert status["size_bytes"] > 0

    def test_nonexistent_db(self, tmp_path: Path):
        status = get_sqlite_status(str(tmp_path / "nope.sqlite"))
        assert status["exists"] is False
        assert status["paper_count"] == 0


class TestFTS5Search:
    def test_search_by_title(self, db_path: str, paper_list: list[Paper]):
        sync_papers_to_sqlite(paper_list, db_path)
        rebuild_fts_index(db_path)
        ids = search_papers_fts(db_path, "Hybrid Retrieval")
        assert len(ids) >= 1

    def test_search_by_abstract_keyword(self, db_path: str, paper_list: list[Paper]):
        sync_papers_to_sqlite(paper_list, db_path)
        rebuild_fts_index(db_path)
        ids = search_papers_fts(db_path, "cloud documentation")
        assert len(ids) >= 1

    def test_no_results(self, db_path: str, paper_list: list[Paper]):
        sync_papers_to_sqlite(paper_list, db_path)
        rebuild_fts_index(db_path)
        ids = search_papers_fts(db_path, "quantum entanglement")
        assert ids == []

    def test_nonexistent_db(self, tmp_path: Path):
        ids = search_papers_fts(str(tmp_path / "nope.sqlite"), "test")
        assert ids == []

    def test_rebuild_returns_count(self, db_path: str, paper_list: list[Paper]):
        sync_papers_to_sqlite(paper_list, db_path)
        count = rebuild_fts_index(db_path)
        assert count == 3
