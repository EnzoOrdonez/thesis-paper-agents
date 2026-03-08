from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models.paper import Paper

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS papers (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        normalized_title TEXT NOT NULL,
        doi TEXT,
        year INTEGER,
        publication_date TEXT,
        venue TEXT,
        publisher TEXT,
        url TEXT,
        abstract TEXT,
        citation_count INTEGER NOT NULL DEFAULT 0,
        relevance_score INTEGER NOT NULL DEFAULT 0,
        relevance_level TEXT NOT NULL,
        covers_gap TEXT,
        scopus_indexed INTEGER,
        doi_verified INTEGER,
        source_api TEXT,
        source_trusted INTEGER,
        date_found TEXT,
        status TEXT NOT NULL,
        notes TEXT,
        apa7_reference TEXT,
        bibtex TEXT,
        authors_json TEXT NOT NULL,
        categories_json TEXT NOT NULL,
        keywords_matched_json TEXT NOT NULL,
        raw_json TEXT NOT NULL,
        row_hash TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)",
    "CREATE INDEX IF NOT EXISTS idx_papers_normalized_title ON papers(normalized_title)",
    "CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)",
    "CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status)",
    "CREATE INDEX IF NOT EXISTS idx_papers_source_api ON papers(source_api)",
    "CREATE INDEX IF NOT EXISTS idx_papers_source_trusted ON papers(source_trusted)",
    "CREATE INDEX IF NOT EXISTS idx_papers_relevance_level ON papers(relevance_level)",
    "CREATE INDEX IF NOT EXISTS idx_papers_date_found ON papers(date_found)",
]

REQUIRED_PAPERS_COLUMNS = {
    "row_hash": "TEXT NOT NULL DEFAULT ''",
}


def _connect(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _paper_payload(paper: Paper) -> tuple[str, str]:
    raw_payload = paper.model_dump()
    raw_json = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True)
    row_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    return raw_json, row_hash


def _paper_record(paper: Paper) -> tuple[Any, ...]:
    raw_json, row_hash = _paper_payload(paper)
    return (
        paper.id,
        paper.title,
        paper.normalized_title(),
        paper.doi,
        paper.year,
        paper.publication_date,
        paper.venue,
        paper.publisher,
        paper.url,
        paper.abstract,
        paper.citation_count,
        paper.relevance_score,
        paper.relevance_level.value,
        paper.covers_gap,
        _bool_to_int(paper.scopus_indexed),
        _bool_to_int(paper.doi_verified),
        paper.source_api,
        _bool_to_int(paper.source_trusted),
        paper.date_found,
        paper.status.value,
        paper.notes,
        paper.apa7_reference,
        paper.bibtex,
        json.dumps(paper.authors, ensure_ascii=False),
        json.dumps(paper.categories, ensure_ascii=False),
        json.dumps(paper.keywords_matched, ensure_ascii=False),
        raw_json,
        row_hash,
    )


def _papers_content_hash(papers: list[Paper]) -> str:
    payload = json.dumps(
        [paper.model_dump() for paper in papers],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata_value(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM sync_metadata WHERE key = ?",
        (key,),
    ).fetchone()
    if row:
        return str(row[0])
    return None


def _ensure_papers_columns(conn: sqlite3.Connection) -> None:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "papers" not in tables:
        return

    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(papers)")}
    for column_name, ddl in REQUIRED_PAPERS_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {column_name} {ddl}")


def ensure_schema(path: str) -> None:
    conn = _connect(path)
    try:
        with conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            _ensure_papers_columns(conn)
    finally:
        conn.close()


def sync_papers_to_sqlite(papers: list[Paper], path: str, *, force: bool = False) -> dict[str, Any]:
    conn = _connect(path)
    desired_hash = _papers_content_hash(papers)

    try:
        with conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            _ensure_papers_columns(conn)

        existing_rows = {str(row[0]): str(row[1] or "") for row in conn.execute("SELECT id, row_hash FROM papers")}

        incoming_ids: set[str] = set()
        records_to_upsert: list[tuple[Any, ...]] = []
        for paper in papers:
            record = _paper_record(paper)
            paper_id = str(record[0])
            row_hash = str(record[-1])
            incoming_ids.add(paper_id)
            if force or existing_rows.get(paper_id) != row_hash:
                records_to_upsert.append(record)

        ids_to_delete = [paper_id for paper_id in existing_rows.keys() if paper_id not in incoming_ids]

        metadata_needs_refresh = force or any(
            [
                _metadata_value(conn, "content_hash") != desired_hash,
                _metadata_value(conn, "paper_count") != str(len(papers)),
                _metadata_value(conn, "sync_mode") != "incremental_upsert",
            ]
        )

        changed = bool(records_to_upsert or ids_to_delete or metadata_needs_refresh)
        if not changed:
            index_count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND tbl_name = 'papers' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            return {
                "paper_count": len(papers),
                "index_count": int(index_count),
                "path": str(Path(path)),
                "changed": False,
                "upserted_count": 0,
                "deleted_count": 0,
            }

        with conn:
            if ids_to_delete:
                conn.executemany(
                    "DELETE FROM papers WHERE id = ?",
                    [(paper_id,) for paper_id in ids_to_delete],
                )

            if records_to_upsert:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO papers (
                        id, title, normalized_title, doi, year, publication_date, venue,
                        publisher, url, abstract, citation_count, relevance_score,
                        relevance_level, covers_gap, scopus_indexed, doi_verified,
                        source_api, source_trusted, date_found, status, notes,
                        apa7_reference, bibtex, authors_json, categories_json,
                        keywords_matched_json, raw_json, row_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records_to_upsert,
                )

            metadata = {
                "last_synced_at": datetime.now().isoformat(),
                "paper_count": str(len(papers)),
                "sync_mode": "incremental_upsert",
                "content_hash": desired_hash,
                "last_upserted_count": str(len(records_to_upsert)),
                "last_deleted_count": str(len(ids_to_delete)),
            }
            for key, value in metadata.items():
                conn.execute(
                    """
                    INSERT INTO sync_metadata (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )

        index_count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND tbl_name = 'papers' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "paper_count": len(papers),
        "index_count": int(index_count),
        "path": str(Path(path)),
        "changed": True,
        "upserted_count": len(records_to_upsert),
        "deleted_count": len(ids_to_delete),
    }


def load_papers_from_sqlite(path: str) -> list[Paper]:
    conn = _connect(path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "papers" not in tables:
            return []

        rows = conn.execute("SELECT raw_json FROM papers ORDER BY rowid").fetchall()
        return [Paper(**json.loads(row[0])) for row in rows]
    finally:
        conn.close()


def get_sqlite_status(path: str) -> dict[str, Any]:
    db_path = Path(path)
    if not db_path.exists():
        return {
            "exists": False,
            "path": str(db_path),
            "paper_count": 0,
            "index_count": 0,
            "last_synced_at": None,
            "size_bytes": 0,
            "sync_mode": None,
            "content_hash": None,
            "last_upserted_count": 0,
            "last_deleted_count": 0,
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        paper_count = 0
        index_count = 0
        last_synced_at = None
        sync_mode = None
        content_hash = None
        last_upserted_count = 0
        last_deleted_count = 0

        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "papers" in tables:
            paper_count = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
            index_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND tbl_name = 'papers' AND name NOT LIKE 'sqlite_%'"
                ).fetchone()[0]
            )
        if "sync_metadata" in tables:
            last_synced_at = _metadata_value(conn, "last_synced_at")
            sync_mode = _metadata_value(conn, "sync_mode")
            content_hash = _metadata_value(conn, "content_hash")
            last_upserted_count = int(_metadata_value(conn, "last_upserted_count") or 0)
            last_deleted_count = int(_metadata_value(conn, "last_deleted_count") or 0)
    finally:
        conn.close()

    size_bytes = db_path.stat().st_size
    wal_path = Path(str(db_path) + "-wal")
    shm_path = Path(str(db_path) + "-shm")
    if wal_path.exists():
        size_bytes += wal_path.stat().st_size
    if shm_path.exists():
        size_bytes += shm_path.stat().st_size

    return {
        "exists": True,
        "path": str(db_path),
        "paper_count": paper_count,
        "index_count": index_count,
        "last_synced_at": last_synced_at,
        "size_bytes": size_bytes,
        "sync_mode": sync_mode,
        "content_hash": content_hash,
        "last_upserted_count": last_upserted_count,
        "last_deleted_count": last_deleted_count,
    }
