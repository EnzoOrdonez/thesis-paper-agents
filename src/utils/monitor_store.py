from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.models.paper import Paper, PaperStatus

JOB_AND_LOCK_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS job_runs (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        trigger TEXT NOT NULL,
        phases_json TEXT NOT NULL,
        api_names_json TEXT NOT NULL,
        dry_run INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        error_message TEXT,
        search_raw_results INTEGER,
        search_final_papers INTEGER,
        search_high_count INTEGER,
        compile_imported_count INTEGER,
        compile_processed_reports INTEGER,
        compile_skipped_reports INTEGER,
        metadata_doi_processed INTEGER,
        metadata_scopus_processed INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_locks (
        lock_key TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        acquired_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_job_runs_started_at ON job_runs(started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_job_runs_status ON job_runs(status)",
]

SORT_FIELDS = {
    "title": "COALESCE(title, '') COLLATE NOCASE",
    "year": "COALESCE(year, 0)",
    "ranking_score": "ranking_score",
    "source_api": "COALESCE(source_api, '') COLLATE NOCASE",
    "status": "COALESCE(status, '') COLLATE NOCASE",
    "date_found": "COALESCE(date_found, '')",
    "citation_count": "COALESCE(citation_count, 0)",
    "relevance_score": "COALESCE(relevance_score, 0)",
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


def ensure_runtime_schema(path: str) -> None:
    conn = _connect(path)
    try:
        with conn:
            for statement in JOB_AND_LOCK_SCHEMA:
                conn.execute(statement)
    finally:
        conn.close()


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _int_to_bool(value: int | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _ranking_sql(config: dict[str, Any]) -> str:
    general = config.get("general", {})
    provisional_penalty = int(general.get("provisional_ranking_penalty", 15))
    missing_doi_penalty = int(general.get("missing_doi_ranking_penalty", 3))
    missing_venue_penalty = int(general.get("missing_venue_ranking_penalty", 2))
    return (
        "MAX(0, relevance_score"
        f" - CASE WHEN COALESCE(source_trusted, 0) = 0 THEN {provisional_penalty} ELSE 0 END"
        f" - CASE WHEN doi IS NULL OR doi = '' THEN {missing_doi_penalty} ELSE 0 END"
        f" - CASE WHEN venue IS NULL OR venue = '' OR venue = 'N/A' THEN {missing_venue_penalty} ELSE 0 END"
        ")"
    )


def _paper_from_row(row: sqlite3.Row) -> Paper:
    return Paper(**json.loads(row["raw_json"]))


def _paper_summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "title": str(row["title"]),
        "authors": json.loads(row["authors_json"] or "[]"),
        "year": row["year"],
        "publication_date": row["publication_date"],
        "venue": row["venue"],
        "publisher": row["publisher"],
        "doi": row["doi"],
        "url": row["url"],
        "abstract": row["abstract"],
        "citation_count": int(row["citation_count"] or 0),
        "relevance_score": int(row["relevance_score"] or 0),
        "ranking_score": int(row["ranking_score"] or 0),
        "relevance_level": row["relevance_level"],
        "covers_gap": row["covers_gap"],
        "scopus_indexed": _int_to_bool(row["scopus_indexed"]),
        "doi_verified": _int_to_bool(row["doi_verified"]),
        "source_api": row["source_api"],
        "source_trusted": _int_to_bool(row["source_trusted"]),
        "date_found": row["date_found"],
        "status": row["status"],
        "notes": row["notes"],
        "categories": json.loads(row["categories_json"] or "[]"),
        "keywords_matched": json.loads(row["keywords_matched_json"] or "[]"),
    }


def _metadata_value(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_metadata WHERE key = ?", (key,)).fetchone()
    if row:
        return str(row[0])
    return None


def _set_metadata_values(conn: sqlite3.Connection, values: dict[str, str]) -> None:
    for key, value in values.items():
        conn.execute(
            """
            INSERT INTO sync_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def _recompute_sync_metadata(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT raw_json FROM papers ORDER BY id").fetchall()
    raw_items = [json.loads(row[0]) for row in rows]
    payload = json.dumps(raw_items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _set_metadata_values(
        conn,
        {
            "last_synced_at": datetime.now().isoformat(),
            "paper_count": str(len(raw_items)),
            "sync_mode": str(_metadata_value(conn, "sync_mode") or "incremental_upsert"),
            "content_hash": __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest(),
            "last_upserted_count": "1",
            "last_deleted_count": "0",
        },
    )


def _write_single_paper(conn: sqlite3.Connection, paper: Paper, config: dict[str, Any]) -> None:
    raw_json = json.dumps(paper.model_dump(), ensure_ascii=False, sort_keys=True)
    row_hash = __import__("hashlib").sha256(raw_json.encode("utf-8")).hexdigest()
    conn.execute(
        """
        UPDATE papers
        SET title = ?, normalized_title = ?, doi = ?, year = ?, publication_date = ?, venue = ?,
            publisher = ?, url = ?, abstract = ?, citation_count = ?, relevance_score = ?,
            relevance_level = ?, covers_gap = ?, scopus_indexed = ?, doi_verified = ?, source_api = ?,
            source_trusted = ?, date_found = ?, status = ?, notes = ?, apa7_reference = ?, bibtex = ?,
            authors_json = ?, categories_json = ?, keywords_matched_json = ?, raw_json = ?, row_hash = ?
        WHERE id = ?
        """,
        (
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
            paper.id,
        ),
    )


def _parse_bool_filter(value: str | bool | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "si", "trusted"}:
        return 1
    if normalized in {"0", "false", "no", "provisional"}:
        return 0
    return None


def get_paper_by_id(path: str, paper_id: str, config: dict[str, Any]) -> dict[str, Any] | None:
    conn = _connect(path)
    try:
        ensure_runtime_schema(path)
        ranking_sql = _ranking_sql(config)
        row = conn.execute(
            f"SELECT papers.*, {ranking_sql} AS ranking_score FROM papers WHERE id = ?",
            (paper_id,),
        ).fetchone()
        if not row:
            return None
        return _paper_summary_from_row(row)
    finally:
        conn.close()


def list_papers_paginated(
    path: str,
    config: dict[str, Any],
    *,
    filters: dict[str, Any] | None = None,
    page: int = 1,
    page_size: int = 50,
    sort: str = "ranking_score",
    descending: bool = True,
) -> dict[str, Any]:
    filters = filters or {}
    page = max(page, 1)
    page_size = max(1, min(page_size, 200))
    normalized_sort = sort if sort in SORT_FIELDS else "ranking_score"
    ranking_sql = _ranking_sql(config)
    order_sql = SORT_FIELDS.get(normalized_sort, "ranking_score")
    direction = "DESC" if descending else "ASC"

    where_clauses: list[str] = []
    params: list[Any] = []

    query_text = str(filters.get("q") or "").strip().lower()
    if query_text:
        like_value = f"%{query_text}%"
        where_clauses.append(
            "(LOWER(title) LIKE ? OR LOWER(COALESCE(abstract, '')) LIKE ? OR LOWER(COALESCE(doi, '')) LIKE ? OR LOWER(COALESCE(authors_json, '')) LIKE ?)"
        )
        params.extend([like_value, like_value, like_value, like_value])

    if filters.get("only_pending"):
        where_clauses.append("status = ?")
        params.append(PaperStatus.NEW.value)
    elif filters.get("status"):
        where_clauses.append("status = ?")
        params.append(str(filters["status"]))

    if filters.get("relevance_level"):
        where_clauses.append("relevance_level = ?")
        params.append(str(filters["relevance_level"]))

    if filters.get("source_api"):
        where_clauses.append("source_api = ?")
        params.append(str(filters["source_api"]))

    source_trusted = _parse_bool_filter(filters.get("source_trusted"))
    if source_trusted is not None:
        where_clauses.append("source_trusted = ?")
        params.append(source_trusted)

    if filters.get("category"):
        where_clauses.append("LOWER(categories_json) LIKE ?")
        params.append(f"%{str(filters['category']).strip().lower()}%")

    if filters.get("year"):
        try:
            where_clauses.append("year = ?")
            params.append(int(filters["year"]))
        except (TypeError, ValueError):
            pass

    doi_verified = _parse_bool_filter(filters.get("doi_verified"))
    if doi_verified is not None:
        where_clauses.append("doi_verified = ?")
        params.append(doi_verified)
    elif str(filters.get("doi_verified") or "").lower() == "unknown":
        where_clauses.append("doi_verified IS NULL")

    scopus_indexed = _parse_bool_filter(filters.get("scopus_indexed"))
    if scopus_indexed is not None:
        where_clauses.append("scopus_indexed = ?")
        params.append(scopus_indexed)
    elif str(filters.get("scopus_indexed") or "").lower() == "unknown":
        where_clauses.append("scopus_indexed IS NULL")

    if filters.get("date_found"):
        where_clauses.append("date_found = ?")
        params.append(str(filters["date_found"]))

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    offset = (page - 1) * page_size

    conn = _connect(path)
    try:
        ensure_runtime_schema(path)
        total_count = int(conn.execute(f"SELECT COUNT(*) FROM papers {where_sql}", params).fetchone()[0])
        rows = conn.execute(
            f"""
            SELECT papers.*, {ranking_sql} AS ranking_score
            FROM papers
            {where_sql}
            ORDER BY {order_sql} {direction},
                     ranking_score DESC,
                     COALESCE(citation_count, 0) DESC,
                     COALESCE(year, 0) DESC,
                     COALESCE(title, '') COLLATE NOCASE ASC
            LIMIT ? OFFSET ?
            """,
            [*params, page_size, offset],
        ).fetchall()
    finally:
        conn.close()

    pages = max(1, math.ceil(total_count / page_size)) if total_count else 1
    return {
        "items": [_paper_summary_from_row(row) for row in rows],
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "pages": pages,
        "sort": normalized_sort,
        "descending": descending,
    }


def list_recent_high_papers(path: str, config: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    conn = _connect(path)
    try:
        ranking_sql = _ranking_sql(config)
        rows = conn.execute(
            f"""
            SELECT papers.*, {ranking_sql} AS ranking_score
            FROM papers
            WHERE relevance_level = 'ALTA'
            ORDER BY {ranking_sql} DESC, citation_count DESC, COALESCE(year, 0) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_paper_summary_from_row(row) for row in rows]
    finally:
        conn.close()


def list_distinct_filter_values(path: str) -> dict[str, list[Any]]:
    conn = _connect(path)
    try:
        source_rows = conn.execute(
            "SELECT DISTINCT source_api FROM papers WHERE source_api IS NOT NULL AND source_api != '' ORDER BY source_api ASC"
        ).fetchall()
        year_rows = conn.execute(
            "SELECT DISTINCT year FROM papers WHERE year IS NOT NULL ORDER BY year DESC"
        ).fetchall()
        date_rows = conn.execute(
            "SELECT DISTINCT date_found FROM papers WHERE date_found IS NOT NULL ORDER BY date_found DESC LIMIT 120"
        ).fetchall()
        category_rows = conn.execute(
            "SELECT categories_json FROM papers WHERE categories_json IS NOT NULL AND categories_json != ''"
        ).fetchall()
    finally:
        conn.close()

    categories: set[str] = set()
    for row in category_rows:
        try:
            for category in json.loads(row[0] or "[]"):
                if category:
                    categories.add(str(category))
        except json.JSONDecodeError:
            continue

    return {
        "source_apis": [str(row[0]) for row in source_rows if row[0]],
        "years": [int(row[0]) for row in year_rows if row[0] is not None],
        "dates_found": [str(row[0]) for row in date_rows if row[0]],
        "categories": sorted(categories),
    }


def get_dashboard_metrics(path: str) -> dict[str, Any]:
    conn = _connect(path)
    try:
        total_papers = int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        status_rows = conn.execute("SELECT status, COUNT(*) AS count FROM papers GROUP BY status").fetchall()
        relevance_rows = conn.execute(
            "SELECT relevance_level, COUNT(*) AS count FROM papers GROUP BY relevance_level"
        ).fetchall()
        source_rows = conn.execute(
            "SELECT source_api, COUNT(*) AS count FROM papers WHERE source_api IS NOT NULL GROUP BY source_api"
        ).fetchall()
        trust_rows = conn.execute(
            "SELECT COALESCE(source_trusted, 0) AS trusted, COUNT(*) AS count FROM papers GROUP BY COALESCE(source_trusted, 0)"
        ).fetchall()
        year_rows = conn.execute(
            "SELECT year, COUNT(*) AS count FROM papers WHERE year IS NOT NULL GROUP BY year ORDER BY year DESC"
        ).fetchall()
        doi_count = int(conn.execute("SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL AND doi != ''").fetchone()[0])
        doi_verified_count = int(conn.execute("SELECT COUNT(*) FROM papers WHERE doi_verified = 1").fetchone()[0])
        scopus_count = int(conn.execute("SELECT COUNT(*) FROM papers WHERE scopus_indexed = 1").fetchone()[0])
        category_rows = conn.execute(
            "SELECT categories_json FROM papers WHERE categories_json IS NOT NULL AND categories_json != ''"
        ).fetchall()
    finally:
        conn.close()

    category_counts: dict[str, int] = {}
    for row in category_rows:
        try:
            categories = json.loads(row[0] or "[]")
        except json.JSONDecodeError:
            continue
        for category in categories:
            category_counts[str(category)] = category_counts.get(str(category), 0) + 1

    return {
        "total_papers": total_papers,
        "status_counts": {str(row["status"]): int(row["count"]) for row in status_rows},
        "relevance_counts": {str(row["relevance_level"]): int(row["count"]) for row in relevance_rows},
        "source_counts": {str(row["source_api"]): int(row["count"]) for row in source_rows if row["source_api"]},
        "trust_counts": {
            ("trusted" if int(row["trusted"] or 0) else "provisional"): int(row["count"]) for row in trust_rows
        },
        "year_counts": {int(row["year"]): int(row["count"]) for row in year_rows if row["year"] is not None},
        "category_counts": dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True)),
        "doi_count": doi_count,
        "doi_verified_count": doi_verified_count,
        "scopus_count": scopus_count,
    }


def update_paper_status(path: str, paper_id: str, status: str, config: dict[str, Any]) -> dict[str, Any] | None:
    conn = _connect(path)
    try:
        row = conn.execute("SELECT raw_json FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            return None
        paper = Paper(**json.loads(row[0]))
        paper.status = PaperStatus(status)
        with conn:
            _write_single_paper(conn, paper, config)
            _recompute_sync_metadata(conn)
        return get_paper_by_id(path, paper_id, config)
    finally:
        conn.close()


def update_paper_notes(path: str, paper_id: str, notes: str | None, config: dict[str, Any]) -> dict[str, Any] | None:
    conn = _connect(path)
    try:
        row = conn.execute("SELECT raw_json FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            return None
        paper = Paper(**json.loads(row[0]))
        paper.notes = notes or None
        with conn:
            _write_single_paper(conn, paper, config)
            _recompute_sync_metadata(conn)
        return get_paper_by_id(path, paper_id, config)
    finally:
        conn.close()


def update_paper_categories(
    path: str, paper_id: str, categories: list[str], config: dict[str, Any]
) -> dict[str, Any] | None:
    conn = _connect(path)
    try:
        row = conn.execute("SELECT raw_json FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if not row:
            return None
        paper = Paper(**json.loads(row[0]))
        paper.categories = categories
        with conn:
            _write_single_paper(conn, paper, config)
            _recompute_sync_metadata(conn)
        return get_paper_by_id(path, paper_id, config)
    finally:
        conn.close()


def create_job_run(
    path: str,
    *,
    trigger: str,
    phases: list[str],
    api_names: list[str] | None,
    dry_run: bool,
    job_id: str | None = None,
) -> str:
    ensure_runtime_schema(path)
    run_id = job_id or str(uuid.uuid4())
    conn = _connect(path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO job_runs (
                    id, started_at, trigger, phases_json, api_names_json, dry_run, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now().isoformat(),
                    trigger,
                    json.dumps(phases, ensure_ascii=False),
                    json.dumps(api_names or [], ensure_ascii=False),
                    int(dry_run),
                    "running",
                ),
            )
        return run_id
    finally:
        conn.close()


def finish_job_run(
    path: str,
    job_id: str,
    *,
    status: str,
    error_message: str | None = None,
    search_summary: dict[str, Any] | None = None,
    compile_summary: dict[str, Any] | None = None,
    metadata_summary: dict[str, Any] | None = None,
) -> None:
    ensure_runtime_schema(path)
    search_summary = search_summary or {}
    compile_summary = compile_summary or {}
    metadata_summary = metadata_summary or {}

    conn = _connect(path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE job_runs
                SET finished_at = ?,
                    status = ?,
                    error_message = ?,
                    search_raw_results = ?,
                    search_final_papers = ?,
                    search_high_count = ?,
                    compile_imported_count = ?,
                    compile_processed_reports = ?,
                    compile_skipped_reports = ?,
                    metadata_doi_processed = ?,
                    metadata_scopus_processed = ?
                WHERE id = ?
                """,
                (
                    datetime.now().isoformat(),
                    status,
                    error_message,
                    search_summary.get("raw_results"),
                    search_summary.get("final_papers"),
                    search_summary.get("high_relevance"),
                    compile_summary.get("imported_count"),
                    compile_summary.get("processed_reports"),
                    compile_summary.get("skipped_reports"),
                    metadata_summary.get("doi_processed"),
                    metadata_summary.get("scopus_processed"),
                    job_id,
                ),
            )
    finally:
        conn.close()


def list_job_runs(path: str, limit: int = 25) -> list[dict[str, Any]]:
    ensure_runtime_schema(path)
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM job_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "trigger": row["trigger"],
                "phases": json.loads(row["phases_json"] or "[]"),
                "api_names": json.loads(row["api_names_json"] or "[]"),
                "dry_run": bool(row["dry_run"]),
                "status": row["status"],
                "error_message": row["error_message"],
                "search_raw_results": row["search_raw_results"],
                "search_final_papers": row["search_final_papers"],
                "search_high_count": row["search_high_count"],
                "compile_imported_count": row["compile_imported_count"],
                "compile_processed_reports": row["compile_processed_reports"],
                "compile_skipped_reports": row["compile_skipped_reports"],
                "metadata_doi_processed": row["metadata_doi_processed"],
                "metadata_scopus_processed": row["metadata_scopus_processed"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def acquire_runtime_lock(
    path: str,
    *,
    lock_key: str,
    owner_id: str,
    ttl_seconds: int = 7200,
) -> tuple[bool, dict[str, Any] | None]:
    ensure_runtime_schema(path)
    now = datetime.now()
    expires_at = now + timedelta(seconds=max(60, ttl_seconds))
    conn = _connect(path)
    try:
        with conn:
            row = conn.execute("SELECT * FROM runtime_locks WHERE lock_key = ?", (lock_key,)).fetchone()
            if row:
                try:
                    current_expires = datetime.fromisoformat(str(row["expires_at"]))
                except ValueError:
                    current_expires = now - timedelta(seconds=1)
                if current_expires > now:
                    return False, {
                        "lock_key": str(row["lock_key"]),
                        "owner_id": str(row["owner_id"]),
                        "acquired_at": str(row["acquired_at"]),
                        "expires_at": str(row["expires_at"]),
                    }
                conn.execute("DELETE FROM runtime_locks WHERE lock_key = ?", (lock_key,))

            conn.execute(
                "INSERT INTO runtime_locks (lock_key, owner_id, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
                (lock_key, owner_id, now.isoformat(), expires_at.isoformat()),
            )
            return True, None
    finally:
        conn.close()


def release_runtime_lock(path: str, *, lock_key: str, owner_id: str | None = None) -> None:
    ensure_runtime_schema(path)
    conn = _connect(path)
    try:
        with conn:
            if owner_id:
                conn.execute(
                    "DELETE FROM runtime_locks WHERE lock_key = ? AND owner_id = ?",
                    (lock_key, owner_id),
                )
            else:
                conn.execute("DELETE FROM runtime_locks WHERE lock_key = ?", (lock_key,))
    finally:
        conn.close()


def get_runtime_lock(path: str, lock_key: str) -> dict[str, Any] | None:
    ensure_runtime_schema(path)
    conn = _connect(path)
    try:
        row = conn.execute("SELECT * FROM runtime_locks WHERE lock_key = ?", (lock_key,)).fetchone()
        if not row:
            return None
        try:
            expires_at = datetime.fromisoformat(str(row["expires_at"]))
        except ValueError:
            expires_at = datetime.now() - timedelta(seconds=1)
        if expires_at <= datetime.now():
            with conn:
                conn.execute("DELETE FROM runtime_locks WHERE lock_key = ?", (lock_key,))
            return None
        return {
            "lock_key": str(row["lock_key"]),
            "owner_id": str(row["owner_id"]),
            "acquired_at": str(row["acquired_at"]),
            "expires_at": str(row["expires_at"]),
        }
    finally:
        conn.close()


def refresh_json_export_from_sqlite(sqlite_path: str, json_path: str) -> bool:
    conn = _connect(sqlite_path)
    try:
        rows = conn.execute("SELECT raw_json FROM papers ORDER BY rowid").fetchall()
        payload = json.dumps([json.loads(row[0]) for row in rows], ensure_ascii=False, indent=2)
    finally:
        conn.close()

    output_path = Path(json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_text(encoding="utf-8") == payload:
        return False

    output_path.write_text(payload, encoding="utf-8")
    return True
