"""Agent 2: Paper Compiler - Maintains database, validates, generates reports."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from src.apis.crossref_api import CrossRefAPI
from src.apis.openalex_api import OpenAlexAPI
from src.models.paper import Paper, PaperStatus, RelevanceLevel
from src.utils.duplicate_detector import add_to_dedup_index, build_dedup_index, find_duplicates_in_list, has_duplicate_in_index
from src.utils.gap_analyzer import generate_gap_report
from src.utils.logger import setup_logger
from src.utils.reference_formatter import format_apa7, format_bibtex
from src.utils.relevance_scorer import check_gap_coverage, is_from_trusted_source, ranking_score, suggest_categories
from src.utils.api_runtime import APIRuntimeTracker
from src.utils.sqlite_store import get_sqlite_status, load_papers_from_sqlite, sync_papers_to_sqlite

logger = setup_logger("paper_compiler")
console = Console()

CROSSREF_RUNTIME_KEY = "crossref_metadata"
OPENALEX_SCOPUS_RUNTIME_KEY = "openalex_scopus"


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)



def get_api_runtime_tracker(config: dict | None = None) -> APIRuntimeTracker:
    """Load the persistent runtime state shared by ingestion and metadata jobs."""
    active_config = config or load_config()
    state_path = active_config.get("output", {}).get("api_runtime_state_path", "data/api_runtime_state.json")
    return APIRuntimeTracker(state_path)


def _format_runtime_value(value: str | None) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def _format_disabled_until(value: str | None) -> str:
    if not value:
        return "-"
    try:
        disabled_until = datetime.fromisoformat(value)
    except ValueError:
        return value
    if disabled_until <= datetime.now(disabled_until.tzinfo):
        return "-"
    return disabled_until.strftime("%Y-%m-%d %H:%M")


def _doi_validation_batch_size(config: dict) -> int:
    return max(1, int(config.get("general", {}).get("doi_validation_batch_size", 25)))


def _scopus_check_batch_size(config: dict) -> int:
    return max(1, int(config.get("general", {}).get("scopus_check_batch_size", 25)))


def _build_crossref_client(config: dict, tracker: APIRuntimeTracker | None = None) -> CrossRefAPI:
    api_config = config.get("apis", {}).get("crossref", {})
    client = CrossRefAPI(
        base_url=api_config.get("base_url", "https://api.crossref.org"),
        email=api_config.get("email", ""),
        rate_limit_per_second=api_config.get("rate_limit_per_second", 50),
        max_retries=api_config.get("max_retries", 3),
        cooldown_seconds=api_config.get("cooldown_seconds", 900),
        shutdown_after_consecutive_failures=api_config.get("shutdown_after_consecutive_failures", 2),
    )
    if tracker:
        tracker.apply_to_client(CROSSREF_RUNTIME_KEY, client)
    return client


def _build_openalex_scopus_client(config: dict, tracker: APIRuntimeTracker | None = None) -> OpenAlexAPI:
    api_config = config.get("apis", {}).get("openalex", {})
    client = OpenAlexAPI(
        base_url=api_config.get("base_url", "https://api.openalex.org"),
        email=api_config.get("email", ""),
        rate_limit_per_second=api_config.get("rate_limit_per_second", 10),
        max_retries=api_config.get("max_retries", 3),
        cooldown_seconds=api_config.get("cooldown_seconds", 900),
        shutdown_after_consecutive_failures=api_config.get("shutdown_after_consecutive_failures", 2),
    )
    if tracker:
        tracker.apply_to_client(OPENALEX_SCOPUS_RUNTIME_KEY, client)
    return client


def show_metadata_status(config: dict) -> None:
    """Print the persisted runtime status for metadata validation providers."""
    tracker = get_api_runtime_tracker(config)
    entries = [
        ("CrossRef DOI validation", CROSSREF_RUNTIME_KEY),
        ("OpenAlex Scopus check", OPENALEX_SCOPUS_RUNTIME_KEY),
    ]

    table = Table(title="Metadata Provider Runtime Status")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Disabled Until", style="yellow")
    table.add_column("Last Run", style="white")
    table.add_column("Batch", style="green")
    table.add_column("Processed", style="green")
    table.add_column("Last Error", style="red")

    for label, runtime_key in entries:
        provider = tracker.get_provider(runtime_key)
        table.add_row(
            label,
            str(provider.get("last_status", "never") or "never"),
            _format_disabled_until(provider.get("disabled_until")),
            _format_runtime_value(provider.get("last_run_finished_at")),
            str(provider.get("last_queries_submitted", 0) or 0),
            str(provider.get("last_results_returned", 0) or 0),
            str(provider.get("last_error", "") or "-")[:60],
        )

    console.print(table)

def load_database(path: str, config: dict[str, Any] | None = None) -> list[Paper]:
    """Load papers from the configured persistence layer, preferring SQLite when available."""
    active_config = config or load_config()
    sqlite_path = _resolve_sqlite_path(path, config=active_config)
    if sqlite_path and Path(sqlite_path).exists():
        try:
            return load_papers_from_sqlite(sqlite_path)
        except Exception as exc:
            logger.warning(f"SQLite load failed for {sqlite_path}, falling back to JSON export: {exc}")

    db_path = Path(path)
    if not db_path.exists():
        return []
    with open(db_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Paper(**item) for item in data]


def _write_text_file_if_changed(path: Path, content: str) -> bool:
    """Write a text file atomically only when content actually changes."""
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return False
    except OSError:
        pass

    tmp_path = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    if path.exists():
        path.unlink()
    tmp_path.rename(path)
    return True


def _write_json_export(papers: list[Paper], path: str) -> bool:
    """Write the compatibility JSON export only when content changes."""
    payload = json.dumps([paper.model_dump() for paper in papers], ensure_ascii=False, indent=2)
    return _write_text_file_if_changed(Path(path), payload)


def save_database(papers: list[Paper], path: str, config: dict[str, Any] | None = None) -> None:
    """Save papers to the configured persistence layer."""
    active_config = config or load_config()
    db_path = Path(path)
    sqlite_primary = _sqlite_primary_storage_enabled(active_config)

    if sqlite_primary:
        _sync_sqlite_mirror(papers, str(db_path), force=False, config=active_config)
        if _json_export_enabled(active_config):
            json_changed = _write_json_export(papers, str(db_path))
            if json_changed:
                logger.info(f"JSON compatibility export updated: {db_path}")
        return

    json_changed = _write_json_export(papers, str(db_path))
    if json_changed:
        logger.info(f"Database saved: {len(papers)} papers to {db_path}")

    _sync_sqlite_mirror(papers, str(db_path), force=json_changed, config=active_config)


def _resolve_sqlite_path(database_path: str, config: dict[str, Any] | None = None) -> str | None:
    """Resolve the configured SQLite storage path."""
    active_config = config or load_config()
    output_config = active_config.get("output", {})
    if not output_config.get("sqlite_enabled", False):
        return None

    sqlite_path = output_config.get("sqlite_database_path")
    if sqlite_path:
        return sqlite_path

    return str(Path(database_path).with_suffix(".sqlite"))


def _sqlite_primary_storage_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return whether SQLite should be treated as the primary persistence layer."""
    active_config = config or load_config()
    output_config = active_config.get("output", {})
    return bool(output_config.get("sqlite_enabled", False) and output_config.get("sqlite_primary_storage", False))


def _json_export_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return whether the JSON compatibility export should be maintained."""
    active_config = config or load_config()
    return bool(active_config.get("output", {}).get("json_export_enabled", True))


def _sync_sqlite_mirror(
    papers: list[Paper],
    database_path: str,
    *,
    force: bool = False,
    config: dict[str, Any] | None = None,
) -> bool:
    """Synchronize the configured SQLite storage when enabled and needed."""
    active_config = config or load_config()
    sqlite_path = _resolve_sqlite_path(database_path, config=active_config)
    if not sqlite_path:
        return False

    stats = sync_papers_to_sqlite(papers, sqlite_path, force=force)
    if stats["changed"]:
        storage_label = "SQLite primary storage" if _sqlite_primary_storage_enabled(active_config) else "SQLite mirror"
        logger.info(
            f"{storage_label} synchronized: {stats['paper_count']} papers, {stats['upserted_count']} upserted, {stats['deleted_count']} deleted, {stats['index_count']} indexes to {sqlite_path}"
        )
    return bool(stats["changed"])


def sync_sqlite_mirror(config: dict, force: bool = True) -> bool:
    """Synchronize the current database state into the configured SQLite storage."""
    db_path = config["output"]["database_path"]
    sqlite_path = _resolve_sqlite_path(db_path, config=config)
    if not sqlite_path:
        console.print("[yellow]SQLite storage disabled in config.[/yellow]")
        return False

    papers = load_database(db_path, config=config)
    stats = sync_papers_to_sqlite(papers, sqlite_path, force=force)
    storage_label = "SQLite primary storage" if _sqlite_primary_storage_enabled(config) else "SQLite mirror"

    if stats["changed"]:
        console.print(
            f"[green]{storage_label} synchronized: {stats['paper_count']} papers, {stats['upserted_count']} upserted, {stats['deleted_count']} deleted, {stats['index_count']} indexes -> {sqlite_path}[/green]"
        )
    else:
        console.print(f"[dim]{storage_label} unchanged: {sqlite_path} ({stats['paper_count']} papers)[/dim]")
    return bool(stats["changed"])


def show_sqlite_status(config: dict) -> None:
    """Display basic information about the SQLite storage."""
    db_path = config["output"]["database_path"]
    sqlite_path = _resolve_sqlite_path(db_path, config=config)
    if not sqlite_path:
        console.print("[yellow]SQLite storage disabled in config.[/yellow]")
        return

    status = get_sqlite_status(sqlite_path)
    if not status["exists"]:
        console.print(f"[yellow]SQLite storage not found: {sqlite_path}[/yellow]")
        return

    size_mb = status["size_bytes"] / (1024 * 1024)
    mode = "primary" if _sqlite_primary_storage_enabled(config) else "mirror"
    console.print("\n[bold cyan]SQLite Storage[/bold cyan]")
    console.print(f"Path: {status['path']}")
    console.print(f"Mode: {mode}")
    console.print(f"Papers: {status['paper_count']}")
    console.print(f"Indexes: {status['index_count']}")
    console.print(f"Last sync: {status['last_synced_at'] or 'unknown'}")
    console.print(f"Sync mode: {status['sync_mode'] or 'unknown'}")
    console.print(f"Last changes: {status['last_upserted_count']} upserted, {status['last_deleted_count']} deleted")
    console.print(f"Size: {size_mb:.2f} MB")

def load_import_manifest(path: str) -> dict[str, dict[str, int]]:
    """Load the incremental import manifest for daily reports."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if isinstance(payload, dict) and isinstance(payload.get("reports"), dict):
        reports = payload["reports"]
    elif isinstance(payload, dict):
        reports = payload
    else:
        return {}

    normalized: dict[str, dict[str, int]] = {}
    for key, value in reports.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        size = value.get("size")
        mtime_ns = value.get("mtime_ns")
        if isinstance(size, int) and isinstance(mtime_ns, int):
            normalized[key] = {"size": size, "mtime_ns": mtime_ns}
    return normalized


def save_import_manifest(manifest: dict[str, dict[str, int]], path: str) -> None:
    """Persist the incremental import manifest atomically."""
    if load_import_manifest(path) == manifest:
        return

    manifest_path = Path(path)
    payload = {
        "updated_at": datetime.now().isoformat(),
        "reports": manifest,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    _write_text_file_if_changed(manifest_path, serialized)


def _report_manifest_key(report_file: Path, daily_dir: Path) -> str:
    """Build a stable manifest key relative to the daily reports directory."""
    try:
        return report_file.relative_to(daily_dir).as_posix()
    except ValueError:
        return report_file.name


def _report_fingerprint(report_file: Path) -> dict[str, int]:
    """Fingerprint a daily report by size and modification time."""
    stat = report_file.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

def ensure_source_trust(papers: list[Paper], db_path: str | None = None) -> int:
    """Backfill source trust metadata for older records when missing."""
    updated = 0
    for paper in papers:
        if paper.source_trusted is None:
            paper.source_trusted = is_from_trusted_source(paper)
            updated += 1

    if updated and db_path:
        save_database(papers, db_path)

    return updated


def _preferred_daily_import_files(daily_dir: Path) -> list[Path]:
    """Prefer structured JSON reports over Markdown for the same day."""
    selected: dict[str, Path] = {}
    for md_file in sorted(daily_dir.glob("*_daily_papers.md")):
        selected[md_file.stem] = md_file
    for json_file in sorted(daily_dir.glob("*_daily_papers.json")):
        selected[json_file.stem] = json_file
    return [selected[key] for key in sorted(selected)]


def import_daily_papers(config: dict) -> dict[str, int]:
    """Import papers from daily report files into the database."""
    db_path = config["output"]["database_path"]
    daily_dir = Path(config["output"]["daily_dir"])
    manifest_path = config.get("output", {}).get("import_manifest_path", "data/import_manifest.json")

    papers = load_database(db_path)
    dedup_index = build_dedup_index(papers)
    import_manifest = load_import_manifest(manifest_path)
    new_manifest: dict[str, dict[str, int]] = {}
    new_count = 0
    processed_reports = 0
    skipped_reports = 0

    if not daily_dir.exists():
        console.print("[yellow]No daily directory found.[/yellow]")
        return {
            "imported_count": 0,
            "processed_reports": 0,
            "skipped_reports": 0,
        }

    for report_file in _preferred_daily_import_files(daily_dir):
        manifest_key = _report_manifest_key(report_file, daily_dir)
        fingerprint = _report_fingerprint(report_file)
        new_manifest[manifest_key] = fingerprint

        if import_manifest.get(manifest_key) == fingerprint:
            skipped_reports += 1
            logger.debug(f"Skipping unchanged daily report: {report_file.name}")
            continue

        processed_reports += 1
        logger.info(f"Importing from {report_file.name}")
        if report_file.suffix.lower() == ".json":
            imported = _parse_daily_json_report(report_file)
        else:
            imported = _parse_daily_report(report_file)

        for paper in imported:
            if has_duplicate_in_index(paper.title, paper.doi, dedup_index):
                continue

            if not paper.categories:
                paper.categories = suggest_categories(paper)
            if not paper.covers_gap:
                paper.covers_gap = check_gap_coverage(paper)
            if paper.source_trusted is None:
                paper.source_trusted = is_from_trusted_source(paper)

            paper.apa7_reference = paper.apa7_reference or format_apa7(paper)
            paper.bibtex = paper.bibtex or format_bibtex(paper)

            papers.append(paper)
            add_to_dedup_index(dedup_index, paper.title, paper.doi)
            new_count += 1

    if new_count > 0:
        save_database(papers, db_path)
    save_import_manifest(new_manifest, manifest_path)

    console.print(f"[green]Imported {new_count} new papers from daily reports.[/green]")
    console.print(f"[cyan]Processed {processed_reports} new or modified daily reports.[/cyan]")
    console.print(f"[dim]Skipped {skipped_reports} unchanged daily reports.[/dim]")
    return {
        "imported_count": new_count,
        "processed_reports": processed_reports,
        "skipped_reports": skipped_reports,
    }


def _parse_daily_json_report(filepath: Path) -> list[Paper]:
    """Parse structured daily JSON output produced by the daily researcher."""
    with open(filepath, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("papers", [])

    return [Paper(**item) for item in items]


def _parse_daily_report(filepath: Path) -> list[Paper]:
    """Parse a daily markdown report and extract papers."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    papers: list[Paper] = []
    sections = re.split(r"\n## \[(ALTA|MEDIA|BAJA)\] ", content)

    for index in range(1, len(sections) - 1, 2):
        level = sections[index]
        body = sections[index + 1]

        lines = body.split("\n")
        title = lines[0].strip() if lines else ""
        paper = Paper(title=title)

        if level == "ALTA":
            paper.relevance_level = RelevanceLevel.HIGH
        elif level == "MEDIA":
            paper.relevance_level = RelevanceLevel.MEDIUM
        else:
            paper.relevance_level = RelevanceLevel.LOW

        parsed_categories: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("- **Autores:**"):
                authors_str = stripped.replace("- **Autores:**", "").strip()
                if authors_str and authors_str != "N/A":
                    paper.authors = [author.strip() for author in authors_str.split(",")]
            elif stripped.startswith("- **Fecha:**"):
                value = stripped.replace("- **Fecha:**", "").strip()
                if value and value != "N/A":
                    paper.publication_date = value
                    try:
                        paper.year = int(value[:4])
                    except (ValueError, IndexError):
                        pass
            elif stripped.startswith("- **Fuente:**"):
                value = stripped.replace("- **Fuente:**", "").strip()
                if value != "N/A":
                    paper.venue = value
            elif stripped.startswith("- **Fuente confiable:**"):
                value = stripped.replace("- **Fuente confiable:**", "").strip().lower()
                paper.source_trusted = value == "si"
            elif stripped.startswith("- **DOI:**"):
                value = stripped.replace("- **DOI:**", "").strip()
                if value != "N/A" and "doi.org/" in value:
                    paper.doi = value.replace("https://doi.org/", "")
            elif stripped.startswith("- **Link:**"):
                value = stripped.replace("- **Link:**", "").strip()
                if value != "N/A":
                    paper.url = value
            elif stripped.startswith("- **Citaciones:**"):
                value = stripped.replace("- **Citaciones:**", "").strip()
                try:
                    paper.citation_count = int(value)
                except ValueError:
                    pass
            elif stripped.startswith("- **Score de relevancia:**"):
                value = stripped.replace("- **Score de relevancia:**", "").strip().replace("/100", "")
                try:
                    paper.relevance_score = int(value)
                except ValueError:
                    pass
            elif stripped.startswith("- **Abstract:**"):
                paper.abstract = stripped.replace("- **Abstract:**", "").strip()
            elif stripped.startswith("- **Cubre algun gap pendiente?**"):
                value = stripped.replace("- **Cubre algun gap pendiente?**", "").strip()
                if value.startswith("Si"):
                    gap_name = value.replace("Si", "").replace("--", "").strip()
                    if gap_name:
                        paper.covers_gap = gap_name
            elif stripped.startswith("- **Keywords match:**"):
                value = stripped.replace("- **Keywords match:**", "").strip()
                if value and value != "[]":
                    cleaned = value.strip("[]")
                    paper.keywords_matched = [
                        keyword.strip().strip("'\"") for keyword in cleaned.split(",") if keyword.strip()
                    ]
            elif re.match(r"^- \[x\] .+", stripped):
                category_name = re.sub(r"^- \[x\] ", "", stripped).strip()
                if category_name:
                    parsed_categories.append(category_name)

        if parsed_categories:
            paper.categories = parsed_categories
        elif paper.title:
            paper.categories = suggest_categories(paper)

        if not paper.covers_gap and paper.title:
            paper.covers_gap = check_gap_coverage(paper)
        if paper.source_trusted is None:
            paper.source_trusted = is_from_trusted_source(paper)

        if paper.title:
            papers.append(paper)

    return papers


def validate_dois(config: dict) -> dict[str, int | bool]:
    """Verify pending DOIs in incremental batches against CrossRef."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)
    tracker = get_api_runtime_tracker(config)
    crossref = _build_crossref_client(config, tracker=tracker)

    pending = [paper for paper in papers if paper.doi and paper.doi_verified is None]
    if not pending:
        console.print("[green]All DOIs already verified.[/green]")
        return {
            "doi_processed": 0,
            "doi_verified": 0,
            "doi_failed": 0,
            "pending_total": 0,
            "pending_after": 0,
            "skipped": False,
        }

    batch_size = min(len(pending), _doi_validation_batch_size(config))
    to_verify = pending[:batch_size]

    if crossref.is_temporarily_disabled():
        tracker.mark_skipped(CROSSREF_RUNTIME_KEY, crossref, batch_size, reason="cooldown")
        tracker.save()
        console.print(f"[yellow]CrossRef validation skipped due to cooldown. Pending DOIs: {len(pending)}[/yellow]")
        return {
            "doi_processed": 0,
            "doi_verified": 0,
            "doi_failed": 0,
            "pending_total": len(pending),
            "pending_after": len(pending),
            "skipped": True,
        }

    console.print(f"Verifying {len(to_verify)} DOIs against CrossRef (pending total: {len(pending)})...")
    tracker.mark_started(CROSSREF_RUNTIME_KEY, len(to_verify))

    verified = 0
    failed = 0
    processed = 0

    for paper in to_verify:
        if crossref.is_temporarily_disabled():
            console.print("[yellow]CrossRef entered cooldown during the batch. Remaining DOIs stay pending.[/yellow]")
            break

        try:
            valid = crossref.verify_doi(paper.doi)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning(f"DOI verification error for {paper.doi}: {exc}")
            break

        if valid:
            paper.doi_verified = True
            verified += 1
            processed += 1
            continue

        if crossref.last_request_failed():
            logger.warning("Stopping DOI validation batch because CrossRef failed before a definitive answer")
            break

        paper.doi_verified = False
        failed += 1
        processed += 1

    tracker.mark_completed(CROSSREF_RUNTIME_KEY, crossref, len(to_verify), processed)
    tracker.save()

    if processed > 0:
        save_database(papers, db_path)

    remaining = max(0, len(pending) - processed)
    console.print(f"[green]Processed DOI batch: {processed}/{len(to_verify)}[/green] (pending after batch: {remaining})")
    console.print(f"[green]Verified: {verified}[/green], [red]Failed: {failed}[/red]")
    return {
        "doi_processed": processed,
        "doi_verified": verified,
        "doi_failed": failed,
        "pending_total": len(pending),
        "pending_after": remaining,
        "skipped": False,
    }


def check_scopus_indexing(config: dict) -> dict[str, int | bool]:
    """Check pending Scopus indexing status via OpenAlex in incremental batches."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)
    tracker = get_api_runtime_tracker(config)
    openalex = _build_openalex_scopus_client(config, tracker=tracker)

    pending = [paper for paper in papers if paper.doi and paper.scopus_indexed is None]
    if not pending:
        console.print("[green]All papers already checked for Scopus indexing.[/green]")
        return {
            "scopus_processed": 0,
            "scopus_indexed": 0,
            "pending_total": 0,
            "pending_after": 0,
            "skipped": False,
        }

    batch_size = min(len(pending), _scopus_check_batch_size(config))
    to_check = pending[:batch_size]

    if openalex.is_temporarily_disabled():
        tracker.mark_skipped(OPENALEX_SCOPUS_RUNTIME_KEY, openalex, batch_size, reason="cooldown")
        tracker.save()
        console.print(f"[yellow]Scopus check skipped due to OpenAlex cooldown. Pending papers: {len(pending)}[/yellow]")
        return {
            "scopus_processed": 0,
            "scopus_indexed": 0,
            "pending_total": len(pending),
            "pending_after": len(pending),
            "skipped": True,
        }

    console.print(f"Checking Scopus indexing for {len(to_check)} papers (pending total: {len(pending)})...")
    tracker.mark_started(OPENALEX_SCOPUS_RUNTIME_KEY, len(to_check))

    indexed = 0
    processed = 0

    for paper in to_check:
        if openalex.is_temporarily_disabled():
            console.print("[yellow]OpenAlex entered cooldown during the batch. Remaining Scopus checks stay pending.[/yellow]")
            break

        try:
            scopus_indexed = openalex.check_scopus_indexed(paper.doi)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning(f"Scopus check error for {paper.doi}: {exc}")
            break

        if openalex.last_request_failed():
            logger.warning("Stopping Scopus batch because OpenAlex failed before a definitive answer")
            break

        paper.scopus_indexed = scopus_indexed
        processed += 1
        if scopus_indexed:
            indexed += 1

    tracker.mark_completed(OPENALEX_SCOPUS_RUNTIME_KEY, openalex, len(to_check), processed)
    tracker.save()

    if processed > 0:
        save_database(papers, db_path)

    remaining = max(0, len(pending) - processed)
    denominator = processed
    console.print(f"[green]Processed Scopus batch: {processed}/{len(to_check)}[/green] (pending after batch: {remaining})")
    console.print(f"[green]Scopus indexed: {indexed}/{denominator}[/green]")
    return {
        "scopus_processed": processed,
        "scopus_indexed": indexed,
        "pending_total": len(pending),
        "pending_after": remaining,
        "skipped": False,
    }

def generate_references(config: dict) -> int:
    """Generate APA7 and BibTeX references for all papers."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)
    updated = 0

    for paper in papers:
        changed = False
        if not paper.apa7_reference:
            paper.apa7_reference = format_apa7(paper)
            changed = True
        if not paper.bibtex:
            paper.bibtex = format_bibtex(paper)
            changed = True
        if changed:
            updated += 1

    if updated > 0:
        save_database(papers, db_path)

    console.print(f"[green]Generated references for {updated} papers.[/green]")
    return updated


def repair_categories(config: dict) -> int:
    """Re-assign categories and gap coverage for papers that have empty categories."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    repaired = 0
    for paper in papers:
        changed = False

        if not paper.categories:
            paper.categories = suggest_categories(paper)
            changed = True

        if not paper.covers_gap:
            gap = check_gap_coverage(paper)
            if gap:
                paper.covers_gap = gap
                changed = True

        if paper.source_trusted is None:
            paper.source_trusted = is_from_trusted_source(paper)
            changed = True

        if changed:
            repaired += 1

    if repaired > 0:
        save_database(papers, db_path)

    console.print(f"[green]Repaired metadata for {repaired} papers.[/green]")
    return repaired


def remove_duplicates(config: dict) -> int:
    """Find and remove duplicate papers from the database."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    duplicates = find_duplicates_in_list(papers)
    if not duplicates:
        console.print("[green]No duplicates found.[/green]")
        return 0

    indices_to_remove: set[int] = set()
    for i, j, _score in duplicates:
        left = papers[i]
        right = papers[j]
        left_quality = (int(bool(left.doi)), int(bool(left.source_trusted)), left.citation_count, len(left.abstract or ""))
        right_quality = (int(bool(right.doi)), int(bool(right.source_trusted)), right.citation_count, len(right.abstract or ""))
        if left_quality >= right_quality:
            indices_to_remove.add(j)
        else:
            indices_to_remove.add(i)

    cleaned = [paper for idx, paper in enumerate(papers) if idx not in indices_to_remove]
    removed = len(papers) - len(cleaned)
    save_database(cleaned, db_path)
    console.print(f"[green]Removed {removed} duplicate papers.[/green]")
    return removed


def interactive_review(config: dict) -> None:
    """Interactively review papers and set their status."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    to_review = [paper for paper in papers if paper.status == PaperStatus.NEW]
    if not to_review:
        console.print("[green]No new papers to review.[/green]")
        return

    console.print(f"\n[bold]Reviewing {len(to_review)} new papers[/bold]\n")

    for index, paper in enumerate(to_review, 1):
        console.print(f"\n[bold cyan]--- Paper {index}/{len(to_review)} ---[/bold cyan]")
        console.print(f"[bold]{paper.title}[/bold]")
        console.print(f"Authors: {', '.join(paper.authors[:5])}")
        console.print(f"Year: {paper.year} | Venue: {paper.venue}")
        console.print(f"Score: {paper.relevance_score}/100 [{paper.relevance_level.value}]")
        console.print(f"Categories: {', '.join(paper.categories)}")
        console.print(f"Trusted source: {'Yes' if paper.source_trusted else 'No'}")
        if paper.covers_gap:
            console.print(f"[green]Covers gap: {paper.covers_gap}[/green]")
        if paper.abstract:
            console.print(f"Abstract: {paper.truncated_abstract(100)}")

        choice = Prompt.ask("Status", choices=["accept", "reject", "skip", "quit"], default="skip")

        if choice == "accept":
            paper.status = PaperStatus.ACCEPTED
        elif choice == "reject":
            paper.status = PaperStatus.REJECTED
        elif choice == "quit":
            break

        notes = Prompt.ask("Notes (optional)", default="")
        if notes:
            paper.notes = notes

    save_database(papers, db_path)
    console.print("[green]Review session saved.[/green]")


def show_statistics(config: dict) -> None:
    """Display statistics about the paper database."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)
    backfilled = ensure_source_trust(papers, db_path=db_path)

    if not papers:
        console.print("[yellow]Database is empty.[/yellow]")
        return

    if backfilled:
        console.print(f"[dim]Source trust metadata backfilled for {backfilled} papers.[/dim]")

    console.print("\n[bold cyan]Database Statistics[/bold cyan]")
    console.print(f"Total papers: {len(papers)}")

    status_counts = Counter(paper.status.value for paper in papers)
    table = Table(title="By Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")
    for status, count in status_counts.most_common():
        table.add_row(status, str(count))
    console.print(table)

    year_counts = Counter(paper.year for paper in papers if paper.year)
    table = Table(title="By Year")
    table.add_column("Year", style="cyan")
    table.add_column("Count", style="green")
    for year in sorted(year_counts.keys(), reverse=True):
        table.add_row(str(year), str(year_counts[year]))
    console.print(table)

    source_counts = Counter(paper.source_api for paper in papers if paper.source_api)
    table = Table(title="By Source API")
    table.add_column("Source", style="cyan")
    table.add_column("Count", style="green")
    for source, count in source_counts.most_common():
        table.add_row(source, str(count))
    console.print(table)

    trust_counts = Counter("trusted" if paper.source_trusted else "provisional" for paper in papers)
    table = Table(title="By Source Trust")
    table.add_column("Trust", style="cyan")
    table.add_column("Count", style="green")
    for trust, count in trust_counts.items():
        table.add_row(trust, str(count))
    console.print(table)

    category_counts: Counter[str] = Counter()
    for paper in papers:
        for category in paper.categories:
            category_counts[category] += 1
    table = Table(title="By Category")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    for category, count in category_counts.most_common():
        table.add_row(category, str(count))
    console.print(table)

    relevance_counts = Counter(paper.relevance_level.value for paper in papers)
    table = Table(title="By Relevance")
    table.add_column("Level", style="cyan")
    table.add_column("Count", style="green")
    for level in ["ALTA", "MEDIA", "BAJA"]:
        table.add_row(level, str(relevance_counts.get(level, 0)))
    console.print(table)


def export_apa7(config: dict) -> bool:
    """Export all accepted papers as APA 7 references."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    accepted = [paper for paper in papers if paper.status in (PaperStatus.ACCEPTED, PaperStatus.NEW)]
    accepted.sort(key=lambda paper: (paper.authors[0] if paper.authors else "", paper.year or 0))

    lines = [f"# Referencias APA 7 -- Generado: {date.today().isoformat()}", ""]
    for paper in accepted:
        lines.append(f"- {paper.apa7_reference or format_apa7(paper)}")
        lines.append("")

    filepath = reports_dir / "references_apa7.md"
    changed = _write_text_file_if_changed(filepath, "\n".join(lines))
    if changed:
        console.print(f"[green]APA 7 references updated: {filepath} ({len(accepted)} papers)[/green]")
    else:
        console.print(f"[dim]APA 7 references unchanged: {filepath}[/dim]")
    return changed


def export_bibtex(config: dict) -> bool:
    """Export all accepted papers as BibTeX."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    accepted = [paper for paper in papers if paper.status in (PaperStatus.ACCEPTED, PaperStatus.NEW)]

    entries = [paper.bibtex or format_bibtex(paper) for paper in accepted]

    filepath = reports_dir / "references.bib"
    changed = _write_text_file_if_changed(filepath, "\n\n".join(entries))
    if changed:
        console.print(f"[green]BibTeX updated: {filepath} ({len(accepted)} entries)[/green]")
    else:
        console.print(f"[dim]BibTeX unchanged: {filepath}[/dim]")
    return changed


def generate_consolidated_report(config: dict) -> bool:
    """Generate the consolidated report organized by category."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    ensure_source_trust(papers, db_path=db_path)
    accepted = [paper for paper in papers if paper.status in (PaperStatus.ACCEPTED, PaperStatus.NEW)]

    lines = [
        f"# Reporte Consolidado de Papers -- Generado: {date.today().isoformat()}",
        "",
        f"Total papers: {len(accepted)}",
        "",
    ]

    by_category: dict[str, list[Paper]] = {}
    for paper in accepted:
        for category in paper.categories:
            by_category.setdefault(category, []).append(paper)

    for category in sorted(by_category.keys()):
        category_papers = by_category[category]
        category_papers.sort(key=lambda paper: (ranking_score(paper), int(bool(paper.source_trusted))), reverse=True)
        lines.append(f"## {category} ({len(category_papers)} papers)")
        lines.append("")

        for paper in category_papers:
            doi_str = f"https://doi.org/{paper.doi}" if paper.doi else "N/A"
            lines.append(f"### [{paper.relevance_level.value}] {paper.title}")
            lines.append(f"- **Autores:** {', '.join(paper.authors[:5])}")
            lines.append(f"- **Ano:** {paper.year} | **Fuente:** {paper.venue}")
            lines.append(f"- **Fuente confiable:** {'Si' if paper.source_trusted else 'No'}")
            lines.append(f"- **DOI:** {doi_str}")
            lines.append(f"- **Score:** {paper.relevance_score}/100")
            if paper.covers_gap:
                lines.append(f"- **Cubre gap:** {paper.covers_gap}")
            lines.append("")

    filepath = reports_dir / "consolidated_report.md"
    changed = _write_text_file_if_changed(filepath, "\n".join(lines))
    if changed:
        console.print(f"[green]Consolidated report updated: {filepath}[/green]")
    else:
        console.print(f"[dim]Consolidated report unchanged: {filepath}[/dim]")
    return changed


def generate_gap_analysis_report(config: dict) -> bool:
    """Generate the gap analysis report."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    report = generate_gap_report(papers)

    filepath = reports_dir / "gap_analysis.md"
    changed = _write_text_file_if_changed(filepath, report)
    if changed:
        console.print(f"[green]Gap analysis report updated: {filepath}[/green]")
    else:
        console.print(f"[dim]Gap analysis report unchanged: {filepath}[/dim]")
    return changed


def generate_statistics_report(config: dict) -> bool:
    """Generate a statistics markdown report."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    ensure_source_trust(papers, db_path=db_path)
    today = date.today().isoformat()

    lines = [f"# Estadisticas -- Generado: {today}", ""]
    lines.append(f"## Total papers en base de datos: {len(papers)}")
    lines.append("")

    year_counts = Counter(paper.year for paper in papers if paper.year)
    lines.append("## Papers por ano")
    lines.append("| Ano | Cantidad |")
    lines.append("|-----|----------|")
    for year in sorted(year_counts.keys(), reverse=True):
        lines.append(f"| {year} | {year_counts[year]} |")
    lines.append("")

    source_counts = Counter(paper.source_api for paper in papers if paper.source_api)
    lines.append("## Papers por fuente API")
    lines.append("| Fuente | Cantidad |")
    lines.append("|--------|----------|")
    for source, count in source_counts.most_common():
        lines.append(f"| {source} | {count} |")
    lines.append("")

    trust_counts = Counter("Confiable" if paper.source_trusted else "Provisional" for paper in papers)
    lines.append("## Papers por confianza de fuente")
    lines.append("| Confianza | Cantidad |")
    lines.append("|-----------|----------|")
    for trust, count in trust_counts.items():
        lines.append(f"| {trust} | {count} |")
    lines.append("")

    category_counts: Counter[str] = Counter()
    for paper in papers:
        for category in paper.categories:
            category_counts[category] += 1
    lines.append("## Papers por categoria")
    lines.append("| Categoria | Cantidad |")
    lines.append("|-----------|----------|")
    for category, count in category_counts.most_common():
        lines.append(f"| {category} | {count} |")
    lines.append("")

    date_counts = Counter(paper.date_found for paper in papers)
    lines.append("## Timeline de busqueda")
    lines.append("| Fecha | Papers encontrados |")
    lines.append("|-------|-------------------|")
    for found_date in sorted(date_counts.keys()):
        lines.append(f"| {found_date} | {date_counts[found_date]} |")
    lines.append("")

    filepath = reports_dir / "statistics.md"
    changed = _write_text_file_if_changed(filepath, "\n".join(lines))
    if changed:
        console.print(f"[green]Statistics report updated: {filepath}[/green]")
    else:
        console.print(f"[dim]Statistics report unchanged: {filepath}[/dim]")
    return changed


def _step_title(step: str) -> str:
    return f"[bold]{step}[/bold]"


def finalize_compiler_outputs(
    config: dict,
    reports_step: str = "Generating reports...",
    sync_step: str | None = None,
) -> dict[str, Any]:
    """Generate derived reports/exports and synchronize the configured storage."""
    console.print(f"\n{_step_title(reports_step)}")
    consolidated_changed = generate_consolidated_report(config)
    gap_changed = generate_gap_analysis_report(config)
    statistics_changed = generate_statistics_report(config)
    apa_changed = export_apa7(config)
    bibtex_changed = export_bibtex(config)

    sync_changed = False
    if sync_step is not None:
        storage_label = "SQLite primary storage" if _sqlite_primary_storage_enabled(config) else "SQLite mirror"
        console.print(f"\n{_step_title(sync_step.format(storage_label=storage_label))}")
        sync_changed = sync_sqlite_mirror(config, force=False)

    return {
        "consolidated_report_changed": consolidated_changed,
        "gap_analysis_changed": gap_changed,
        "statistics_report_changed": statistics_changed,
        "apa_changed": apa_changed,
        "bibtex_changed": bibtex_changed,
        "sqlite_sync_changed": sync_changed,
    }


def run_compilation_phase(
    config: dict,
    import_step: str = "Importing daily papers...",
    repair_step: str = "Repairing categories for existing papers...",
    dedup_step: str = "Removing duplicates...",
    references_step: str = "Generating references...",
    finalize_outputs: bool = False,
) -> dict[str, Any]:
    """Run the core compilation stage without metadata validation."""
    console.print(_step_title(import_step))
    import_summary = import_daily_papers(config)

    console.print(f"\n{_step_title(repair_step)}")
    repaired_count = repair_categories(config)

    console.print(f"\n{_step_title(dedup_step)}")
    removed_duplicates = remove_duplicates(config)

    console.print(f"\n{_step_title(references_step)}")
    generated_references = generate_references(config)

    summary: dict[str, Any] = {
        **import_summary,
        "repaired_count": repaired_count,
        "removed_duplicates": removed_duplicates,
        "generated_references": generated_references,
    }

    if finalize_outputs:
        summary["finalize"] = finalize_compiler_outputs(
            config,
            reports_step="Generating reports...",
            sync_step="Syncing {storage_label}...",
        )

    return summary


def run_metadata_phase(
    config: dict,
    doi_step: str = "Validating DOIs...",
    scopus_step: str = "Checking Scopus indexing...",
    finalize_outputs: bool = False,
) -> dict[str, Any]:
    """Run the metadata validation stage for DOI and Scopus signals."""
    console.print(_step_title(doi_step))
    doi_summary = validate_dois(config)

    console.print(f"\n{_step_title(scopus_step)}")
    scopus_summary = check_scopus_indexing(config)

    summary: dict[str, Any] = {
        **doi_summary,
        **scopus_summary,
    }

    if finalize_outputs:
        summary["finalize"] = finalize_compiler_outputs(
            config,
            reports_step="Generating reports...",
            sync_step="Syncing {storage_label}...",
        )

    return summary


def compile_all(config: dict) -> None:
    """Run the full compilation pipeline."""
    console.print("\n[bold cyan]Paper Compiler Agent[/bold cyan]\n")

    run_compilation_phase(
        config,
        import_step="1. Importing daily papers...",
        repair_step="2. Repairing categories for existing papers...",
        dedup_step="3. Removing duplicates...",
        references_step="4. Generating references...",
        finalize_outputs=False,
    )

    console.print("")
    run_metadata_phase(
        config,
        doi_step="5. Validating DOIs...",
        scopus_step="6. Checking Scopus indexing...",
        finalize_outputs=False,
    )

    finalize_compiler_outputs(
        config,
        reports_step="7. Generating reports...",
        sync_step="8. Syncing {storage_label}...",
    )

    console.print("\n[bold green]Compilation complete![/bold green]")

