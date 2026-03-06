"""Agent 1: Daily Researcher — Searches for new papers related to the thesis."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.apis.semantic_scholar import SemanticScholarAPI
from src.apis.arxiv_api import ArxivAPI
from src.apis.openalex_api import OpenAlexAPI
from src.models.paper import Paper
from src.utils.duplicate_detector import (
    is_duplicate_by_doi,
    is_duplicate_by_title,
    is_duplicate_of_existing,
)
from src.utils.relevance_scorer import (
    check_gap_coverage,
    is_from_trusted_source,
    score_paper,
    suggest_categories,
)
from src.utils.logger import setup_logger

logger = setup_logger("daily_researcher")
console = Console()


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    """Load the main configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_keywords(path: str = "config/keywords.yaml") -> dict[str, list[str]]:
    """Load keyword groups for searching."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("keyword_groups", {})


def get_cache_path(config: dict) -> Path:
    """Get the cache directory path."""
    cache_dir = Path(config.get("output", {}).get("cache_dir", "data/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(api_name: str, query: str, date_str: str) -> str:
    """Generate a cache key for a search query."""
    raw = f"{api_name}:{query}:{date_str}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cache(cache_dir: Path, cache_key: str) -> list[dict] | None:
    """Load cached results if they exist and are less than 24 hours old."""
    cache_file = cache_dir / f"{cache_key}.json"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        cached_time = datetime.fromisoformat(data["timestamp"])
        if datetime.now() - cached_time > timedelta(hours=24):
            return None
        return data["results"]
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def save_cache(cache_dir: Path, cache_key: str, results: list[dict]) -> None:
    """Save search results to cache."""
    cache_file = cache_dir / f"{cache_key}.json"
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def search_semantic_scholar(
    query: str,
    config: dict,
    cache_dir: Path,
    date_from: str,
    max_results: int = 20,
) -> list[Paper]:
    """Search Semantic Scholar with caching."""
    cache_key = get_cache_key("semantic_scholar", query, date_from)
    cached = load_cache(cache_dir, cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for Semantic Scholar: '{query}'")
        return [Paper(**p) for p in cached]

    api_config = config.get("apis", {}).get("semantic_scholar", {})
    if not api_config.get("enabled", True):
        return []

    api = SemanticScholarAPI(
        base_url=api_config.get("base_url", "https://api.semanticscholar.org/graph/v1"),
        rate_limit=api_config.get("rate_limit_per_second", 1),
    )

    year_start = date_from[:4] if date_from else "2020"
    papers = api.search(query, limit=max_results, year_range=f"{year_start}-")

    # Cache results
    save_cache(cache_dir, cache_key, [p.model_dump() for p in papers])
    return papers


def search_arxiv(
    query: str,
    config: dict,
    cache_dir: Path,
    date_from: str,
    max_results: int = 20,
) -> list[Paper]:
    """Search arXiv with caching."""
    cache_key = get_cache_key("arxiv", query, date_from)
    cached = load_cache(cache_dir, cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for arXiv: '{query}'")
        return [Paper(**p) for p in cached]

    api_config = config.get("apis", {}).get("arxiv", {})
    if not api_config.get("enabled", True):
        return []

    api = ArxivAPI(
        base_url=api_config.get("base_url", "http://export.arxiv.org/api/query"),
        rate_limit_seconds=api_config.get("rate_limit_seconds", 3),
        categories=api_config.get("categories", ["cs.IR", "cs.CL", "cs.AI", "cs.LG"]),
    )

    papers = api.search(query, max_results=max_results, date_from=date_from)

    save_cache(cache_dir, cache_key, [p.model_dump() for p in papers])
    return papers


def search_openalex(
    query: str,
    config: dict,
    cache_dir: Path,
    date_from: str,
    max_results: int = 20,
) -> list[Paper]:
    """Search OpenAlex with caching."""
    cache_key = get_cache_key("openalex", query, date_from)
    cached = load_cache(cache_dir, cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for OpenAlex: '{query}'")
        return [Paper(**p) for p in cached]

    api_config = config.get("apis", {}).get("openalex", {})
    if not api_config.get("enabled", True):
        return []

    api = OpenAlexAPI(
        base_url=api_config.get("base_url", "https://api.openalex.org"),
        email=api_config.get("email", ""),
    )

    papers = api.search(query, limit=max_results, from_date=date_from)

    save_cache(cache_dir, cache_key, [p.model_dump() for p in papers])
    return papers


def run_daily_search(days: int = 7, dry_run: bool = False) -> list[Paper]:
    """Execute the full daily search across all APIs and keyword groups.

    Args:
        days: Number of days back to search.
        dry_run: If True, search but don't write output files.

    Returns:
        List of all found and scored papers.
    """
    config = load_config()
    keywords = load_keywords()
    cache_dir = get_cache_path(config)

    date_from = (date.today() - timedelta(days=days)).isoformat()
    max_results = config.get("general", {}).get("max_results_per_query", 20)
    min_year = config.get("general", {}).get("min_year", 2020)

    console.print(f"\n[bold cyan]Daily Researcher Agent[/bold cyan]")
    console.print(f"Searching papers from {date_from} to today")
    console.print(f"Keyword groups: {len(keywords)}")
    console.print()

    all_papers: list[Paper] = []
    stats = {
        "total_queries": 0,
        "semantic_scholar_results": 0,
        "arxiv_results": 0,
        "openalex_results": 0,
        "duplicates_removed": 0,
        "untrusted_removed": 0,
        "below_min_year": 0,
    }

    total_queries = sum(len(kws) for kws in keywords.values()) * 3  # 3 APIs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching papers...", total=total_queries)

        for group_name, keyword_list in keywords.items():
            for query in keyword_list:
                # Search Semantic Scholar
                progress.update(task, description=f"[cyan]S2:[/cyan] {query[:50]}...")
                papers = search_semantic_scholar(query, config, cache_dir, date_from, max_results)
                stats["semantic_scholar_results"] += len(papers)
                all_papers.extend(papers)
                stats["total_queries"] += 1
                progress.advance(task)

                # Search arXiv
                progress.update(task, description=f"[green]arXiv:[/green] {query[:50]}...")
                papers = search_arxiv(query, config, cache_dir, date_from, max_results)
                stats["arxiv_results"] += len(papers)
                all_papers.extend(papers)
                stats["total_queries"] += 1
                progress.advance(task)

                # Search OpenAlex
                progress.update(task, description=f"[yellow]OA:[/yellow] {query[:50]}...")
                papers = search_openalex(query, config, cache_dir, date_from, max_results)
                stats["openalex_results"] += len(papers)
                all_papers.extend(papers)
                stats["total_queries"] += 1
                progress.advance(task)

    console.print(f"\n[bold]Raw results:[/bold] {len(all_papers)} papers")

    # --- Deduplication ---
    console.print("[dim]Removing duplicates...[/dim]")
    unique_papers: list[Paper] = []
    for paper in all_papers:
        # Skip if year is below minimum
        if paper.year and paper.year < min_year:
            stats["below_min_year"] += 1
            continue

        # Skip if duplicate of existing thesis papers
        if is_duplicate_of_existing(paper.title, paper.doi):
            stats["duplicates_removed"] += 1
            continue

        # Skip if duplicate within current batch
        if paper.doi and is_duplicate_by_doi(paper.doi, unique_papers):
            stats["duplicates_removed"] += 1
            continue
        if is_duplicate_by_title(paper.title, unique_papers):
            stats["duplicates_removed"] += 1
            continue

        unique_papers.append(paper)

    console.print(f"[bold]After deduplication:[/bold] {len(unique_papers)} papers")

    # --- Score and filter ---
    console.print("[dim]Scoring and filtering...[/dim]")
    scored_papers: list[Paper] = []
    for paper in unique_papers:
        # Check trusted source
        if not is_from_trusted_source(paper):
            stats["untrusted_removed"] += 1
            continue

        # Score
        paper = score_paper(paper)
        paper.categories = suggest_categories(paper)
        paper.covers_gap = check_gap_coverage(paper)
        scored_papers.append(paper)

    # Sort by relevance
    scored_papers.sort(key=lambda p: p.relevance_score, reverse=True)

    high = [p for p in scored_papers if p.relevance_level.value == "ALTA"]
    medium = [p for p in scored_papers if p.relevance_level.value == "MEDIA"]
    low = [p for p in scored_papers if p.relevance_level.value == "BAJA"]

    # --- Display summary ---
    table = Table(title="Search Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total raw results", str(len(all_papers)))
    table.add_row("Semantic Scholar", str(stats["semantic_scholar_results"]))
    table.add_row("arXiv", str(stats["arxiv_results"]))
    table.add_row("OpenAlex", str(stats["openalex_results"]))
    table.add_row("Duplicates removed", str(stats["duplicates_removed"]))
    table.add_row("Untrusted source removed", str(stats["untrusted_removed"]))
    table.add_row("Below min year removed", str(stats["below_min_year"]))
    table.add_row("Final papers", str(len(scored_papers)))
    table.add_row("HIGH relevance", str(len(high)))
    table.add_row("MEDIUM relevance", str(len(medium)))
    table.add_row("LOW relevance", str(len(low)))
    console.print(table)

    # Papers covering gaps
    gap_papers = [p for p in scored_papers if p.covers_gap]
    if gap_papers:
        console.print("\n[bold green]Papers covering gaps:[/bold green]")
        for p in gap_papers:
            console.print(f"  - [bold]{p.covers_gap}[/bold]: {p.title}")

    # --- Write output ---
    if not dry_run:
        write_daily_report(scored_papers, stats, config)
    else:
        console.print("\n[yellow]Dry run — no files written.[/yellow]")

    return scored_papers


def write_daily_report(papers: list[Paper], stats: dict, config: dict) -> None:
    """Write the daily report markdown file."""
    today = date.today().isoformat()
    output_dir = Path(config.get("output", {}).get("daily_dir", "output/daily"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to temp file first, then rename (atomic)
    filepath = output_dir / f"{today}_daily_papers.md"
    tmp_filepath = output_dir / f"{today}_daily_papers.md.tmp"

    high = [p for p in papers if p.relevance_level.value == "ALTA"]
    medium = [p for p in papers if p.relevance_level.value == "MEDIA"]
    low = [p for p in papers if p.relevance_level.value == "BAJA"]
    gap_papers = [p for p in papers if p.covers_gap]

    lines: list[str] = []

    # Header
    lines.append(f"# Reporte Diario de Papers -- {today}")
    lines.append("")
    lines.append("## Resumen")
    lines.append(f"- Papers encontrados: {len(papers)}")
    lines.append(f"- Papers de alta relevancia: {len(high)}")
    lines.append(f"- Papers de media relevancia: {len(medium)}")
    lines.append(f"- Papers de baja relevancia: {len(low)}")
    lines.append(f"- APIs consultadas: Semantic Scholar, arXiv, OpenAlex")
    keyword_groups = load_keywords()
    lines.append(f"- Keywords groups procesados: {len(keyword_groups)}/{len(keyword_groups)}")
    lines.append("")

    # Gaps covered
    lines.append("## Gaps cubiertos hoy:")
    if gap_papers:
        for p in gap_papers:
            lines.append(f"- [x] {p.covers_gap} -- {p.title}")
    else:
        lines.append("- Ninguno nuevo cubierto hoy")
    lines.append("")

    # Top 5
    lines.append("## Top 5 Papers del dia:")
    for i, p in enumerate(papers[:5], 1):
        lines.append(f"{i}. [{p.relevance_level.value}] {p.title} (Score: {p.relevance_score}/100)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # All papers
    for paper in papers:
        lines.append(f"## [{paper.relevance_level.value}] {paper.title}")
        lines.append(f"- **Autores:** {', '.join(paper.authors) if paper.authors else 'N/A'}")
        lines.append(f"- **Fecha:** {paper.publication_date or 'N/A'}")
        lines.append(f"- **Fuente:** {paper.venue or 'N/A'}")
        doi_str = f"https://doi.org/{paper.doi}" if paper.doi else "N/A"
        lines.append(f"- **DOI:** {doi_str}")
        lines.append(f"- **Link:** {paper.url or 'N/A'}")
        lines.append(f"- **Citaciones:** {paper.citation_count}")
        lines.append(f"- **Score de relevancia:** {paper.relevance_score}/100")
        lines.append(f"- **Abstract:** {paper.truncated_abstract(200)}")
        lines.append(f"- **Keywords match:** {paper.keywords_matched}")

        lines.append("- **Categoria sugerida para mi tesis:**")
        all_cats = [
            "Sistemas RAG Hibridos", "Normalizacion/Preprocesamiento",
            "Segmentacion/Chunking", "Modelos de Embedding",
            "Estrategias de Recuperacion", "Re-ranking",
            "Alucinaciones en LLMs", "Evaluacion de sistemas RAG",
            "Documentacion Cloud", "Metricas de evaluacion", "Vector Databases",
        ]
        for cat in all_cats:
            check = "x" if cat in paper.categories else " "
            lines.append(f"  - [{check}] {cat}")

        gap_text = f"Si -- {paper.covers_gap}" if paper.covers_gap else "No"
        lines.append(f"- **Cubre algun gap pendiente?** {gap_text}")
        lines.append(f"- **Notas:** Paper encontrado via {paper.source_api}")
        lines.append("")

    content = "\n".join(lines)

    with open(tmp_filepath, "w", encoding="utf-8") as f:
        f.write(content)

    # Atomic rename
    if filepath.exists():
        filepath.unlink()
    tmp_filepath.rename(filepath)

    console.print(f"\n[bold green]Report written:[/bold green] {filepath}")
    logger.info(f"Daily report written to {filepath}")
