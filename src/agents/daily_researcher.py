"""Agent 1: Daily Researcher - Searches for new papers related to the thesis."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from src.apis.arxiv_api import ArxivAPI
from src.apis.openalex_api import OpenAlexAPI
from src.apis.semantic_scholar import SemanticScholarAPI
from src.models.paper import Paper
from src.utils.duplicate_detector import DedupIndex, add_to_dedup_index, has_duplicate_in_index, load_existing_dedup_index
from src.utils.logger import setup_logger
from src.utils.relevance_scorer import check_gap_coverage, is_from_trusted_source, ranking_score, score_paper, suggest_categories

logger = setup_logger("daily_researcher")
console = Console()

QUERY_GENERIC_TERMS = {
    "and", "the", "for", "with", "without", "using", "based", "from", "into",
    "system", "systems", "model", "models", "method", "methods", "framework",
    "analysis", "approach", "approaches", "study", "studies", "comparison",
    "strategy", "strategies", "pipeline",
}

QUERY_CONCEPT_ALIASES = {
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "hybrid": ["hybrid", "hybrid retrieval", "hybrid search", "hybrid rag", "lexical and semantic", "sparse and dense"],
    "bm25": ["bm25", "okapi"],
    "dense": ["dense retrieval", "dense passage retrieval", "vector search"],
    "colbert": ["colbert", "late interaction"],
    "sentence_bert": ["sentence bert", "sentence-bert", "sbert"],
    "bge": ["bge", "bge-m3", "baai general embedding", "baai"],
    "reranking": ["reranking", "re-ranking", "re ranking", "cross encoder", "cross-encoder"],
    "chunking": ["chunking", "chunk", "segmentation", "text segmentation"],
    "hallucination": ["hallucination", "faithfulness", "groundedness"],
    "documentation": ["documentation", "technical documentation", "api documentation", "developer documentation", "knowledge base"],
    "cloud": ["cloud", "aws", "azure", "gcp", "google cloud"],
    "vector_db": ["vector database", "faiss", "chromadb", "pinecone"],
    "ann": ["approximate nearest neighbor", "nearest neighbor", "ann", "hnsw"],
    "ragas": ["ragas", "rag assessment"],
}


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _contains_any_phrase(text: str, phrases: list[str]) -> bool:
    normalized_text = _normalize_text(text)
    return any(_normalize_text(phrase) in normalized_text for phrase in phrases if phrase)


def _query_concept_groups(query: str) -> list[list[str]]:
    normalized_query = _normalize_text(query)
    groups: list[list[str]] = []
    for aliases in QUERY_CONCEPT_ALIASES.values():
        normalized_aliases = [_normalize_text(alias) for alias in aliases if alias]
        if any(alias and alias in normalized_query for alias in normalized_aliases):
            groups.append(normalized_aliases)
    return groups


def _count_concept_hits(text: str, groups: list[list[str]]) -> int:
    normalized_text = _normalize_text(text)
    return sum(1 for group in groups if any(alias in normalized_text for alias in group))


def _extract_query_terms(query: str) -> tuple[list[str], list[str]]:
    tokens = [token for token in _normalize_text(query).split() if len(token) >= 3]
    strong_tokens = [token for token in tokens if token not in QUERY_GENERIC_TERMS]
    return tokens, strong_tokens


def _paper_matches_query(paper: Paper, query: str) -> bool:
    """Require a minimum overlap while accepting common thesis-domain aliases."""
    combined_text = _normalize_text(f"{paper.title or ''} {paper.abstract or ''}")
    if not combined_text:
        return False

    title_text = _normalize_text(paper.title or "")
    normalized_query = _normalize_text(query)
    if normalized_query and normalized_query in combined_text:
        return True

    concept_groups = _query_concept_groups(query)
    if concept_groups:
        concept_hits = _count_concept_hits(combined_text, concept_groups)
        title_concept_hits = _count_concept_hits(title_text, concept_groups)
        required_concepts = 1 if len(concept_groups) <= 2 else 2
        if paper.source_api == "arxiv":
            required_concepts = max(1, required_concepts - 1)
        if concept_hits >= required_concepts and (title_concept_hits >= 1 or paper.source_api == "arxiv"):
            return True

    query_tokens, strong_tokens = _extract_query_terms(query)
    if not query_tokens:
        return True

    combined_tokens = set(combined_text.split())
    title_tokens = set(title_text.split())

    total_matches = sum(1 for token in query_tokens if token in combined_tokens)
    strong_matches = sum(1 for token in strong_tokens if token in combined_tokens)
    title_strong_matches = sum(1 for token in strong_tokens if token in title_tokens)

    required_total = 1 if len(query_tokens) <= 2 else 2
    required_strong = 1 if len(strong_tokens) <= 2 else 2
    if paper.source_api == "arxiv":
        required_total = max(1, required_total - 1)
        if strong_tokens:
            required_strong = max(1, required_strong - 1)

    if total_matches < required_total:
        return False

    if strong_tokens and strong_matches < required_strong and title_strong_matches < required_strong:
        return False

    return True


def _filter_papers_for_query(query: str, papers: list[Paper]) -> list[Paper]:
    return [paper for paper in papers if _paper_matches_query(paper, query)]


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
    payload = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_api_clients(config: dict) -> dict[str, Any]:
    """Instantiate reusable API clients for a search run."""
    api_config = config.get("apis", {})
    clients: dict[str, Any] = {}

    semantic_scholar = api_config.get("semantic_scholar", {})
    if semantic_scholar.get("enabled", True):
        clients["semantic_scholar"] = SemanticScholarAPI(
            base_url=semantic_scholar.get("base_url", "https://api.semanticscholar.org/graph/v1"),
            rate_limit=semantic_scholar.get("rate_limit_per_second", 1),
            fields=semantic_scholar.get(
                "fields",
                "title,abstract,year,venue,externalIds,authors,citationCount,publicationDate,url",
            ),
            max_retries_with_key=semantic_scholar.get("max_retries_with_key", 4),
            max_retries_without_key=semantic_scholar.get("max_retries_without_key", 2),
            cooldown_seconds=semantic_scholar.get("cooldown_seconds", 1800),
            shutdown_after_consecutive_failures=semantic_scholar.get("shutdown_after_consecutive_failures", 2),
        )

    arxiv = api_config.get("arxiv", {})
    if arxiv.get("enabled", True):
        clients["arxiv"] = ArxivAPI(
            base_url=arxiv.get("base_url", "http://export.arxiv.org/api/query"),
            rate_limit_seconds=arxiv.get("rate_limit_seconds", 3),
            categories=arxiv.get("categories", ["cs.IR", "cs.CL", "cs.AI", "cs.LG"]),
        )

    openalex = api_config.get("openalex", {})
    if openalex.get("enabled", True):
        clients["openalex"] = OpenAlexAPI(
            base_url=openalex.get("base_url", "https://api.openalex.org"),
            email=openalex.get("email", ""),
            rate_limit_per_second=openalex.get("rate_limit_per_second", 10),
        )

    return clients


def search_semantic_scholar(
    query: str,
    api: SemanticScholarAPI,
    cache_dir: Path,
    date_from: str,
    max_results: int = 20,
) -> list[Paper]:
    """Search Semantic Scholar with caching."""
    cache_key = get_cache_key("semantic_scholar", query, date_from)
    cached = load_cache(cache_dir, cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for Semantic Scholar: '{query}'")
        return [Paper(**paper) for paper in cached]

    year_start = date_from[:4] if date_from else "2020"
    papers = api.search(query, limit=max_results, year_range=f"{year_start}-")
    save_cache(cache_dir, cache_key, [paper.model_dump() for paper in papers])
    return papers


def search_arxiv(
    query: str,
    api: ArxivAPI,
    cache_dir: Path,
    date_from: str,
    max_results: int = 20,
) -> list[Paper]:
    """Search arXiv with caching."""
    cache_key = get_cache_key("arxiv", query, date_from)
    cached = load_cache(cache_dir, cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for arXiv: '{query}'")
        return [Paper(**paper) for paper in cached]

    papers = api.search(query, max_results=max_results, date_from=date_from)
    save_cache(cache_dir, cache_key, [paper.model_dump() for paper in papers])
    return papers


def search_openalex(
    query: str,
    api: OpenAlexAPI,
    cache_dir: Path,
    date_from: str,
    max_results: int = 20,
) -> list[Paper]:
    """Search OpenAlex with caching."""
    cache_key = get_cache_key("openalex", query, date_from)
    cached = load_cache(cache_dir, cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for OpenAlex: '{query}'")
        return [Paper(**paper) for paper in cached]

    papers = api.search(query, limit=max_results, from_date=date_from)
    save_cache(cache_dir, cache_key, [paper.model_dump() for paper in papers])
    return papers


def _write_structured_daily_report(papers: list[Paper], stats: dict[str, int], output_dir: Path, today: str) -> None:
    """Persist the daily report in JSON so downstream steps do not need to parse Markdown."""
    filepath = output_dir / f"{today}_daily_papers.json"
    tmp_filepath = output_dir / f"{today}_daily_papers.json.tmp"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "stats": stats,
        "papers": [paper.model_dump() for paper in papers],
    }

    with open(tmp_filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if filepath.exists():
        filepath.unlink()
    tmp_filepath.rename(filepath)

    logger.info(f"Structured daily report written to {filepath}")


def run_daily_search(days: int = 7, dry_run: bool = False) -> list[Paper]:
    """Execute the full daily search across all APIs and keyword groups."""
    config = load_config()
    keywords = load_keywords()
    cache_dir = get_cache_path(config)
    clients = build_api_clients(config)

    date_from = (date.today() - timedelta(days=days)).isoformat()
    general_config = config.get("general", {})
    max_results = general_config.get("max_results_per_query", 20)
    min_year = general_config.get("min_year", 2020)
    strict_source_filter = general_config.get("strict_source_filter", False)
    untrusted_keep_threshold = general_config.get("untrusted_keep_score_threshold", 65)
    min_keep_score = general_config.get("min_keep_score", 35)
    worker_count = min(general_config.get("search_workers_per_query", 3), max(len(clients), 1))

    console.print("\n[bold cyan]Daily Researcher Agent[/bold cyan]")
    console.print(f"Searching papers from {date_from} to today")
    console.print(f"Keyword groups: {len(keywords)}")
    console.print(f"Enabled APIs: {', '.join(clients.keys())}")
    console.print()

    search_plan: list[tuple[str, str, Any, str]] = []
    if "semantic_scholar" in clients:
        search_plan.append(("Semantic Scholar", "semantic_scholar_results", search_semantic_scholar, "semantic_scholar"))
    if "arxiv" in clients:
        search_plan.append(("arXiv", "arxiv_results", search_arxiv, "arxiv"))
    if "openalex" in clients:
        search_plan.append(("OpenAlex", "openalex_results", search_openalex, "openalex"))

    if not search_plan:
        console.print("[yellow]No search APIs are enabled in config.[/yellow]")
        return []

    expected_queries = sum(len(group_keywords) for group_keywords in keywords.values()) * len(search_plan)

    all_papers: list[Paper] = []
    stats = {
        "expected_queries": expected_queries,
        "total_queries": 0,
        "semantic_scholar_results": 0,
        "arxiv_results": 0,
        "openalex_results": 0,
        "duplicates_removed": 0,
        "query_mismatch_removed": 0,
        "low_score_removed": 0,
        "untrusted_removed": 0,
        "trusted_kept": 0,
        "provisional_kept": 0,
        "below_min_year": 0,
    }

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Searching papers...", total=expected_queries)

            for group_name, keyword_list in keywords.items():
                for query in keyword_list:
                    progress.update(task, description=f"[cyan]{group_name}[/cyan] {query[:60]}")
                    future_map = {}
                    for label, metric_key, search_fn, client_key in search_plan:
                        future = executor.submit(search_fn, query, clients[client_key], cache_dir, date_from, max_results)
                        future_map[future] = (label, metric_key)

                    for future in as_completed(future_map):
                        label, metric_key = future_map[future]
                        try:
                            papers = future.result()
                        except Exception as exc:
                            logger.error(f"{label} search failed for '{query}': {exc}")
                            papers = []

                        filtered_papers = _filter_papers_for_query(query, papers)
                        stats["query_mismatch_removed"] += max(0, len(papers) - len(filtered_papers))
                        stats[metric_key] += len(filtered_papers)
                        stats["total_queries"] += 1
                        all_papers.extend(filtered_papers)
                        progress.update(task, description=f"[cyan]{label}[/cyan] {query[:50]}...")
                        progress.advance(task)

    console.print(f"\n[bold]Raw results:[/bold] {len(all_papers)} papers")

    existing_index = load_existing_dedup_index(config.get("output", {}).get("existing_papers_path", "data/existing_papers.json"))
    batch_index = DedupIndex()
    unique_papers: list[Paper] = []

    console.print("[dim]Removing duplicates...[/dim]")
    for paper in all_papers:
        if paper.year and paper.year < min_year:
            stats["below_min_year"] += 1
            continue

        if has_duplicate_in_index(paper.title, paper.doi, existing_index):
            stats["duplicates_removed"] += 1
            continue

        if has_duplicate_in_index(paper.title, paper.doi, batch_index):
            stats["duplicates_removed"] += 1
            continue

        add_to_dedup_index(batch_index, paper.title, paper.doi)
        unique_papers.append(paper)

    console.print(f"[bold]After deduplication:[/bold] {len(unique_papers)} papers")

    console.print("[dim]Scoring and filtering...[/dim]")
    scored_papers: list[Paper] = []
    for paper in unique_papers:
        paper.source_trusted = is_from_trusted_source(paper)
        paper = score_paper(paper)
        paper.categories = suggest_categories(paper)
        paper.covers_gap = check_gap_coverage(paper)

        if strict_source_filter:
            if not paper.source_trusted:
                stats["untrusted_removed"] += 1
                continue
        elif not paper.source_trusted and paper.relevance_score < untrusted_keep_threshold and not paper.covers_gap:
            stats["untrusted_removed"] += 1
            continue

        if paper.relevance_score < min_keep_score and not paper.covers_gap:
            stats["low_score_removed"] += 1
            continue

        if paper.source_trusted:
            stats["trusted_kept"] += 1
        else:
            stats["provisional_kept"] += 1

        scored_papers.append(paper)

    scored_papers.sort(
        key=lambda paper: (
            ranking_score(paper, config),
            int(bool(paper.source_trusted)),
            paper.citation_count,
            paper.year or 0,
        ),
        reverse=True,
    )

    high = [paper for paper in scored_papers if paper.relevance_level.value == "ALTA"]
    medium = [paper for paper in scored_papers if paper.relevance_level.value == "MEDIA"]
    low = [paper for paper in scored_papers if paper.relevance_level.value == "BAJA"]

    table = Table(title="Search Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total raw results", str(len(all_papers)))
    table.add_row("Semantic Scholar", str(stats["semantic_scholar_results"]))
    table.add_row("arXiv", str(stats["arxiv_results"]))
    table.add_row("OpenAlex", str(stats["openalex_results"]))
    table.add_row("Query mismatches removed", str(stats["query_mismatch_removed"]))
    table.add_row("Duplicates removed", str(stats["duplicates_removed"]))
    table.add_row("Low score removed", str(stats["low_score_removed"]))
    table.add_row("Untrusted removed", str(stats["untrusted_removed"]))
    table.add_row("Trusted kept", str(stats["trusted_kept"]))
    table.add_row("Provisional kept", str(stats["provisional_kept"]))
    table.add_row("Below min year removed", str(stats["below_min_year"]))
    table.add_row("Final papers", str(len(scored_papers)))
    table.add_row("HIGH relevance", str(len(high)))
    table.add_row("MEDIUM relevance", str(len(medium)))
    table.add_row("LOW relevance", str(len(low)))
    console.print(table)

    gap_papers = [paper for paper in scored_papers if paper.covers_gap]
    if gap_papers:
        console.print("\n[bold green]Papers covering gaps:[/bold green]")
        for paper in gap_papers:
            console.print(f"  - [bold]{paper.covers_gap}[/bold]: {paper.title}")

    if not dry_run:
        write_daily_report(scored_papers, stats, config)
    else:
        console.print("\n[yellow]Dry run - no files written.[/yellow]")

    return scored_papers


def write_daily_report(papers: list[Paper], stats: dict[str, int], config: dict) -> None:
    """Write the daily report in Markdown plus a structured JSON mirror."""
    today = date.today().isoformat()
    output_dir = Path(config.get("output", {}).get("daily_dir", "output/daily"))
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{today}_daily_papers.md"
    tmp_filepath = output_dir / f"{today}_daily_papers.md.tmp"

    high = [paper for paper in papers if paper.relevance_level.value == "ALTA"]
    medium = [paper for paper in papers if paper.relevance_level.value == "MEDIA"]
    low = [paper for paper in papers if paper.relevance_level.value == "BAJA"]
    gap_papers = [paper for paper in papers if paper.covers_gap]

    lines: list[str] = []
    lines.append(f"# Reporte Diario de Papers -- {today}")
    lines.append("")
    lines.append("## Resumen")
    lines.append(f"- Papers encontrados: {len(papers)}")
    lines.append(f"- Papers de alta relevancia: {len(high)}")
    lines.append(f"- Papers de media relevancia: {len(medium)}")
    lines.append(f"- Papers de baja relevancia: {len(low)}")
    lines.append(f"- Papers de fuente confiable: {stats.get('trusted_kept', 0)}")
    lines.append(f"- Papers provisionales conservados: {stats.get('provisional_kept', 0)}")
    lines.append(f"- Resultados descartados por no parecer responder a la query: {stats.get('query_mismatch_removed', 0)}")
    lines.append(f"- Papers descartados por score demasiado bajo: {stats.get('low_score_removed', 0)}")
    lines.append(f"- APIs consultadas: {stats.get('total_queries', 0)}/{stats.get('expected_queries', 0)} llamadas")
    lines.append("")

    lines.append("## Gaps cubiertos hoy")
    if gap_papers:
        for paper in gap_papers:
            lines.append(f"- [x] {paper.covers_gap} -- {paper.title}")
    else:
        lines.append("- Ninguno nuevo cubierto hoy")
    lines.append("")

    lines.append("## Top 5 Papers del dia")
    for index, paper in enumerate(papers[:5], 1):
        trust_flag = "confiable" if paper.source_trusted else "provisional"
        lines.append(f"{index}. [{paper.relevance_level.value}] {paper.title} (Score: {paper.relevance_score}/100, prioridad: {ranking_score(paper, config)}, fuente: {trust_flag})")
    lines.append("")
    lines.append("---")
    lines.append("")

    all_categories = [
        "Sistemas RAG Hibridos",
        "Normalizacion/Preprocesamiento",
        "Segmentacion/Chunking",
        "Modelos de Embedding",
        "Estrategias de Recuperacion",
        "Re-ranking",
        "Alucinaciones en LLMs",
        "Evaluacion de sistemas RAG",
        "Documentacion Cloud",
        "Metricas de evaluacion",
        "Vector Databases",
    ]

    for paper in papers:
        lines.append(f"## [{paper.relevance_level.value}] {paper.title}")
        lines.append(f"- **Autores:** {', '.join(paper.authors) if paper.authors else 'N/A'}")
        lines.append(f"- **Fecha:** {paper.publication_date or 'N/A'}")
        lines.append(f"- **Fuente:** {paper.venue or 'N/A'}")
        lines.append(f"- **Fuente confiable:** {'Si' if paper.source_trusted else 'No'}")
        doi_str = f"https://doi.org/{paper.doi}" if paper.doi else "N/A"
        lines.append(f"- **DOI:** {doi_str}")
        lines.append(f"- **Link:** {paper.url or 'N/A'}")
        lines.append(f"- **Citaciones:** {paper.citation_count}")
        lines.append(f"- **Score de relevancia:** {paper.relevance_score}/100")
        lines.append(f"- **Prioridad de ranking:** {ranking_score(paper, config)}")
        lines.append(f"- **Abstract:** {paper.truncated_abstract(200)}")
        lines.append(f"- **Keywords match:** {paper.keywords_matched}")
        lines.append("- **Categoria sugerida para mi tesis:**")
        for category in all_categories:
            check = "x" if category in paper.categories else " "
            lines.append(f"  - [{check}] {category}")
        gap_text = f"Si -- {paper.covers_gap}" if paper.covers_gap else "No"
        lines.append(f"- **Cubre algun gap pendiente?** {gap_text}")
        lines.append(f"- **Notas:** Paper encontrado via {paper.source_api}")
        lines.append("")

    content = "\n".join(lines)
    with open(tmp_filepath, "w", encoding="utf-8") as f:
        f.write(content)

    if filepath.exists():
        filepath.unlink()
    tmp_filepath.rename(filepath)

    _write_structured_daily_report(papers, stats, output_dir, today)

    console.print(f"\n[bold green]Report written:[/bold green] {filepath}")
    logger.info(f"Daily report written to {filepath}")






