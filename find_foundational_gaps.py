#!/usr/bin/env python3
"""Find and add foundational gap papers to the database.

Searches for specific foundational papers by DOI (primary) and title (fallback)
using Semantic Scholar and CrossRef APIs, then adds them to the database with
correct categories and gap coverage.

Usage:
    python find_foundational_gaps.py
"""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table

from src.apis.crossref_api import CrossRefAPI
from src.apis.semantic_scholar import SemanticScholarAPI
from src.models.paper import Paper, RelevanceLevel
from src.utils.duplicate_detector import is_duplicate_by_doi, is_duplicate_by_title
from src.utils.logger import setup_logger
from src.utils.reference_formatter import format_apa7, format_bibtex
from src.utils.relevance_scorer import score_paper, suggest_categories

logger = setup_logger("find_foundational_gaps")
console = Console()

# ── Foundational papers to find ──────────────────────────────────────────────

FOUNDATIONAL_PAPERS = [
    {
        "name": "ColBERT original",
        "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
        "doi": "10.1145/3397271.3401075",
        "gap_name": "ColBERT original",
        "categories": ["Estrategias de Recuperacion", "Modelos de Embedding"],
    },
    {
        "name": "DPR original",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "doi": "10.18653/v1/2020.emnlp-main.550",
        "gap_name": "DPR original",
        "categories": ["Estrategias de Recuperacion", "Modelos de Embedding"],
    },
    {
        "name": "Sentence-BERT original",
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "doi": "10.18653/v1/D19-1410",
        "gap_name": "Sentence-BERT original",
        "categories": ["Modelos de Embedding"],
    },
    {
        "name": "BGE (C-Pack)",
        "title": "C-Pack: Packed Resources For General Chinese Embeddings",
        "doi": "10.1145/3626772.3657878",
        "gap_name": "BGE embeddings",
        "categories": ["Modelos de Embedding"],
    },
    {
        "name": "BM25 fundacional",
        "title": "The Probabilistic Relevance Framework: BM25 and Beyond",
        "doi": "10.1561/1500000019",
        "gap_name": "BM25 fundacional",
        "categories": ["Estrategias de Recuperacion"],
    },
    {
        "name": "RAGAS",
        "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "doi": "10.18653/v1/2024.eacl-demo.16",
        "gap_name": "RAGAS evaluation framework",
        "categories": ["Evaluacion de sistemas RAG", "Metricas de evaluacion"],
    },
]

# Broader search queries for gaps without specific DOIs
SEARCH_QUERIES = [
    {
        "gap_name": "Normalizacion terminologica en dominios tecnicos",
        "queries": [
            "terminology normalization technical documentation",
            "domain-specific terminology normalization NLP",
        ],
        "categories": ["Normalizacion/Preprocesamiento"],
    },
    {
        "gap_name": "Documentacion cloud como objeto de estudio",
        "queries": [
            "cloud documentation usability analysis",
            "technical documentation complexity analysis cloud",
        ],
        "categories": ["Documentacion Cloud"],
    },
]


def load_database(path: str) -> list[Paper]:
    db_path = Path(path)
    if not db_path.exists():
        return []
    with open(db_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Paper(**item) for item in data]


def save_database(papers: list[Paper], path: str) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in papers], f, ensure_ascii=False, indent=2)
    if db_path.exists():
        db_path.unlink()
    tmp_path.rename(db_path)


def fetch_by_doi(doi: str, ss: SemanticScholarAPI, cr: CrossRefAPI) -> Paper | None:
    """Try to fetch a paper by DOI: Semantic Scholar first, then CrossRef."""
    console.print(f"  Buscando por DOI: [cyan]{doi}[/cyan]")

    paper = ss.get_paper_by_doi(doi)
    if paper:
        console.print(f"    [green]Encontrado en Semantic Scholar[/green]")
        return paper

    paper = cr.get_paper_by_doi(doi)
    if paper:
        console.print(f"    [green]Encontrado en CrossRef[/green]")
        return paper

    console.print(f"    [yellow]No encontrado por DOI[/yellow]")
    return None


def fetch_by_title(title: str, ss: SemanticScholarAPI, cr: CrossRefAPI) -> Paper | None:
    """Try to fetch a paper by title: Semantic Scholar first, then CrossRef."""
    console.print(f"  Buscando por titulo: [cyan]{title[:60]}...[/cyan]")

    paper = ss.get_paper_by_title(title)
    if paper:
        console.print(f"    [green]Encontrado en Semantic Scholar[/green]")
        return paper

    results = cr.search(title, limit=5)
    if results:
        title_lower = title.lower().strip()
        for r in results:
            if r.title.lower().strip() == title_lower:
                console.print(f"    [green]Encontrado en CrossRef (titulo exacto)[/green]")
                return r
        # Fallback: best match
        console.print(f"    [green]Encontrado en CrossRef (mejor match)[/green]")
        return results[0]

    console.print(f"    [yellow]No encontrado por titulo[/yellow]")
    return None


def search_for_gap(queries: list[str], ss: SemanticScholarAPI, cr: CrossRefAPI) -> list[Paper]:
    """Search for papers matching gap-related queries."""
    found: list[Paper] = []
    seen_titles: set[str] = set()

    for query in queries:
        console.print(f"  Buscando: [cyan]{query}[/cyan]")

        # Semantic Scholar
        results = ss.search(query, limit=5)
        for p in results:
            norm = p.title.lower().strip()
            if norm not in seen_titles:
                seen_titles.add(norm)
                found.append(p)

        # CrossRef
        results = cr.search(query, limit=5)
        for p in results:
            norm = p.title.lower().strip()
            if norm not in seen_titles:
                seen_titles.add(norm)
                found.append(p)

    return found


def main() -> None:
    console.print("\n[bold cyan]===== Buscador de Gaps Fundacionales =====[/bold cyan]\n")

    db_path = "data/papers_database.json"
    papers = load_database(db_path)
    console.print(f"Papers en base de datos: {len(papers)}\n")

    ss = SemanticScholarAPI()
    cr = CrossRefAPI()

    added_papers: list[Paper] = []
    skipped: list[str] = []
    failed: list[str] = []

    # ── Phase 1: Foundational papers with known DOIs ─────────────────────────

    console.print("[bold]Fase 1: Buscando papers fundacionales por DOI/titulo[/bold]\n")

    for entry in FOUNDATIONAL_PAPERS:
        console.print(f"[bold yellow]{entry['name']}[/bold yellow]")

        # Check if already in DB
        if entry["doi"] and is_duplicate_by_doi(entry["doi"], papers):
            console.print(f"  [dim]Ya existe en la base de datos (DOI match)[/dim]")
            # Update gap coverage on existing paper
            for p in papers:
                if p.doi and p.doi.lower() == entry["doi"].lower():
                    if not p.covers_gap:
                        p.covers_gap = entry["gap_name"]
                        console.print(f"  [green]Actualizado covers_gap: {entry['gap_name']}[/green]")
                    break
            skipped.append(entry["name"])
            continue

        if is_duplicate_by_title(entry["title"], papers):
            console.print(f"  [dim]Ya existe en la base de datos (titulo match)[/dim]")
            skipped.append(entry["name"])
            continue

        # Try DOI first, then title
        paper = None
        if entry["doi"]:
            paper = fetch_by_doi(entry["doi"], ss, cr)
        if not paper:
            paper = fetch_by_title(entry["title"], ss, cr)

        if paper:
            # Set correct metadata
            paper.categories = entry["categories"]
            paper.covers_gap = entry["gap_name"]
            paper.date_found = date.today().isoformat()
            if entry["doi"] and not paper.doi:
                paper.doi = entry["doi"]

            # Score and generate references
            paper = score_paper(paper)
            # Foundational papers get a minimum score of 80
            if paper.relevance_score < 80:
                paper.relevance_score = 80
                paper.relevance_level = RelevanceLevel.HIGH
            paper.apa7_reference = format_apa7(paper)
            paper.bibtex = format_bibtex(paper)
            paper.notes = "Paper fundacional agregado automaticamente"

            papers.append(paper)
            added_papers.append(paper)
            console.print(f"  [green]Agregado a la base de datos (score: {paper.relevance_score})[/green]")
        else:
            failed.append(entry["name"])
            console.print(f"  [red]No se pudo encontrar[/red]")

        console.print()

    # ── Phase 2: Broader searches for remaining gaps ─────────────────────────

    console.print("\n[bold]Fase 2: Buscando papers para gaps tematicos[/bold]\n")

    for entry in SEARCH_QUERIES:
        console.print(f"[bold yellow]Gap: {entry['gap_name']}[/bold yellow]")

        results = search_for_gap(entry["queries"], ss, cr)
        gap_added = 0

        for paper in results:
            if paper.doi and is_duplicate_by_doi(paper.doi, papers):
                continue
            if is_duplicate_by_title(paper.title, papers):
                continue

            paper.categories = entry["categories"]
            paper.covers_gap = entry["gap_name"]
            paper.date_found = date.today().isoformat()

            paper = score_paper(paper)
            paper.apa7_reference = format_apa7(paper)
            paper.bibtex = format_bibtex(paper)
            paper.notes = f"Encontrado via busqueda de gap: {entry['gap_name']}"

            papers.append(paper)
            added_papers.append(paper)
            gap_added += 1

            if gap_added >= 3:
                break

        console.print(f"  [green]Agregados {gap_added} papers para este gap[/green]\n")

    # ── Save ─────────────────────────────────────────────────────────────────

    save_database(papers, db_path)

    # ── Summary ──────────────────────────────────────────────────────────────

    console.print("\n[bold cyan]===== Resumen =====[/bold cyan]\n")

    table = Table(title="Papers Fundacionales Agregados")
    table.add_column("Paper", style="cyan", max_width=60)
    table.add_column("DOI", style="dim", max_width=30)
    table.add_column("Gap Cubierto", style="green")
    table.add_column("Score", style="yellow", justify="right")

    for p in added_papers:
        table.add_row(
            p.title[:60],
            p.doi or "N/A",
            p.covers_gap or "N/A",
            str(p.relevance_score),
        )

    console.print(table)
    console.print(f"\n[green]Total agregados: {len(added_papers)}[/green]")
    if skipped:
        console.print(f"[dim]Omitidos (ya existian): {', '.join(skipped)}[/dim]")
    if failed:
        console.print(f"[red]No encontrados: {', '.join(failed)}[/red]")

    console.print(f"\nTotal papers en DB: {len(papers)}")


if __name__ == "__main__":
    main()
