#!/usr/bin/env python3
"""Search for a specific paper by title, author, or DOI.

Usage:
    python search_specific.py --title "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
    python search_specific.py --author "Khattab" --keyword "ColBERT"
    python search_specific.py --doi "10.1145/3397271.3401075"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

from src.apis.crossref_api import CrossRefAPI
from src.apis.openalex_api import OpenAlexAPI
from src.apis.semantic_scholar import SemanticScholarAPI
from src.models.paper import Paper
from src.utils.reference_formatter import format_apa7
from src.utils.relevance_scorer import check_gap_coverage, score_paper, suggest_categories

console = Console()


def search_by_doi(doi: str) -> list[Paper]:
    """Search all APIs for a paper by DOI."""
    papers: list[Paper] = []

    console.print(f"[dim]Searching by DOI: {doi}[/dim]")

    # Semantic Scholar
    try:
        api = SemanticScholarAPI()
        paper = api.get_paper_by_doi(doi)
        if paper:
            papers.append(paper)
            console.print("[green]Found in Semantic Scholar[/green]")
    except Exception as e:
        console.print(f"[red]Semantic Scholar error: {e}[/red]")

    # OpenAlex
    try:
        api = OpenAlexAPI()
        paper = api.get_paper_by_doi(doi)
        if paper:
            papers.append(paper)
            console.print("[green]Found in OpenAlex[/green]")
    except Exception as e:
        console.print(f"[red]OpenAlex error: {e}[/red]")

    # CrossRef
    try:
        api = CrossRefAPI()
        paper = api.get_paper_by_doi(doi)
        if paper:
            papers.append(paper)
            console.print("[green]Found in CrossRef[/green]")
    except Exception as e:
        console.print(f"[red]CrossRef error: {e}[/red]")

    return papers


def search_by_title(title: str) -> list[Paper]:
    """Search all APIs for a paper by title."""
    papers: list[Paper] = []

    console.print(f"[dim]Searching by title: {title}[/dim]")

    try:
        api = SemanticScholarAPI()
        results = api.search(title, limit=5)
        papers.extend(results)
        console.print(f"[green]Semantic Scholar: {len(results)} results[/green]")
    except Exception as e:
        console.print(f"[red]Semantic Scholar error: {e}[/red]")

    try:
        api = OpenAlexAPI()
        results = api.search(title, limit=5)
        papers.extend(results)
        console.print(f"[green]OpenAlex: {len(results)} results[/green]")
    except Exception as e:
        console.print(f"[red]OpenAlex error: {e}[/red]")

    try:
        api = CrossRefAPI()
        results = api.search(title, limit=5)
        papers.extend(results)
        console.print(f"[green]CrossRef: {len(results)} results[/green]")
    except Exception as e:
        console.print(f"[red]CrossRef error: {e}[/red]")

    return papers


def search_by_author_keyword(author: str, keyword: str | None = None) -> list[Paper]:
    """Search by author name, optionally filtered by keyword."""
    query = author
    if keyword:
        query = f"{author} {keyword}"

    console.print(f"[dim]Searching by author/keyword: {query}[/dim]")

    papers: list[Paper] = []

    try:
        api = SemanticScholarAPI()
        results = api.search(query, limit=10)
        papers.extend(results)
        console.print(f"[green]Semantic Scholar: {len(results)} results[/green]")
    except Exception as e:
        console.print(f"[red]Semantic Scholar error: {e}[/red]")

    try:
        api = OpenAlexAPI()
        results = api.search(query, limit=10)
        papers.extend(results)
        console.print(f"[green]OpenAlex: {len(results)} results[/green]")
    except Exception as e:
        console.print(f"[red]OpenAlex error: {e}[/red]")

    return papers


def display_results(papers: list[Paper]) -> None:
    """Display search results in a formatted table."""
    if not papers:
        console.print("[yellow]No papers found.[/yellow]")
        return

    # Deduplicate by title similarity
    seen_titles: set[str] = set()
    unique: list[Paper] = []
    for p in papers:
        norm = p.normalized_title()
        if norm not in seen_titles:
            seen_titles.add(norm)
            unique.append(p)

    console.print(f"\n[bold]Found {len(unique)} unique results:[/bold]\n")

    for i, paper in enumerate(unique, 1):
        paper = score_paper(paper)
        paper.categories = suggest_categories(paper)
        paper.covers_gap = check_gap_coverage(paper)

        console.print(f"[bold cyan]--- Result {i} ---[/bold cyan]")
        console.print(f"[bold]{paper.title}[/bold]")
        console.print(f"Authors: {', '.join(paper.authors[:5])}")
        console.print(f"Year: {paper.year} | Venue: {paper.venue}")
        console.print(f"DOI: {paper.doi or 'N/A'}")
        console.print(f"Citations: {paper.citation_count}")
        console.print(f"Score: {paper.relevance_score}/100 [{paper.relevance_level.value}]")
        console.print(f"Categories: {', '.join(paper.categories)}")
        if paper.covers_gap:
            console.print(f"[green]Covers gap: {paper.covers_gap}[/green]")
        console.print(f"Source API: {paper.source_api}")
        console.print(f"URL: {paper.url or 'N/A'}")
        console.print()

        # Show APA7 reference
        apa = format_apa7(paper)
        console.print(f"[dim]APA 7: {apa}[/dim]")
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for a specific paper.")
    parser.add_argument("--title", type=str, help="Search by title")
    parser.add_argument("--author", type=str, help="Search by author name")
    parser.add_argument("--keyword", type=str, help="Additional keyword filter (use with --author)")
    parser.add_argument("--doi", type=str, help="Search by DOI")

    args = parser.parse_args()

    if not any([args.title, args.author, args.doi]):
        parser.print_help()
        sys.exit(1)

    papers: list[Paper] = []

    if args.doi:
        papers = search_by_doi(args.doi)
    elif args.title:
        papers = search_by_title(args.title)
    elif args.author:
        papers = search_by_author_keyword(args.author, args.keyword)

    display_results(papers)


if __name__ == "__main__":
    main()
