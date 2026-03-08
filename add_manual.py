#!/usr/bin/env python3
"""Add a paper manually to the database by DOI or metadata.

Usage:
    python add_manual.py --doi "10.1145/3397271.3401075"
    python add_manual.py --title "Paper Title" --authors "Author1, Author2" --year 2024 --venue "IEEE"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

from src.agents.paper_compiler import load_config, load_database, save_database
from src.apis.crossref_api import CrossRefAPI
from src.apis.openalex_api import OpenAlexAPI
from src.apis.semantic_scholar import SemanticScholarAPI
from src.models.paper import Paper, PaperStatus
from src.utils.duplicate_detector import is_duplicate_by_doi, is_duplicate_by_title
from src.utils.logger import setup_logger
from src.utils.reference_formatter import format_apa7, format_bibtex
from src.utils.relevance_scorer import check_gap_coverage, score_paper, suggest_categories

logger = setup_logger("add_manual")
console = Console()


def add_by_doi(doi: str) -> Paper | None:
    """Fetch a paper by DOI and add it to the database."""
    console.print(f"[dim]Fetching paper with DOI: {doi}[/dim]")

    paper = None

    # Try Semantic Scholar first
    try:
        api = SemanticScholarAPI()
        paper = api.get_paper_by_doi(doi)
        if paper:
            console.print("[green]Found in Semantic Scholar[/green]")
    except Exception as e:
        logger.warning(f"Semantic Scholar error: {e}")

    # Try CrossRef as fallback
    if not paper:
        try:
            api = CrossRefAPI()
            paper = api.get_paper_by_doi(doi)
            if paper:
                console.print("[green]Found in CrossRef[/green]")
        except Exception as e:
            logger.warning(f"CrossRef error: {e}")

    # Try OpenAlex
    if not paper:
        try:
            api = OpenAlexAPI()
            paper = api.get_paper_by_doi(doi)
            if paper:
                console.print("[green]Found in OpenAlex[/green]")
        except Exception as e:
            logger.warning(f"OpenAlex error: {e}")

    if not paper:
        console.print("[red]Paper not found in any API.[/red]")
        return None

    paper.doi = doi
    return paper


def add_by_metadata(
    title: str,
    authors: list[str],
    year: int | None = None,
    venue: str | None = None,
    doi: str | None = None,
) -> Paper:
    """Create a paper from manual metadata."""
    return Paper(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        source_api="manual",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a paper manually to the database.")
    parser.add_argument("--doi", type=str, help="DOI to fetch paper metadata")
    parser.add_argument("--title", type=str, help="Paper title")
    parser.add_argument("--authors", type=str, help="Comma-separated author names")
    parser.add_argument("--year", type=int, help="Publication year")
    parser.add_argument("--venue", type=str, help="Venue/journal name")

    args = parser.parse_args()

    if not args.doi and not args.title:
        parser.print_help()
        sys.exit(1)

    config = load_config()
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    paper: Paper | None = None

    if args.doi:
        # Check for duplicates first
        if is_duplicate_by_doi(args.doi, papers):
            console.print("[yellow]This paper (DOI) is already in the database.[/yellow]")
            return

        paper = add_by_doi(args.doi)
    else:
        if args.title and is_duplicate_by_title(args.title, papers):
            console.print("[yellow]A paper with a similar title is already in the database.[/yellow]")
            return

        authors = [a.strip() for a in args.authors.split(",")] if args.authors else []
        paper = add_by_metadata(
            title=args.title,  # type: ignore
            authors=authors,
            year=args.year,
            venue=args.venue,
            doi=args.doi,
        )

    if paper is None:
        return

    # Score and categorize
    paper = score_paper(paper)
    paper.categories = suggest_categories(paper)
    paper.covers_gap = check_gap_coverage(paper)
    paper.apa7_reference = format_apa7(paper)
    paper.bibtex = format_bibtex(paper)
    paper.status = PaperStatus.ACCEPTED

    papers.append(paper)
    save_database(papers, db_path)

    console.print("\n[bold green]Paper added successfully![/bold green]")
    console.print(f"Title: {paper.title}")
    console.print(f"Authors: {', '.join(paper.authors[:5])}")
    console.print(f"Year: {paper.year}")
    console.print(f"Score: {paper.relevance_score}/100 [{paper.relevance_level.value}]")
    console.print(f"Categories: {', '.join(paper.categories)}")
    if paper.covers_gap:
        console.print(f"[green]Covers gap: {paper.covers_gap}[/green]")
    console.print(f"\nAPA 7: {paper.apa7_reference}")


if __name__ == "__main__":
    main()
