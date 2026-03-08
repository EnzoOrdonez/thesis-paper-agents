#!/usr/bin/env python3
"""Entry point for Agent 2: Paper Compiler.

Usage:
    python paper_compiler.py                  # Full compilation pipeline
    python paper_compiler.py --review         # Interactive review of new papers
    python paper_compiler.py --stats          # Show statistics
    python paper_compiler.py --export-apa     # Export APA 7 references
    python paper_compiler.py --export-bibtex  # Export BibTeX
    python paper_compiler.py --gap-analysis   # Generate gap analysis
    python paper_compiler.py --validate-dois  # Run the CrossRef DOI batch only
    python paper_compiler.py --check-scopus   # Run the OpenAlex Scopus batch only
    python paper_compiler.py --metadata-status # Show metadata provider runtime status
    python paper_compiler.py --sqlite-sync    # Sync SQLite storage
    python paper_compiler.py --sqlite-status  # Show SQLite storage status
    python paper_compiler.py --cleanup        # Remove low-quality papers
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

from src.agents.paper_compiler import (
    check_scopus_indexing,
    compile_all,
    export_apa7,
    export_bibtex,
    generate_gap_analysis_report,
    interactive_review,
    load_config,
    load_database,
    save_database,
    show_metadata_status,
    show_sqlite_status,
    show_statistics,
    sync_sqlite_mirror,
    validate_dois,
)
from src.models.paper import Paper

console = Console()


def cleanup_database(config: dict) -> None:
    """Remove low-quality papers from the database.

    Removes:
    - Papers with relevance score < 20
    - Papers with no DOI AND no abstract AND no venue
    """
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)
    original_count = len(papers)

    low_score = 0
    no_metadata = 0
    cleaned: list[Paper] = []

    for paper in papers:
        if paper.covers_gap:
            cleaned.append(paper)
            continue

        if paper.relevance_score < 20:
            low_score += 1
            continue

        if not paper.doi and not paper.abstract and not paper.venue:
            no_metadata += 1
            continue

        cleaned.append(paper)

    removed = original_count - len(cleaned)
    if removed > 0:
        save_database(cleaned, db_path)

    console.print("\n[bold cyan]Limpieza de base de datos[/bold cyan]")
    console.print(f"  Papers antes: {original_count}")
    console.print(f"  Eliminados por score < 20: [red]{low_score}[/red]")
    console.print(f"  Eliminados sin DOI/abstract/venue: [red]{no_metadata}[/red]")
    console.print(f"  Papers despues: [green]{len(cleaned)}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper Compiler Agent - Maintain database, validate, and generate reports."
    )
    parser.add_argument("--review", action="store_true", help="Interactive review of new papers")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--export-apa", action="store_true", help="Export APA 7 references")
    parser.add_argument("--export-bibtex", action="store_true", help="Export BibTeX file")
    parser.add_argument("--gap-analysis", action="store_true", help="Generate gap analysis report")
    parser.add_argument("--validate-dois", action="store_true", help="Run the CrossRef DOI validation batch")
    parser.add_argument("--check-scopus", action="store_true", help="Run the OpenAlex Scopus batch")
    parser.add_argument("--metadata-status", action="store_true", help="Show metadata provider runtime status")
    parser.add_argument("--sqlite-sync", action="store_true", help="Synchronize the SQLite storage")
    parser.add_argument("--sqlite-status", action="store_true", help="Show SQLite storage status")
    parser.add_argument("--cleanup", action="store_true", help="Remove low-quality papers")

    args = parser.parse_args()
    config = load_config()

    if args.review:
        interactive_review(config)
    elif args.stats:
        show_statistics(config)
    elif args.export_apa:
        export_apa7(config)
    elif args.export_bibtex:
        export_bibtex(config)
    elif args.gap_analysis:
        generate_gap_analysis_report(config)
    elif args.validate_dois:
        validate_dois(config)
    elif args.check_scopus:
        check_scopus_indexing(config)
    elif args.metadata_status:
        show_metadata_status(config)
    elif args.sqlite_sync:
        sync_sqlite_mirror(config)
    elif args.sqlite_status:
        show_sqlite_status(config)
    elif args.cleanup:
        cleanup_database(config)
    else:
        compile_all(config)


if __name__ == "__main__":
    main()
