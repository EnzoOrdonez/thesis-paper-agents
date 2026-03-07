#!/usr/bin/env python3
"""Enhanced interactive paper review with rich UI.

Usage:
    python review_papers.py                          # Review all pending
    python review_papers.py --high-only              # Only alta relevancia
    python review_papers.py --category "Embeddings"  # Only one category
    python review_papers.py --unreviewed             # Only status=new
"""

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.agents.paper_compiler import load_config, load_database as load_papers_database, save_database as save_papers_database
from src.models.paper import Paper, PaperStatus, RelevanceLevel

console = Console()

CONFIG = load_config()
DB_PATH = CONFIG["output"]["database_path"]
THESIS_CATEGORIES = [
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


def load_database() -> list[Paper]:
    return load_papers_database(DB_PATH)



def save_database(papers: list[Paper]) -> None:
    save_papers_database(papers, DB_PATH)


def display_paper(paper: Paper, index: int, total: int) -> None:
    """Display a paper with rich formatting."""
    # Relevance badge
    if paper.relevance_level == RelevanceLevel.HIGH:
        level_str = "[bold green]ALTA[/bold green]"
    elif paper.relevance_level == RelevanceLevel.MEDIUM:
        level_str = "[bold yellow]MEDIA[/bold yellow]"
    else:
        level_str = "[dim]BAJA[/dim]"

    console.print()
    console.print(Rule(style="cyan"))
    console.print(
        f"  [bold cyan]Paper {index}/{total}[/bold cyan] - "
        f"Score: [bold yellow]{paper.relevance_score}/100[/bold yellow] [{level_str}]"
    )
    console.print(Rule(style="cyan"))

    console.print(f"\n  [bold]{paper.title}[/bold]")

    # Authors
    authors_str = ", ".join(paper.authors[:5])
    if len(paper.authors) > 5:
        authors_str += f" (+{len(paper.authors) - 5} mas)"
    console.print(f"  [dim]Autores:[/dim] {authors_str}")

    # Metadata line
    year_str = str(paper.year) if paper.year else "N/A"
    venue_str = paper.venue or "N/A"
    console.print(f"  [dim]Ano:[/dim] {year_str}  |  [dim]Venue:[/dim] {venue_str}")

    # DOI
    if paper.doi:
        console.print(f"  [dim]DOI:[/dim] https://doi.org/{paper.doi}")

    # Categories
    cats_str = ", ".join(paper.categories) if paper.categories else "Sin categoria"
    console.print(f"  [dim]Categorias:[/dim] [cyan]{cats_str}[/cyan]")

    # Gap coverage
    if paper.covers_gap:
        console.print(f"  [green]Cubre gap: {paper.covers_gap}[/green]")

    # Citations
    console.print(f"  [dim]Citaciones:[/dim] {paper.citation_count}")

    # Abstract
    if paper.abstract:
        abstract_text = paper.truncated_abstract(100)
        console.print(f"\n  [dim]Abstract:[/dim] {abstract_text}")

    console.print()
    console.print(Rule(style="dim"))


def ask_category_assignment() -> list[str] | None:
    """Ask user which thesis section(s) this paper belongs to."""
    console.print("\n  [bold]En que seccion(es) de la tesis lo usarias?[/bold]")
    for i, cat in enumerate(THESIS_CATEGORIES, 1):
        console.print(f"    {i:2d}. {cat}")
    console.print(f"     0. Mantener categorias actuales")

    choice = Prompt.ask("  Secciones (numeros separados por coma, 0=skip)", default="0")
    if choice.strip() == "0":
        return None

    selected: list[str] = []
    for num in choice.split(","):
        try:
            idx = int(num.strip())
            if 1 <= idx <= len(THESIS_CATEGORIES):
                selected.append(THESIS_CATEGORIES[idx - 1])
        except ValueError:
            pass

    return selected if selected else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced paper review")
    parser.add_argument("--high-only", action="store_true", help="Only high relevance papers")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (partial match)")
    parser.add_argument("--unreviewed", action="store_true", help="Only unreviewed (status=new)")

    args = parser.parse_args()

    papers = load_database()
    if not papers:
        console.print("[yellow]Base de datos vacia.[/yellow]")
        return

    # Filter papers
    to_review = papers.copy()

    if args.unreviewed:
        to_review = [p for p in to_review if p.status == PaperStatus.NEW]
    else:
        # Default: show new papers
        to_review = [p for p in to_review if p.status == PaperStatus.NEW]

    if args.high_only:
        to_review = [p for p in to_review if p.relevance_level == RelevanceLevel.HIGH]

    if args.category:
        cat_filter = args.category.lower()
        to_review = [
            p for p in to_review
            if any(cat_filter in c.lower() for c in p.categories)
        ]

    # Sort: high relevance first, then by score descending
    to_review.sort(key=lambda p: p.relevance_score, reverse=True)

    if not to_review:
        console.print("[green]No hay papers pendientes de revision con los filtros aplicados.[/green]")
        return

    console.print(Panel.fit(
        f"[bold cyan]Revision de Papers - {len(to_review)} pendientes[/bold cyan]",
        border_style="cyan",
    ))

    # Session stats
    accepted = 0
    rejected = 0
    skipped = 0

    for i, paper in enumerate(to_review, 1):
        display_paper(paper, i, len(to_review))

        # Action prompt
        console.print("  [bold][a][/bold] Aceptar  "
                       "[bold][r][/bold] Rechazar  "
                       "[bold][s][/bold] Skip  "
                       "[bold][n][/bold] Notas  "
                       "[bold][q][/bold] Quit")

        choice = Prompt.ask("  Accion", choices=["a", "r", "s", "n", "q"], default="s")

        if choice == "a":
            paper.status = PaperStatus.ACCEPTED
            accepted += 1

            # Ask for thesis section
            new_cats = ask_category_assignment()
            if new_cats:
                paper.categories = new_cats

            # Optional notes
            notes = Prompt.ask("  Notas (Enter para skip)", default="")
            if notes:
                paper.notes = notes

            console.print("  [green]Aceptado[/green]")

        elif choice == "r":
            paper.status = PaperStatus.REJECTED
            rejected += 1
            notes = Prompt.ask("  Razon de rechazo (Enter para skip)", default="")
            if notes:
                paper.notes = notes
            console.print("  [red]Rechazado[/red]")

        elif choice == "n":
            notes = Prompt.ask("  Notas")
            if notes:
                paper.notes = notes
            console.print("  [yellow]Nota guardada, paper sigue pendiente[/yellow]")
            skipped += 1

        elif choice == "q":
            console.print("\n  [yellow]Sesion terminada.[/yellow]")
            break

        else:
            skipped += 1

        # Save after each action
        save_database(papers)

    # Session summary
    remaining = sum(1 for p in papers if p.status == PaperStatus.NEW)

    console.print()
    console.print(Rule(title="Resumen de Sesion", style="cyan"))

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column(style="cyan")
    summary_table.add_column(style="green", justify="right")
    summary_table.add_row("Papers revisados hoy:", str(accepted + rejected + skipped))
    summary_table.add_row("Aceptados:", str(accepted))
    summary_table.add_row("Rechazados:", str(rejected))
    summary_table.add_row("Saltados:", str(skipped))
    summary_table.add_row("Pendientes restantes:", str(remaining))
    console.print(summary_table)
    console.print()


if __name__ == "__main__":
    main()
