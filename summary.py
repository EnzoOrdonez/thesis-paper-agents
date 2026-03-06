#!/usr/bin/env python3
"""Dashboard — Complete visual status of the paper database.

Usage:
    python summary.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.models.paper import Paper, PaperStatus, RelevanceLevel

console = Console()

DB_PATH = "data/papers_database.json"
GAPS_PATH = "config/trusted_sources.yaml"
AUTO_LOG_PATH = "logs/auto_execution.log"


def load_database() -> list[Paper]:
    db_path = Path(DB_PATH)
    if not db_path.exists():
        return []
    with open(db_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Paper(**item) for item in data]


def load_gaps() -> list[dict]:
    with open(GAPS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("pending_gaps", [])


def make_bar(value: int, total: int, width: int = 20, color: str = "green") -> str:
    """Create a visual progress bar (ASCII-safe for Windows consoles)."""
    if total == 0:
        return " " * width
    filled = int((value / total) * width)
    empty = width - filled
    bar = "[" + color + "]" + "#" * filled + "[/]" + "[dim]-[/dim]" * empty
    return bar


def main() -> None:
    papers = load_database()

    if not papers:
        console.print("[yellow]La base de datos esta vacia.[/yellow]")
        return

    total = len(papers)

    # ── Status counts ────────────────────────────────────────────────────────
    status_counts = Counter(p.status.value for p in papers)
    accepted = status_counts.get("accepted", 0)
    rejected = status_counts.get("rejected", 0)
    pending = status_counts.get("new", 0)
    reviewed = status_counts.get("reviewed", 0)

    # ── Relevance counts ─────────────────────────────────────────────────────
    rel_counts = Counter(p.relevance_level.value for p in papers)
    alta = rel_counts.get("ALTA", 0)
    media = rel_counts.get("MEDIA", 0)
    baja = rel_counts.get("BAJA", 0)

    # ── Last search date ─────────────────────────────────────────────────────
    dates = [p.date_found for p in papers if p.date_found]
    last_search = max(dates) if dates else "N/A"

    # ── Next auto search ─────────────────────────────────────────────────────
    next_auto = "No configurado"
    auto_log = Path(AUTO_LOG_PATH)
    if auto_log.exists():
        lines = auto_log.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            next_auto = "Ultima ejecucion: " + lines[-1][:19]

    # ═══════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════

    header_text = Text()
    header_text.append("THESIS PAPER AGENTS", style="bold cyan")
    header_text.append(" — DASHBOARD", style="dim")

    console.print()
    console.print(Panel(
        header_text,
        border_style="cyan",
        padding=(0, 2),
    ))

    # ── Main stats row ───────────────────────────────────────────────────────

    stats_lines = []
    stats_lines.append(f"  [bold]Base de datos:[/bold] [cyan]{total}[/cyan] papers")
    stats_lines.append(f"  [bold]Ultima busqueda:[/bold] {last_search}")
    stats_lines.append(f"  [bold]Relevancia:[/bold]  "
                       f"[green]ALTA: {alta}[/green]  "
                       f"[yellow]MEDIA: {media}[/yellow]  "
                       f"[dim]BAJA: {baja}[/dim]")
    console.print("\n".join(stats_lines))

    # ── Review status ────────────────────────────────────────────────────────

    review_table = Table(title="Estado de Revision", show_header=False, box=None, padding=(0, 1))
    review_table.add_column(style="cyan", min_width=15)
    review_table.add_column(min_width=25)
    review_table.add_column(style="green", justify="right", min_width=6)
    review_table.add_column(style="dim", justify="right", min_width=5)

    review_table.add_row(
        "Aceptados",
        make_bar(accepted, total, 20, "green"),
        str(accepted),
        f"{(accepted/total*100):.0f}%"
    )
    review_table.add_row(
        "Rechazados",
        make_bar(rejected, total, 20, "red"),
        str(rejected),
        f"{(rejected/total*100):.0f}%"
    )
    review_table.add_row(
        "Pendientes",
        make_bar(pending, total, 20, "yellow"),
        str(pending),
        f"{(pending/total*100):.0f}%"
    )

    console.print()
    console.print(review_table)

    # ── Gaps ─────────────────────────────────────────────────────────────────

    gaps = load_gaps()
    covered_gaps: list[str] = []
    pending_gaps: list[str] = []
    gap_papers: dict[str, str] = {}

    for gap in gaps:
        gap_name = gap["name"]
        covered = False
        for p in papers:
            if p.covers_gap and p.covers_gap == gap_name:
                covered = True
                gap_papers[gap_name] = p.title[:50]
                break
        if covered:
            covered_gaps.append(gap_name)
        else:
            pending_gaps.append(gap_name)

    gap_bar = make_bar(len(covered_gaps), len(gaps), 30, "green")
    console.print()
    console.print(f"  [bold]Gaps Fundacionales:[/bold] {gap_bar} {len(covered_gaps)}/{len(gaps)}")

    if pending_gaps:
        for name in pending_gaps:
            console.print(f"    [red]✗[/red] {name}")
    else:
        console.print(f"    [green]Todos los gaps cubiertos[/green]")

    # ── Papers por categoria ─────────────────────────────────────────────────

    cat_counts: Counter[str] = Counter()
    for p in papers:
        for cat in p.categories:
            cat_counts[cat] += 1

    max_count = max(cat_counts.values()) if cat_counts else 1

    cat_table = Table(title="Papers por Categoria", show_header=True, padding=(0, 1))
    cat_table.add_column("Categoria", style="cyan", min_width=30)
    cat_table.add_column("Barra", min_width=22)
    cat_table.add_column("Total", style="green", justify="right")

    for cat, count in cat_counts.most_common():
        bar = make_bar(count, max_count, 18, "cyan")
        cat_table.add_row(cat, bar, str(count))

    console.print()
    console.print(cat_table)

    # ── Ultimos papers de ALTA relevancia ─────────────────────────────────────

    high_papers = [p for p in papers if p.relevance_level == RelevanceLevel.HIGH]
    high_papers.sort(key=lambda p: p.date_found, reverse=True)

    if high_papers:
        console.print()
        console.print(f"  [bold]Ultimos papers de ALTA relevancia:[/bold]")
        for i, p in enumerate(high_papers[:5], 1):
            title_short = p.title[:55] + ("..." if len(p.title) > 55 else "")
            year_str = str(p.year) if p.year else "N/A"
            console.print(
                f"    {i}. [cyan]{title_short}[/cyan] "
                f"({year_str}) — Score: [yellow]{p.relevance_score}[/yellow]"
            )

    # ── Papers por ano ────────────────────────────────────────────────────────

    year_counts = Counter(p.year for p in papers if p.year)
    if year_counts:
        year_table = Table(title="Papers por Ano", show_header=True, padding=(0, 1))
        year_table.add_column("Ano", style="cyan", justify="center")
        year_table.add_column("Cantidad", style="green", justify="right")
        year_table.add_column("Barra", min_width=15)

        max_year_count = max(year_counts.values())
        for yr in sorted(year_counts.keys(), reverse=True):
            count = year_counts[yr]
            bar = make_bar(count, max_year_count, 12, "cyan")
            year_table.add_row(str(yr), str(count), bar)

        console.print()
        console.print(year_table)

    # ── Source API distribution ───────────────────────────────────────────────

    source_counts = Counter(p.source_api for p in papers if p.source_api)
    if source_counts:
        console.print()
        console.print(f"  [bold]Fuentes API:[/bold]  ", end="")
        parts = []
        for src, cnt in source_counts.most_common():
            parts.append(f"[cyan]{src}[/cyan]: {cnt}")
        console.print("  |  ".join(parts))

    # ── DOI and Scopus stats ─────────────────────────────────────────────────
    has_doi = sum(1 for p in papers if p.doi)
    doi_verified = sum(1 for p in papers if p.doi_verified)
    scopus = sum(1 for p in papers if p.scopus_indexed)

    console.print()
    console.print(f"  [bold]Calidad:[/bold]  "
                  f"Con DOI: {has_doi}/{total}  |  "
                  f"DOI verificado: {doi_verified}  |  "
                  f"Scopus indexado: {scopus}")
    console.print()


if __name__ == "__main__":
    main()
