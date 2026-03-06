"""Agent 2: Paper Compiler — Maintains database, validates, generates reports."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.apis.crossref_api import CrossRefAPI
from src.apis.openalex_api import OpenAlexAPI
from src.models.paper import Paper, PaperStatus
from src.utils.duplicate_detector import (
    find_duplicates_in_list,
    is_duplicate_by_doi,
    is_duplicate_by_title,
)
from src.utils.gap_analyzer import generate_gap_report
from src.utils.logger import setup_logger
from src.utils.reference_formatter import format_apa7, format_bibtex
from src.utils.relevance_scorer import check_gap_coverage, score_paper, suggest_categories

logger = setup_logger("paper_compiler")
console = Console()


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_database(path: str) -> list[Paper]:
    """Load papers from the JSON database."""
    db_path = Path(path)
    if not db_path.exists():
        return []
    with open(db_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Paper(**item) for item in data]


def save_database(papers: list[Paper], path: str) -> None:
    """Save papers to the JSON database atomically."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".json.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in papers], f, ensure_ascii=False, indent=2)

    if db_path.exists():
        db_path.unlink()
    tmp_path.rename(db_path)
    logger.info(f"Database saved: {len(papers)} papers to {db_path}")


def import_daily_papers(config: dict) -> int:
    """Import papers from daily report files into the database.

    Parses the markdown reports in output/daily/ and adds new papers.

    Returns:
        Number of new papers added.
    """
    db_path = config["output"]["database_path"]
    daily_dir = Path(config["output"]["daily_dir"])

    papers = load_database(db_path)
    new_count = 0

    if not daily_dir.exists():
        console.print("[yellow]No daily directory found.[/yellow]")
        return 0

    for md_file in sorted(daily_dir.glob("*_daily_papers.md")):
        logger.info(f"Importing from {md_file.name}")
        imported = _parse_daily_report(md_file)

        for paper in imported:
            # Check for duplicates
            if paper.doi and is_duplicate_by_doi(paper.doi, papers):
                continue
            if is_duplicate_by_title(paper.title, papers):
                continue

            # Ensure categories are always populated
            if not paper.categories:
                paper.categories = suggest_categories(paper)
            if not paper.covers_gap:
                paper.covers_gap = check_gap_coverage(paper)

            # Generate references
            paper.apa7_reference = format_apa7(paper)
            paper.bibtex = format_bibtex(paper)
            papers.append(paper)
            new_count += 1

    save_database(papers, db_path)
    console.print(f"[green]Imported {new_count} new papers from daily reports.[/green]")
    return new_count


def _parse_daily_report(filepath: Path) -> list[Paper]:
    """Parse a daily markdown report and extract papers.

    This is a best-effort parser that extracts structured data from the markdown
    format produced by the daily researcher agent.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    papers: list[Paper] = []
    # Split by paper sections (## [ALTA/MEDIA/BAJA] Title)
    sections = re.split(r"\n## \[(ALTA|MEDIA|BAJA)\] ", content)

    # sections[0] is the header, then alternating: level, content
    for i in range(1, len(sections) - 1, 2):
        level = sections[i]
        body = sections[i + 1]

        lines = body.split("\n")
        title = lines[0].strip() if lines else ""

        paper = Paper(title=title)

        if level == "ALTA":
            paper.relevance_level = "ALTA"  # type: ignore
        elif level == "MEDIA":
            paper.relevance_level = "MEDIA"  # type: ignore
        else:
            paper.relevance_level = "BAJA"  # type: ignore

        parsed_categories: list[str] = []
        for line in lines[1:]:
            line_stripped = line.strip()
            if line_stripped.startswith("- **Autores:**"):
                authors_str = line_stripped.replace("- **Autores:**", "").strip()
                if authors_str and authors_str != "N/A":
                    paper.authors = [a.strip() for a in authors_str.split(",")]
            elif line_stripped.startswith("- **Fecha:**"):
                val = line_stripped.replace("- **Fecha:**", "").strip()
                if val and val != "N/A":
                    paper.publication_date = val
                    try:
                        paper.year = int(val[:4])
                    except (ValueError, IndexError):
                        pass
            elif line_stripped.startswith("- **Fuente:**"):
                val = line_stripped.replace("- **Fuente:**", "").strip()
                if val != "N/A":
                    paper.venue = val
            elif line_stripped.startswith("- **DOI:**"):
                val = line_stripped.replace("- **DOI:**", "").strip()
                if val != "N/A" and "doi.org/" in val:
                    paper.doi = val.replace("https://doi.org/", "")
            elif line_stripped.startswith("- **Link:**"):
                val = line_stripped.replace("- **Link:**", "").strip()
                if val != "N/A":
                    paper.url = val
            elif line_stripped.startswith("- **Citaciones:**"):
                val = line_stripped.replace("- **Citaciones:**", "").strip()
                try:
                    paper.citation_count = int(val)
                except ValueError:
                    pass
            elif line_stripped.startswith("- **Score de relevancia:**"):
                val = line_stripped.replace("- **Score de relevancia:**", "").strip().replace("/100", "")
                try:
                    paper.relevance_score = int(val)
                except ValueError:
                    pass
            elif line_stripped.startswith("- **Abstract:**"):
                paper.abstract = line_stripped.replace("- **Abstract:**", "").strip()
            elif line_stripped.startswith("- **Cubre algun gap pendiente?**"):
                val = line_stripped.replace("- **Cubre algun gap pendiente?**", "").strip()
                if val.startswith("Si"):
                    gap_name = val.replace("Si", "").replace("--", "").strip()
                    if gap_name:
                        paper.covers_gap = gap_name
            elif line_stripped.startswith("- **Keywords match:**"):
                val = line_stripped.replace("- **Keywords match:**", "").strip()
                if val and val != "[]":
                    # Parse list like ['kw1', 'kw2']
                    cleaned = val.strip("[]")
                    paper.keywords_matched = [
                        k.strip().strip("'\"") for k in cleaned.split(",") if k.strip()
                    ]
            # Parse category checkboxes: "- [x] Category Name"
            elif re.match(r"^- \[x\] .+", line_stripped):
                cat_name = re.sub(r"^- \[x\] ", "", line_stripped).strip()
                if cat_name:
                    parsed_categories.append(cat_name)

        # Assign parsed categories; if none were found in markdown, use scorer as fallback
        if parsed_categories:
            paper.categories = parsed_categories
        elif paper.title:
            paper.categories = suggest_categories(paper)

        # Also detect gap coverage via scorer if not already set from markdown
        if not paper.covers_gap and paper.title:
            paper.covers_gap = check_gap_coverage(paper)

        if paper.title:
            papers.append(paper)

    return papers


def validate_dois(config: dict) -> None:
    """Verify all DOIs in the database against CrossRef."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    crossref = CrossRefAPI()
    to_verify = [p for p in papers if p.doi and p.doi_verified is None]

    if not to_verify:
        console.print("[green]All DOIs already verified.[/green]")
        return

    console.print(f"Verifying {len(to_verify)} DOIs against CrossRef...")
    verified = 0
    failed = 0

    for paper in to_verify:
        try:
            if crossref.verify_doi(paper.doi):  # type: ignore
                paper.doi_verified = True
                verified += 1
            else:
                paper.doi_verified = False
                failed += 1
        except Exception as e:
            logger.warning(f"DOI verification error for {paper.doi}: {e}")
            paper.doi_verified = False
            failed += 1

    save_database(papers, db_path)
    console.print(f"[green]Verified: {verified}[/green], [red]Failed: {failed}[/red]")


def check_scopus_indexing(config: dict) -> None:
    """Check Scopus indexing status via OpenAlex."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    openalex = OpenAlexAPI()
    to_check = [p for p in papers if p.doi and p.scopus_indexed is None]

    if not to_check:
        console.print("[green]All papers already checked for Scopus indexing.[/green]")
        return

    console.print(f"Checking Scopus indexing for {len(to_check)} papers...")
    indexed = 0

    for paper in to_check:
        try:
            paper.scopus_indexed = openalex.check_scopus_indexed(paper.doi)  # type: ignore
            if paper.scopus_indexed:
                indexed += 1
        except Exception as e:
            logger.warning(f"Scopus check error for {paper.doi}: {e}")

    save_database(papers, db_path)
    console.print(f"[green]Scopus indexed: {indexed}/{len(to_check)}[/green]")


def generate_references(config: dict) -> None:
    """Generate APA7 and BibTeX references for all papers."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)
    updated = 0

    for paper in papers:
        if not paper.apa7_reference:
            paper.apa7_reference = format_apa7(paper)
            updated += 1
        if not paper.bibtex:
            paper.bibtex = format_bibtex(paper)

    save_database(papers, db_path)
    console.print(f"[green]Generated references for {updated} papers.[/green]")


def repair_categories(config: dict) -> int:
    """Re-assign categories and gap coverage for papers that have empty categories.

    This fixes papers imported before the category-parsing bug was fixed.

    Returns:
        Number of papers repaired.
    """
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

        if changed:
            repaired += 1

    if repaired > 0:
        save_database(papers, db_path)

    console.print(f"[green]Repaired categories for {repaired} papers.[/green]")
    return repaired


def remove_duplicates(config: dict) -> int:
    """Find and remove duplicate papers from the database."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    dupes = find_duplicates_in_list(papers)
    if not dupes:
        console.print("[green]No duplicates found.[/green]")
        return 0

    # Remove the second paper in each duplicate pair (keep the one with more info)
    indices_to_remove: set[int] = set()
    for i, j, score in dupes:
        # Keep the one with more data (higher citation count, DOI, etc.)
        if papers[i].citation_count >= papers[j].citation_count:
            indices_to_remove.add(j)
        else:
            indices_to_remove.add(i)

    cleaned = [p for idx, p in enumerate(papers) if idx not in indices_to_remove]
    removed = len(papers) - len(cleaned)
    save_database(cleaned, db_path)
    console.print(f"[green]Removed {removed} duplicate papers.[/green]")
    return removed


def interactive_review(config: dict) -> None:
    """Interactively review papers and set their status."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    to_review = [p for p in papers if p.status == PaperStatus.NEW]
    if not to_review:
        console.print("[green]No new papers to review.[/green]")
        return

    console.print(f"\n[bold]Reviewing {len(to_review)} new papers[/bold]\n")

    for i, paper in enumerate(to_review, 1):
        console.print(f"\n[bold cyan]--- Paper {i}/{len(to_review)} ---[/bold cyan]")
        console.print(f"[bold]{paper.title}[/bold]")
        console.print(f"Authors: {', '.join(paper.authors[:5])}")
        console.print(f"Year: {paper.year} | Venue: {paper.venue}")
        console.print(f"Score: {paper.relevance_score}/100 [{paper.relevance_level.value}]")
        console.print(f"Categories: {', '.join(paper.categories)}")
        if paper.covers_gap:
            console.print(f"[green]Covers gap: {paper.covers_gap}[/green]")
        if paper.abstract:
            console.print(f"Abstract: {paper.truncated_abstract(100)}")

        choice = Prompt.ask(
            "Status",
            choices=["accept", "reject", "skip", "quit"],
            default="skip",
        )

        if choice == "accept":
            paper.status = PaperStatus.ACCEPTED
        elif choice == "reject":
            paper.status = PaperStatus.REJECTED
        elif choice == "quit":
            break
        # "skip" keeps status as NEW

        notes = Prompt.ask("Notes (optional)", default="")
        if notes:
            paper.notes = notes

    save_database(papers, db_path)
    console.print("[green]Review session saved.[/green]")


def show_statistics(config: dict) -> None:
    """Display statistics about the paper database."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    if not papers:
        console.print("[yellow]Database is empty.[/yellow]")
        return

    console.print(f"\n[bold cyan]Database Statistics[/bold cyan]")
    console.print(f"Total papers: {len(papers)}")

    # By status
    status_counts = Counter(p.status.value for p in papers)
    table = Table(title="By Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")
    for status, count in status_counts.most_common():
        table.add_row(status, str(count))
    console.print(table)

    # By year
    year_counts = Counter(p.year for p in papers if p.year)
    table = Table(title="By Year")
    table.add_column("Year", style="cyan")
    table.add_column("Count", style="green")
    for year in sorted(year_counts.keys(), reverse=True):
        table.add_row(str(year), str(year_counts[year]))
    console.print(table)

    # By source API
    source_counts = Counter(p.source_api for p in papers if p.source_api)
    table = Table(title="By Source API")
    table.add_column("Source", style="cyan")
    table.add_column("Count", style="green")
    for source, count in source_counts.most_common():
        table.add_row(source, str(count))
    console.print(table)

    # By category
    cat_counts: Counter[str] = Counter()
    for p in papers:
        for cat in p.categories:
            cat_counts[cat] += 1
    table = Table(title="By Category")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    for cat, count in cat_counts.most_common():
        table.add_row(cat, str(count))
    console.print(table)

    # By relevance
    rel_counts = Counter(p.relevance_level.value for p in papers)
    table = Table(title="By Relevance")
    table.add_column("Level", style="cyan")
    table.add_column("Count", style="green")
    for level in ["ALTA", "MEDIA", "BAJA"]:
        table.add_row(level, str(rel_counts.get(level, 0)))
    console.print(table)


def export_apa7(config: dict) -> None:
    """Export all accepted papers as APA 7 references."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    accepted = [p for p in papers if p.status in (PaperStatus.ACCEPTED, PaperStatus.NEW)]
    accepted.sort(key=lambda p: (p.authors[0] if p.authors else "", p.year or 0))

    lines = [f"# Referencias APA 7 -- Generado: {date.today().isoformat()}", ""]

    for paper in accepted:
        ref = paper.apa7_reference or format_apa7(paper)
        lines.append(f"- {ref}")
        lines.append("")

    filepath = reports_dir / "references_apa7.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[green]APA 7 references exported to {filepath} ({len(accepted)} papers)[/green]")


def export_bibtex(config: dict) -> None:
    """Export all accepted papers as BibTeX."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    accepted = [p for p in papers if p.status in (PaperStatus.ACCEPTED, PaperStatus.NEW)]

    entries: list[str] = []
    for paper in accepted:
        bib = paper.bibtex or format_bibtex(paper)
        entries.append(bib)

    filepath = reports_dir / "references.bib"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))

    console.print(f"[green]BibTeX exported to {filepath} ({len(accepted)} entries)[/green]")


def generate_consolidated_report(config: dict) -> None:
    """Generate the consolidated report organized by category."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    accepted = [p for p in papers if p.status in (PaperStatus.ACCEPTED, PaperStatus.NEW)]

    lines = [
        f"# Reporte Consolidado de Papers -- Generado: {date.today().isoformat()}",
        f"",
        f"Total papers: {len(accepted)}",
        "",
    ]

    # Group by category
    by_category: dict[str, list[Paper]] = {}
    for p in accepted:
        for cat in p.categories:
            by_category.setdefault(cat, []).append(p)

    for cat in sorted(by_category.keys()):
        cat_papers = by_category[cat]
        cat_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        lines.append(f"## {cat} ({len(cat_papers)} papers)")
        lines.append("")

        for p in cat_papers:
            doi_str = f"https://doi.org/{p.doi}" if p.doi else "N/A"
            lines.append(f"### [{p.relevance_level.value}] {p.title}")
            lines.append(f"- **Autores:** {', '.join(p.authors[:5])}")
            lines.append(f"- **Ano:** {p.year} | **Fuente:** {p.venue}")
            lines.append(f"- **DOI:** {doi_str}")
            lines.append(f"- **Score:** {p.relevance_score}/100")
            if p.covers_gap:
                lines.append(f"- **Cubre gap:** {p.covers_gap}")
            lines.append("")

    filepath = reports_dir / "consolidated_report.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Consolidated report: {filepath}[/green]")


def generate_gap_analysis_report(config: dict) -> None:
    """Generate the gap analysis report."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    report = generate_gap_report(papers)

    filepath = reports_dir / "gap_analysis.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    console.print(f"[green]Gap analysis report: {filepath}[/green]")


def generate_statistics_report(config: dict) -> None:
    """Generate a statistics markdown report."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    today = date.today().isoformat()

    lines = [f"# Estadisticas -- Generado: {today}", ""]

    lines.append(f"## Total papers en base de datos: {len(papers)}")
    lines.append("")

    # By year
    year_counts = Counter(p.year for p in papers if p.year)
    lines.append("## Papers por ano")
    lines.append("| Ano | Cantidad |")
    lines.append("|-----|----------|")
    for yr in sorted(year_counts.keys(), reverse=True):
        lines.append(f"| {yr} | {year_counts[yr]} |")
    lines.append("")

    # By source
    source_counts = Counter(p.source_api for p in papers if p.source_api)
    lines.append("## Papers por fuente API")
    lines.append("| Fuente | Cantidad |")
    lines.append("|--------|----------|")
    for src, cnt in source_counts.most_common():
        lines.append(f"| {src} | {cnt} |")
    lines.append("")

    # By category
    cat_counts: Counter[str] = Counter()
    for p in papers:
        for cat in p.categories:
            cat_counts[cat] += 1
    lines.append("## Papers por categoria")
    lines.append("| Categoria | Cantidad |")
    lines.append("|-----------|----------|")
    for cat, cnt in cat_counts.most_common():
        lines.append(f"| {cat} | {cnt} |")
    lines.append("")

    # Search timeline
    date_counts = Counter(p.date_found for p in papers)
    lines.append("## Timeline de busqueda")
    lines.append("| Fecha | Papers encontrados |")
    lines.append("|-------|-------------------|")
    for d in sorted(date_counts.keys()):
        lines.append(f"| {d} | {date_counts[d]} |")
    lines.append("")

    filepath = reports_dir / "statistics.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Statistics report: {filepath}[/green]")


def compile_all(config: dict) -> None:
    """Run the full compilation pipeline."""
    console.print("\n[bold cyan]Paper Compiler Agent[/bold cyan]\n")

    console.print("[bold]1. Importing daily papers...[/bold]")
    import_daily_papers(config)

    console.print("\n[bold]2. Repairing categories for existing papers...[/bold]")
    repair_categories(config)

    console.print("\n[bold]3. Removing duplicates...[/bold]")
    remove_duplicates(config)

    console.print("\n[bold]4. Generating references...[/bold]")
    generate_references(config)

    console.print("\n[bold]5. Validating DOIs...[/bold]")
    validate_dois(config)

    console.print("\n[bold]6. Checking Scopus indexing...[/bold]")
    check_scopus_indexing(config)

    console.print("\n[bold]7. Generating reports...[/bold]")
    generate_consolidated_report(config)
    generate_gap_analysis_report(config)
    generate_statistics_report(config)
    export_apa7(config)
    export_bibtex(config)

    console.print("\n[bold green]Compilation complete![/bold green]")
