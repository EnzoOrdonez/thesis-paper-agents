"""Agent 2: Paper Compiler - Maintains database, validates, generates reports."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
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
        json.dump([paper.model_dump() for paper in papers], f, ensure_ascii=False, indent=2)

    if db_path.exists():
        db_path.unlink()
    tmp_path.rename(db_path)
    logger.info(f"Database saved: {len(papers)} papers to {db_path}")


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


def import_daily_papers(config: dict) -> int:
    """Import papers from daily report files into the database."""
    db_path = config["output"]["database_path"]
    daily_dir = Path(config["output"]["daily_dir"])

    papers = load_database(db_path)
    dedup_index = build_dedup_index(papers)
    new_count = 0

    if not daily_dir.exists():
        console.print("[yellow]No daily directory found.[/yellow]")
        return 0

    for report_file in _preferred_daily_import_files(daily_dir):
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

    save_database(papers, db_path)
    console.print(f"[green]Imported {new_count} new papers from daily reports.[/green]")
    return new_count


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


def validate_dois(config: dict) -> None:
    """Verify all DOIs in the database against CrossRef."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    crossref = CrossRefAPI()
    to_verify = [paper for paper in papers if paper.doi and paper.doi_verified is None]

    if not to_verify:
        console.print("[green]All DOIs already verified.[/green]")
        return

    console.print(f"Verifying {len(to_verify)} DOIs against CrossRef...")
    verified = 0
    failed = 0

    for paper in to_verify:
        try:
            if crossref.verify_doi(paper.doi):  # type: ignore[arg-type]
                paper.doi_verified = True
                verified += 1
            else:
                paper.doi_verified = False
                failed += 1
        except Exception as exc:
            logger.warning(f"DOI verification error for {paper.doi}: {exc}")
            paper.doi_verified = False
            failed += 1

    save_database(papers, db_path)
    console.print(f"[green]Verified: {verified}[/green], [red]Failed: {failed}[/red]")


def check_scopus_indexing(config: dict) -> None:
    """Check Scopus indexing status via OpenAlex."""
    db_path = config["output"]["database_path"]
    papers = load_database(db_path)

    openalex = OpenAlexAPI()
    to_check = [paper for paper in papers if paper.doi and paper.scopus_indexed is None]

    if not to_check:
        console.print("[green]All papers already checked for Scopus indexing.[/green]")
        return

    console.print(f"Checking Scopus indexing for {len(to_check)} papers...")
    indexed = 0

    for paper in to_check:
        try:
            paper.scopus_indexed = openalex.check_scopus_indexed(paper.doi)  # type: ignore[arg-type]
            if paper.scopus_indexed:
                indexed += 1
        except Exception as exc:
            logger.warning(f"Scopus check error for {paper.doi}: {exc}")

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


def export_apa7(config: dict) -> None:
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
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[green]APA 7 references exported to {filepath} ({len(accepted)} papers)[/green]")


def export_bibtex(config: dict) -> None:
    """Export all accepted papers as BibTeX."""
    db_path = config["output"]["database_path"]
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    papers = load_database(db_path)
    accepted = [paper for paper in papers if paper.status in (PaperStatus.ACCEPTED, PaperStatus.NEW)]

    entries = [paper.bibtex or format_bibtex(paper) for paper in accepted]

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






