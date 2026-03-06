"""Gap analysis for thesis coverage based on paper categories and pending gaps."""

from __future__ import annotations

from collections import Counter
from datetime import date

import yaml

from src.models.paper import Paper, PaperStatus


def load_pending_gaps(path: str = "config/trusted_sources.yaml") -> list[dict]:
    """Load the list of pending foundational gaps."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("pending_gaps", [])


def load_thesis_categories(path: str = "config/trusted_sources.yaml") -> list[str]:
    """Load the list of thesis categories."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("thesis_categories", [])


def analyze_category_coverage(papers: list[Paper]) -> dict[str, dict]:
    """Analyze how many papers cover each thesis category.

    Returns:
        Dict mapping category name to {count, status_emoji, status_text, action}.
    """
    categories = load_thesis_categories()
    accepted = [p for p in papers if p.status in (PaperStatus.NEW, PaperStatus.ACCEPTED)]

    cat_counts: Counter[str] = Counter()
    for p in accepted:
        for cat in p.categories:
            cat_counts[cat] += 1

    result: dict[str, dict] = {}
    for cat in categories:
        count = cat_counts.get(cat, 0)
        if count >= 5:
            status_emoji = "Excelente"
            action = "Mantener"
        elif count >= 3:
            status_emoji = "Bueno"
            action = "Mantener"
        elif count >= 1:
            status_emoji = "Parcial"
            action = "Buscar mas"
        else:
            status_emoji = "Insuficiente"
            action = "URGENTE"

        result[cat] = {
            "count": count,
            "status": status_emoji,
            "action": action,
        }

    return result


def analyze_gap_coverage(papers: list[Paper]) -> list[dict]:
    """Check which foundational gaps have been covered by found papers.

    Returns:
        List of dicts: {name, description, covered, covered_by}.
    """
    gaps = load_pending_gaps()
    accepted = [p for p in papers if p.status in (PaperStatus.NEW, PaperStatus.ACCEPTED)]

    results: list[dict] = []
    for gap in gaps:
        covered = False
        covered_by = None
        for p in accepted:
            if p.covers_gap and p.covers_gap == gap["name"]:
                covered = True
                covered_by = p.title
                break
        results.append({
            "name": gap["name"],
            "description": gap.get("description", ""),
            "covered": covered,
            "covered_by": covered_by,
        })

    return results


def generate_gap_report(papers: list[Paper]) -> str:
    """Generate a full gap analysis report in Markdown."""
    today = date.today().isoformat()
    lines = [
        f"# Analisis de Gaps -- Generado: {today}",
        "",
        "## Estado por Categoria:",
        "",
        "| Categoria | Papers | Status | Accion |",
        "|-----------|--------|--------|--------|",
    ]

    coverage = analyze_category_coverage(papers)
    for cat, info in coverage.items():
        status_icon = {
            "Excelente": "Excelente",
            "Bueno": "Bueno",
            "Parcial": "Parcial",
            "Insuficiente": "Insuficiente",
        }.get(info["status"], info["status"])
        lines.append(f"| {cat} | {info['count']} | {status_icon} | {info['action']} |")

    lines.extend(["", "## Gaps Fundacionales Pendientes:", ""])

    gap_results = analyze_gap_coverage(papers)
    for gap in gap_results:
        if gap["covered"]:
            lines.append(f"- [x] {gap['name']} ({gap['description']}) -- CUBIERTO por: {gap['covered_by']}")
        else:
            lines.append(f"- [ ] {gap['name']} ({gap['description']})")

    covered_count = sum(1 for g in gap_results if g["covered"])
    total = len(gap_results)
    lines.extend([
        "",
        f"## Resumen: {covered_count}/{total} gaps fundacionales cubiertos",
        "",
    ])

    return "\n".join(lines)
