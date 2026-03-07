#!/usr/bin/env python3
"""Export thesis materials: state-of-art table, literature map, draft paragraphs.

Usage:
    python export_thesis.py                              # Export all
    python export_thesis.py --format markdown             # Markdown only
    python export_thesis.py --only-accepted               # Only accepted papers
    python export_thesis.py --category "Sistemas RAG"     # Only one category
    python export_thesis.py --for-rag                     # Export for RAG system
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from rich.console import Console

from src.agents.paper_compiler import load_config, load_database as load_papers_database
from src.models.paper import Paper, PaperStatus, RelevanceLevel
from src.utils.reference_formatter import format_apa7

console = Console()

CONFIG = load_config()
DB_PATH = CONFIG["output"]["database_path"]
OUTPUT_DIR = Path("output/thesis")


def load_database() -> list[Paper]:
    return load_papers_database(DB_PATH)



def load_categories() -> list[str]:
    with open("config/trusted_sources.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("thesis_categories", [])


def _author_cite(paper: Paper) -> str:
    """Generate short citation like (Sawarkar et al., 2024)."""
    if not paper.authors:
        return f"(Unknown, {paper.year or 'n.d.'})"

    first = paper.authors[0]
    # Extract surname
    parts = first.split()
    surname = parts[-1] if parts else first

    if len(paper.authors) == 1:
        return f"({surname}, {paper.year or 'n.d.'})"
    elif len(paper.authors) == 2:
        second = paper.authors[1].split()
        surname2 = second[-1] if second else paper.authors[1]
        return f"({surname} & {surname2}, {paper.year or 'n.d.'})"
    else:
        return f"({surname} et al., {paper.year or 'n.d.'})"


def _infer_approach(paper: Paper) -> str:
    """Infer the paper's approach/focus from title and abstract."""
    combined = f"{paper.title or ''} {paper.abstract or ''}".lower()

    approaches = []
    if "hybrid" in combined and ("bm25" in combined or "lexical" in combined):
        approaches.append("Hybrid retrieval (BM25 + dense)")
    elif "hybrid" in combined:
        approaches.append("Hybrid approach")

    if "rag" in combined or "retrieval augmented" in combined:
        approaches.append("RAG system")
    if "embedding" in combined or "dense representation" in combined:
        approaches.append("Dense embeddings")
    if "evaluation" in combined or "benchmark" in combined:
        approaches.append("Evaluation/Benchmark")
    if "hallucination" in combined or "faithfulness" in combined:
        approaches.append("Hallucination mitigation")
    if "chunk" in combined or "segment" in combined:
        approaches.append("Document segmentation")
    if "rerank" in combined or "re-rank" in combined:
        approaches.append("Neural reranking")
    if "cloud" in combined or "documentation" in combined:
        approaches.append("Technical documentation")
    if "vector" in combined and ("database" in combined or "search" in combined):
        approaches.append("Vector database")
    if "survey" in combined or "review" in combined:
        approaches.append("Survey/Review")

    return "; ".join(approaches[:2]) if approaches else "N/A"


def _infer_relevance_to_thesis(paper: Paper) -> str:
    """Infer why this paper is relevant to the thesis."""
    cats = paper.categories
    if not cats:
        return "Relevancia general al tema de investigacion"

    cat_relevance = {
        "Sistemas RAG Hibridos": "Directamente relacionado con el modelo hibrido propuesto",
        "Estrategias de Recuperacion": "Fundamenta las estrategias de recuperacion del modelo",
        "Modelos de Embedding": "Base para la componente de embeddings semanticos",
        "Segmentacion/Chunking": "Informa la estrategia de segmentacion de documentos",
        "Re-ranking": "Contribuye al pipeline de re-ranking del modelo",
        "Alucinaciones en LLMs": "Aborda problema critico de calidad en sistemas RAG",
        "Evaluacion de sistemas RAG": "Metodologia de evaluacion del sistema propuesto",
        "Documentacion Cloud": "Contexto del dominio de aplicacion (AWS, Azure, GCP)",
        "Metricas de evaluacion": "Metricas para validar el modelo propuesto",
        "Vector Databases": "Infraestructura de almacenamiento vectorial",
        "Normalizacion/Preprocesamiento": "Preprocesamiento de terminologia tecnica",
    }

    return cat_relevance.get(cats[0], "Relevancia general")


# Exporters

def export_estado_del_arte_tabla(papers: list[Paper], category_filter: str | None = None) -> None:
    """Generate state-of-the-art table in Markdown."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    # Group by category
    by_category: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        for cat in p.categories:
            if category_filter and category_filter.lower() not in cat.lower():
                continue
            by_category[cat].append(p)

    lines = [
        f"# Tabla de Estado del Arte",
        f"_Generado: {today}_",
        "",
    ]

    categories = load_categories()
    for cat in categories:
        if cat not in by_category:
            continue

        cat_papers = by_category[cat]
        cat_papers.sort(key=lambda p: (p.year or 0, p.relevance_score), reverse=True)

        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| Autor(es) | Ano | Titulo | Enfoque | Relevancia para mi tesis |")
        lines.append("|-----------|-----|--------|---------|--------------------------|")

        for p in cat_papers:
            authors = _author_cite(p).strip("()")
            approach = _infer_approach(p)
            relevance = _infer_relevance_to_thesis(p)
            title_short = p.title[:60] + ("..." if len(p.title) > 60 else "")
            lines.append(f"| {authors} | {p.year or 'N/A'} | {title_short} | {approach} | {relevance} |")

        lines.append("")

    filepath = OUTPUT_DIR / "estado_del_arte_tabla.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Tabla de estado del arte: {filepath}[/green]")


def export_literature_map(papers: list[Paper]) -> None:
    """Generate a Mermaid diagram of the literature map."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by category and year
    by_category: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        for cat in p.categories:
            by_category[cat].append(p)

    lines = [
        "---",
        "title: Mapa de Literatura - Tesis RAG Hibrido",
        "---",
        "graph TD",
        "",
        '    THESIS["Modelo Semantico Hibrido<br/>para RAG sobre<br/>Documentacion Cloud"]',
        "",
    ]

    # Create category nodes
    cat_ids = {}
    for i, cat in enumerate(sorted(by_category.keys())):
        cat_id = f"CAT{i}"
        cat_ids[cat] = cat_id
        safe_cat = cat.replace("/", "_").replace(" ", "_")
        lines.append(f'    {cat_id}["{cat}<br/>({len(by_category[cat])} papers)"]')

    lines.append("")

    # Connect thesis to categories
    for cat, cat_id in cat_ids.items():
        lines.append(f"    THESIS --> {cat_id}")

    lines.append("")

    # Add top papers per category (max 3 per cat)
    paper_id = 0
    for cat, cat_papers in by_category.items():
        top = sorted(cat_papers, key=lambda p: p.relevance_score, reverse=True)[:3]
        cat_id = cat_ids[cat]

        for p in top:
            pid = f"P{paper_id}"
            paper_id += 1
            author_short = _author_cite(p).strip("()")
            lines.append(f'    {cat_id} --> {pid}["{author_short}"]')

    lines.append("")

    # Styling
    lines.append("    classDef thesis fill:#1a73e8,color:#fff,stroke:#1557b0")
    lines.append("    classDef category fill:#34a853,color:#fff,stroke:#1e8e3e")
    lines.append("    class THESIS thesis")
    for cat_id in cat_ids.values():
        lines.append(f"    class {cat_id} category")

    filepath = OUTPUT_DIR / "literature_map.mermaid"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Mapa de literatura: {filepath}[/green]")


def export_estado_del_arte_borrador(papers: list[Paper], category_filter: str | None = None) -> None:
    """Generate draft paragraphs for each category's state of the art section."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    by_category: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        for cat in p.categories:
            if category_filter and category_filter.lower() not in cat.lower():
                continue
            by_category[cat].append(p)

    lines = [
        f"# Borrador de Estado del Arte",
        f"_Generado: {today} - BORRADOR para revision del autor_",
        "",
        "> **Nota:** Este es un borrador generado automaticamente. ",
        "> Debe ser revisado, editado y enriquecido por el autor de la tesis.",
        "",
    ]

    categories = load_categories()
    for cat in categories:
        if cat not in by_category:
            continue

        cat_papers = by_category[cat]
        cat_papers.sort(key=lambda p: (p.year or 0), reverse=True)

        lines.append(f"## {cat}")
        lines.append("")

        # Generate summary paragraph
        recent = [p for p in cat_papers if p.year and p.year >= 2023]
        older = [p for p in cat_papers if p.year and p.year < 2023]

        if recent:
            para_parts = []

            # Opening sentence
            para_parts.append(
                f"En el area de {cat.lower()}, la literatura reciente muestra "
                f"un crecimiento significativo en los ultimos anos, con "
                f"{len(recent)} publicaciones relevantes identificadas desde 2023."
            )

            # Cite top papers
            top_papers = sorted(recent, key=lambda p: p.relevance_score, reverse=True)[:5]
            for p in top_papers:
                cite = _author_cite(p)
                approach = _infer_approach(p)
                para_parts.append(
                    f"{cite} presenta un enfoque de {approach.lower()} "
                    f"que contribuye al entendimiento actual del tema."
                )

            # Closing
            if older:
                para_parts.append(
                    f"Estos trabajos se construyen sobre investigaciones previas, "
                    f"incluyendo {len(older)} publicaciones anteriores a 2023 "
                    f"que establecieron las bases del campo."
                )

            paragraph = " ".join(para_parts)
            lines.append(paragraph)
        else:
            lines.append(
                f"La investigacion en {cat.lower()} cuenta con "
                f"{len(cat_papers)} publicaciones identificadas en la revision sistematica."
            )

        lines.append("")

        # Add citation list
        lines.append(f"**Referencias de esta seccion ({len(cat_papers)}):**")
        for p in cat_papers[:10]:
            lines.append(f"- {_author_cite(p)} - _{p.title}_")
        if len(cat_papers) > 10:
            lines.append(f"- _... y {len(cat_papers) - 10} referencias adicionales_")
        lines.append("")

    filepath = OUTPUT_DIR / "estado_del_arte_borrador.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Borrador estado del arte: {filepath}[/green]")


def export_referencias_tesis(papers: list[Paper]) -> None:
    """Export references only for accepted papers."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    accepted = [p for p in papers if p.status == PaperStatus.ACCEPTED]
    accepted.sort(key=lambda p: (p.authors[0] if p.authors else "", p.year or 0))
    lines = [f"# Referencias para la Tesis - Generado: {today}", ""]
    lines.append(f"Total papers aceptados: {len(accepted)}")
    lines.append("")

    for p in accepted:
        ref = p.apa7_reference or format_apa7(p)
        lines.append(f"- {ref}")
        lines.append("")

    filepath = OUTPUT_DIR / "referencias_tesis.md"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"[green]Referencias tesis: {filepath} ({len(accepted)} papers)[/green]")


def export_for_rag(papers: list[Paper]) -> None:
    """Export accepted papers metadata for the future RAG system."""
    rag_dir = Path("output/rag_corpus")
    rag_dir.mkdir(parents=True, exist_ok=True)

    # Include accepted and high-relevance new papers
    eligible = [
        p for p in papers
        if p.status == PaperStatus.ACCEPTED
        or (p.status == PaperStatus.NEW and p.relevance_level == RelevanceLevel.HIGH)
    ]

    rag_entries = []
    for p in eligible:
        entry = {
            "id": p.id,
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "doi": p.doi,
            "abstract": p.abstract,
            "categories": p.categories,
            "relevance_score": p.relevance_score,
            "citation_count": p.citation_count,
            "venue": p.venue,
            "url": p.url or (f"https://doi.org/{p.doi}" if p.doi else None),
        }
        rag_entries.append(entry)

    filepath = rag_dir / "accepted_papers_metadata.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(rag_entries, f, ensure_ascii=False, indent=2)

    console.print(f"[green]RAG corpus exportado: {filepath} ({len(rag_entries)} papers)[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis materials")
    parser.add_argument("--format", type=str, choices=["markdown", "all"], default="all")
    parser.add_argument("--only-accepted", action="store_true", help="Only accepted papers")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--for-rag", action="store_true", help="Export for RAG system")

    args = parser.parse_args()

    papers = load_database()
    if not papers:
        console.print("[yellow]Base de datos vacia.[/yellow]")
        return

    if args.only_accepted:
        papers = [p for p in papers if p.status == PaperStatus.ACCEPTED]

    console.print(f"\n[bold cyan]===== Exportacion para Tesis =====[/bold cyan]")
    console.print(f"Papers en seleccion: {len(papers)}\n")

    if args.for_rag:
        export_for_rag(papers)
        return

    export_estado_del_arte_tabla(papers, args.category)
    export_literature_map(papers)
    export_estado_del_arte_borrador(papers, args.category)
    export_referencias_tesis(papers)
    export_for_rag(papers)

    console.print(f"\n[bold green]Exportacion completada en {OUTPUT_DIR}/[/bold green]")


if __name__ == "__main__":
    main()
