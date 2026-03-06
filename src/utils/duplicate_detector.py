"""Duplicate detection for academic papers using DOI and fuzzy title matching."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rapidfuzz import fuzz

from src.models.paper import Paper


TITLE_SIMILARITY_THRESHOLD = 90


def normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, strip punctuation."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()


def is_duplicate_by_doi(doi: str, papers: list[Paper]) -> bool:
    """Check if a DOI already exists in the paper list."""
    if not doi:
        return False
    doi_lower = doi.lower().strip()
    return any(
        p.doi and p.doi.lower().strip() == doi_lower
        for p in papers
    )


def is_duplicate_by_title(title: str, papers: list[Paper], threshold: int = TITLE_SIMILARITY_THRESHOLD) -> bool:
    """Check if a similar title already exists using fuzzy matching."""
    if not title:
        return False
    norm_title = normalize_title(title)
    for p in papers:
        norm_existing = normalize_title(p.title)
        ratio = fuzz.ratio(norm_title, norm_existing)
        if ratio >= threshold:
            return True
    return False


def is_duplicate_of_existing(
    title: str,
    doi: str | None,
    existing_path: str = "data/existing_papers.json",
    threshold: int = TITLE_SIMILARITY_THRESHOLD,
) -> bool:
    """Check if paper is a duplicate of one already in the thesis (existing_papers.json)."""
    path = Path(existing_path)
    if not path.exists():
        return False

    with open(path, encoding="utf-8") as f:
        existing = json.load(f)

    if doi:
        doi_lower = doi.lower().strip()
        for ep in existing:
            if ep.get("doi") and ep["doi"].lower().strip() == doi_lower:
                return True

    if title:
        norm_title = normalize_title(title)
        for ep in existing:
            norm_existing = normalize_title(ep.get("title", ""))
            ratio = fuzz.ratio(norm_title, norm_existing)
            if ratio >= threshold:
                return True

    return False


def find_duplicates_in_list(papers: list[Paper]) -> list[tuple[int, int, float]]:
    """Find duplicate pairs within a list of papers.

    Returns:
        List of (index_a, index_b, similarity_score) tuples.
    """
    duplicates: list[tuple[int, int, float]] = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            # Check DOI first
            if papers[i].doi and papers[j].doi:
                if papers[i].doi.lower().strip() == papers[j].doi.lower().strip():
                    duplicates.append((i, j, 100.0))
                    continue

            # Check title similarity
            ratio = fuzz.ratio(
                normalize_title(papers[i].title),
                normalize_title(papers[j].title),
            )
            if ratio >= TITLE_SIMILARITY_THRESHOLD:
                duplicates.append((i, j, ratio))

    return duplicates
