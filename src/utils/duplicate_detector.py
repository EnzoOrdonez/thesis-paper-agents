"""Duplicate detection for academic papers using DOI and fuzzy title matching."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from src.models.paper import Paper

TITLE_SIMILARITY_THRESHOLD = 90
TITLE_BLOCK_PREFIX_LENGTH = 18
TITLE_MIN_TOKEN_LENGTH = 3


@dataclass
class DedupIndex:
    doi_values: set[str] = field(default_factory=set)
    exact_titles: set[str] = field(default_factory=set)
    title_blocks: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))


def normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, strip punctuation."""
    return re.sub(r"[^\w\s]", "", (title or "").lower()).strip()


def normalize_doi(doi: str | None) -> str | None:
    """Normalize DOI values to a canonical lowercase form."""
    if not doi:
        return None
    normalized = doi.lower().strip()
    normalized = normalized.removeprefix("https://doi.org/")
    normalized = normalized.removeprefix("http://doi.org/")
    return normalized or None


def _title_block_keys(title: str) -> set[str]:
    normalized = normalize_title(title)
    if not normalized:
        return set()

    tokens = [token for token in normalized.split() if len(token) >= TITLE_MIN_TOKEN_LENGTH]
    keys = {f"prefix:{normalized[:TITLE_BLOCK_PREFIX_LENGTH]}"}
    if tokens:
        keys.add(f"lead:{' '.join(tokens[:3])}")
        keys.add(f"edge:{tokens[0]}:{tokens[-1]}")
    return keys


def _extract_title(item: Paper | Mapping[str, Any]) -> str:
    if isinstance(item, Paper):
        return item.title
    return str(item.get("title", ""))


def _extract_doi(item: Paper | Mapping[str, Any]) -> str | None:
    if isinstance(item, Paper):
        return item.doi
    raw = item.get("doi")
    return str(raw) if raw else None


def add_to_dedup_index(index: DedupIndex, title: str, doi: str | None = None) -> None:
    """Add a paper signature to a dedup index."""
    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        index.doi_values.add(normalized_doi)

    normalized_title = normalize_title(title)
    if not normalized_title:
        return

    index.exact_titles.add(normalized_title)
    for key in _title_block_keys(title):
        index.title_blocks[key].add(normalized_title)


def build_dedup_index(items: Iterable[Paper | Mapping[str, Any]]) -> DedupIndex:
    """Build a dedup index from papers or raw mappings."""
    index = DedupIndex()
    for item in items:
        add_to_dedup_index(index, _extract_title(item), _extract_doi(item))
    return index


def has_duplicate_in_index(
    title: str,
    doi: str | None,
    index: DedupIndex,
    threshold: int = TITLE_SIMILARITY_THRESHOLD,
) -> bool:
    """Check if a title/DOI is already present in a dedup index."""
    normalized_doi = normalize_doi(doi)
    if normalized_doi and normalized_doi in index.doi_values:
        return True

    normalized_title = normalize_title(title)
    if not normalized_title:
        return False
    if normalized_title in index.exact_titles:
        return True

    candidate_titles: set[str] = set()
    for key in _title_block_keys(title):
        candidate_titles.update(index.title_blocks.get(key, set()))

    for candidate in candidate_titles:
        if abs(len(candidate) - len(normalized_title)) > 40:
            continue
        if fuzz.ratio(normalized_title, candidate) >= threshold:
            return True

    return False


def load_existing_dedup_index(existing_path: str = "data/existing_papers.json") -> DedupIndex:
    """Load existing thesis papers into a dedup index."""
    path = Path(existing_path)
    if not path.exists():
        return DedupIndex()
    with open(path, encoding="utf-8") as f:
        existing = json.load(f)
    return build_dedup_index(existing)


def is_duplicate_by_doi(doi: str, papers: list[Paper]) -> bool:
    """Check if a DOI already exists in the paper list."""
    normalized_doi = normalize_doi(doi)
    if not normalized_doi:
        return False
    return any(normalize_doi(paper.doi) == normalized_doi for paper in papers)


def is_duplicate_by_title(title: str, papers: list[Paper], threshold: int = TITLE_SIMILARITY_THRESHOLD) -> bool:
    """Check if a similar title already exists using fuzzy matching."""
    if not title:
        return False
    normalized_title = normalize_title(title)
    for paper in papers:
        normalized_existing = normalize_title(paper.title)
        if fuzz.ratio(normalized_title, normalized_existing) >= threshold:
            return True
    return False


def is_duplicate_of_existing(
    title: str,
    doi: str | None,
    existing_path: str = "data/existing_papers.json",
    threshold: int = TITLE_SIMILARITY_THRESHOLD,
) -> bool:
    """Check if a paper is already represented in existing thesis papers."""
    index = load_existing_dedup_index(existing_path)
    return has_duplicate_in_index(title, doi, index, threshold=threshold)


def find_duplicates_in_list(papers: list[Paper]) -> list[tuple[int, int, float]]:
    """Find duplicate pairs within a list using DOI and title blocking."""
    duplicates: list[tuple[int, int, float]] = []
    seen_pairs: set[tuple[int, int]] = set()
    compared_pairs: set[tuple[int, int]] = set()
    normalized_titles = [normalize_title(paper.title) for paper in papers]

    doi_groups: dict[str, list[int]] = defaultdict(list)
    title_blocks: dict[str, set[int]] = defaultdict(set)

    for index, paper in enumerate(papers):
        normalized_doi = normalize_doi(paper.doi)
        if normalized_doi:
            doi_groups[normalized_doi].append(index)
        for key in _title_block_keys(paper.title):
            title_blocks[key].add(index)

    for indices in doi_groups.values():
        if len(indices) < 2:
            continue
        for i, j in combinations(indices, 2):
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            duplicates.append((pair[0], pair[1], 100.0))
            seen_pairs.add(pair)

    for title_indices in title_blocks.values():
        if len(title_indices) < 2:
            continue
        for i, j in combinations(sorted(title_indices), 2):
            pair = (i, j)
            if pair in seen_pairs or pair in compared_pairs:
                continue
            compared_pairs.add(pair)

            left_title = normalized_titles[i]
            right_title = normalized_titles[j]
            if not left_title or not right_title:
                continue
            if abs(len(left_title) - len(right_title)) > 40:
                continue

            ratio = fuzz.ratio(left_title, right_title)
            if ratio >= TITLE_SIMILARITY_THRESHOLD:
                duplicates.append((i, j, ratio))
                seen_pairs.add(pair)

    return duplicates
