"""APA 7 and BibTeX reference formatting for academic papers."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from src.models.paper import Paper


def _format_author_apa(name: str) -> str:
    """Format a single author name for APA 7: 'Last, F. M.'

    Handles common formats:
      - "First Last" -> "Last, F."
      - "First Middle Last" -> "Last, F. M."
      - "Last, First" -> "Last, F."
    """
    name = name.strip()
    if not name:
        return name

    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        last = parts[0]
        given = parts[1] if len(parts) > 1 else ""
        initials = " ".join(f"{w[0]}." for w in given.split() if w)
        return f"{last}, {initials}".strip()

    parts = name.split()
    if len(parts) == 1:
        return parts[0]

    last = parts[-1]
    initials = " ".join(f"{w[0]}." for w in parts[:-1] if w)
    return f"{last}, {initials}"


def format_apa7(paper: Paper) -> str:
    """Format a paper reference in APA 7 style.

    Patterns:
      Journal article:
        Author, A. A., & Author, B. B. (Year). Title. *Journal*, vol(issue), pages. https://doi.org/xxxx
      Conference paper:
        Author, A. A. (Year). Title. In *Proceedings of Conference* (pp. pages). Publisher. https://doi.org/xxxx
      Preprint:
        Author, A. A. (Year). Title. *arXiv*. https://doi.org/xxxx
    """
    # Format authors
    if not paper.authors:
        author_str = "Unknown Author"
    elif len(paper.authors) == 1:
        author_str = _format_author_apa(paper.authors[0])
    elif len(paper.authors) == 2:
        author_str = f"{_format_author_apa(paper.authors[0])} & {_format_author_apa(paper.authors[1])}"
    elif len(paper.authors) <= 20:
        formatted = [_format_author_apa(a) for a in paper.authors[:-1]]
        author_str = ", ".join(formatted) + ", & " + _format_author_apa(paper.authors[-1])
    else:
        formatted = [_format_author_apa(a) for a in paper.authors[:19]]
        author_str = ", ".join(formatted) + ", ... " + _format_author_apa(paper.authors[-1])

    year = paper.year or "n.d."
    title = paper.title.rstrip(".")

    # Build reference
    parts = [f"{author_str} ({year}). {title}."]

    if paper.venue:
        parts.append(f" *{paper.venue}*.")

    if paper.doi:
        doi_url = paper.doi if paper.doi.startswith("http") else f"https://doi.org/{paper.doi}"
        parts.append(f" {doi_url}")
    elif paper.url:
        parts.append(f" {paper.url}")

    return "".join(parts)


def _make_bibtex_key(paper: Paper) -> str:
    """Generate a BibTeX citation key: firstauthorlastname + year + first_title_word."""
    first_author = paper.authors[0] if paper.authors else "unknown"
    # Extract last name
    if "," in first_author:
        last = first_author.split(",")[0].strip()
    else:
        parts = first_author.split()
        last = parts[-1] if parts else "unknown"

    # Clean for BibTeX key
    last = re.sub(r"[^\w]", "", last.lower())
    year = str(paper.year) if paper.year else "nd"

    # First meaningful word of title
    title_words = re.sub(r"[^\w\s]", "", paper.title.lower()).split()
    stop_words = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with"}
    first_word = ""
    for w in title_words:
        if w not in stop_words:
            first_word = w
            break
    if not first_word and title_words:
        first_word = title_words[0]

    return f"{last}{year}{first_word}"


def format_bibtex(paper: Paper) -> str:
    """Format a paper as a BibTeX entry."""
    key = _make_bibtex_key(paper)
    authors_bib = " and ".join(paper.authors) if paper.authors else "Unknown"

    entry_type = "article"
    venue_lower = (paper.venue or "").lower()
    if any(kw in venue_lower for kw in ("conference", "proceedings", "workshop", "symposium")):
        entry_type = "inproceedings"
    elif "arxiv" in venue_lower or paper.source_api == "arxiv":
        entry_type = "misc"

    lines = [f"@{entry_type}{{{key},"]
    lines.append(f"  title = {{{paper.title}}},")
    lines.append(f"  author = {{{authors_bib}}},")
    if paper.year:
        lines.append(f"  year = {{{paper.year}}},")
    if paper.venue:
        if entry_type == "inproceedings":
            lines.append(f"  booktitle = {{{paper.venue}}},")
        else:
            lines.append(f"  journal = {{{paper.venue}}},")
    if paper.doi:
        doi_val = paper.doi
        if doi_val.startswith("https://doi.org/"):
            doi_val = doi_val[len("https://doi.org/"):]
        elif doi_val.startswith("http://doi.org/"):
            doi_val = doi_val[len("http://doi.org/"):]
        lines.append(f"  doi = {{{doi_val}}},")
    if paper.url:
        lines.append(f"  url = {{{paper.url}}},")
    if paper.abstract:
        # Truncate abstract for BibTeX
        abstract_short = paper.truncated_abstract(100)
        lines.append(f"  abstract = {{{abstract_short}}},")
    lines.append("}")

    return "\n".join(lines)
