"""Pydantic models for academic papers."""

from __future__ import annotations

import re
import uuid
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class RelevanceLevel(StrEnum):
    HIGH = "ALTA"
    MEDIUM = "MEDIA"
    LOW = "BAJA"


class PaperStatus(StrEnum):
    NEW = "new"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Paper(BaseModel):
    """Represents an academic paper with all metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    publication_date: str | None = None
    venue: str | None = None
    publisher: str | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    citation_count: int = 0
    categories: list[str] = Field(default_factory=list)
    relevance_score: int = 0
    relevance_level: RelevanceLevel = RelevanceLevel.LOW
    keywords_matched: list[str] = Field(default_factory=list)
    covers_gap: str | None = None
    scopus_indexed: bool | None = None
    doi_verified: bool | None = None
    source_api: str | None = None
    source_trusted: bool | None = None
    date_found: str = Field(default_factory=lambda: date.today().isoformat())
    status: PaperStatus = PaperStatus.NEW
    notes: str | None = None
    apa7_reference: str | None = None
    bibtex: str | None = None

    def truncated_abstract(self, max_words: int = 200) -> str:
        """Return abstract truncated to max_words."""
        if not self.abstract:
            return ""
        words = self.abstract.split()
        if len(words) <= max_words:
            return self.abstract
        return " ".join(words[:max_words]) + "..."

    def normalized_title(self) -> str:
        """Return lowercase title stripped of punctuation for comparison."""
        return re.sub(r"[^\w\s]", "", self.title.lower()).strip()


class ExistingPaper(BaseModel):
    """Simplified model for papers already in the thesis."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
