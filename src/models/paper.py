"""Pydantic models for academic papers."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RelevanceLevel(str, Enum):
    HIGH = "ALTA"
    MEDIUM = "MEDIA"
    LOW = "BAJA"


class PaperStatus(str, Enum):
    NEW = "new"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Paper(BaseModel):
    """Represents an academic paper with all metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    publication_date: Optional[str] = None
    venue: Optional[str] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    citation_count: int = 0
    categories: list[str] = Field(default_factory=list)
    relevance_score: int = 0
    relevance_level: RelevanceLevel = RelevanceLevel.LOW
    keywords_matched: list[str] = Field(default_factory=list)
    covers_gap: Optional[str] = None
    scopus_indexed: Optional[bool] = None
    doi_verified: Optional[bool] = None
    source_api: Optional[str] = None
    date_found: str = Field(default_factory=lambda: date.today().isoformat())
    status: PaperStatus = PaperStatus.NEW
    notes: Optional[str] = None
    apa7_reference: Optional[str] = None
    bibtex: Optional[str] = None

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
        import re
        return re.sub(r"[^\w\s]", "", self.title.lower()).strip()


class ExistingPaper(BaseModel):
    """Simplified model for papers already in the thesis."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
