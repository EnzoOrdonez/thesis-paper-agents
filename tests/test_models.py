"""Tests for Pydantic paper models."""

from __future__ import annotations

from src.models.paper import ExistingPaper, Paper, PaperStatus, RelevanceLevel


class TestPaperCreation:
    def test_default_fields(self):
        paper = Paper(title="Test Paper")
        assert paper.title == "Test Paper"
        assert paper.id  # auto-generated UUID
        assert paper.authors == []
        assert paper.citation_count == 0
        assert paper.relevance_score == 0
        assert paper.relevance_level == RelevanceLevel.LOW
        assert paper.status == PaperStatus.NEW
        assert paper.date_found  # auto-generated date

    def test_full_construction(self, sample_paper: Paper):
        assert sample_paper.year == 2024
        assert sample_paper.doi == "10.1234/test.2024.001"
        assert len(sample_paper.authors) == 2
        assert sample_paper.relevance_level == RelevanceLevel.HIGH

    def test_unique_ids(self):
        p1 = Paper(title="Paper A")
        p2 = Paper(title="Paper B")
        assert p1.id != p2.id


class TestNormalizedTitle:
    def test_basic_normalization(self):
        paper = Paper(title="Hello, World! A Test-Paper (2024)")
        assert paper.normalized_title() == "hello world a testpaper 2024"

    def test_uppercase(self):
        paper = Paper(title="ALL CAPS TITLE")
        assert paper.normalized_title() == "all caps title"

    def test_empty_result_stripped(self):
        paper = Paper(title="...!!!")
        assert paper.normalized_title() == ""


class TestTruncatedAbstract:
    def test_short_abstract_unchanged(self):
        paper = Paper(title="T", abstract="Short abstract here.")
        assert paper.truncated_abstract() == "Short abstract here."

    def test_long_abstract_truncated(self):
        words = ["word"] * 250
        paper = Paper(title="T", abstract=" ".join(words))
        result = paper.truncated_abstract(max_words=200)
        assert result.endswith("...")
        # "..." is appended to last word, so split gives 200 tokens
        assert len(result.split()) == 200

    def test_no_abstract(self):
        paper = Paper(title="T")
        assert paper.truncated_abstract() == ""


class TestEnumValues:
    def test_relevance_levels(self):
        assert RelevanceLevel.HIGH == "ALTA"
        assert RelevanceLevel.MEDIUM == "MEDIA"
        assert RelevanceLevel.LOW == "BAJA"

    def test_paper_statuses(self):
        assert PaperStatus.NEW == "new"
        assert PaperStatus.REVIEWED == "reviewed"
        assert PaperStatus.ACCEPTED == "accepted"
        assert PaperStatus.REJECTED == "rejected"


class TestExistingPaper:
    def test_creation(self):
        ep = ExistingPaper(title="Existing", authors=["A"], year=2023, doi="10.1/x")
        assert ep.title == "Existing"
        assert ep.year == 2023

    def test_defaults(self):
        ep = ExistingPaper(title="Minimal")
        assert ep.authors == []
        assert ep.year is None
        assert ep.doi is None
