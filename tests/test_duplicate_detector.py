"""Tests for duplicate detection logic."""

from __future__ import annotations

from src.models.paper import Paper
from src.utils.duplicate_detector import (
    DedupIndex,
    _title_block_keys,
    add_to_dedup_index,
    build_dedup_index,
    find_duplicates_in_list,
    has_duplicate_in_index,
    is_duplicate_by_doi,
    is_duplicate_by_title,
    normalize_doi,
    normalize_title,
)


class TestNormalization:
    def test_normalize_title_basic(self):
        assert normalize_title("Hello, World!") == "hello world"

    def test_normalize_title_empty(self):
        assert normalize_title("") == ""

    def test_normalize_title_none(self):
        assert normalize_title(None) == ""

    def test_normalize_doi_strips_prefix(self):
        assert normalize_doi("https://doi.org/10.1234/test") == "10.1234/test"
        assert normalize_doi("http://doi.org/10.1234/test") == "10.1234/test"

    def test_normalize_doi_lowercase(self):
        assert normalize_doi("10.1234/TEST") == "10.1234/test"

    def test_normalize_doi_none(self):
        assert normalize_doi(None) is None
        assert normalize_doi("") is None


class TestTitleBlockKeys:
    def test_generates_keys(self):
        keys = _title_block_keys("Hybrid Retrieval Augmented Generation for Cloud")
        assert any(k.startswith("prefix:") for k in keys)
        assert any(k.startswith("lead:") for k in keys)
        assert any(k.startswith("edge:") for k in keys)

    def test_empty_title(self):
        assert _title_block_keys("") == set()

    def test_short_title(self):
        keys = _title_block_keys("Hi")
        # "hi" has no tokens >= 3 chars, so only prefix key
        assert len(keys) >= 1


class TestDedupIndex:
    def test_doi_duplicate(self):
        index = DedupIndex()
        add_to_dedup_index(index, "Paper A", doi="10.1234/a")
        assert has_duplicate_in_index("Different Title", "10.1234/a", index)

    def test_exact_title_duplicate(self):
        index = DedupIndex()
        add_to_dedup_index(index, "Hybrid RAG for Cloud Docs", doi=None)
        assert has_duplicate_in_index("Hybrid RAG for Cloud Docs", None, index)

    def test_fuzzy_title_duplicate(self):
        index = DedupIndex()
        add_to_dedup_index(index, "Hybrid Retrieval-Augmented Generation for Cloud Documentation")
        assert has_duplicate_in_index(
            "Hybrid Retrieval Augmented Generation for Cloud Documentation",
            None,
            index,
        )

    def test_no_duplicate(self):
        index = DedupIndex()
        add_to_dedup_index(index, "Paper About Machine Learning")
        assert not has_duplicate_in_index("Quantum Computing Survey", None, index)

    def test_build_from_papers(self, paper_list: list[Paper]):
        index = build_dedup_index(paper_list)
        assert len(index.exact_titles) == 3
        assert has_duplicate_in_index(paper_list[0].title, paper_list[0].doi, index)


class TestDuplicateByDoi:
    def test_finds_duplicate(self, paper_list: list[Paper]):
        assert is_duplicate_by_doi("10.1234/test.2024.001", paper_list)

    def test_no_duplicate(self, paper_list: list[Paper]):
        assert not is_duplicate_by_doi("10.9999/nonexistent", paper_list)

    def test_empty_doi(self, paper_list: list[Paper]):
        assert not is_duplicate_by_doi("", paper_list)


class TestDuplicateByTitle:
    def test_exact_match(self, paper_list: list[Paper]):
        assert is_duplicate_by_title(paper_list[0].title, paper_list)

    def test_no_match(self, paper_list: list[Paper]):
        assert not is_duplicate_by_title("Completely Unrelated Title About Nothing", paper_list)

    def test_empty_title(self, paper_list: list[Paper]):
        assert not is_duplicate_by_title("", paper_list)


class TestFindDuplicatesInList:
    def test_no_duplicates(self, paper_list: list[Paper]):
        dupes = find_duplicates_in_list(paper_list)
        assert dupes == []

    def test_finds_doi_duplicates(self, sample_paper: Paper):
        p2 = Paper(title="Same DOI Paper", doi=sample_paper.doi)
        dupes = find_duplicates_in_list([sample_paper, p2])
        assert len(dupes) == 1
        assert dupes[0][2] == 100.0  # DOI match = 100%

    def test_empty_list(self):
        assert find_duplicates_in_list([]) == []
