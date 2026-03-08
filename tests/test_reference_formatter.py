"""Tests for APA 7 and BibTeX reference formatting."""

from __future__ import annotations

from src.models.paper import Paper
from src.utils.reference_formatter import format_apa7, format_bibtex


class TestAPA7:
    def test_single_author(self):
        paper = Paper(title="Test Paper", authors=["John Smith"], year=2024, doi="10.1234/test")
        ref = format_apa7(paper)
        assert "Smith, J." in ref
        assert "(2024)" in ref
        assert "Test Paper" in ref
        assert "https://doi.org/10.1234/test" in ref

    def test_two_authors(self):
        paper = Paper(title="Test", authors=["Alice Smith", "Bob Jones"], year=2024)
        ref = format_apa7(paper)
        assert "Smith, A." in ref
        assert "& Jones, B." in ref

    def test_no_authors(self):
        paper = Paper(title="Orphan Paper", year=2023)
        ref = format_apa7(paper)
        assert "Unknown Author" in ref

    def test_no_year(self):
        paper = Paper(title="Timeless", authors=["A"])
        ref = format_apa7(paper)
        assert "(n.d.)" in ref

    def test_venue_included(self):
        paper = Paper(title="Test", authors=["A B"], year=2024, venue="SIGIR 2024")
        ref = format_apa7(paper)
        assert "*SIGIR 2024*" in ref

    def test_url_fallback(self):
        paper = Paper(title="Test", authors=["A B"], year=2024, url="https://example.com")
        ref = format_apa7(paper)
        assert "https://example.com" in ref


class TestBibTeX:
    def test_article_format(self):
        paper = Paper(
            title="Dense Retrieval Methods",
            authors=["Alice Smith", "Bob Jones"],
            year=2024,
            venue="Information Retrieval Journal",
            doi="10.1234/irj.2024",
        )
        bib = format_bibtex(paper)
        assert bib.startswith("@article{")
        assert "title = {Dense Retrieval Methods}" in bib
        assert "author = {Alice Smith and Bob Jones}" in bib
        assert "year = {2024}" in bib
        assert "journal = {Information Retrieval Journal}" in bib
        assert "doi = {10.1234/irj.2024}" in bib

    def test_inproceedings_format(self):
        paper = Paper(title="Test", authors=["A"], year=2024, venue="Proceedings of SIGIR")
        bib = format_bibtex(paper)
        assert bib.startswith("@inproceedings{")
        assert "booktitle = {Proceedings of SIGIR}" in bib

    def test_arxiv_misc(self):
        paper = Paper(title="Test", authors=["A"], year=2024, source_api="arxiv")
        bib = format_bibtex(paper)
        assert bib.startswith("@misc{")

    def test_no_authors(self):
        paper = Paper(title="Orphan Paper", year=2023)
        bib = format_bibtex(paper)
        assert "author = {Unknown}" in bib

    def test_doi_prefix_stripped(self):
        paper = Paper(title="T", authors=["A"], doi="https://doi.org/10.1234/x")
        bib = format_bibtex(paper)
        assert "doi = {10.1234/x}" in bib
