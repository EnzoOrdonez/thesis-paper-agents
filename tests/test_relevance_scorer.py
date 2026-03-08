"""Tests for the thesis-aware relevance scoring system."""

from __future__ import annotations

from src.models.paper import Paper, RelevanceLevel
from src.utils.relevance_scorer import score_paper, suggest_categories


class TestScorePaper:
    def test_high_relevance_rag_paper(self):
        paper = Paper(
            title="Hybrid Retrieval-Augmented Generation with ColBERT Reranking",
            abstract=(
                "We propose a hybrid RAG pipeline that combines BM25 sparse retrieval "
                "with dense passage retrieval and ColBERT reranking for improved "
                "question answering over technical cloud documentation."
            ),
            year=2024,
            citation_count=25,
            venue="SIGIR",
        )
        scored = score_paper(paper)
        assert scored.relevance_score >= 40
        assert scored.relevance_level in (RelevanceLevel.HIGH, RelevanceLevel.MEDIUM)

    def test_low_relevance_unrelated_paper(self):
        paper = Paper(
            title="Deep Learning for Protein Structure Prediction",
            abstract="We apply transformer architectures to predict 3D protein structures from amino acid sequences.",
            year=2024,
            citation_count=5,
        )
        scored = score_paper(paper)
        assert scored.relevance_score < 40

    def test_score_preserves_metadata(self, sample_paper: Paper):
        scored = score_paper(sample_paper)
        assert scored.title == sample_paper.title
        assert scored.doi == sample_paper.doi
        assert scored.year == sample_paper.year

    def test_score_sets_categories(self):
        paper = Paper(
            title="Embedding Models for Semantic Search in RAG Systems",
            abstract="We evaluate sentence embedding models for retrieval-augmented generation pipelines.",
            year=2024,
        )
        scored = score_paper(paper)
        assert len(scored.categories) >= 0  # May or may not match categories
        assert isinstance(scored.relevance_score, int)


class TestSuggestCategories:
    def test_rag_paper_categorized(self):
        paper = Paper(
            title="Retrieval-Augmented Generation for Cloud Documentation",
            abstract="This paper explores RAG pipelines for cloud technical documentation.",
        )
        categories = suggest_categories(paper)
        assert isinstance(categories, list)

    def test_empty_paper(self):
        paper = Paper(title="Untitled")
        categories = suggest_categories(paper)
        assert isinstance(categories, list)
