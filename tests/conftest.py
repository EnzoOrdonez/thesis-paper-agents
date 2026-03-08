"""Shared fixtures for thesis-paper-agents tests."""

from __future__ import annotations

import pytest

from src.models.paper import Paper, PaperStatus, RelevanceLevel


@pytest.fixture
def sample_paper() -> Paper:
    """A typical RAG-related paper for testing."""
    return Paper(
        title="Hybrid Retrieval-Augmented Generation for Cloud Documentation",
        authors=["Alice Smith", "Bob Jones"],
        year=2024,
        doi="10.1234/test.2024.001",
        abstract="This paper presents a hybrid RAG system combining dense and sparse retrieval for cloud documentation.",
        citation_count=15,
        relevance_score=75,
        relevance_level=RelevanceLevel.HIGH,
        categories=["Sistemas RAG Hibridos"],
        keywords_matched=["hybrid rag", "retrieval-augmented generation", "cloud documentation"],
        source_api="semantic_scholar",
        source_trusted=True,
        status=PaperStatus.NEW,
    )


@pytest.fixture
def sample_paper_low() -> Paper:
    """A low-relevance paper for contrast in tests."""
    return Paper(
        title="Machine Learning for Weather Prediction Using Satellite Data",
        authors=["Jane Doe"],
        year=2023,
        abstract="We apply deep learning models to predict weather patterns from satellite imagery.",
        citation_count=3,
        relevance_score=10,
        relevance_level=RelevanceLevel.LOW,
        source_api="openalex",
    )


@pytest.fixture
def paper_list(sample_paper: Paper, sample_paper_low: Paper) -> list[Paper]:
    """A small list of papers for dedup and list operation tests."""
    return [
        sample_paper,
        sample_paper_low,
        Paper(
            title="Dense Passage Retrieval for Open-Domain Question Answering",
            authors=["Vladimir Karpukhin", "Barlas Oguz"],
            year=2020,
            doi="10.18653/v1/2020.emnlp-main.550",
            abstract="Dense retrieval using BERT-based encoders outperforms BM25 on several QA benchmarks.",
            citation_count=3200,
            relevance_score=85,
            relevance_level=RelevanceLevel.HIGH,
            categories=["Metodos de Recuperacion"],
            keywords_matched=["dense retrieval", "passage retrieval"],
            source_api="semantic_scholar",
        ),
    ]
