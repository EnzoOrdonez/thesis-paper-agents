"""Relevance scoring for academic papers based on thesis criteria.

Enhanced with:
- Cached config/data loading to avoid repeated disk I/O per paper
- TF-IDF-inspired keyword density scoring
- Age-relative citation scoring
- Genericity penalty for off-topic papers mentioning RAG tangentially
- Bonus for papers citing existing thesis references
- Stricter foundational gap detection to reduce false positives
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.models.paper import Paper, RelevanceLevel

FOUNDATIONAL_GAP_RULES: dict[str, dict[str, Any]] = {
    "ColBERT original": {
        "title_any": ["colbert"],
        "text_any": ["late interaction", "passage search"],
        "author_any": ["khattab", "zahia", "zaharia"],
        "year_min": 2020,
        "year_max": 2021,
    },
    "DPR original": {
        "title_any": ["dense passage retrieval"],
        "text_any": ["open domain question answering", "open-domain question answering"],
        "author_any": ["karpukhin"],
        "year_min": 2020,
        "year_max": 2021,
    },
    "Sentence-BERT original": {
        "title_any": ["sentence bert", "sentence-bert"],
        "text_any": ["siamese bert networks", "sentence embeddings"],
        "author_any": ["reimers", "gurevych"],
        "year_min": 2019,
        "year_max": 2021,
    },
    "BGE embeddings": {
        "title_any": ["bge-m3", "bge m3", "c-pack", "baai general embedding", "general embeddings"],
        "text_any": ["baai", "general embeddings", "shitao xiao", "zheng liu", "c-pack"],
        "year_min": 2023,
    },
    "BM25 fundacional": {
        "title_any": ["bm25", "okapi"],
        "text_any": ["robertson", "zaragoza", "probabilistic relevance", "okapi"],
        "year_max": 2012,
    },
    "Documentacion cloud como objeto de estudio": {
        "text_all": [["cloud", "documentation"]],
        "text_any": [
            "technical documentation",
            "api documentation",
            "usability",
            "developer documentation",
            "documentation usability",
            "documentation complexity",
        ],
    },
    "RAGAS evaluation framework": {
        "title_any": ["ragas"],
        "text_any": ["ragas", "retrieval augmented generation assessment"],
        "year_min": 2023,
    },
    "Normalizacion terminologica en dominios tecnicos": {
        "text_all": [["terminology", "normalization"], ["acronym", "normalization"], ["term", "normalization"]],
        "text_any": [
            "technical domain",
            "technical documentation",
            "domain specific terminology",
            "terminology normalization",
            "acronym normalization",
        ],
    },
}


def _cache_key(path: str) -> tuple[str, int | None]:
    p = Path(path)
    try:
        return str(p.resolve(strict=False)), p.stat().st_mtime_ns
    except FileNotFoundError:
        return str(p.resolve(strict=False)), None


@lru_cache(maxsize=16)
def _load_yaml(cache_key: tuple[str, int | None]) -> dict[str, Any]:
    resolved_path, modified_ns = cache_key
    if modified_ns is None:
        return {}
    with open(resolved_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=16)
def _load_json(cache_key: tuple[str, int | None]) -> list[dict]:
    resolved_path, modified_ns = cache_key
    if modified_ns is None:
        return []
    with open(resolved_path, encoding="utf-8") as f:
        result: list[dict] = json.load(f)
        return result


def _load_trusted_sources(path: str = "config/trusted_sources.yaml") -> dict[str, Any]:
    return _load_yaml(_cache_key(path))


def _load_app_config(path: str = "config/config.yaml") -> dict[str, Any]:
    return _load_yaml(_cache_key(path))


def _load_keywords(path: str = "config/keywords.yaml") -> dict[str, list[str]]:
    data = _load_yaml(_cache_key(path))
    result: dict[str, list[str]] = data.get("keyword_groups", {})
    return result


def _load_existing_papers(path: str = "data/existing_papers.json") -> list[dict]:
    return _load_json(_cache_key(path))


def _load_pending_gaps(path: str = "config/trusted_sources.yaml") -> list[dict]:
    data = _load_yaml(_cache_key(path))
    result: list[dict] = data.get("pending_gaps", [])
    return result


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _text_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    return set(normalized.split())


def _text_contains_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return which keywords appear in the text using token-aware matching."""
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return []

    tokens = set(normalized_text.split())
    matched: list[str] = []
    for keyword in keywords:
        normalized_keyword = _normalize_text(keyword)
        if not normalized_keyword:
            continue
        keyword_tokens = normalized_keyword.split()
        if len(keyword_tokens) == 1:
            if keyword_tokens[0] in tokens:
                matched.append(keyword)
        elif normalized_keyword in normalized_text or all(token in tokens for token in keyword_tokens):
            matched.append(keyword)
    return matched


def _keyword_density_score(text: str, keywords: list[str]) -> float:
    """Calculate a TF-IDF-inspired density score from thesis-relevant tokens."""
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return 0.0

    words = normalized_text.split()
    total_words = max(len(words), 1)

    keyword_tokens: set[str] = set()
    for keyword in keywords:
        for token in _normalize_text(keyword).split():
            if len(token) > 3:
                keyword_tokens.add(token)

    hits = sum(1 for word in words if word in keyword_tokens)
    density = hits / total_words
    return min(density / 0.10, 1.0)


def _venue_tier(venue: str | None, publisher: str | None, sources: dict) -> int:
    """Return points based on venue/publisher tier."""
    if not venue and not publisher:
        return 0
    combined = f"{venue or ''} {publisher or ''}".lower()
    trusted_sources = sources.get("trusted_sources", {})

    for conference in trusted_sources.get("conferences_tier1", []):
        if conference.lower() in combined:
            return 20
    for publisher_name in trusted_sources.get("publishers", []):
        if publisher_name.lower() in combined:
            return 20
    for journal in trusted_sources.get("journals", []):
        if journal.lower() in combined:
            return 20
    for conference in trusted_sources.get("conferences_tier2", []):
        if conference.lower() in combined:
            return 10
    for preprint in trusted_sources.get("preprints", []):
        if preprint.lower() in combined:
            return 5
    return 0


def source_tier(paper: Paper, sources: dict | None = None) -> int:
    """Expose the source tier score for downstream filtering/reporting."""
    if sources is None:
        sources = _load_trusted_sources()
    return _venue_tier(paper.venue, paper.publisher, sources)


def _year_score(year: int | None) -> int:
    """Return points based on publication year."""
    if year is None:
        return 0
    if year >= 2025:
        return 15
    if year == 2024:
        return 10
    if year == 2023:
        return 5
    return 0


def _citation_score_relative(count: int, year: int | None) -> int:
    """Score citations relative to paper age."""
    if year is None:
        if count >= 100:
            return 15
        if count >= 50:
            return 10
        if count >= 10:
            return 5
        return 0

    current_year = date.today().year
    age = max(current_year - year, 0)

    if age == 0:
        if count >= 10:
            return 15
        if count >= 5:
            return 10
        return 5

    cites_per_year = count / max(age, 1)

    if cites_per_year >= 50:
        return 15
    if cites_per_year >= 20:
        return 12
    if count >= 100:
        return 15
    if count >= 50:
        return 10
    if count >= 10:
        return 5
    return 0


def _genericity_penalty(paper: Paper) -> int:
    """Penalize papers that mention RAG/retrieval only tangentially."""
    combined = f"{paper.title or ''} {paper.abstract or ''}".lower()

    off_topic_indicators = [
        "medical",
        "clinical",
        "patient",
        "diagnosis",
        "healthcare",
        "legal",
        "court",
        "jurisdiction",
        "statute",
        "financial",
        "stock",
        "trading",
        "portfolio",
        "biology",
        "genomic",
        "protein",
        "molecular",
        "chemistry",
        "chemical",
        "reaction",
        "agriculture",
        "crop",
        "soil",
        "physics",
        "quantum",
        "particle",
    ]

    on_topic_core = [
        "information retrieval",
        "natural language processing",
        "text retrieval",
        "document retrieval",
        "passage retrieval",
        "semantic search",
        "embedding",
        "vector search",
        "question answering",
        "knowledge base",
        "retrieval augmented",
        "rag system",
        "rag pipeline",
        "language model",
        "transformer",
        "bert",
        "chunking",
        "reranking",
        "re-ranking",
        "bm25",
        "dense retrieval",
        "hybrid search",
        "cloud documentation",
        "technical documentation",
    ]

    off_topic_count = sum(1 for keyword in off_topic_indicators if keyword in combined)
    on_topic_count = sum(1 for keyword in on_topic_core if keyword in combined)

    if off_topic_count >= 3 and on_topic_count <= 1:
        return -20
    if off_topic_count >= 2 and on_topic_count == 0:
        return -15

    return 0


def _methodology_focus_bonus(paper: Paper) -> int:
    """Reward generalizable retrieval/system papers over domain-only applications."""
    combined = f"{paper.title or ''} {paper.abstract or ''}".lower()
    title_text = (paper.title or "").lower()
    bonus = 0

    retrieval_method_signals = [
        "retriever",
        "retrievers",
        "retrieval pipeline",
        "retrieval system",
        "reranker",
        "reranking",
        "re-ranking",
        "similarity metric",
        "similarity metrics",
        "index",
        "indexing",
        "vector database",
        "faiss",
        "milvus",
        "pgvector",
        "bm25",
        "dense retrieval",
        "dense passage retrieval",
        "colbert",
        "cross-encoder",
        "late interaction",
        "approximate nearest neighbor",
        "chunking",
        "segmentation",
        "embedding model",
        "embedding models",
    ]
    evaluation_signals = [
        "empirical evaluation",
        "systematic evaluation",
        "benchmark",
        "benchmarks",
        "comparison",
        "comparative",
        "ablation",
        "effect size",
        "anova",
        "ndcg",
        "mrr",
        "recall",
        "precision",
        "latency",
        "cost",
    ]
    cloud_doc_signals = [
        "technical documentation",
        "api documentation",
        "developer documentation",
        "cloud documentation",
        "documentation retrieval",
        "knowledge base",
        "aws",
        "azure",
        "gcp",
        "google cloud",
    ]

    retrieval_hits = sum(1 for signal in retrieval_method_signals if signal in combined)
    evaluation_hits = sum(1 for signal in evaluation_signals if signal in combined)
    cloud_hits = sum(1 for signal in cloud_doc_signals if signal in combined)

    if retrieval_hits >= 4:
        bonus += 8
    elif retrieval_hits >= 2:
        bonus += 5
    elif retrieval_hits >= 1:
        bonus += 2

    if retrieval_hits >= 2 and evaluation_hits >= 3:
        bonus += 4
    elif retrieval_hits >= 1 and evaluation_hits >= 2:
        bonus += 2

    if cloud_hits >= 2:
        bonus += 10
    elif cloud_hits >= 1:
        bonus += 6

    if (retrieval_hits >= 1 or cloud_hits >= 1) and any(
        signal in title_text for signal in ["benchmark", "evaluation", "comparison", "survey"]
    ):
        bonus += 2

    return bonus


def _applied_domain_penalty(paper: Paper) -> int:
    """Penalize papers whose main contribution is a niche application outside thesis scope."""
    combined = f"{paper.title or ''} {paper.abstract or ''}".lower()

    domain_groups = {
        "medical": [
            "medical",
            "clinical",
            "healthcare",
            "patient",
            "pubmedqa",
            "medqa",
            "medmcqa",
            "biomedical",
            "hospital",
        ],
        "telecom": [
            "6g",
            "wireless",
            "radio frequency",
            "rf sensing",
            "spectrogram",
            "radar",
            "communications intelligence",
            "telecommunication",
        ],
        "finance": [
            "financial",
            "fintech",
            "stock",
            "trading",
            "portfolio",
            "compliance domain",
        ],
        "legal": [
            "legal",
            "court",
            "jurisdiction",
            "regulation",
            "statute",
        ],
        "science": [
            "materials science",
            "graphene",
            "molecular",
            "genomic",
            "protein",
            "chemistry",
        ],
        "sensor": [
            "eeg",
            "wearable",
            "sensor data",
            "disaster impact",
            "museum exhibition",
        ],
    }
    thesis_anchor_signals = [
        "technical documentation",
        "api documentation",
        "developer documentation",
        "cloud documentation",
        "aws",
        "azure",
        "gcp",
        "google cloud",
    ]

    domain_hits = sum(1 for signals in domain_groups.values() if any(signal in combined for signal in signals))
    anchor_hits = sum(1 for signal in thesis_anchor_signals if signal in combined)

    if any(signal in combined for signal in ["technical documentation", "api documentation", "cloud documentation"]):
        return 0

    if domain_hits >= 2:
        penalty = 20
    elif domain_hits == 1:
        penalty = 16
    else:
        return 0

    if anchor_hits >= 2:
        penalty -= 6
    elif anchor_hits == 1:
        penalty -= 3

    return -max(penalty, 0)


def _existing_paper_bonus(paper: Paper) -> int:
    """Give bonus if paper cites or references papers already in our thesis."""
    existing = _load_existing_papers()
    if not existing:
        return 0

    combined = f"{paper.title or ''} {paper.abstract or ''}".lower()
    bonus = 0

    key_authors = set()
    for existing_paper in existing:
        for author in existing_paper.get("authors", []):
            cleaned = author.replace("et al.", "").strip()
            parts = cleaned.split()
            if parts:
                surname = parts[-1].lower()
                if len(surname) > 2:
                    key_authors.add(surname)

    matched_authors = sum(1 for author in key_authors if author in combined)
    if matched_authors >= 2:
        bonus = 15
    elif matched_authors >= 1:
        bonus = 8

    return bonus


def _thesis_alignment_bonus(paper: Paper) -> int:
    """Reward papers that align with the exact thesis problem, not just generic RAG."""
    combined = f"{paper.title or ''} {paper.abstract or ''}".lower()
    bonus = 0

    hybrid_signals = [
        "hybrid retrieval",
        "hybrid search",
        "bm25",
        "lexical",
        "sparse",
        "dense retrieval",
        "semantic search",
        "late interaction",
    ]
    cloud_doc_signals = [
        "technical documentation",
        "api documentation",
        "developer documentation",
        "cloud documentation",
        "aws",
        "azure",
        "gcp",
        "google cloud",
    ]
    comparison_signals = [
        "benchmark",
        "comparison",
        "compare",
        "versus",
        "vs",
        "ablation",
        "evaluation",
    ]

    hybrid_matches = sum(1 for signal in hybrid_signals if signal in combined)
    cloud_matches = sum(1 for signal in cloud_doc_signals if signal in combined)
    comparison_matches = sum(1 for signal in comparison_signals if signal in combined)

    if hybrid_matches >= 2:
        bonus += 8
    if cloud_matches >= 1:
        bonus += 6
    if hybrid_matches >= 1 and comparison_matches >= 1:
        bonus += 4

    return bonus


def is_from_trusted_source(paper: Paper, sources: dict | None = None) -> bool:
    """Check if a paper comes from a trusted source."""
    return source_tier(paper, sources) > 0 or paper.source_api == "arxiv"


def ranking_score(paper: Paper, config: dict[str, Any] | None = None) -> int:
    """Return a trust-adjusted ranking score for ordering reports and top lists."""
    general_config = (config or _load_app_config()).get("general", {})
    score = paper.relevance_score

    if not paper.source_trusted:
        score -= general_config.get("provisional_ranking_penalty", 15)
    if not paper.doi:
        score -= general_config.get("missing_doi_ranking_penalty", 3)
    if not paper.venue or paper.venue == "N/A":
        score -= general_config.get("missing_venue_ranking_penalty", 2)

    return max(score, 0)


def score_paper(paper: Paper) -> Paper:
    """Calculate relevance score for a paper and update it in-place."""
    sources = _load_trusted_sources()
    keyword_groups = _load_keywords()
    general_config = _load_app_config().get("general", {})
    high_threshold = general_config.get("relevance_threshold_high", 70)
    medium_threshold = general_config.get("relevance_threshold_medium", 40)

    total = 0
    all_matched: list[str] = []

    all_keyword_tokens: list[str] = []
    for keywords in keyword_groups.values():
        all_keyword_tokens.extend(keywords)

    title_match_found = False
    for keywords in keyword_groups.values():
        matched = _text_contains_keywords(paper.title or "", keywords)
        if matched:
            title_match_found = True
            all_matched.extend(matched)
    if title_match_found:
        density = _keyword_density_score(paper.title or "", all_keyword_tokens)
        total += 20 + int(density * 5)

    abstract_groups_matched = 0
    for keywords in keyword_groups.values():
        matched = _text_contains_keywords(paper.abstract or "", keywords)
        if matched:
            abstract_groups_matched += 1
            all_matched.extend(matched)
    if abstract_groups_matched > 0:
        base_points = min(15, abstract_groups_matched * 4)
        density = _keyword_density_score(paper.abstract or "", all_keyword_tokens)
        total += base_points + int(density * 5)

    total += _venue_tier(paper.venue, paper.publisher, sources)
    total += _year_score(paper.year)
    total += _citation_score_relative(paper.citation_count, paper.year)
    total += _existing_paper_bonus(paper)
    total += _thesis_alignment_bonus(paper)
    total += _methodology_focus_bonus(paper)
    total += _applied_domain_penalty(paper)
    total += _genericity_penalty(paper)

    total = max(0, min(100, total))

    paper.relevance_score = total
    paper.keywords_matched = list(dict.fromkeys(all_matched))

    if total >= high_threshold:
        paper.relevance_level = RelevanceLevel.HIGH
    elif total >= medium_threshold:
        paper.relevance_level = RelevanceLevel.MEDIUM
    else:
        paper.relevance_level = RelevanceLevel.LOW

    return paper


def suggest_categories(paper: Paper) -> list[str]:
    """Suggest thesis categories based on keyword matches and content."""
    keyword_groups = _load_keywords()
    title_text = (paper.title or "").lower()
    abstract_text = (paper.abstract or "").lower()
    combined_text = f"{title_text} {abstract_text}"

    group_to_category = {
        "core_rag": "Sistemas RAG Hibridos",
        "retrieval_methods": "Estrategias de Recuperacion",
        "embeddings": "Modelos de Embedding",
        "chunking_preprocessing": "Segmentacion/Chunking",
        "reranking": "Re-ranking",
        "hallucination": "Alucinaciones en LLMs",
        "evaluation": "Evaluacion de sistemas RAG",
        "cloud_documentation": "Documentacion Cloud",
        "vector_databases": "Vector Databases",
    }

    category_scores: dict[str, float] = {}

    for group_name, keywords in keyword_groups.items():
        if group_name not in group_to_category:
            continue
        category = group_to_category[group_name]

        title_matched = _text_contains_keywords(title_text, keywords)
        abstract_matched = _text_contains_keywords(abstract_text, keywords)

        score = len(title_matched) * 3.0 + len(abstract_matched) * 1.0
        if score > 0:
            category_scores[category] = score

    rag_indicators = ["rag", "retrieval augmented generation", "retrieval-augmented generation"]
    has_rag = any(indicator in combined_text for indicator in rag_indicators)
    has_rag_title = any(indicator in title_text for indicator in rag_indicators)

    hybrid_indicators = [
        "hybrid retrieval",
        "hybrid search",
        "hybrid rag",
        "bm25 and semantic",
        "bm25 and dense",
        "bm25 + dense",
        "bm25 + semantic",
        "lexical and semantic",
        "lexical and dense",
        "sparse and dense",
        "hybrid model",
        "hybrid approach",
    ]
    has_hybrid = any(indicator in combined_text for indicator in hybrid_indicators)

    if "Sistemas RAG Hibridos" in category_scores:
        if has_rag and (has_hybrid or has_rag_title):
            pass
        elif has_rag:
            category_scores["Sistemas RAG Hibridos"] *= 0.3
        else:
            del category_scores["Sistemas RAG Hibridos"]
            if any(keyword in combined_text for keyword in ["evaluation", "benchmark", "metric"]):
                category_scores.setdefault("Evaluacion de sistemas RAG", 1.0)
            elif any(keyword in combined_text for keyword in ["retrieval", "search", "ranking"]):
                category_scores.setdefault("Estrategias de Recuperacion", 1.0)

    normalization_keywords = [
        "terminology normalization",
        "term normalization",
        "acronym normalization",
        "preprocessing pipeline",
        "text preprocessing",
        "text normalization",
    ]
    if any(keyword in combined_text for keyword in normalization_keywords):
        category_scores.setdefault("Normalizacion/Preprocesamiento", 2.0)

    metric_keywords = [
        "ndcg",
        "mrr",
        "recall@",
        "precision@",
        "f1 score",
        "exact match",
        "mean reciprocal rank",
        "evaluation metric",
    ]
    if any(keyword in combined_text for keyword in metric_keywords):
        category_scores.setdefault("Metricas de evaluacion", 2.0)

    sorted_categories = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    categories = [category for category, _ in sorted_categories[:3]]

    if not categories:
        if any(indicator in combined_text for indicator in rag_indicators):
            return ["Sistemas RAG Hibridos"]
        if any(keyword in combined_text for keyword in ["hallucination", "faithfulness", "groundedness"]):
            return ["Alucinaciones en LLMs"]
        if any(keyword in combined_text for keyword in ["rerank", "re-rank", "cross-encoder"]):
            return ["Re-ranking"]
        if any(keyword in combined_text for keyword in ["embedding", "vector representation", "sentence embedding"]):
            return ["Modelos de Embedding"]
        if any(
            keyword in combined_text
            for keyword in ["vector database", "faiss", "chromadb", "pinecone", "nearest neighbor"]
        ):
            return ["Vector Databases"]
        if any(keyword in combined_text for keyword in ["chunk", "segment", "split"]):
            return ["Segmentacion/Chunking"]
        if any(keyword in combined_text for keyword in ["cloud", "aws", "azure", "gcp"]):
            return ["Documentacion Cloud"]
        if any(keyword in combined_text for keyword in ["normalization", "preprocessing", "terminology"]):
            return ["Normalizacion/Preprocesamiento"]
        if any(keyword in combined_text for keyword in ["evaluation", "benchmark", "metric"]):
            return ["Evaluacion de sistemas RAG"]
        if any(keyword in combined_text for keyword in ["retrieval", "search", "information retrieval"]):
            return ["Estrategias de Recuperacion"]
        return ["Sistemas RAG Hibridos"]

    return categories


def _contains_any_phrase(text: str, phrases: list[str]) -> bool:
    normalized_text = _normalize_text(text)
    return any(_normalize_text(phrase) in normalized_text for phrase in phrases if phrase)


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    tokens = _text_tokens(text)
    normalized_terms = [_normalize_text(term) for term in terms if term]
    for term in normalized_terms:
        if not term:
            continue
        term_tokens = term.split()
        if not all(token in tokens for token in term_tokens):
            return False
    return True


def _matches_gap_rule(paper: Paper, gap_name: str) -> bool:
    rule = FOUNDATIONAL_GAP_RULES.get(gap_name)
    if not rule:
        return False

    title_text = paper.title or ""
    authors_text = " ".join(paper.authors)
    combined_text = f"{paper.title or ''} {paper.abstract or ''} {authors_text}"

    if rule.get("year_min") and (paper.year or 0) < rule["year_min"]:
        return False
    if rule.get("year_max") and paper.year and paper.year > rule["year_max"]:
        return False

    title_any = rule.get("title_any", [])
    if title_any and not _contains_any_phrase(title_text, title_any):
        return False

    text_any = rule.get("text_any", [])
    if text_any and not _contains_any_phrase(combined_text, text_any):
        return False

    text_all_groups = rule.get("text_all", [])
    if text_all_groups and not any(_contains_all_terms(combined_text, group) for group in text_all_groups):
        return False

    author_any = rule.get("author_any", [])
    if author_any and not _contains_any_phrase(authors_text, author_any):
        return False

    return True


def check_gap_coverage(paper: Paper, gaps_path: str = "config/trusted_sources.yaml") -> str | None:
    """Check if a paper covers any of the pending gaps."""
    gaps = _load_pending_gaps(gaps_path)
    combined = f"{paper.title or ''} {paper.abstract or ''} {' '.join(paper.authors)}".lower()

    for gap in gaps:
        gap_name = gap["name"]
        if _matches_gap_rule(paper, gap_name):
            result: str = gap_name
            return result

        if gap_name in FOUNDATIONAL_GAP_RULES:
            continue

        for hint in gap.get("search_hints", []):
            words = hint.lower().split()
            if len(words) <= 1:
                continue
            if all(word in combined for word in words):
                hint_result: str = gap_name
                return hint_result
        gap_words = gap_name.lower().split()
        if len(gap_words) > 1 and all(word in combined for word in gap_words):
            words_result: str = gap_name
            return words_result

    return None
