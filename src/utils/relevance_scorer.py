"""Relevance scoring for academic papers based on thesis criteria.

Enhanced with:
- TF-IDF-inspired keyword density scoring
- Age-relative citation scoring
- Genericity penalty for off-topic papers mentioning RAG tangentially
- Bonus for papers citing existing thesis references
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.models.paper import Paper, RelevanceLevel


def _load_trusted_sources(path: str = "config/trusted_sources.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_keywords(path: str = "config/keywords.yaml") -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("keyword_groups", {})


def _load_existing_papers(path: str = "data/existing_papers.json") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _text_contains_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return which keywords appear in the text (case-insensitive)."""
    text_lower = text.lower()
    matched: list[str] = []
    for kw in keywords:
        words = kw.lower().split()
        if all(w in text_lower for w in words):
            matched.append(kw)
    return matched


def _keyword_density_score(text: str, keywords: list[str]) -> float:
    """Calculate a TF-IDF-inspired density score.

    Instead of just checking presence, count how many times thesis-relevant
    keywords appear normalized by text length. Returns 0.0 to 1.0.
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    # Build keyword tokens from all keyword phrases
    kw_tokens: set[str] = set()
    for kw in keywords:
        for token in kw.lower().split():
            if len(token) > 3:  # skip short common words
                kw_tokens.add(token)

    hits = sum(1 for w in words if w in kw_tokens)
    density = hits / total_words
    # Normalize to 0-1 range (typical density is 0.01-0.15)
    return min(density / 0.10, 1.0)


def _venue_tier(venue: str | None, publisher: str | None, sources: dict) -> int:
    """Return points based on venue/publisher tier."""
    if not venue and not publisher:
        return 0
    combined = f"{venue or ''} {publisher or ''}".lower()
    ts = sources.get("trusted_sources", {})

    for conf in ts.get("conferences_tier1", []):
        if conf.lower() in combined:
            return 20
    for pub in ts.get("publishers", []):
        if pub.lower() in combined:
            return 20
    for journal in ts.get("journals", []):
        if journal.lower() in combined:
            return 20
    for conf in ts.get("conferences_tier2", []):
        if conf.lower() in combined:
            return 10
    for pre in ts.get("preprints", []):
        if pre.lower() in combined:
            return 5
    return 0


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
    """Score citations relative to paper age.

    A paper with 100 citations in 1 year is more impressive than 100 in 5 years.
    Papers less than 6 months old get a pass (too new to evaluate).
    """
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

    # Too new: less than 1 year - don't penalize for low citations
    if age == 0:
        if count >= 10:
            return 15  # Very impressive for brand new
        if count >= 5:
            return 10
        return 5  # Give benefit of the doubt

    # Calculate citations per year
    cites_per_year = count / max(age, 1)

    if cites_per_year >= 50:
        return 15  # Foundational / highly influential
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
    """Penalize papers that mention RAG/retrieval only tangentially.

    If a paper's primary topic is outside IR/NLP (e.g., medicine, law, finance)
    but mentions RAG in the abstract, apply a penalty.
    """
    combined = f"{paper.title or ''} {paper.abstract or ''}".lower()

    # Off-topic domains that sometimes mention RAG tangentially
    off_topic_indicators = [
        "medical", "clinical", "patient", "diagnosis", "healthcare",
        "legal", "court", "jurisdiction", "statute",
        "financial", "stock", "trading", "portfolio",
        "biology", "genomic", "protein", "molecular",
        "chemistry", "chemical", "reaction",
        "agriculture", "crop", "soil",
        "physics", "quantum", "particle",
    ]

    # Core IR/NLP terms that indicate the paper IS about our field
    on_topic_core = [
        "information retrieval", "natural language processing",
        "text retrieval", "document retrieval", "passage retrieval",
        "semantic search", "embedding", "vector search",
        "question answering", "knowledge base",
        "retrieval augmented", "rag system", "rag pipeline",
        "language model", "transformer", "bert",
        "chunking", "reranking", "re-ranking",
        "bm25", "dense retrieval", "hybrid search",
        "cloud documentation", "technical documentation",
    ]

    off_topic_count = sum(1 for kw in off_topic_indicators if kw in combined)
    on_topic_count = sum(1 for kw in on_topic_core if kw in combined)

    # If heavily off-topic and weakly on-topic, penalize
    if off_topic_count >= 3 and on_topic_count <= 1:
        return -20
    if off_topic_count >= 2 and on_topic_count == 0:
        return -15

    return 0


def _existing_paper_bonus(paper: Paper) -> int:
    """Give bonus if paper cites or references papers already in our thesis."""
    existing = _load_existing_papers()
    if not existing:
        return 0

    combined = f"{paper.title or ''} {paper.abstract or ''}".lower()
    bonus = 0

    # Check if our thesis papers' key author surnames appear
    key_authors = set()
    for ep in existing:
        for author in ep.get("authors", []):
            # Extract surname (last word, or before "et al.")
            cleaned = author.replace("et al.", "").strip()
            parts = cleaned.split()
            if parts:
                surname = parts[-1].lower()
                if len(surname) > 2:
                    key_authors.add(surname)

    matched_authors = sum(1 for a in key_authors if a in combined)
    if matched_authors >= 2:
        bonus = 15
    elif matched_authors >= 1:
        bonus = 8

    return bonus


def is_from_trusted_source(paper: Paper, sources: dict | None = None) -> bool:
    """Check if a paper comes from a trusted source."""
    if sources is None:
        sources = _load_trusted_sources()

    ts = sources.get("trusted_sources", {})
    combined = f"{paper.venue or ''} {paper.publisher or ''}".lower()

    all_trusted: list[str] = []
    for key in ("publishers", "conferences_tier1", "conferences_tier2", "journals", "preprints"):
        all_trusted.extend(ts.get(key, []))

    for name in all_trusted:
        if name.lower() in combined:
            return True

    if paper.source_api == "arxiv":
        return True

    return False


def score_paper(paper: Paper) -> Paper:
    """Calculate relevance score for a paper and update it in-place.

    Enhanced scoring breakdown (0-100):
      - Title keyword match: up to +25 (with density bonus)
      - Abstract keyword match: up to +20 (with density bonus)
      - Source tier: up to +20
      - Year: up to +15
      - Citations (age-relative): up to +15
      - Existing paper bonus: up to +15
      - Genericity penalty: up to -20
    """
    sources = _load_trusted_sources()
    keyword_groups = _load_keywords()

    total = 0
    all_matched: list[str] = []

    # Collect all keyword tokens for density calc
    all_kw_tokens: list[str] = []
    for kws in keyword_groups.values():
        all_kw_tokens.extend(kws)

    # Title keyword matching (+25 max, with density)
    title_match_found = False
    for group_name, keywords in keyword_groups.items():
        matched = _text_contains_keywords(paper.title or "", keywords)
        if matched:
            title_match_found = True
            all_matched.extend(matched)
    if title_match_found:
        density = _keyword_density_score(paper.title or "", all_kw_tokens)
        total += 20 + int(density * 5)  # 20-25 based on density

    # Abstract keyword matching (+20 max, with density bonus)
    abstract_groups_matched = 0
    for group_name, keywords in keyword_groups.items():
        matched = _text_contains_keywords(paper.abstract or "", keywords)
        if matched:
            abstract_groups_matched += 1
            all_matched.extend(matched)
    if abstract_groups_matched > 0:
        base_points = min(15, abstract_groups_matched * 4)
        density = _keyword_density_score(paper.abstract or "", all_kw_tokens)
        total += base_points + int(density * 5)  # Up to +20

    # Source tier (+20 max)
    total += _venue_tier(paper.venue, paper.publisher, sources)

    # Year (+15 max)
    total += _year_score(paper.year)

    # Citations — age-relative (+15 max)
    total += _citation_score_relative(paper.citation_count, paper.year)

    # Bonus: cites our existing papers (+15 max)
    total += _existing_paper_bonus(paper)

    # Penalty: off-topic papers (-20 max)
    total += _genericity_penalty(paper)

    # Clamp to 0-100
    total = max(0, min(100, total))

    # Deduplicate matched keywords
    unique_matched = list(dict.fromkeys(all_matched))

    paper.relevance_score = total
    paper.keywords_matched = unique_matched

    if total >= 70:
        paper.relevance_level = RelevanceLevel.HIGH
    elif total >= 40:
        paper.relevance_level = RelevanceLevel.MEDIUM
    else:
        paper.relevance_level = RelevanceLevel.LOW

    return paper


def suggest_categories(paper: Paper) -> list[str]:
    """Suggest thesis categories based on keyword matches and content.

    Each paper is assigned to at most 3 categories (the most relevant ones).
    'Sistemas RAG Hibridos' requires explicit hybrid indicators in title or abstract.
    """
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

    # Score each category by how strongly it matches (title matches weigh more)
    category_scores: dict[str, float] = {}

    for group_name, keywords in keyword_groups.items():
        if group_name not in group_to_category:
            continue
        cat = group_to_category[group_name]

        title_matched = _text_contains_keywords(title_text, keywords)
        abstract_matched = _text_contains_keywords(abstract_text, keywords)

        score = len(title_matched) * 3.0 + len(abstract_matched) * 1.0
        if score > 0:
            category_scores[cat] = score

    # ── Strict filter for "Sistemas RAG Hibridos" ────────────────────────────
    rag_indicators = ["rag", "retrieval augmented generation", "retrieval-augmented generation"]
    has_rag = any(ind in combined_text for ind in rag_indicators)
    has_rag_title = any(ind in title_text for ind in rag_indicators)

    hybrid_indicators = [
        "hybrid retrieval", "hybrid search", "hybrid rag",
        "bm25 and semantic", "bm25 and dense", "bm25 + dense",
        "bm25 + semantic", "lexical and semantic", "lexical and dense",
        "sparse and dense", "hybrid model", "hybrid approach",
    ]
    has_hybrid = any(ind in combined_text for ind in hybrid_indicators)

    if "Sistemas RAG Hibridos" in category_scores:
        if has_rag and (has_hybrid or has_rag_title):
            pass  # Keep it
        elif has_rag:
            category_scores["Sistemas RAG Hibridos"] *= 0.3
        else:
            del category_scores["Sistemas RAG Hibridos"]
            if any(kw in combined_text for kw in ["evaluation", "benchmark", "metric"]):
                category_scores.setdefault("Evaluacion de sistemas RAG", 1.0)
            elif any(kw in combined_text for kw in ["retrieval", "search", "ranking"]):
                category_scores.setdefault("Estrategias de Recuperacion", 1.0)

    # ── Additional specific category checks ──────────────────────────────────
    normalization_kws = ["terminology normalization", "term normalization",
                         "acronym normalization", "preprocessing pipeline",
                         "text preprocessing", "text normalization"]
    if any(kw in combined_text for kw in normalization_kws):
        category_scores.setdefault("Normalizacion/Preprocesamiento", 2.0)

    metric_kws = ["ndcg", "mrr", "recall@", "precision@", "f1 score",
                   "exact match", "mean reciprocal rank", "evaluation metric"]
    if any(kw in combined_text for kw in metric_kws):
        category_scores.setdefault("Metricas de evaluacion", 2.0)

    # Sort categories by score descending and take top 3
    sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    categories = [cat for cat, _ in sorted_cats[:3]]

    if not categories:
        # Fallback: try to infer from broad content with priority order
        if any(ind in combined_text for ind in rag_indicators):
            return ["Sistemas RAG Hibridos"]
        if any(kw in combined_text for kw in ["hallucination", "faithfulness", "groundedness"]):
            return ["Alucinaciones en LLMs"]
        if any(kw in combined_text for kw in ["rerank", "re-rank", "cross-encoder"]):
            return ["Re-ranking"]
        if any(kw in combined_text for kw in ["embedding", "vector representation", "sentence embedding"]):
            return ["Modelos de Embedding"]
        if any(kw in combined_text for kw in ["vector database", "faiss", "chromadb", "pinecone", "nearest neighbor"]):
            return ["Vector Databases"]
        if any(kw in combined_text for kw in ["chunk", "segment", "split"]):
            return ["Segmentacion/Chunking"]
        if any(kw in combined_text for kw in ["cloud", "aws", "azure", "gcp"]):
            return ["Documentacion Cloud"]
        if any(kw in combined_text for kw in ["normalization", "preprocessing", "terminology"]):
            return ["Normalizacion/Preprocesamiento"]
        if any(kw in combined_text for kw in ["evaluation", "benchmark", "metric"]):
            return ["Evaluacion de sistemas RAG"]
        if any(kw in combined_text for kw in ["retrieval", "search", "information retrieval"]):
            return ["Estrategias de Recuperacion"]
        return ["Sistemas RAG Hibridos"]

    return categories


def check_gap_coverage(paper: Paper, gaps_path: str = "config/trusted_sources.yaml") -> str | None:
    """Check if a paper covers any of the pending gaps."""
    with open(gaps_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    gaps = data.get("pending_gaps", [])
    combined = f"{paper.title or ''} {paper.abstract or ''} {' '.join(paper.authors)}".lower()

    for gap in gaps:
        for hint in gap.get("search_hints", []):
            words = hint.lower().split()
            if all(w in combined for w in words):
                return gap["name"]
        gap_words = gap["name"].lower().split()
        if all(w in combined for w in gap_words):
            return gap["name"]

    return None
