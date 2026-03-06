"""Semantic Scholar API client for paper search."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from src.models.paper import Paper
from src.utils.logger import setup_logger

logger = setup_logger("semantic_scholar")

DEFAULT_BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = "title,abstract,year,venue,externalIds,authors,citationCount,publicationDate,url"


class SemanticScholarAPI:
    """Client for the Semantic Scholar Academic Graph API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        rate_limit: float = 1.0,
        fields: str = DEFAULT_FIELDS,
    ):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.fields = fields
        self._last_request_time: float = 0
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = (1.0 / self.rate_limit) - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def _request_with_backoff(self, url: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Make a GET request with exponential backoff on 429."""
        max_retries = 4
        delay = 2.0
        for attempt in range(max_retries + 1):
            self._wait_rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    logger.warning(f"Rate limited (429), waiting {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                logger.error(f"Semantic Scholar API error {resp.status_code}: {resp.text[:200]}")
                return None
            except requests.RequestException as e:
                logger.error(f"Semantic Scholar request error: {e}")
                if attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    return None
        return None

    def search(
        self,
        query: str,
        limit: int = 20,
        year_range: str | None = None,
    ) -> list[Paper]:
        """Search for papers matching a query.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            year_range: Optional year filter, e.g. "2023-2025".

        Returns:
            List of Paper objects.
        """
        url = f"{self.base_url}/paper/search"
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": self.fields,
        }
        if year_range:
            params["year"] = year_range

        logger.debug(f"Searching Semantic Scholar: query='{query}', year={year_range}")
        data = self._request_with_backoff(url, params)
        if not data:
            return []

        papers: list[Paper] = []
        for item in data.get("data", []):
            try:
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse paper: {e}")

        logger.info(f"Semantic Scholar: found {len(papers)} papers for '{query}'")
        return papers

    def get_paper_by_doi(self, doi: str) -> Paper | None:
        """Fetch a single paper by DOI."""
        url = f"{self.base_url}/paper/DOI:{doi}"
        params = {"fields": self.fields}
        data = self._request_with_backoff(url, params)
        if not data:
            return None
        return self._parse_paper(data)

    def get_paper_by_title(self, title: str) -> Paper | None:
        """Search for a paper by exact title match."""
        papers = self.search(title, limit=5)
        title_lower = title.lower().strip()
        for p in papers:
            if p.title.lower().strip() == title_lower:
                return p
        return papers[0] if papers else None

    def _parse_paper(self, data: dict[str, Any]) -> Paper | None:
        """Parse a Semantic Scholar result into a Paper model."""
        title = data.get("title")
        if not title:
            return None

        authors = []
        for a in data.get("authors", []):
            name = a.get("name")
            if name:
                authors.append(name)

        external_ids = data.get("externalIds") or {}
        doi = external_ids.get("DOI")

        return Paper(
            title=title,
            authors=authors,
            year=data.get("year"),
            publication_date=data.get("publicationDate"),
            venue=data.get("venue") or None,
            doi=doi,
            url=data.get("url"),
            abstract=data.get("abstract"),
            citation_count=data.get("citationCount", 0) or 0,
            source_api="semantic_scholar",
        )
