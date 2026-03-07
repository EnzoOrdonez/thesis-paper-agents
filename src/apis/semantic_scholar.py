"""Semantic Scholar API client for paper search."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
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
        max_retries_with_key: int = 4,
        max_retries_without_key: int = 2,
        cooldown_seconds: int = 1800,
        shutdown_after_consecutive_failures: int = 2,
    ):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.fields = fields
        self.max_retries_with_key = max_retries_with_key
        self.max_retries_without_key = max_retries_without_key
        self.cooldown_seconds = cooldown_seconds
        self.shutdown_after_consecutive_failures = shutdown_after_consecutive_failures
        self._last_request_time: float = 0
        self._consecutive_rate_limit_failures = 0
        self._disabled_until: float = 0
        self._last_error: str = ""
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = (1.0 / self.rate_limit) - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def _is_disabled(self) -> bool:
        return time.time() < self._disabled_until

    def is_temporarily_disabled(self) -> bool:
        """Return whether the provider is currently in cooldown."""
        return self._is_disabled()

    def _disabled_until_iso(self) -> str | None:
        if not self._disabled_until or time.time() >= self._disabled_until:
            return None
        return datetime.fromtimestamp(self._disabled_until, tz=timezone.utc).isoformat()

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        """Restore persisted cooldown metadata from a previous run."""
        disabled_until = state.get("disabled_until")
        if disabled_until:
            try:
                self._disabled_until = datetime.fromisoformat(disabled_until).timestamp()
            except ValueError:
                self._disabled_until = 0
        self._consecutive_rate_limit_failures = int(state.get("consecutive_failures", 0) or 0)
        self._last_error = str(state.get("last_error", "") or "")

    def export_runtime_state(self) -> dict[str, Any]:
        """Export cooldown metadata so daily runs can persist provider state."""
        return {
            "disabled_until": self._disabled_until_iso(),
            "consecutive_failures": self._consecutive_rate_limit_failures,
            "last_error": self._last_error,
        }

    def _trip_circuit_breaker(self) -> None:
        self._disabled_until = time.time() + self.cooldown_seconds
        logger.warning(
            "Semantic Scholar disabled for %.0f minutes after repeated failures",
            self.cooldown_seconds / 60,
        )

    def _request_with_backoff(self, url: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Make a GET request with exponential backoff on retryable failures."""
        if self._is_disabled():
            logger.warning("Skipping Semantic Scholar request because the client is temporarily disabled")
            return None

        max_retries = self.max_retries_with_key if self.api_key else self.max_retries_without_key
        delay = 2.0

        for attempt in range(max_retries + 1):
            self._wait_rate_limit()
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    self._consecutive_rate_limit_failures = 0
                    self._last_error = ""
                    return response.json()
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning("Rate limited (429), waiting %ss (attempt %s)", int(delay), attempt + 1)
                    if attempt >= max_retries:
                        self._consecutive_rate_limit_failures += 1
                        self._last_error = "HTTP 429"
                        if self._consecutive_rate_limit_failures >= self.shutdown_after_consecutive_failures:
                            self._trip_circuit_breaker()
                        return None
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                logger.error("Semantic Scholar API error %s: %s", response.status_code, response.text[:200])
                self._consecutive_rate_limit_failures += 1
                self._last_error = f"HTTP {response.status_code}"
                if self._consecutive_rate_limit_failures >= self.shutdown_after_consecutive_failures:
                    self._trip_circuit_breaker()
                return None
            except requests.RequestException as exc:
                logger.error("Semantic Scholar request error: %s", exc)
                self._last_error = str(exc)
                if attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                else:
                    self._consecutive_rate_limit_failures += 1
                    if self._consecutive_rate_limit_failures >= self.shutdown_after_consecutive_failures:
                        self._trip_circuit_breaker()
                    return None
        return None

    def search(
        self,
        query: str,
        limit: int = 20,
        year_range: str | None = None,
    ) -> list[Paper]:
        """Search for papers matching a query."""
        if self._is_disabled():
            logger.info("Semantic Scholar skipped for '%s' because the client is in cooldown", query)
            return []

        url = f"{self.base_url}/paper/search"
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": self.fields,
        }
        if year_range:
            params["year"] = year_range

        logger.debug("Searching Semantic Scholar: query='%s', year=%s", query, year_range)
        data = self._request_with_backoff(url, params)
        if not data:
            return []

        papers: list[Paper] = []
        for item in data.get("data", []):
            try:
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)
            except Exception as exc:
                logger.warning("Failed to parse paper: %s", exc)

        logger.info("Semantic Scholar: found %s papers for '%s'", len(papers), query)
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
        for paper in papers:
            if paper.title.lower().strip() == title_lower:
                return paper
        return papers[0] if papers else None

    def _parse_paper(self, data: dict[str, Any]) -> Paper | None:
        """Parse a Semantic Scholar result into a Paper model."""
        title = data.get("title")
        if not title:
            return None

        authors = []
        for author in data.get("authors", []):
            name = author.get("name")
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
