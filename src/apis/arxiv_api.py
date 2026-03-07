"""arXiv API client for paper search."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import feedparser
import requests

from src.models.paper import Paper
from src.utils.logger import setup_logger

logger = setup_logger("arxiv_api")

DEFAULT_BASE_URL = "http://export.arxiv.org/api/query"
VALID_CATEGORIES = {"cs.IR", "cs.CL", "cs.AI", "cs.LG"}


class ArxivAPI:
    """Client for the arXiv API (Atom feed)."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        rate_limit_seconds: float = 3.0,
        categories: list[str] | None = None,
        max_retries: int = 2,
        cooldown_seconds: int = 900,
        shutdown_after_consecutive_failures: int = 2,
    ):
        self.base_url = base_url
        self.rate_limit_seconds = rate_limit_seconds
        self.categories = categories or list(VALID_CATEGORIES)
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.shutdown_after_consecutive_failures = shutdown_after_consecutive_failures
        self._last_request_time: float = 0
        self._consecutive_failures = 0
        self._disabled_until: float = 0
        self._last_error: str = ""
        self.session = requests.Session()

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = self.rate_limit_seconds - elapsed
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
        self._consecutive_failures = int(state.get("consecutive_failures", 0) or 0)
        self._last_error = str(state.get("last_error", "") or "")

    def export_runtime_state(self) -> dict[str, Any]:
        """Export cooldown metadata so daily runs can persist provider state."""
        return {
            "disabled_until": self._disabled_until_iso(),
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error,
        }

    def _trip_circuit_breaker(self) -> None:
        self._disabled_until = time.time() + self.cooldown_seconds
        logger.warning(
            "arXiv disabled for %.0f minutes after repeated failures",
            self.cooldown_seconds / 60,
        )

    def _register_failure(self, error: str) -> None:
        self._consecutive_failures += 1
        self._last_error = error
        if self._consecutive_failures >= self.shutdown_after_consecutive_failures:
            self._trip_circuit_breaker()

    def _request(self, params: dict[str, Any]) -> str | None:
        """Fetch the raw Atom feed with retry and cooldown support."""
        if self._is_disabled():
            logger.warning("Skipping arXiv request because the client is temporarily disabled")
            return None

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._wait_rate_limit()
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                if response.status_code == 200:
                    self._consecutive_failures = 0
                    self._last_error = ""
                    return response.text
                if response.status_code in {429, 500, 502, 503, 504}:
                    logger.warning("arXiv retryable error %s, waiting %ss", response.status_code, int(delay))
                    if attempt >= self.max_retries:
                        self._register_failure(f"HTTP {response.status_code}")
                        return None
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.error("arXiv request error: %s", exc)
                self._last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    self._register_failure(str(exc))
                    return None
        return None

    def search(
        self,
        query: str,
        max_results: int = 20,
        date_from: str | None = None,
    ) -> list[Paper]:
        """Search arXiv for papers matching a query."""
        if self._is_disabled():
            logger.info("arXiv skipped for '%s' because the client is in cooldown", query)
            return []

        cat_filter = " OR ".join(f"cat:{category}" for category in self.categories)
        search_query = f"all:{query} AND ({cat_filter})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        logger.debug("Searching arXiv: query='%s'", query)
        response_text = self._request(params)
        if not response_text:
            return []

        feed = feedparser.parse(response_text)
        papers: list[Paper] = []

        for entry in feed.entries:
            try:
                paper = self._parse_entry(entry)
                if paper:
                    if date_from and paper.publication_date and paper.publication_date < date_from:
                        continue
                    papers.append(paper)
            except Exception as exc:
                logger.warning("Failed to parse arXiv entry: %s", exc)

        logger.info("arXiv: found %s papers for '%s'", len(papers), query)
        return papers

    def _parse_entry(self, entry: Any) -> Paper | None:
        """Parse a feedparser entry into a Paper model."""
        title = entry.get("title", "").replace("\n", " ").strip()
        if not title:
            return None

        authors = []
        for author in entry.get("authors", []):
            name = author.get("name", "").strip()
            if name:
                authors.append(name)

        published = entry.get("published", "")
        publication_date = None
        year = None
        if published:
            try:
                parsed_date = datetime.strptime(published[:10], "%Y-%m-%d")
                publication_date = parsed_date.strftime("%Y-%m-%d")
                year = parsed_date.year
            except ValueError:
                pass

        doi = None
        arxiv_id = None
        for link in entry.get("links", []):
            href = link.get("href", "")
            if "doi.org" in href:
                doi = href.replace("https://doi.org/", "").replace("http://doi.org/", "")
            if "arxiv.org/abs/" in href:
                arxiv_id = href.split("/abs/")[-1]

        abstract = entry.get("summary", "").replace("\n", " ").strip()

        primary_category = ""
        if hasattr(entry, "arxiv_primary_category"):
            primary_category = entry.arxiv_primary_category.get("term", "")

        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry.get("link", "")

        return Paper(
            title=title,
            authors=authors,
            year=year,
            publication_date=publication_date,
            venue=f"arXiv:{primary_category}" if primary_category else "arXiv",
            publisher="arXiv",
            doi=doi,
            url=url,
            abstract=abstract,
            source_api="arxiv",
        )
