"""CrossRef API client for DOI validation and metadata retrieval."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

from src.models.paper import Paper
from src.utils.logger import setup_logger

logger = setup_logger("crossref_api")

DEFAULT_BASE_URL = "https://api.crossref.org"


class CrossRefAPI:
    """Client for the CrossRef API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        email: str | None = None,
        rate_limit_per_second: float = 50.0,
        max_retries: int = 3,
        cooldown_seconds: int = 900,
        shutdown_after_consecutive_failures: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.email = email or os.getenv("CROSSREF_EMAIL", "")
        self.rate_limit_per_second = rate_limit_per_second
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.shutdown_after_consecutive_failures = shutdown_after_consecutive_failures
        self._last_request_time: float = 0
        self._consecutive_failures = 0
        self._disabled_until: float = 0
        self._last_error: str = ""
        self._last_outcome: str = "never"
        self.session = requests.Session()
        if self.email:
            self.session.headers["User-Agent"] = f"ThesisPaperAgents/1.0 (mailto:{self.email})"

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = (1.0 / self.rate_limit_per_second) - elapsed
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
        """Export cooldown metadata so compiler runs can persist provider state."""
        return {
            "disabled_until": self._disabled_until_iso(),
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error,
        }

    def last_request_failed(self) -> bool:
        """Return whether the last request failed at transport or rate-limit level."""
        return self._last_outcome in {"failed", "skipped"}

    def _trip_circuit_breaker(self) -> None:
        self._disabled_until = time.time() + self.cooldown_seconds
        logger.warning(
            "CrossRef disabled for %.0f minutes after repeated failures",
            self.cooldown_seconds / 60,
        )

    def _register_failure(self, error: str) -> None:
        self._consecutive_failures += 1
        self._last_error = error
        self._last_outcome = "failed"
        if self._consecutive_failures >= self.shutdown_after_consecutive_failures:
            self._trip_circuit_breaker()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Make a GET request with backoff."""
        if self._is_disabled():
            logger.warning("Skipping CrossRef request because the client is temporarily disabled")
            self._last_outcome = "skipped"
            return None

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._wait_rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    self._consecutive_failures = 0
                    self._last_error = ""
                    self._last_outcome = "success"
                    return resp.json()
                if resp.status_code == 404:
                    logger.debug("CrossRef: not found at %s", url)
                    self._last_error = ""
                    self._last_outcome = "not_found"
                    return None
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning("CrossRef rate limited, waiting %ss", int(delay))
                    if attempt >= self.max_retries:
                        self._register_failure("HTTP 429")
                        return None
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                logger.error("CrossRef error %s: %s", resp.status_code, resp.text[:200])
                self._register_failure(f"HTTP {resp.status_code}")
                return None
            except requests.RequestException as exc:
                logger.error("CrossRef request error: %s", exc)
                self._last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    self._register_failure(str(exc))
                    return None
        return None

    def verify_doi(self, doi: str) -> bool:
        """Verify that a DOI is valid and resolves in CrossRef."""
        clean_doi = doi.strip()
        if clean_doi.startswith("https://doi.org/"):
            clean_doi = clean_doi[len("https://doi.org/"):]
        elif clean_doi.startswith("http://doi.org/"):
            clean_doi = clean_doi[len("http://doi.org/"):]

        url = f"{self.base_url}/works/{clean_doi}"
        data = self._get(url)
        return data is not None

    def get_paper_by_doi(self, doi: str) -> Paper | None:
        """Fetch paper metadata from CrossRef by DOI."""
        clean_doi = doi.strip()
        if clean_doi.startswith("https://doi.org/"):
            clean_doi = clean_doi[len("https://doi.org/"):]
        elif clean_doi.startswith("http://doi.org/"):
            clean_doi = clean_doi[len("http://doi.org/"):]

        url = f"{self.base_url}/works/{clean_doi}"
        data = self._get(url)
        if not data:
            return None

        return self._parse_work(data.get("message", {}))

    def search(
        self,
        query: str,
        limit: int = 20,
        from_date: str | None = None,
    ) -> list[Paper]:
        """Search CrossRef for papers."""
        if self._is_disabled():
            logger.info("CrossRef skipped for '%s' because the client is in cooldown", query)
            self._last_outcome = "skipped"
            return []

        url = f"{self.base_url}/works"
        params: dict[str, Any] = {
            "query": query,
            "rows": min(limit, 50),
            "sort": "published",
            "order": "desc",
        }
        if from_date:
            params["filter"] = f"from-pub-date:{from_date}"

        logger.debug("Searching CrossRef: query='%s'", query)
        data = self._get(url, params)
        if not data:
            return []

        papers: list[Paper] = []
        for item in data.get("message", {}).get("items", []):
            try:
                paper = self._parse_work(item)
                if paper:
                    papers.append(paper)
            except Exception as exc:
                logger.warning("Failed to parse CrossRef work: %s", exc)

        logger.info("CrossRef: found %s papers for '%s'", len(papers), query)
        return papers

    def _parse_work(self, data: dict[str, Any]) -> Paper | None:
        """Parse a CrossRef work item into a Paper."""
        titles = data.get("title", [])
        title = titles[0] if titles else None
        if not title:
            return None

        authors: list[str] = []
        for author_data in data.get("author", []):
            given = author_data.get("given", "")
            family = author_data.get("family", "")
            if given and family:
                authors.append(f"{given} {family}")
            elif family:
                authors.append(family)

        doi = data.get("DOI")

        pub_date_parts = data.get("published-print", data.get("published-online", {}))
        date_parts = pub_date_parts.get("date-parts", [[]])[0] if pub_date_parts else []
        year = date_parts[0] if len(date_parts) >= 1 else None
        month = date_parts[1] if len(date_parts) >= 2 else 1
        day = date_parts[2] if len(date_parts) >= 3 else 1
        pub_date = None
        if year:
            pub_date = f"{year}-{month:02d}-{day:02d}"

        container = data.get("container-title", [])
        venue = container[0] if container else None
        publisher = data.get("publisher")

        url = data.get("URL") or (f"https://doi.org/{doi}" if doi else None)

        abstract = data.get("abstract")
        if abstract:
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        return Paper(
            title=title,
            authors=authors,
            year=year,
            publication_date=pub_date,
            venue=venue,
            publisher=publisher,
            doi=doi,
            url=url,
            abstract=abstract,
            citation_count=data.get("is-referenced-by-count", 0) or 0,
            source_api="crossref",
        )