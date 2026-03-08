"""OpenAlex API client for paper search and Scopus verification."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

import requests

from src.models.paper import Paper
from src.utils.logger import setup_logger

logger = setup_logger("openalex_api")

DEFAULT_BASE_URL = "https://api.openalex.org"


class OpenAlexAPI:
    """Client for the OpenAlex API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        email: str | None = None,
        rate_limit_per_second: float = 10.0,
        max_retries: int = 3,
        cooldown_seconds: int = 900,
        shutdown_after_consecutive_failures: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.email = email or os.getenv("OPENALEX_EMAIL", "")
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
        return datetime.fromtimestamp(self._disabled_until, tz=UTC).isoformat()

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

    def last_request_failed(self) -> bool:
        """Return whether the last request failed at transport or rate-limit level."""
        return self._last_outcome in {"failed", "skipped"}

    def _trip_circuit_breaker(self) -> None:
        self._disabled_until = time.time() + self.cooldown_seconds
        logger.warning(
            "OpenAlex disabled for %.0f minutes after repeated failures",
            self.cooldown_seconds / 60,
        )

    def _register_failure(self, error: str) -> None:
        self._consecutive_failures += 1
        self._last_error = error
        self._last_outcome = "failed"
        if self._consecutive_failures >= self.shutdown_after_consecutive_failures:
            self._trip_circuit_breaker()

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Make a GET request with backoff."""
        if self._is_disabled():
            logger.warning("Skipping OpenAlex request because the client is temporarily disabled")
            self._last_outcome = "skipped"
            return None

        url = f"{self.base_url}{endpoint}"
        if self.email:
            params["mailto"] = self.email

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
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning("OpenAlex rate limited, waiting %ss", int(delay))
                    if attempt >= self.max_retries:
                        self._register_failure("HTTP 429")
                        return None
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                logger.error("OpenAlex error %s: %s", resp.status_code, resp.text[:200])
                self._register_failure(f"HTTP {resp.status_code}")
                return None
            except requests.RequestException as exc:
                logger.error("OpenAlex request error: %s", exc)
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
        limit: int = 20,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[Paper]:
        """Search for papers matching a query."""
        if self._is_disabled():
            logger.info("OpenAlex skipped for '%s' because the client is in cooldown", query)
            self._last_outcome = "skipped"
            return []

        params: dict[str, Any] = {
            "search": query,
            "per_page": min(limit, 50),
            "sort": "publication_date:desc",
        }

        filters: list[str] = []
        if from_date:
            filters.append(f"from_publication_date:{from_date}")
        if to_date:
            filters.append(f"to_publication_date:{to_date}")
        if filters:
            params["filter"] = ",".join(filters)

        logger.debug("Searching OpenAlex: query='%s'", query)
        data = self._get("/works", params)
        if not data:
            return []

        papers: list[Paper] = []
        for item in data.get("results", []):
            try:
                paper = self._parse_work(item)
                if paper:
                    papers.append(paper)
            except Exception as exc:
                logger.warning("Failed to parse OpenAlex work: %s", exc)

        logger.info("OpenAlex: found %s papers for '%s'", len(papers), query)
        return papers

    def check_scopus_indexed(self, doi: str) -> bool:
        """Check if a paper with given DOI is indexed in Scopus via OpenAlex."""
        if not doi:
            return False

        params: dict[str, Any] = {"filter": f"doi:{doi}"}
        data = self._get("/works", params)
        if not data:
            return False

        results = data.get("results", [])
        if not results:
            return False

        work = results[0]
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}

        issn = source.get("issn_l") or source.get("issn")
        if issn:
            return True

        return source.get("type") == "journal"

    def get_paper_by_doi(self, doi: str) -> Paper | None:
        """Fetch a single paper by DOI."""
        clean_doi = doi.strip()
        if clean_doi.startswith("https://doi.org/"):
            clean_doi = clean_doi[len("https://doi.org/") :]

        params: dict[str, Any] = {"filter": f"doi:{clean_doi}"}
        data = self._get("/works", params)
        if not data or not data.get("results"):
            return None
        return self._parse_work(data["results"][0])

    def _parse_work(self, data: dict[str, Any]) -> Paper | None:
        """Parse an OpenAlex work into a Paper model."""
        title = data.get("title")
        if not title:
            return None

        authors: list[str] = []
        for authorship in data.get("authorships", []):
            author_data = authorship.get("author", {})
            name = author_data.get("display_name")
            if name:
                authors.append(name)

        doi = data.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/") :]

        primary = data.get("primary_location") or {}
        source = primary.get("source") or {}
        venue = source.get("display_name")
        publisher = data.get("host_venue", {}).get("publisher") if "host_venue" in data else None

        pub_date = data.get("publication_date")
        year = data.get("publication_year")

        return Paper(
            title=title,
            authors=authors,
            year=year,
            publication_date=pub_date,
            venue=venue,
            publisher=publisher,
            doi=doi,
            url=data.get("id"),
            abstract=self._reconstruct_abstract(data),
            citation_count=data.get("cited_by_count", 0) or 0,
            source_api="openalex",
        )

    def _reconstruct_abstract(self, data: dict[str, Any]) -> str | None:
        """Reconstruct abstract from OpenAlex inverted index."""
        inverted = data.get("abstract_inverted_index")
        if not inverted:
            return None

        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            for pos in positions:
                word_positions.append((pos, word))

        word_positions.sort(key=lambda item: item[0])
        return " ".join(word for _, word in word_positions)
