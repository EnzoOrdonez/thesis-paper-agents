"""OpenAlex API client for paper search and Scopus verification."""

from __future__ import annotations

import os
import time
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
    ):
        self.base_url = base_url.rstrip("/")
        self.email = email or os.getenv("OPENALEX_EMAIL", "")
        self.rate_limit_per_second = rate_limit_per_second
        self._last_request_time: float = 0
        self.session = requests.Session()
        if self.email:
            self.session.headers["User-Agent"] = f"ThesisPaperAgents/1.0 (mailto:{self.email})"

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = (1.0 / self.rate_limit_per_second) - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Make a GET request with backoff."""
        url = f"{self.base_url}{endpoint}"
        if self.email:
            params["mailto"] = self.email

        max_retries = 3
        delay = 2.0
        for attempt in range(max_retries + 1):
            self._wait_rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    logger.warning(f"OpenAlex rate limited, waiting {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                logger.error(f"OpenAlex error {resp.status_code}: {resp.text[:200]}")
                return None
            except requests.RequestException as e:
                logger.error(f"OpenAlex request error: {e}")
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                else:
                    return None
        return None

    def search(
        self,
        query: str,
        limit: int = 20,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[Paper]:
        """Search for papers matching a query.

        Args:
            query: Text search query.
            limit: Maximum number of results.
            from_date: Filter papers from this date (YYYY-MM-DD).
            to_date: Filter papers up to this date (YYYY-MM-DD).

        Returns:
            List of Paper objects.
        """
        params: dict[str, Any] = {
            "search": query,
            "per_page": min(limit, 50),
            "sort": "publication_date:desc",
        }

        # Build filter
        filters: list[str] = []
        if from_date:
            filters.append(f"from_publication_date:{from_date}")
        if to_date:
            filters.append(f"to_publication_date:{to_date}")
        if filters:
            params["filter"] = ",".join(filters)

        logger.debug(f"Searching OpenAlex: query='{query}'")
        data = self._get("/works", params)
        if not data:
            return []

        papers: list[Paper] = []
        for item in data.get("results", []):
            try:
                paper = self._parse_work(item)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse OpenAlex work: {e}")

        logger.info(f"OpenAlex: found {len(papers)} papers for '{query}'")
        return papers

    def check_scopus_indexed(self, doi: str) -> bool:
        """Check if a paper with given DOI is indexed in Scopus via OpenAlex.

        OpenAlex ingests Scopus data, so if a work is found and has a primary_location
        with a source that has type 'journal', it's very likely Scopus-indexed.
        """
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

        # Check if it has an ISSN (good indicator of Scopus indexing)
        issn = source.get("issn_l") or source.get("issn")
        if issn:
            return True

        # Check biblio for journal metadata
        return source.get("type") == "journal"

    def get_paper_by_doi(self, doi: str) -> Paper | None:
        """Fetch a single paper by DOI."""
        clean_doi = doi.strip()
        if clean_doi.startswith("https://doi.org/"):
            clean_doi = clean_doi[len("https://doi.org/"):]

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

        # Authors
        authors: list[str] = []
        for authorship in data.get("authorships", []):
            author_data = authorship.get("author", {})
            name = author_data.get("display_name")
            if name:
                authors.append(name)

        # DOI
        doi = data.get("doi")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]

        # Venue info
        primary = data.get("primary_location") or {}
        source = primary.get("source") or {}
        venue = source.get("display_name")
        publisher = data.get("host_venue", {}).get("publisher") if "host_venue" in data else None

        # Publication date
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
            url=data.get("id"),  # OpenAlex URL
            abstract=self._reconstruct_abstract(data),
            citation_count=data.get("cited_by_count", 0) or 0,
            source_api="openalex",
        )

    def _reconstruct_abstract(self, data: dict[str, Any]) -> str | None:
        """Reconstruct abstract from OpenAlex inverted index."""
        inverted = data.get("abstract_inverted_index")
        if not inverted:
            return None

        # Rebuild from inverted index
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            for pos in positions:
                word_positions.append((pos, word))

        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)
