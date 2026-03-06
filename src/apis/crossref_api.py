"""CrossRef API client for DOI validation and metadata retrieval."""

from __future__ import annotations

import os
import time
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
    ):
        self.base_url = base_url.rstrip("/")
        self.email = email or os.getenv("CROSSREF_EMAIL", "")
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

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Make a GET request with backoff."""
        max_retries = 3
        delay = 2.0
        for attempt in range(max_retries + 1):
            self._wait_rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 404:
                    logger.debug(f"CrossRef: not found at {url}")
                    return None
                if resp.status_code == 429:
                    logger.warning(f"CrossRef rate limited, waiting {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                logger.error(f"CrossRef error {resp.status_code}: {resp.text[:200]}")
                return None
            except requests.RequestException as e:
                logger.error(f"CrossRef request error: {e}")
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
                else:
                    return None
        return None

    def verify_doi(self, doi: str) -> bool:
        """Verify that a DOI is valid and resolves in CrossRef.

        Args:
            doi: The DOI to verify (with or without URL prefix).

        Returns:
            True if the DOI exists in CrossRef.
        """
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
        """Search CrossRef for papers.

        Args:
            query: Search query.
            limit: Maximum results.
            from_date: Filter from this date (YYYY-MM-DD).

        Returns:
            List of Paper objects.
        """
        url = f"{self.base_url}/works"
        params: dict[str, Any] = {
            "query": query,
            "rows": min(limit, 50),
            "sort": "published",
            "order": "desc",
        }
        if from_date:
            params["filter"] = f"from-pub-date:{from_date}"

        logger.debug(f"Searching CrossRef: query='{query}'")
        data = self._get(url, params)
        if not data:
            return []

        papers: list[Paper] = []
        for item in data.get("message", {}).get("items", []):
            try:
                paper = self._parse_work(item)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse CrossRef work: {e}")

        logger.info(f"CrossRef: found {len(papers)} papers for '{query}'")
        return papers

    def _parse_work(self, data: dict[str, Any]) -> Paper | None:
        """Parse a CrossRef work item into a Paper."""
        # Title
        titles = data.get("title", [])
        title = titles[0] if titles else None
        if not title:
            return None

        # Authors
        authors: list[str] = []
        for a in data.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            if given and family:
                authors.append(f"{given} {family}")
            elif family:
                authors.append(family)

        # DOI
        doi = data.get("DOI")

        # Publication date
        pub_date_parts = data.get("published-print", data.get("published-online", {}))
        date_parts = pub_date_parts.get("date-parts", [[]])[0] if pub_date_parts else []
        year = date_parts[0] if len(date_parts) >= 1 else None
        month = date_parts[1] if len(date_parts) >= 2 else 1
        day = date_parts[2] if len(date_parts) >= 3 else 1
        pub_date = None
        if year:
            pub_date = f"{year}-{month:02d}-{day:02d}"

        # Venue
        container = data.get("container-title", [])
        venue = container[0] if container else None
        publisher = data.get("publisher")

        # URL
        url = data.get("URL") or (f"https://doi.org/{doi}" if doi else None)

        # Abstract (CrossRef sometimes includes it)
        abstract = data.get("abstract")
        if abstract:
            # Strip HTML tags that CrossRef sometimes includes
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
