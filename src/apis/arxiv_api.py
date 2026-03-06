"""arXiv API client for paper search."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
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
    ):
        self.base_url = base_url
        self.rate_limit_seconds = rate_limit_seconds
        self.categories = categories or list(VALID_CATEGORIES)
        self._last_request_time: float = 0

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        wait = self.rate_limit_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 20,
        date_from: str | None = None,
    ) -> list[Paper]:
        """Search arXiv for papers matching a query.

        Args:
            query: Search query (will be searched in title+abstract).
            max_results: Maximum number of results.
            date_from: Optional start date for filtering (YYYY-MM-DD).

        Returns:
            List of Paper objects.
        """
        # Build arXiv query: search in title and abstract, restrict to categories
        cat_filter = " OR ".join(f"cat:{c}" for c in self.categories)
        search_query = f"all:{query} AND ({cat_filter})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        logger.debug(f"Searching arXiv: query='{query}'")
        self._wait_rate_limit()

        try:
            resp = requests.get(self.base_url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"arXiv request error: {e}")
            return []

        feed = feedparser.parse(resp.text)
        papers: list[Paper] = []

        for entry in feed.entries:
            try:
                paper = self._parse_entry(entry)
                if paper:
                    # Filter by date if specified
                    if date_from and paper.publication_date:
                        if paper.publication_date < date_from:
                            continue
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse arXiv entry: {e}")

        logger.info(f"arXiv: found {len(papers)} papers for '{query}'")
        return papers

    def _parse_entry(self, entry: Any) -> Paper | None:
        """Parse a feedparser entry into a Paper model."""
        title = entry.get("title", "").replace("\n", " ").strip()
        if not title:
            return None

        authors = []
        for a in entry.get("authors", []):
            name = a.get("name", "").strip()
            if name:
                authors.append(name)

        # Extract publication date
        published = entry.get("published", "")
        pub_date = None
        year = None
        if published:
            try:
                dt = datetime.strptime(published[:10], "%Y-%m-%d")
                pub_date = dt.strftime("%Y-%m-%d")
                year = dt.year
            except ValueError:
                pass

        # Extract DOI if available
        doi = None
        arxiv_id = None
        for link in entry.get("links", []):
            href = link.get("href", "")
            if "doi.org" in href:
                doi = href.replace("https://doi.org/", "").replace("http://doi.org/", "")
            if "arxiv.org/abs/" in href:
                arxiv_id = href.split("/abs/")[-1]

        # Get abstract
        abstract = entry.get("summary", "").replace("\n", " ").strip()

        # Primary category
        primary_cat = ""
        if hasattr(entry, "arxiv_primary_category"):
            primary_cat = entry.arxiv_primary_category.get("term", "")

        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry.get("link", "")

        return Paper(
            title=title,
            authors=authors,
            year=year,
            publication_date=pub_date,
            venue=f"arXiv:{primary_cat}" if primary_cat else "arXiv",
            publisher="arXiv",
            doi=doi,
            url=url,
            abstract=abstract,
            source_api="arxiv",
        )
