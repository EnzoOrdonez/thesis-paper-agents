"""arXiv API client for paper search."""

from __future__ import annotations

import time
from datetime import datetime
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
        self.session = requests.Session()

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
        """Search arXiv for papers matching a query."""
        cat_filter = " OR ".join(f"cat:{category}" for category in self.categories)
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
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"arXiv request error: {exc}")
            return []

        feed = feedparser.parse(response.text)
        papers: list[Paper] = []

        for entry in feed.entries:
            try:
                paper = self._parse_entry(entry)
                if paper:
                    if date_from and paper.publication_date and paper.publication_date < date_from:
                        continue
                    papers.append(paper)
            except Exception as exc:
                logger.warning(f"Failed to parse arXiv entry: {exc}")

        logger.info(f"arXiv: found {len(papers)} papers for '{query}'")
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
