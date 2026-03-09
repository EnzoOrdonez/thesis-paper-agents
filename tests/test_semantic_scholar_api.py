"""Tests for the Semantic Scholar API client."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.apis.semantic_scholar import SemanticScholarAPI
from src.models.paper import Paper


def _mock_response(status_code=200, json_data=None, text="", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.headers = headers or {}
    return resp


SEMANTIC_SCHOLAR_SEARCH_RESPONSE = {
    "total": 1,
    "data": [
        {
            "paperId": "abc123",
            "title": "Test Paper on Dense Retrieval",
            "abstract": "Test abstract about dense passage retrieval.",
            "year": 2024,
            "venue": "Test Conf",
            "citationCount": 5,
            "authors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
            "externalIds": {"DOI": "10.1234/test"},
            "publicationDate": "2024-06-15",
            "url": "https://semanticscholar.org/paper/abc123",
        }
    ],
}

SEMANTIC_SCHOLAR_PAPER_RESPONSE = {
    "paperId": "def456",
    "title": "Single Paper by DOI",
    "abstract": "Abstract of a paper fetched by DOI.",
    "year": 2023,
    "venue": "EMNLP",
    "citationCount": 100,
    "authors": [{"name": "Alice Smith"}],
    "externalIds": {"DOI": "10.5678/paper"},
    "publicationDate": "2023-12-01",
    "url": "https://semanticscholar.org/paper/def456",
}


class TestSemanticScholarSearch:
    @patch("time.sleep", return_value=None)
    def test_successful_search(self, _mock_sleep):
        client = SemanticScholarAPI(rate_limit=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=200, json_data=SEMANTIC_SCHOLAR_SEARCH_RESPONSE)

        papers = client.search("dense retrieval")

        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, Paper)
        assert paper.title == "Test Paper on Dense Retrieval"
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert paper.year == 2024
        assert paper.publication_date == "2024-06-15"
        assert paper.doi == "10.1234/test"
        assert paper.source_api == "semantic_scholar"
        assert paper.citation_count == 5
        assert paper.venue == "Test Conf"

    @patch("time.sleep", return_value=None)
    def test_disabled_returns_empty(self, _mock_sleep):
        client = SemanticScholarAPI(rate_limit=9999)
        client.session = MagicMock()
        client._disabled_until = time.time() + 9999

        papers = client.search("any query")

        assert papers == []
        client.session.get.assert_not_called()

    @patch("time.sleep", return_value=None)
    def test_search_with_year_range(self, _mock_sleep):
        client = SemanticScholarAPI(rate_limit=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=200, json_data=SEMANTIC_SCHOLAR_SEARCH_RESPONSE)

        papers = client.search("dense retrieval", year_range="2023-2024")

        assert len(papers) == 1
        # Verify year param was passed
        call_kwargs = client.session.get.call_args
        assert call_kwargs[1]["params"]["year"] == "2023-2024"


class TestSemanticScholarGetByDoi:
    @patch("time.sleep", return_value=None)
    def test_found(self, _mock_sleep):
        client = SemanticScholarAPI(rate_limit=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=200, json_data=SEMANTIC_SCHOLAR_PAPER_RESPONSE)

        paper = client.get_paper_by_doi("10.5678/paper")

        assert paper is not None
        assert paper.title == "Single Paper by DOI"
        assert paper.doi == "10.5678/paper"
        assert paper.year == 2023
        assert paper.citation_count == 100

    @patch("time.sleep", return_value=None)
    def test_not_found(self, _mock_sleep):
        client = SemanticScholarAPI(rate_limit=9999, max_retries_without_key=0)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=404, text="Not Found")

        paper = client.get_paper_by_doi("10.9999/nonexistent")

        assert paper is None


class TestSemanticScholarCircuitBreaker:
    @patch("random.uniform", return_value=5.0)
    @patch("time.sleep", return_value=None)
    def test_trips_with_jitter(self, _mock_sleep, mock_random):
        client = SemanticScholarAPI(
            rate_limit=9999,
            max_retries_without_key=0,
            cooldown_seconds=1800,
            shutdown_after_consecutive_failures=2,
        )
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=500, text="Internal Server Error")

        # First failure
        result1 = client._request_with_backoff("https://api.test/paper/search", {})
        assert result1 is None
        assert client._consecutive_rate_limit_failures == 1
        assert not client.is_temporarily_disabled()

        # Second failure trips the breaker
        result2 = client._request_with_backoff("https://api.test/paper/search", {})
        assert result2 is None
        assert client._consecutive_rate_limit_failures == 2
        assert client.is_temporarily_disabled()

        # Verify random.uniform was called for jitter
        mock_random.assert_called_with(0, 30)


class TestSemanticScholarRuntimeState:
    def test_export_restore(self):
        client1 = SemanticScholarAPI(rate_limit=9999)
        client1._consecutive_rate_limit_failures = 1
        client1._last_error = "HTTP 429"
        client1._disabled_until = time.time() + 600

        state = client1.export_runtime_state()

        assert state["consecutive_failures"] == 1
        assert state["last_error"] == "HTTP 429"
        assert state["disabled_until"] is not None

        client2 = SemanticScholarAPI(rate_limit=9999)
        assert client2._consecutive_rate_limit_failures == 0
        client2.restore_runtime_state(state)

        assert client2._consecutive_rate_limit_failures == 1
        assert client2._last_error == "HTTP 429"
        assert client2.is_temporarily_disabled()
