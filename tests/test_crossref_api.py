"""Tests for the CrossRef API client."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.apis.crossref_api import CrossRefAPI
from src.models.paper import Paper


def _mock_response(status_code=200, json_data=None, text="", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.headers = headers or {}
    return resp


CROSSREF_SEARCH_RESPONSE = {
    "status": "ok",
    "message": {
        "items": [
            {
                "DOI": "10.1234/test",
                "title": ["Test Paper on RAG"],
                "author": [{"given": "John", "family": "Doe"}],
                "published-print": {"date-parts": [[2024, 6, 15]]},
                "container-title": ["Test Journal"],
                "abstract": "<p>Test abstract about retrieval.</p>",
                "publisher": "Test Publisher",
                "URL": "https://doi.org/10.1234/test",
                "is-referenced-by-count": 42,
            }
        ]
    },
}

CROSSREF_DOI_RESPONSE = {
    "status": "ok",
    "message": {
        "DOI": "10.1234/test",
        "title": ["Test Paper on RAG"],
        "author": [{"given": "John", "family": "Doe"}],
        "published-print": {"date-parts": [[2024, 6, 15]]},
        "container-title": ["Test Journal"],
        "abstract": "Test abstract about retrieval.",
        "publisher": "Test Publisher",
        "URL": "https://doi.org/10.1234/test",
        "is-referenced-by-count": 42,
    },
}


class TestCrossRefSearch:
    @patch("time.sleep", return_value=None)
    def test_successful_search(self, _mock_sleep):
        client = CrossRefAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=200, json_data=CROSSREF_SEARCH_RESPONSE)

        papers = client.search("retrieval augmented generation")

        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, Paper)
        assert paper.title == "Test Paper on RAG"
        assert paper.authors == ["John Doe"]
        assert paper.year == 2024
        assert paper.publication_date == "2024-06-15"
        assert paper.doi == "10.1234/test"
        assert paper.source_api == "crossref"
        assert paper.citation_count == 42
        # abstract should have HTML tags stripped
        assert "<p>" not in (paper.abstract or "")

    @patch("time.sleep", return_value=None)
    def test_disabled_returns_empty(self, _mock_sleep):
        client = CrossRefAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client._disabled_until = time.time() + 9999

        papers = client.search("any query")

        assert papers == []
        client.session.get.assert_not_called()


class TestCrossRefVerifyDoi:
    @patch("time.sleep", return_value=None)
    def test_valid_doi(self, _mock_sleep):
        client = CrossRefAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=200, json_data=CROSSREF_DOI_RESPONSE)

        result = client.verify_doi("10.1234/test")

        assert result is True

    @patch("time.sleep", return_value=None)
    def test_invalid_doi(self, _mock_sleep):
        client = CrossRefAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=404)

        result = client.verify_doi("10.9999/nonexistent")

        assert result is False


class TestCrossRefGetByDoi:
    @patch("time.sleep", return_value=None)
    def test_found(self, _mock_sleep):
        client = CrossRefAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=200, json_data=CROSSREF_DOI_RESPONSE)

        paper = client.get_paper_by_doi("10.1234/test")

        assert paper is not None
        assert paper.title == "Test Paper on RAG"
        assert paper.doi == "10.1234/test"

    @patch("time.sleep", return_value=None)
    def test_not_found(self, _mock_sleep):
        client = CrossRefAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=404)

        paper = client.get_paper_by_doi("10.9999/nonexistent")

        assert paper is None


class TestCrossRefCircuitBreaker:
    @patch("time.sleep", return_value=None)
    def test_trips_after_failures(self, _mock_sleep):
        client = CrossRefAPI(
            rate_limit_per_second=9999,
            max_retries=0,
            shutdown_after_consecutive_failures=2,
        )
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=500, text="Internal Server Error")

        # First failure
        result1 = client._get("https://api.crossref.org/works", {})
        assert result1 is None
        assert client._consecutive_failures == 1
        assert not client.is_temporarily_disabled()

        # Second failure trips the breaker
        result2 = client._get("https://api.crossref.org/works", {})
        assert result2 is None
        assert client._consecutive_failures == 2
        assert client.is_temporarily_disabled()


class TestCrossRefRuntimeState:
    def test_export_restore(self):
        client1 = CrossRefAPI(rate_limit_per_second=9999)
        client1._consecutive_failures = 1
        client1._last_error = "HTTP 500"
        client1._disabled_until = time.time() + 600

        state = client1.export_runtime_state()

        assert state["consecutive_failures"] == 1
        assert state["last_error"] == "HTTP 500"
        assert state["disabled_until"] is not None

        client2 = CrossRefAPI(rate_limit_per_second=9999)
        assert client2._consecutive_failures == 0
        client2.restore_runtime_state(state)

        assert client2._consecutive_failures == 1
        assert client2._last_error == "HTTP 500"
        assert client2.is_temporarily_disabled()
