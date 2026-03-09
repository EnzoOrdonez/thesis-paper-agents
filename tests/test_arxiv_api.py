"""Tests for the arXiv API client."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.apis.arxiv_api import ArxivAPI
from src.models.paper import Paper


def _mock_response(status_code=200, json_data=None, text="", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.headers = headers or {}
    return resp


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>Retrieval Augmented Generation for Cloud Docs</title>
    <author><name>Alice Smith</name></author>
    <published>2024-06-15T00:00:00Z</published>
    <summary>A study on RAG systems for cloud documentation.</summary>
    <link href="http://arxiv.org/abs/2406.12345" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2406.12345" title="pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.IR"/>
  </entry>
</feed>"""

ARXIV_EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""


class TestArxivSearch:
    @patch("time.sleep", return_value=None)
    def test_successful_search(self, _mock_sleep):
        client = ArxivAPI(rate_limit_seconds=0)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=200, text=ARXIV_FEED)

        papers = client.search("retrieval augmented generation")

        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, Paper)
        assert paper.title == "Retrieval Augmented Generation for Cloud Docs"
        assert paper.authors == ["Alice Smith"]
        assert paper.year == 2024
        assert paper.publication_date == "2024-06-15"
        assert paper.source_api == "arxiv"
        assert paper.abstract == "A study on RAG systems for cloud documentation."
        assert "2406.12345" in paper.url
        client.session.get.assert_called_once()

    @patch("time.sleep", return_value=None)
    def test_empty_results(self, _mock_sleep):
        client = ArxivAPI(rate_limit_seconds=0)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=200, text=ARXIV_EMPTY_FEED)

        papers = client.search("nonexistent topic xyz")

        assert papers == []

    @patch("time.sleep", return_value=None)
    def test_disabled_returns_empty(self, _mock_sleep):
        client = ArxivAPI(rate_limit_seconds=0)
        client.session = MagicMock()
        # Set disabled_until far in the future
        client._disabled_until = time.time() + 9999

        papers = client.search("any query")

        assert papers == []
        # Should not make any HTTP request
        client.session.get.assert_not_called()


class TestArxivCircuitBreaker:
    @patch("time.sleep", return_value=None)
    def test_consecutive_failures_trip_breaker(self, _mock_sleep):
        client = ArxivAPI(
            rate_limit_seconds=0,
            max_retries=0,
            shutdown_after_consecutive_failures=2,
        )
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=500)

        # First failure
        result1 = client._request({"search_query": "test"})
        assert result1 is None
        assert client._consecutive_failures == 1
        assert not client.is_temporarily_disabled()

        # Second failure should trip the breaker
        result2 = client._request({"search_query": "test"})
        assert result2 is None
        assert client._consecutive_failures == 2
        assert client.is_temporarily_disabled()


class TestArxivRuntimeState:
    def test_export_and_restore(self):
        client1 = ArxivAPI(rate_limit_seconds=0)
        client1._consecutive_failures = 1
        client1._last_error = "HTTP 500"
        client1._disabled_until = time.time() + 600

        state = client1.export_runtime_state()

        assert state["consecutive_failures"] == 1
        assert state["last_error"] == "HTTP 500"
        assert state["disabled_until"] is not None

        # Restore into a fresh instance
        client2 = ArxivAPI(rate_limit_seconds=0)
        assert client2._consecutive_failures == 0
        client2.restore_runtime_state(state)

        assert client2._consecutive_failures == 1
        assert client2._last_error == "HTTP 500"
        assert client2.is_temporarily_disabled()
