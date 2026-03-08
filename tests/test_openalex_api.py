"""Tests for the OpenAlex API client."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.apis.openalex_api import OpenAlexAPI
from src.models.paper import Paper


def _mock_response(status_code=200, json_data=None, text="", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.headers = headers or {}
    return resp


OPENALEX_SEARCH_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1234/test",
            "title": "Test Paper on Information Retrieval",
            "authorships": [
                {"author": {"display_name": "John Doe"}},
                {"author": {"display_name": "Jane Smith"}},
            ],
            "publication_year": 2024,
            "publication_date": "2024-06-15",
            "primary_location": {
                "source": {"display_name": "Test Journal"}
            },
            "cited_by_count": 10,
            "abstract_inverted_index": {
                "This": [0],
                "is": [1],
                "a": [2],
                "test": [3],
                "abstract": [4],
            },
        }
    ]
}

OPENALEX_SCOPUS_INDEXED_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W456",
            "doi": "https://doi.org/10.1234/indexed",
            "title": "Indexed Paper",
            "primary_location": {
                "source": {
                    "display_name": "Scopus Journal",
                    "issn_l": "1234-5678",
                    "type": "journal",
                }
            },
        }
    ]
}

OPENALEX_NOT_INDEXED_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W789",
            "doi": "https://doi.org/10.1234/notindexed",
            "title": "Not Indexed Paper",
            "primary_location": {
                "source": {
                    "display_name": "Some Repository",
                    "type": "repository",
                }
            },
        }
    ]
}


class TestOpenAlexSearch:
    @patch("time.sleep", return_value=None)
    def test_successful_search(self, _mock_sleep):
        client = OpenAlexAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(
            status_code=200, json_data=OPENALEX_SEARCH_RESPONSE
        )

        papers = client.search("information retrieval")

        assert len(papers) == 1
        paper = papers[0]
        assert isinstance(paper, Paper)
        assert paper.title == "Test Paper on Information Retrieval"
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert paper.year == 2024
        assert paper.publication_date == "2024-06-15"
        assert paper.doi == "10.1234/test"
        assert paper.source_api == "openalex"
        assert paper.citation_count == 10
        assert paper.venue == "Test Journal"
        # Abstract should be reconstructed from inverted index
        assert paper.abstract == "This is a test abstract"

    @patch("time.sleep", return_value=None)
    def test_disabled_returns_empty(self, _mock_sleep):
        client = OpenAlexAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client._disabled_until = time.time() + 9999

        papers = client.search("any query")

        assert papers == []
        client.session.get.assert_not_called()


class TestOpenAlexScopus:
    @patch("time.sleep", return_value=None)
    def test_indexed(self, _mock_sleep):
        client = OpenAlexAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(
            status_code=200, json_data=OPENALEX_SCOPUS_INDEXED_RESPONSE
        )

        result = client.check_scopus_indexed("10.1234/indexed")

        assert result is True

    @patch("time.sleep", return_value=None)
    def test_not_indexed(self, _mock_sleep):
        client = OpenAlexAPI(rate_limit_per_second=9999)
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(
            status_code=200, json_data=OPENALEX_NOT_INDEXED_RESPONSE
        )

        result = client.check_scopus_indexed("10.1234/notindexed")

        assert result is False

    def test_empty_doi_returns_false(self):
        client = OpenAlexAPI(rate_limit_per_second=9999)
        result = client.check_scopus_indexed("")
        assert result is False


class TestOpenAlexAbstract:
    def test_reconstruct_from_inverted_index(self):
        client = OpenAlexAPI(rate_limit_per_second=9999)
        data = {
            "abstract_inverted_index": {
                "Retrieval": [0],
                "augmented": [1],
                "generation": [2],
                "improves": [3],
                "accuracy.": [4],
            }
        }
        abstract = client._reconstruct_abstract(data)
        assert abstract == "Retrieval augmented generation improves accuracy."

    def test_no_inverted_index_returns_none(self):
        client = OpenAlexAPI(rate_limit_per_second=9999)
        data = {}
        abstract = client._reconstruct_abstract(data)
        assert abstract is None

    def test_inverted_index_with_repeated_words(self):
        client = OpenAlexAPI(rate_limit_per_second=9999)
        data = {
            "abstract_inverted_index": {
                "the": [0, 3],
                "model": [1, 4],
                "uses": [2],
                "well": [5],
            }
        }
        abstract = client._reconstruct_abstract(data)
        assert abstract == "the model uses the model well"


class TestOpenAlexCircuitBreaker:
    @patch("time.sleep", return_value=None)
    def test_trips_after_failures(self, _mock_sleep):
        client = OpenAlexAPI(
            rate_limit_per_second=9999,
            max_retries=0,
            shutdown_after_consecutive_failures=2,
        )
        client.session = MagicMock()
        client.session.get.return_value = _mock_response(status_code=500, text="Internal Server Error")

        # First failure
        result1 = client._get("/works", {})
        assert result1 is None
        assert client._consecutive_failures == 1
        assert not client.is_temporarily_disabled()

        # Second failure trips the breaker
        result2 = client._get("/works", {})
        assert result2 is None
        assert client._consecutive_failures == 2
        assert client.is_temporarily_disabled()


class TestOpenAlexRuntimeState:
    def test_export_restore(self):
        client1 = OpenAlexAPI(rate_limit_per_second=9999)
        client1._consecutive_failures = 1
        client1._last_error = "HTTP 429"
        client1._disabled_until = time.time() + 600

        state = client1.export_runtime_state()

        assert state["consecutive_failures"] == 1
        assert state["last_error"] == "HTTP 429"
        assert state["disabled_until"] is not None

        client2 = OpenAlexAPI(rate_limit_per_second=9999)
        assert client2._consecutive_failures == 0
        client2.restore_runtime_state(state)

        assert client2._consecutive_failures == 1
        assert client2._last_error == "HTTP 429"
        assert client2.is_temporarily_disabled()
