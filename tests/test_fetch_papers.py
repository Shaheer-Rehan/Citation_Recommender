"""
test_fetch_papers.py
--------------------
Unit tests for data/fetch_papers.py.

Pure functions (deduplicate_papers, filter_papers, normalise_paper,
build_dataframe, get_headers) are tested directly. make_request and
fetch_page use unittest.mock.patch to intercept requests.get.
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from data.fetch_papers import (
    get_headers,
    make_request,
    deduplicate_papers,
    filter_papers,
    normalise_paper,
    build_dataframe,
)


# ── get_headers ────────────────────────────────────────────────────────────────

class TestGetHeaders:

    def test_without_api_key_no_x_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("S2_API_KEY", None)
            headers = get_headers()
        assert "x-api-key" not in headers
        assert headers["Accept"] == "application/json"

    def test_with_api_key_included_in_headers(self):
        with patch.dict(os.environ, {"S2_API_KEY": "test-key-123"}):
            headers = get_headers()
        assert headers["x-api-key"] == "test-key-123"

    def test_always_includes_accept_json(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("S2_API_KEY", None)
            headers = get_headers()
        assert "Accept" in headers


# ── make_request ───────────────────────────────────────────────────────────────

class TestMakeRequest:

    def _mock_response(self, status_code, json_data=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data or {}
        mock_resp.text = str(json_data)
        return mock_resp

    @patch("data.fetch_papers.time.sleep")
    @patch("data.fetch_papers.requests.get")
    def test_200_returns_json(self, mock_get, mock_sleep):
        mock_get.return_value = self._mock_response(200, {"data": [1, 2, 3]})
        result = make_request("http://example.com", {}, {})
        assert result == {"data": [1, 2, 3]}
        mock_sleep.assert_not_called()

    @patch("data.fetch_papers.time.sleep")
    @patch("data.fetch_papers.requests.get")
    def test_429_retries_and_succeeds(self, mock_get, mock_sleep):
        # First call: rate-limited. Second: success.
        mock_get.side_effect = [
            self._mock_response(429),
            self._mock_response(200, {"data": "ok"}),
        ]
        result = make_request("http://example.com", {}, {}, max_retries=3)
        assert result == {"data": "ok"}
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("data.fetch_papers.time.sleep")
    @patch("data.fetch_papers.requests.get")
    def test_429_all_retries_exhausted_returns_none(self, mock_get, mock_sleep):
        mock_get.return_value = self._mock_response(429)
        result = make_request("http://example.com", {}, {}, max_retries=3)
        assert result is None
        assert mock_get.call_count == 3

    @patch("data.fetch_papers.time.sleep")
    @patch("data.fetch_papers.requests.get")
    def test_non_200_non_429_returns_none(self, mock_get, mock_sleep):
        mock_get.return_value = self._mock_response(403)
        result = make_request("http://example.com", {}, {})
        assert result is None

    @patch("data.fetch_papers.time.sleep")
    @patch("data.fetch_papers.requests.get")
    def test_504_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            self._mock_response(504),
            self._mock_response(200, {"ok": True}),
        ]
        result = make_request("http://example.com", {}, {}, max_retries=3)
        assert result == {"ok": True}

    @patch("data.fetch_papers.time.sleep")
    @patch("data.fetch_papers.requests.get")
    def test_request_exception_returns_none(self, mock_get, mock_sleep):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("no connection")
        result = make_request("http://example.com", {}, {}, max_retries=2)
        assert result is None

    @patch("data.fetch_papers.time.sleep")
    @patch("data.fetch_papers.requests.get")
    def test_timeout_retries(self, mock_get, mock_sleep):
        import requests
        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            self._mock_response(200, {"data": "recovered"}),
        ]
        result = make_request("http://example.com", {}, {}, max_retries=3)
        assert result == {"data": "recovered"}


# ── deduplicate_papers ─────────────────────────────────────────────────────────

class TestDeduplicatePapers:

    def _paper(self, pid):
        return {"paperId": pid, "title": f"Title {pid}"}

    def test_empty_list_returns_empty(self):
        assert deduplicate_papers([]) == []

    def test_no_duplicates_returns_same_length(self):
        papers = [self._paper("a"), self._paper("b"), self._paper("c")]
        result = deduplicate_papers(papers)
        assert len(result) == 3

    def test_all_duplicates_returns_one(self):
        papers = [self._paper("a")] * 5
        result = deduplicate_papers(papers)
        assert len(result) == 1
        assert result[0]["paperId"] == "a"

    def test_mixed_duplicates_correct_count(self):
        papers = [self._paper("a"), self._paper("b"), self._paper("a"), self._paper("c")]
        result = deduplicate_papers(papers)
        assert len(result) == 3

    def test_preserves_insertion_order(self):
        papers = [self._paper("c"), self._paper("a"), self._paper("b"),
                  self._paper("a"), self._paper("c")]
        result = deduplicate_papers(papers)
        ids = [p["paperId"] for p in result]
        assert ids == ["c", "a", "b"]

    def test_single_paper_returned(self):
        papers = [self._paper("solo")]
        result = deduplicate_papers(papers)
        assert len(result) == 1


# ── filter_papers ──────────────────────────────────────────────────────────────

class TestFilterPapers:

    def _paper(self, abstract):
        return {"paperId": "x", "abstract": abstract}

    def test_empty_list_returns_empty(self):
        assert filter_papers([]) == []

    def test_paper_with_none_abstract_removed(self):
        paper = {"paperId": "x", "abstract": None}
        assert filter_papers([paper]) == []

    def test_paper_with_empty_abstract_removed(self):
        assert filter_papers([self._paper("")]) == []

    def test_paper_with_whitespace_abstract_removed(self):
        assert filter_papers([self._paper("   ")]) == []

    def test_paper_below_min_word_threshold_removed(self):
        short = "Only five words here."   # < 20 words
        assert filter_papers([self._paper(short)]) == []

    def test_paper_at_min_word_threshold_kept(self):
        # Exactly 20 words
        abstract = " ".join(["word"] * 20)
        result = filter_papers([self._paper(abstract)])
        assert len(result) == 1

    def test_paper_above_min_words_kept(self):
        abstract = " ".join(["word"] * 50)
        result = filter_papers([self._paper(abstract)])
        assert len(result) == 1

    def test_custom_min_abstract_words_respected(self):
        five_word_abstract = "One two three four five."
        result = filter_papers([self._paper(five_word_abstract)], min_abstract_words=5)
        assert len(result) == 1

    def test_mixed_papers_filtered_correctly(self):
        papers = [
            self._paper(""),                    # filtered
            self._paper(" ".join(["w"] * 25)), # kept
            self._paper("Too short."),          # filtered
            self._paper(" ".join(["w"] * 30)), # kept
        ]
        result = filter_papers(papers)
        assert len(result) == 2


# ── normalise_paper ────────────────────────────────────────────────────────────

class TestNormalisePaper:

    def _base_paper(self, **overrides):
        paper = {
            "paperId": "pid_001",
            "title": "Test Paper Title",
            "abstract": "A sufficiently long test abstract for normalisation.",
            "year": 2022,
            "citationCount": 42,
            "fieldsOfStudy": [{"category": "Computer Science", "source": "external"}],
            "references": [
                {"paperId": "ref_001", "title": "Ref 1"},
                {"paperId": "ref_002", "title": "Ref 2"},
            ],
            "externalIds": {"ArXiv": "2301.12345"},
        }
        paper.update(overrides)
        return paper

    def test_basic_field_mapping(self):
        result = normalise_paper(self._base_paper())
        assert result["paper_id"]       == "pid_001"
        assert result["title"]          == "Test Paper Title"
        assert result["year"]           == 2022
        assert result["citation_count"] == 42

    def test_references_extracted_as_id_list(self):
        result = normalise_paper(self._base_paper())
        assert result["references"] == ["ref_001", "ref_002"]

    def test_references_none_returns_empty_list(self):
        result = normalise_paper(self._base_paper(references=None))
        assert result["references"] == []

    def test_references_without_paper_id_excluded(self):
        paper = self._base_paper(references=[
            {"paperId": "good_ref", "title": "OK"},
            {"title": "No paperId here"},          # missing paperId
            {"paperId": None, "title": "Null ID"}, # None paperId
        ])
        result = normalise_paper(paper)
        assert result["references"] == ["good_ref"]

    def test_fields_of_study_from_dict_list(self):
        paper = self._base_paper(
            fieldsOfStudy=[{"category": "Computer Science"}, {"category": "Mathematics"}]
        )
        result = normalise_paper(paper)
        assert "Computer Science" in result["fields_of_study"]
        assert "Mathematics" in result["fields_of_study"]

    def test_fields_of_study_from_string_list(self):
        paper = self._base_paper(fieldsOfStudy=["Computer Science", "Physics"])
        result = normalise_paper(paper)
        assert result["fields_of_study"] == ["Computer Science", "Physics"]

    def test_fields_of_study_none_returns_empty(self):
        result = normalise_paper(self._base_paper(fieldsOfStudy=None))
        assert result["fields_of_study"] == []

    def test_arxiv_id_extracted(self):
        result = normalise_paper(self._base_paper())
        assert result["arxiv_id"] == "2301.12345"

    def test_missing_arxiv_id_returns_empty_string(self):
        result = normalise_paper(self._base_paper(externalIds={}))
        assert result["arxiv_id"] == ""

    def test_external_ids_none_returns_empty_arxiv(self):
        result = normalise_paper(self._base_paper(externalIds=None))
        assert result["arxiv_id"] == ""

    def test_none_title_becomes_empty_string(self):
        result = normalise_paper(self._base_paper(title=None))
        assert result["title"] == ""

    def test_none_abstract_becomes_empty_string(self):
        result = normalise_paper(self._base_paper(abstract=None))
        assert result["abstract"] == ""

    def test_none_citation_count_becomes_zero(self):
        result = normalise_paper(self._base_paper(citationCount=None))
        assert result["citation_count"] == 0


# ── build_dataframe ────────────────────────────────────────────────────────────

class TestBuildDataframe:

    def _papers(self):
        return [
            {
                "paperId": f"pid_{i}",
                "title": f"Title {i}",
                "abstract": f"Abstract {i} " * 10,
                "year": 2020 + i,
                "citationCount": i * 5,
                "fieldsOfStudy": [{"category": "CS"}],
                "references": [{"paperId": f"ref_{i}"}],
                "externalIds": {"ArXiv": f"230{i}.00001"},
            }
            for i in range(5)
        ]

    def test_returns_dataframe(self):
        df = build_dataframe(self._papers())
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = build_dataframe(self._papers())
        assert len(df) == 5

    def test_required_columns_present(self):
        df = build_dataframe(self._papers())
        for col in ["paper_id", "title", "abstract", "year", "citation_count",
                    "fields_of_study", "references", "arxiv_id"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_year_is_nullable_int64(self):
        df = build_dataframe(self._papers())
        assert str(df["year"].dtype) == "Int64"

    def test_year_none_becomes_na(self):
        papers = self._papers()
        papers[0]["year"] = None
        df = build_dataframe(papers)
        assert pd.isna(df.loc[0, "year"])

    def test_references_are_lists(self):
        df = build_dataframe(self._papers())
        assert isinstance(df.loc[0, "references"], list)

    def test_fields_of_study_are_lists(self):
        df = build_dataframe(self._papers())
        assert isinstance(df.loc[0, "fields_of_study"], list)
