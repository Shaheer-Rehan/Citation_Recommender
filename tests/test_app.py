"""
test_app.py
-----------
Unit tests for app.py helper functions.

Streamlit UI rendering functions (display_result_card, render_sidebar, etc.)
are not tested here as they require a live Streamlit runtime. Only the pure
utility helpers are covered: clean_arxiv_id, apply_filters, make_paper_url,
and get_title_lookup.
"""

import pytest
from app import clean_arxiv_id, apply_filters, make_paper_url, get_title_lookup


# ── clean_arxiv_id ─────────────────────────────────────────────────────────────

class TestCleanArxivId:

    def test_plain_id_unchanged(self):
        assert clean_arxiv_id("2310.06825") == "2310.06825"

    def test_id_with_version_unchanged(self):
        assert clean_arxiv_id("2310.06825v2") == "2310.06825v2"

    def test_strips_abs_url_prefix(self):
        assert clean_arxiv_id("https://arxiv.org/abs/2310.06825") == "2310.06825"

    def test_strips_pdf_url_prefix(self):
        assert clean_arxiv_id("https://arxiv.org/pdf/2310.06825") == "2310.06825"

    def test_strips_pdf_extension(self):
        assert clean_arxiv_id("https://arxiv.org/pdf/2310.06825.pdf") == "2310.06825"

    def test_http_url_also_stripped(self):
        assert clean_arxiv_id("http://arxiv.org/abs/2310.06825") == "2310.06825"

    def test_leading_trailing_whitespace_stripped(self):
        assert clean_arxiv_id("  2310.06825  ") == "2310.06825"

    def test_abs_url_with_version_stripped_correctly(self):
        assert clean_arxiv_id("https://arxiv.org/abs/2310.06825v3") == "2310.06825v3"

    def test_returns_string(self):
        result = clean_arxiv_id("1234.5678")
        assert isinstance(result, str)


# ── apply_filters ──────────────────────────────────────────────────────────────

class TestApplyFilters:

    def _candidate(self, year, citations):
        return {
            "paper_id":        "pid",
            "title":           "Title",
            "year":            year,
            "citation_count":  citations,
            "score":           0.8,
        }

    def test_no_filters_returns_all(self):
        candidates = [
            self._candidate(2010, 0),
            self._candidate(2020, 500),
            self._candidate(2015, 50),
        ]
        result = apply_filters(candidates, min_year=2000, min_citations=0)
        assert len(result) == 3

    def test_year_filter_removes_old_papers(self):
        candidates = [
            self._candidate(2009, 10),
            self._candidate(2015, 10),
            self._candidate(2022, 10),
        ]
        result = apply_filters(candidates, min_year=2010, min_citations=0)
        assert len(result) == 2
        years = [r["year"] for r in result]
        assert 2009 not in years

    def test_year_boundary_is_inclusive(self):
        candidates = [self._candidate(2010, 0)]
        result = apply_filters(candidates, min_year=2010, min_citations=0)
        assert len(result) == 1

    def test_citation_filter_removes_low_cited(self):
        candidates = [
            self._candidate(2020, 5),
            self._candidate(2020, 100),
            self._candidate(2020, 0),
        ]
        result = apply_filters(candidates, min_year=2000, min_citations=10)
        assert len(result) == 1
        assert result[0]["citation_count"] == 100

    def test_citation_boundary_is_inclusive(self):
        candidates = [self._candidate(2020, 10)]
        result = apply_filters(candidates, min_year=2000, min_citations=10)
        assert len(result) == 1

    def test_both_filters_applied_simultaneously(self):
        candidates = [
            self._candidate(2008, 500),   # too old
            self._candidate(2020, 3),     # too few citations
            self._candidate(2018, 50),    # passes both
            self._candidate(2022, 1000),  # passes both
        ]
        result = apply_filters(candidates, min_year=2010, min_citations=10)
        assert len(result) == 2

    def test_none_year_paper_is_kept(self):
        # Papers without a publication year should not be penalised
        candidates = [self._candidate(None, 100)]
        result = apply_filters(candidates, min_year=2015, min_citations=0)
        assert len(result) == 1

    def test_empty_candidates_returns_empty(self):
        assert apply_filters([], min_year=2010, min_citations=0) == []

    def test_all_filtered_returns_empty(self):
        candidates = [self._candidate(2005, 0), self._candidate(2008, 2)]
        result = apply_filters(candidates, min_year=2020, min_citations=100)
        assert result == []


# ── make_paper_url ─────────────────────────────────────────────────────────────

class TestMakePaperUrl:

    def _result(self, paper_id, arxiv_id=""):
        return {"paper_id": paper_id, "arxiv_id": arxiv_id}

    def test_arxiv_id_present_returns_arxiv_url(self):
        result = make_paper_url(self._result("pid_abc", "2310.06825"))
        assert result == "https://arxiv.org/abs/2310.06825"
        assert "semanticscholar" not in result

    def test_no_arxiv_id_returns_semantic_scholar_url(self):
        result = make_paper_url(self._result("pid_abc123", ""))
        assert "semanticscholar.org" in result
        assert "pid_abc123" in result

    def test_none_arxiv_id_falls_back_to_semantic_scholar(self):
        record = {"paper_id": "pid_xyz", "arxiv_id": None}
        result = make_paper_url(record)
        assert "semanticscholar.org" in result

    def test_whitespace_arxiv_id_treated_as_missing(self):
        record = {"paper_id": "pid_xyz", "arxiv_id": "   "}
        result = make_paper_url(record)
        # strip() in make_paper_url turns whitespace → "" → fallback
        assert "semanticscholar.org" in result

    def test_url_is_string(self):
        result = make_paper_url(self._result("pid_000", "1234.5678"))
        assert isinstance(result, str)


# ── get_title_lookup ───────────────────────────────────────────────────────────

class TestGetTitleLookup:

    def test_lookup_returns_dict(self, sample_resources):
        result = get_title_lookup(sample_resources)
        assert isinstance(result, dict)

    def test_lookup_length_matches_metadata(self, sample_resources):
        result = get_title_lookup(sample_resources)
        assert len(result) == len(sample_resources["metadata"])

    def test_lookup_keys_are_lowercase(self, sample_resources):
        result = get_title_lookup(sample_resources)
        for key in result:
            assert key == key.lower()

    def test_lookup_values_are_metadata_dicts(self, sample_resources):
        result = get_title_lookup(sample_resources)
        for val in result.values():
            assert "paper_id" in val
            assert "title" in val

    def test_exact_title_match_found_case_insensitive(self, sample_resources):
        # Take a known title from the metadata, look it up with different casing
        first_title = sample_resources["metadata"][0]["title"]
        result = get_title_lookup(sample_resources)
        # Should find it regardless of case
        assert first_title.lower().strip() in result

    def test_nonexistent_title_not_in_lookup(self, sample_resources):
        result = get_title_lookup(sample_resources)
        assert "this title definitely does not exist in the corpus" not in result
