"""
test_retrieve.py
----------------
Unit tests for retrieval/retrieve.py.

_format_input and _normalise are pure — tested directly.
search_index uses the real small_faiss_index fixture (no mock needed — FAISS is fast).
embed_single_paper and embed_catalogue patch _embed_texts to avoid loading a real model.
"""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from retrieval.retrieve import (
    _format_input,
    _normalise,
    search_index,
    embed_single_paper,
    embed_catalogue,
)

DIM = 768


# ── _format_input ──────────────────────────────────────────────────────────────

class TestFormatInput:

    def test_title_and_abstract_combined_with_sep(self):
        result = _format_input("My Title", "My abstract text.", "[SEP]")
        assert result == "My Title[SEP]My abstract text."

    def test_empty_title_returns_only_abstract(self):
        result = _format_input("", "Abstract only.", "[SEP]")
        assert result == "Abstract only."

    def test_none_title_returns_only_abstract(self):
        result = _format_input(None, "Abstract only.", "[SEP]")
        assert result == "Abstract only."

    def test_whitespace_title_treated_as_empty(self):
        # strip() in the function converts "   " to "", triggering abstract-only path
        result = _format_input("   ", "Abstract only.", "[SEP]")
        assert result == "Abstract only."

    def test_empty_abstract_with_title(self):
        result = _format_input("Title Here", "", "[SEP]")
        # No abstract → just title + sep (sep_token is always appended)
        assert "Title Here" in result

    def test_both_empty_returns_empty_or_minimal(self):
        result = _format_input("", "", "[SEP]")
        assert isinstance(result, str)

    def test_sep_token_preserved_exactly(self):
        result = _format_input("T", "A", ">>SEP<<")
        assert ">>SEP<<" in result


# ── _normalise ─────────────────────────────────────────────────────────────────

class TestNormalise:

    def test_1d_vector_becomes_unit_norm(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = _normalise(v)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)

    def test_2d_matrix_all_rows_unit_norm(self):
        np.random.seed(5)
        mat = np.random.randn(8, DIM).astype(np.float32)
        result = _normalise(mat)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_already_unit_norm_unchanged(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _normalise(v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_zero_1d_vector_no_crash(self):
        v = np.zeros(DIM, dtype=np.float32)
        result = _normalise(v)   # should not raise ZeroDivisionError
        assert not np.any(np.isnan(result))

    def test_zero_2d_row_no_crash(self):
        mat = np.zeros((3, DIM), dtype=np.float32)
        result = _normalise(mat)
        assert not np.any(np.isnan(result))

    def test_direction_preserved_after_normalisation(self):
        v = np.array([3.0, 0.0, 4.0], dtype=np.float32)
        result = _normalise(v)
        # Direction: ratio of components must be preserved
        assert result[0] / result[2] == pytest.approx(3.0 / 4.0, abs=1e-5)

    def test_returns_float32(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _normalise(v.astype(np.float32))
        assert result.dtype == np.float32


# ── search_index ───────────────────────────────────────────────────────────────

class TestSearchIndex:

    def test_returns_list_of_dicts(self, sample_resources, sample_embeddings):
        query_vec = sample_embeddings[0]
        results = search_index(query_vec, sample_resources, k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_returns_at_most_k_results(self, sample_resources, sample_embeddings):
        query_vec = sample_embeddings[0]
        results = search_index(query_vec, sample_resources, k=3)
        assert len(results) <= 3

    def test_each_result_has_score_field(self, sample_resources, sample_embeddings):
        query_vec = sample_embeddings[0]
        results = search_index(query_vec, sample_resources, k=5)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_results_sorted_by_score_descending(self, sample_resources, sample_embeddings):
        query_vec = sample_embeddings[0]
        results = search_index(query_vec, sample_resources, k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exclude_ids_removes_papers(self, sample_resources, sample_embeddings, sample_paper_ids):
        query_vec = sample_embeddings[0]
        exclude = [sample_paper_ids[0]]
        results = search_index(query_vec, sample_resources, k=5, exclude_ids=exclude)
        returned_ids = [r["paper_id"] for r in results]
        assert sample_paper_ids[0] not in returned_ids

    def test_exclude_multiple_ids(self, sample_resources, sample_embeddings, sample_paper_ids):
        query_vec = sample_embeddings[0]
        exclude = sample_paper_ids[:3]
        results = search_index(query_vec, sample_resources, k=5, exclude_ids=exclude)
        returned_ids = [r["paper_id"] for r in results]
        for eid in exclude:
            assert eid not in returned_ids

    def test_k_larger_than_corpus_returns_all_minus_excluded(self, sample_resources, sample_embeddings):
        query_vec = sample_embeddings[0]
        results = search_index(query_vec, sample_resources, k=100)
        # Index has 10 vectors; at most 10 results
        assert len(results) <= 10

    def test_no_exclude_ids_returns_k_results(self, sample_resources, sample_embeddings):
        query_vec = sample_embeddings[0]
        k = 4
        results = search_index(query_vec, sample_resources, k=k)
        assert len(results) == k

    def test_results_contain_expected_metadata_keys(self, sample_resources, sample_embeddings):
        query_vec = sample_embeddings[0]
        results = search_index(query_vec, sample_resources, k=1)
        required = {"paper_id", "title", "abstract", "year",
                    "citation_count", "references", "score"}
        assert required.issubset(set(results[0].keys()))


# ── embed_single_paper ─────────────────────────────────────────────────────────

class TestEmbedSinglePaper:

    def _fake_embed_texts(self, texts, tokenizer, model, device):
        """Returns a unit-norm random vector for each input text."""
        np.random.seed(7)
        vecs  = np.random.randn(len(texts), DIM).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_returns_1d_array_of_correct_dim(self, sample_resources):
        with patch("retrieval.retrieve._embed_texts", side_effect=self._fake_embed_texts):
            result = embed_single_paper("Title", "Abstract text.", sample_resources)
        assert result.shape == (DIM,)

    def test_output_is_unit_norm(self, sample_resources):
        with patch("retrieval.retrieve._embed_texts", side_effect=self._fake_embed_texts):
            result = embed_single_paper("Title", "Abstract text.", sample_resources)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_output_dtype_is_float32(self, sample_resources):
        with patch("retrieval.retrieve._embed_texts", side_effect=self._fake_embed_texts):
            result = embed_single_paper("Title", "Abstract text.", sample_resources)
        assert result.dtype == np.float32


# ── embed_catalogue ────────────────────────────────────────────────────────────

class TestEmbedCatalogue:

    def _fake_embed_texts(self, texts, tokenizer, model, device):
        np.random.seed(8)
        vecs  = np.random.randn(len(texts), DIM).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_empty_list_returns_zero_vector(self, sample_resources):
        result = embed_catalogue([], sample_resources)
        assert result.shape == (DIM,)
        assert np.all(result == 0.0)

    def test_single_paper_returns_unit_norm_vector(self, sample_resources):
        papers = [{"title": "A title", "abstract": "An abstract text for testing."}]
        with patch("retrieval.retrieve._embed_texts", side_effect=self._fake_embed_texts):
            result = embed_catalogue(papers, sample_resources)
        assert result.shape == (DIM,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_multiple_papers_returns_unit_norm_vector(self, sample_resources):
        papers = [
            {"title": f"Paper {i}", "abstract": f"Abstract {i} content."}
            for i in range(4)
        ]
        with patch("retrieval.retrieve._embed_texts", side_effect=self._fake_embed_texts):
            result = embed_catalogue(papers, sample_resources)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_output_is_float32(self, sample_resources):
        papers = [{"title": "T", "abstract": "Abstract content here."}]
        with patch("retrieval.retrieve._embed_texts", side_effect=self._fake_embed_texts):
            result = embed_catalogue(papers, sample_resources)
        assert result.dtype == np.float32
