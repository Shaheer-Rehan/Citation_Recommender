"""
test_explain.py
---------------
Unit tests for retrieval/explain.py.

split_sentences and _cosine_similarity_matrix are pure — tested directly.
find_best_pair and generate_explanation use the mock_sentence_model fixture.
"""

import numpy as np
import pytest
from retrieval.explain import (
    split_sentences,
    _cosine_similarity_matrix,
    find_best_pair,
    generate_explanation,
    explain_results,
    _FALLBACK_EXPLANATION,
    _MIN_SENTENCE_WORDS,
)


# ── split_sentences ────────────────────────────────────────────────────────────

class TestSplitSentences:

    def test_empty_string_returns_empty_list(self):
        assert split_sentences("") == []

    def test_none_like_empty_returns_empty(self):
        assert split_sentences("   ") == []

    def test_single_long_sentence_returned(self):
        text = "This is a sufficiently long sentence with many words in it for testing."
        result = split_sentences(text)
        assert len(result) >= 1
        assert any("sufficiently" in s for s in result)

    def test_short_sentences_filtered_out(self):
        # Each sentence is below _MIN_SENTENCE_WORDS
        text = "OK. Yes. No. Fine."
        result = split_sentences(text)
        assert result == []

    def test_mixed_length_sentences(self):
        long_sent  = "This sentence is definitely long enough to survive the filter threshold."
        short_sent = "Too short."
        text = f"{long_sent} {short_sent}"
        result = split_sentences(text)
        assert any("definitely" in s for s in result)
        # Short sentence should be filtered
        assert not any(s.strip() == short_sent for s in result)

    def test_multiple_long_sentences_all_kept(self):
        sents = [
            "First sentence with enough words to pass the minimum word filter.",
            "Second sentence also containing sufficient words to survive filtering.",
            "Third independent sentence here with words to meet threshold requirements.",
        ]
        text = " ".join(sents)
        result = split_sentences(text)
        assert len(result) >= 2

    def test_minimum_word_boundary(self):
        # Construct a sentence with exactly _MIN_SENTENCE_WORDS words
        at_boundary = " ".join(["word"] * _MIN_SENTENCE_WORDS) + "."
        below       = " ".join(["word"] * (_MIN_SENTENCE_WORDS - 1)) + "."
        assert len(split_sentences(at_boundary)) == 1
        assert len(split_sentences(below)) == 0

    def test_returns_list_of_strings(self):
        text = "This is a sentence with plenty of words to clear the filter easily."
        result = split_sentences(text)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)


# ── _cosine_similarity_matrix ──────────────────────────────────────────────────

class TestCosineSimilarityMatrix:

    def _unit(self, v):
        return v / np.linalg.norm(v)

    def test_identity_vector_similarity_is_one(self):
        v = self._unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        a = v.reshape(1, -1)
        result = _cosine_similarity_matrix(a, a)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_similarity_is_zero(self):
        v1 = self._unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = self._unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        a = v1.reshape(1, -1)
        b = v2.reshape(1, -1)
        result = _cosine_similarity_matrix(a, b)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_output_shape_is_n_by_m(self):
        np.random.seed(1)
        a = np.random.randn(3, 128).astype(np.float32)
        b = np.random.randn(5, 128).astype(np.float32)
        # Normalise so IP = cosine
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        result = _cosine_similarity_matrix(a, b)
        assert result.shape == (3, 5)

    def test_symmetric_when_same_inputs(self):
        np.random.seed(2)
        vecs = np.random.randn(4, 64).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        result = _cosine_similarity_matrix(vecs, vecs)
        np.testing.assert_allclose(result, result.T, atol=1e-5)

    def test_similarity_bounded_for_unit_vectors(self):
        np.random.seed(3)
        a = np.random.randn(5, 64).astype(np.float32)
        b = np.random.randn(5, 64).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        result = _cosine_similarity_matrix(a, b)
        assert result.min() >= -1.01
        assert result.max() <= 1.01

    def test_identical_matrix_diagonal_is_one(self):
        np.random.seed(4)
        vecs = np.random.randn(3, 64).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        result = _cosine_similarity_matrix(vecs, vecs)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-5)


# ── find_best_pair ─────────────────────────────────────────────────────────────

class TestFindBestPair:

    def test_returns_none_triple_when_query_too_short(self, mock_sentence_model):
        short = "Too short."
        long  = "This is a sufficiently long candidate abstract with plenty of words."
        q_sent, c_sent, score = find_best_pair(short, long, mock_sentence_model)
        assert q_sent is None
        assert c_sent is None
        assert score == 0.0

    def test_returns_none_triple_when_candidate_too_short(self, mock_sentence_model):
        long  = "This is a sufficiently long query abstract containing many descriptive words."
        short = "Brief."
        q_sent, c_sent, score = find_best_pair(long, short, mock_sentence_model)
        assert q_sent is None

    def test_returns_strings_on_valid_input(self, mock_sentence_model):
        q = "This paper proposes a novel deep learning approach for language modelling."
        c = "We introduce a transformer model trained on large corpora of text data."
        q_sent, c_sent, score = find_best_pair(q, c, mock_sentence_model)
        assert isinstance(q_sent, str)
        assert isinstance(c_sent, str)
        assert isinstance(score, float)

    def test_score_is_bounded(self, mock_sentence_model):
        q = "We study deep learning methods applied to natural language generation tasks."
        c = "Neural network approaches to text generation have shown remarkable progress."
        _, _, score = find_best_pair(q, c, mock_sentence_model)
        assert -1.01 <= score <= 1.01


# ── generate_explanation ───────────────────────────────────────────────────────

class TestGenerateExplanation:

    def test_empty_query_abstract_returns_fallback(self, mock_sentence_model):
        result = generate_explanation("", "Some long enough candidate abstract text here.", mock_sentence_model)
        assert result == _FALLBACK_EXPLANATION

    def test_empty_candidate_abstract_returns_fallback(self, mock_sentence_model):
        result = generate_explanation("Some long enough query abstract text here.", "", mock_sentence_model)
        assert result == _FALLBACK_EXPLANATION

    def test_both_empty_returns_fallback(self, mock_sentence_model):
        result = generate_explanation("", "", mock_sentence_model)
        assert result == _FALLBACK_EXPLANATION

    def test_valid_abstracts_returns_non_empty_string(self, mock_sentence_model):
        q = "This paper studies attention mechanisms in transformer architectures for NLP."
        c = "We present a self-attention model trained on large text corpora achieving SOTA."
        result = generate_explanation(q, c, mock_sentence_model)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_valid_abstracts_result_is_not_fallback(self, mock_sentence_model):
        q = "We propose a deep learning framework for document summarisation at scale."
        c = "Neural network models for text summarisation achieve strong benchmark results."
        result = generate_explanation(q, c, mock_sentence_model)
        assert result != _FALLBACK_EXPLANATION

    def test_never_raises_on_exception(self, mock_sentence_model):
        # Simulate a model that raises unexpectedly
        mock_sentence_model.encode.side_effect = RuntimeError("simulated crash")
        result = generate_explanation("Some long query abstract words here.", "Some long candidate words.", mock_sentence_model)
        assert result == _FALLBACK_EXPLANATION
        # Restore for subsequent tests
        mock_sentence_model.encode.side_effect = None


# ── explain_results ────────────────────────────────────────────────────────────

class TestExplainResults:

    def test_empty_results_returned_unchanged(self, mock_sentence_model):
        assert explain_results([], "some abstract", mock_sentence_model) == []

    def test_explanation_field_added_to_each_result(self, mock_sentence_model, sample_candidates):
        results = [dict(c) for c in sample_candidates[:3]]
        query_abstract = "A long query abstract about deep learning and NLP research."
        explain_results(results, query_abstract, mock_sentence_model)
        for r in results:
            assert "explanation" in r
            assert isinstance(r["explanation"], str)

    def test_returns_same_list_reference(self, mock_sentence_model, sample_candidates):
        results = [dict(c) for c in sample_candidates[:2]]
        returned = explain_results(results, "abstract text here", mock_sentence_model)
        assert returned is results
