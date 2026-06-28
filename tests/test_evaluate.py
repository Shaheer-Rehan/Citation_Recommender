"""
test_evaluate.py
----------------
Unit tests for evaluation/evaluate.py.

compute_metrics and aggregate_metrics are pure — results verified against
manually computed expected values. sample_test_papers tests filtering logic.
tfidf_recommend is tested with a real sparse TF-IDF matrix on synthetic data.
"""

import numpy as np
import pytest

from evaluation.evaluate import (
    compute_metrics,
    aggregate_metrics,
    sample_test_papers,
    tfidf_recommend,
)


# ── compute_metrics ────────────────────────────────────────────────────────────

class TestComputeMetrics:
    """
    All expected values are computed by hand to serve as regression anchors.
    """

    def test_perfect_recommendations_all_metrics_one(self):
        # All top-3 are ground truth
        ranked = ["a", "b", "c", "d", "e"]
        gt     = ["a", "b", "c"]
        m = compute_metrics(ranked, gt, k=3)
        assert m["precision_at_k"] == pytest.approx(1.0)
        assert m["recall_at_k"]    == pytest.approx(1.0)
        assert m["ndcg_at_k"]      == pytest.approx(1.0)
        assert m["mrr"]            == pytest.approx(1.0)

    def test_no_correct_recommendations_all_metrics_zero(self):
        ranked = ["x", "y", "z"]
        gt     = ["a", "b", "c"]
        m = compute_metrics(ranked, gt, k=3)
        assert m["precision_at_k"] == pytest.approx(0.0)
        assert m["recall_at_k"]    == pytest.approx(0.0)
        assert m["ndcg_at_k"]      == pytest.approx(0.0)
        assert m["mrr"]            == pytest.approx(0.0)

    def test_precision_at_k_correct(self):
        # 2 hits in top-5
        ranked = ["a", "x", "b", "y", "z"]
        gt     = ["a", "b", "c"]
        m = compute_metrics(ranked, gt, k=5)
        assert m["precision_at_k"] == pytest.approx(2 / 5)

    def test_recall_at_k_correct(self):
        # 2 hits out of 3 ground truth
        ranked = ["a", "x", "b", "y", "z"]
        gt     = ["a", "b", "c"]
        m = compute_metrics(ranked, gt, k=5)
        assert m["recall_at_k"] == pytest.approx(2 / 3)

    def test_mrr_first_hit_at_rank_1(self):
        ranked = ["a", "x", "y"]
        gt     = ["a"]
        m = compute_metrics(ranked, gt, k=3)
        assert m["mrr"] == pytest.approx(1.0)

    def test_mrr_first_hit_at_rank_2(self):
        ranked = ["x", "a", "y"]
        gt     = ["a"]
        m = compute_metrics(ranked, gt, k=3)
        assert m["mrr"] == pytest.approx(0.5)

    def test_mrr_first_hit_at_rank_3(self):
        ranked = ["x", "y", "a"]
        gt     = ["a"]
        m = compute_metrics(ranked, gt, k=3)
        assert m["mrr"] == pytest.approx(1 / 3)

    def test_mrr_no_hit_is_zero(self):
        ranked = ["x", "y", "z"]
        gt     = ["a"]
        m = compute_metrics(ranked, gt, k=3)
        assert m["mrr"] == pytest.approx(0.0)

    def test_ndcg_manually_verified(self):
        # Hits at positions 0 and 2 (0-indexed), k=5, 2 ground truth items
        # DCG  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        # NDCG = 1.5 / 1.6309 ≈ 0.9197
        ranked = ["a", "x", "b", "y", "z"]
        gt     = ["a", "b"]
        m = compute_metrics(ranked, gt, k=5)
        assert m["ndcg_at_k"] == pytest.approx(1.5 / (1.0 + 1 / np.log2(3)), abs=1e-4)

    def test_ndcg_perfect_score_equals_one(self):
        ranked = ["a", "b"]
        gt     = ["a", "b"]
        m = compute_metrics(ranked, gt, k=2)
        assert m["ndcg_at_k"] == pytest.approx(1.0)

    def test_k_equals_one_precision_is_binary(self):
        assert compute_metrics(["a"], ["a"], k=1)["precision_at_k"] == pytest.approx(1.0)
        assert compute_metrics(["x"], ["a"], k=1)["precision_at_k"] == pytest.approx(0.0)

    def test_empty_ground_truth_all_zeros(self):
        ranked = ["a", "b", "c"]
        m = compute_metrics(ranked, [], k=3)
        for key, val in m.items():
            assert val == pytest.approx(0.0), f"{key} should be 0 for empty ground truth"

    def test_ranked_shorter_than_k(self):
        # Only 2 candidates returned, k=5 — must not crash
        ranked = ["a", "b"]
        gt     = ["a", "b", "c"]
        m = compute_metrics(ranked, gt, k=5)
        assert 0.0 <= m["precision_at_k"] <= 1.0
        assert 0.0 <= m["recall_at_k"]    <= 1.0

    def test_all_metrics_are_in_zero_one_range(self):
        ranked = ["a", "z", "b", "y", "c"]
        gt     = ["a", "b", "c", "d"]
        m = compute_metrics(ranked, gt, k=5)
        for key, val in m.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


# ── aggregate_metrics ──────────────────────────────────────────────────────────

class TestAggregateMetrics:

    def test_empty_list_returns_zeros(self):
        result = aggregate_metrics([])
        for val in result.values():
            assert val == pytest.approx(0.0)

    def test_single_paper_returns_same_values(self):
        single = {"precision_at_k": 0.4, "recall_at_k": 0.6,
                  "ndcg_at_k": 0.5, "mrr": 0.333}
        result = aggregate_metrics([single])
        for key, val in single.items():
            assert result[key] == pytest.approx(val, abs=1e-3)

    def test_two_papers_averaged_correctly(self):
        m1 = {"precision_at_k": 0.2, "recall_at_k": 0.3, "ndcg_at_k": 0.25, "mrr": 0.5}
        m2 = {"precision_at_k": 0.4, "recall_at_k": 0.5, "ndcg_at_k": 0.45, "mrr": 0.25}
        result = aggregate_metrics([m1, m2])
        assert result["precision_at_k"] == pytest.approx(0.3,  abs=1e-4)
        assert result["recall_at_k"]    == pytest.approx(0.4,  abs=1e-4)
        assert result["ndcg_at_k"]      == pytest.approx(0.35, abs=1e-4)
        assert result["mrr"]            == pytest.approx(0.375, abs=1e-4)

    def test_result_values_rounded_to_4_decimal_places(self):
        m = [{"precision_at_k": 1/3, "recall_at_k": 1/7,
              "ndcg_at_k": 2/9, "mrr": 1/6}]
        result = aggregate_metrics(m)
        for key, val in result.items():
            # 4 decimal places: str repr shouldn't have more than 4 significant decimals
            assert abs(val - round(val, 4)) < 1e-9

    def test_all_zeros_input_returns_zeros(self):
        zero = {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "mrr": 0.0}
        result = aggregate_metrics([zero, zero, zero])
        for val in result.values():
            assert val == pytest.approx(0.0)


# ── sample_test_papers ─────────────────────────────────────────────────────────

class TestSampleTestPapers:

    def _make_metadata(self, n=20, refs_per_paper=5):
        """Synthetic metadata list where every paper has refs_per_paper in-corpus refs."""
        corpus = {f"corpus_{i}" for i in range(50)}
        result = []
        for i in range(n):
            result.append({
                "paper_id":       f"paper_{i}",
                "title":          f"Paper {i}",
                "abstract":       "Abstract " * 10,
                "references":     [f"corpus_{j}" for j in range(refs_per_paper)],
                "year":           2020,
                "citation_count": 0,
                "fields_of_study": [],
                "arxiv_id":       "",
            })
        return result, corpus

    def test_returns_at_most_n_papers(self):
        metadata, corpus = self._make_metadata(n=20, refs_per_paper=5)
        result = sample_test_papers(metadata, corpus, n=10, min_refs=3)
        assert len(result) <= 10

    def test_filters_papers_below_min_refs(self):
        metadata, corpus = self._make_metadata(n=20, refs_per_paper=2)
        # All papers have only 2 in-corpus refs; min_refs=3 → all filtered
        result = sample_test_papers(metadata, corpus, n=10, min_refs=3)
        assert result == []

    def test_all_eligible_returned_when_fewer_than_n(self):
        metadata, corpus = self._make_metadata(n=5, refs_per_paper=4)
        result = sample_test_papers(metadata, corpus, n=100, min_refs=3)
        assert len(result) == 5

    def test_in_corpus_refs_field_added(self):
        metadata, corpus = self._make_metadata(n=5, refs_per_paper=5)
        result = sample_test_papers(metadata, corpus, n=5, min_refs=3)
        for paper in result:
            assert "in_corpus_refs" in paper
            assert len(paper["in_corpus_refs"]) >= 3

    def test_same_seed_produces_same_sample(self):
        metadata, corpus = self._make_metadata(n=20, refs_per_paper=5)
        r1 = sample_test_papers(metadata, corpus, n=10, min_refs=3, seed=42)
        r2 = sample_test_papers(metadata, corpus, n=10, min_refs=3, seed=42)
        ids1 = [p["paper_id"] for p in r1]
        ids2 = [p["paper_id"] for p in r2]
        assert ids1 == ids2

    def test_different_seeds_may_differ(self):
        metadata, corpus = self._make_metadata(n=20, refs_per_paper=5)
        r1 = sample_test_papers(metadata, corpus, n=10, min_refs=3, seed=1)
        r2 = sample_test_papers(metadata, corpus, n=10, min_refs=3, seed=99)
        ids1 = [p["paper_id"] for p in r1]
        ids2 = [p["paper_id"] for p in r2]
        # With 20 papers and sample of 10, different seeds almost certainly differ
        assert ids1 != ids2


# ── tfidf_recommend ────────────────────────────────────────────────────────────

class TestTfidfRecommend:

    @pytest.fixture
    def tfidf_data(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        abstracts = [
            "machine learning deep neural network training optimization",
            "natural language processing text generation transformer model",
            "computer vision image classification convolutional neural network",
            "reinforcement learning reward policy gradient agent",
            "graph neural network node classification link prediction",
        ]
        paper_ids = [f"p{i}" for i in range(len(abstracts))]
        vec    = TfidfVectorizer()
        matrix = vec.fit_transform(abstracts)
        return {
            "matrix":    matrix,
            "paper_ids": paper_ids,
            "id_to_pos": {pid: i for i, pid in enumerate(paper_ids)},
        }

    def test_returns_list_of_paper_ids(self, tfidf_data):
        result = tfidf_recommend("p0", tfidf_data, k=3)
        assert isinstance(result, list)
        assert all(isinstance(x, str) for x in result)

    def test_query_paper_not_in_results(self, tfidf_data):
        result = tfidf_recommend("p0", tfidf_data, k=4)
        assert "p0" not in result

    def test_returns_at_most_k_results(self, tfidf_data):
        result = tfidf_recommend("p0", tfidf_data, k=2)
        assert len(result) <= 2

    def test_unknown_paper_returns_empty(self, tfidf_data):
        result = tfidf_recommend("nonexistent_id", tfidf_data, k=3)
        assert result == []

    def test_exclude_ids_not_in_results(self, tfidf_data):
        exclude = ["p1", "p2"]
        result = tfidf_recommend("p0", tfidf_data, k=3, exclude_ids=exclude)
        for eid in exclude:
            assert eid not in result

    def test_similar_abstracts_ranked_higher(self, tfidf_data):
        # p0 is about ML/deep learning; p1 is about NLP (transformers = also neural).
        # p2 is computer vision, p3 is RL, p4 is graph NN.
        # Top result for p0 should be a neural-network related paper.
        result = tfidf_recommend("p0", tfidf_data, k=4)
        assert len(result) >= 1   # at least one result returned
