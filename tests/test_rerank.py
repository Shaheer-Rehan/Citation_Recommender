"""
test_rerank.py
--------------
Unit tests for retrieval/rerank.py.
All functions here are pure logic — no mocking required.
"""

import pytest
from retrieval.rerank import compute_citation_overlap, _compute_hybrid_score, rerank


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_candidate(paper_id, score, refs=None):
    """Minimal candidate dict matching the shape returned by search_index()."""
    return {
        "paper_id":        paper_id,
        "title":           f"Paper {paper_id}",
        "abstract":        "Some abstract text here for testing purposes.",
        "year":            2021,
        "citation_count":  42,
        "fields_of_study": ["Computer Science"],
        "references":      refs or [],
        "arxiv_id":        "",
        "score":           score,
    }


# ── compute_citation_overlap ───────────────────────────────────────────────────

class TestComputeCitationOverlap:

    def test_empty_query_refs_returns_zero(self):
        """No query references means no overlap signal — must return 0."""
        assert compute_citation_overlap([], ["a", "b"]) == 0.0

    def test_empty_candidate_refs_returns_zero(self):
        assert compute_citation_overlap(["a", "b"], []) == 0.0

    def test_both_empty_returns_zero(self):
        assert compute_citation_overlap([], []) == 0.0

    def test_completely_disjoint_refs(self):
        assert compute_citation_overlap(["a", "b"], ["c", "d"]) == 0.0

    def test_full_overlap_returns_one(self):
        refs = ["pid_001", "pid_002", "pid_003"]
        assert compute_citation_overlap(refs, refs) == pytest.approx(1.0)

    def test_partial_overlap_correct_fraction(self):
        # query has 4 refs, 2 shared with candidate → 2/4 = 0.5
        q = ["a", "b", "c", "d"]
        c = ["b", "c", "e", "f"]
        assert compute_citation_overlap(q, c) == pytest.approx(0.5)

    def test_one_of_three_shared(self):
        assert compute_citation_overlap(["a", "b", "c"], ["a"]) == pytest.approx(1 / 3)

    def test_metric_is_not_symmetric(self):
        # query has 2 refs, candidate has 4. Overlap = 2/2 = 1.0 from query perspective.
        # Reversed: 2/4 = 0.5. Intentionally asymmetric.
        q = ["a", "b"]
        c = ["a", "b", "c", "d"]
        assert compute_citation_overlap(q, c) == pytest.approx(1.0)
        assert compute_citation_overlap(c, q) == pytest.approx(0.5)

    def test_duplicate_query_refs_handled_via_sets(self):
        # Duplicates collapsed: set(["a","a","b"]) = {a, b}, intersection with ["a"] = {a}
        # overlap = 1/2 = 0.5
        assert compute_citation_overlap(["a", "a", "b"], ["a"]) == pytest.approx(0.5)

    def test_single_ref_full_hit(self):
        assert compute_citation_overlap(["only_one"], ["only_one", "extra"]) == pytest.approx(1.0)

    def test_single_ref_no_hit(self):
        assert compute_citation_overlap(["only_one"], ["other"]) == pytest.approx(0.0)


# ── _compute_hybrid_score ──────────────────────────────────────────────────────

class TestComputeHybridScore:

    def test_alpha_one_is_pure_semantic(self):
        assert _compute_hybrid_score(0.8, 0.5, alpha=1.0) == pytest.approx(0.8)

    def test_alpha_zero_is_pure_citation(self):
        assert _compute_hybrid_score(0.8, 0.5, alpha=0.0) == pytest.approx(0.5)

    def test_alpha_half_averages(self):
        assert _compute_hybrid_score(0.6, 0.4, alpha=0.5) == pytest.approx(0.5)

    def test_alpha_point_seven_correct_weighting(self):
        s, c, a = 0.8, 0.4, 0.7
        expected = 0.7 * 0.8 + 0.3 * 0.4
        assert _compute_hybrid_score(s, c, a) == pytest.approx(expected)

    def test_both_inputs_zero_returns_zero(self):
        assert _compute_hybrid_score(0.0, 0.0, 0.7) == pytest.approx(0.0)

    def test_both_inputs_one_returns_one(self):
        assert _compute_hybrid_score(1.0, 1.0, 0.7) == pytest.approx(1.0)

    def test_output_range_between_inputs(self):
        # hybrid must lie between the two component scores
        s, c, a = 0.9, 0.1, 0.7
        result = _compute_hybrid_score(s, c, a)
        assert c <= result <= s


# ── rerank ─────────────────────────────────────────────────────────────────────

class TestRerank:

    def test_empty_candidates_returns_empty(self):
        assert rerank([], query_refs=[], alpha=0.7, top_k=10) == []

    def test_single_candidate_returned(self):
        candidates = [make_candidate("a", 0.9)]
        result = rerank(candidates, query_refs=[], alpha=0.7, top_k=10)
        assert len(result) == 1
        assert result[0]["paper_id"] == "a"

    def test_results_sorted_by_hybrid_score_descending(self):
        candidates = [
            make_candidate("low",  0.5),
            make_candidate("high", 0.9),
            make_candidate("mid",  0.7),
        ]
        # alpha=1.0 → hybrid = semantic, so ordering by cosine sim
        result = rerank(candidates, query_refs=[], alpha=1.0, top_k=3)
        scores = [r["hybrid_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self):
        candidates = [make_candidate(f"p{i}", 0.9 - i * 0.05) for i in range(8)]
        result = rerank(candidates, query_refs=[], alpha=0.7, top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_candidates_returns_all(self):
        candidates = [make_candidate("a", 0.8), make_candidate("b", 0.6)]
        result = rerank(candidates, query_refs=[], alpha=0.7, top_k=20)
        assert len(result) == 2

    def test_citation_overlap_can_change_ranking(self):
        # Paper B has lower semantic score but perfectly shares references.
        candidates = [
            make_candidate("semantic_winner", score=0.9, refs=["x"]),
            make_candidate("citation_winner", score=0.6, refs=["a", "b", "c"]),
        ]
        query_refs = ["a", "b", "c"]
        # alpha=0 → ignore semantics, rank purely by citation overlap
        result = rerank(candidates, query_refs=query_refs, alpha=0.0, top_k=2)
        assert result[0]["paper_id"] == "citation_winner"

    def test_no_query_refs_gives_zero_citation_overlap(self):
        candidates = [make_candidate("a", 0.8, refs=["x", "y", "z"])]
        result = rerank(candidates, query_refs=[], alpha=0.7, top_k=5)
        assert result[0]["citation_overlap"] == pytest.approx(0.0)

    def test_hybrid_score_field_added_to_output(self):
        candidates = [make_candidate("a", 0.8)]
        result = rerank(candidates, query_refs=[], alpha=0.7, top_k=5)
        assert "hybrid_score" in result[0]

    def test_citation_overlap_field_added_to_output(self):
        candidates = [make_candidate("a", 0.8)]
        result = rerank(candidates, query_refs=[], alpha=0.7, top_k=5)
        assert "citation_overlap" in result[0]

    def test_original_score_field_preserved(self):
        candidates = [make_candidate("a", 0.85)]
        result = rerank(candidates, query_refs=[], alpha=1.0, top_k=5)
        assert result[0]["score"] == pytest.approx(0.85)

    def test_original_candidate_dicts_not_mutated(self):
        """rerank() should not modify the input list in-place."""
        candidates = [make_candidate("a", 0.8)]
        original_keys = set(candidates[0].keys())
        rerank(candidates, query_refs=[], alpha=0.7, top_k=5)
        assert set(candidates[0].keys()) == original_keys

    def test_alpha_boundary_zero_correct_scores(self):
        candidates = [make_candidate("a", 0.9, refs=["x"]), make_candidate("b", 0.6, refs=[])]
        # With alpha=0: hybrid = citation_overlap only.
        # Both have empty query_refs → overlap=0 for all → hybrid=0 for all.
        result = rerank(candidates, query_refs=[], alpha=0.0, top_k=2)
        for r in result:
            assert r["hybrid_score"] == pytest.approx(0.0)
