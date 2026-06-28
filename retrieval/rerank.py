"""
rerank.py
---------
Re-ranks the top-K FAISS candidates using a hybrid score that combines
semantic similarity (from SPECTER2 embeddings) with citation graph overlap.

The core insight is that pure semantic similarity can surface papers that share
vocabulary but belong to different research sub-communities. Two papers can
use the word "attention" in very different senses. Citation overlap is a
complementary structural signal: if paper A and paper B both cite many of the
same prior works, they are in the same academic conversation — regardless of
surface-level wording.

Hybrid scoring formula:
    hybrid_score = alpha * cosine_similarity + (1 - alpha) * citation_overlap

    cosine_similarity:
        The inner product between the query and candidate SPECTER2 vectors
        (already computed by FAISS and stored in candidate['score']).
        Range: approximately [0, 1] for thematically related papers.

    citation_overlap:
        Fraction of the query paper's references that also appear in the
        candidate's reference list.
        Formula: |refs(query) ∩ refs(candidate)| / max(|refs(query)|, 1)
        Range: [0, 1].
        This is a recall-based metric from the query's perspective: "what
        fraction of papers the query cites does the candidate also cite?"
        It is NOT symmetric — that is intentional. We care about whether the
        candidate draws from the same intellectual tradition as the query,
        not whether their reference lists are identical.

    alpha (default 0.7):
        Controls the trade-off. At 0.7, semantic similarity dominates but
        citation overlap provides a meaningful boost to structurally adjacent
        papers. Lower alpha = more weight on citation structure; higher = purer
        semantic similarity.

Graceful degradation:
    When the query paper has no known references (e.g. the user pasted an
    unpublished draft or a paper not in our corpus), citation_overlap = 0 for
    all candidates and the hybrid score reduces to alpha * cosine_similarity.
    The ranking is still semantically meaningful; the citation boost simply
    doesn't apply.

Pipeline position:
    retrieve.py  →  [rerank.py]  →  explain.py  →  app.py
"""

import logging

log = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────────────

def compute_citation_overlap(
    query_refs: list,
    candidate_refs: list,
) -> float:
    """
    Compute the citation overlap between a query paper and a candidate paper.

    Measures what fraction of the query paper's references are also cited by
    the candidate. A high overlap means both papers draw from the same pool of
    prior work — a strong indicator of academic adjacency independent of
    surface-level language similarity.

    The formula is recall-oriented from the query's perspective:
        overlap = |refs(query) ∩ refs(candidate)| / max(|refs(query)|, 1)

    Example:
        query cites     : [A, B, C, D, E]  (5 references)
        candidate cites : [B, C, F, G]
        intersection    : [B, C]            (2 shared)
        overlap         : 2 / 5 = 0.40

    Args:
        query_refs:     List of paper_id strings cited by the query paper.
        candidate_refs: List of paper_id strings cited by the candidate paper.

    Returns:
        Float in [0, 1]. Returns 0.0 if query_refs is empty (no information
        available to compute overlap — does not penalise the candidate).
    """
    if not query_refs:
        # No references available for the query — return 0 so the hybrid score
        # degrades cleanly to pure semantic similarity (alpha * cosine_sim).
        return 0.0

    query_set     = set(query_refs)
    candidate_set = set(candidate_refs)
    intersection  = query_set & candidate_set

    return len(intersection) / len(query_set)


def _compute_hybrid_score(
    cosine_sim: float,
    citation_overlap: float,
    alpha: float,
) -> float:
    """
    Combine cosine similarity and citation overlap into a single hybrid score.

    Both inputs are expected to be in [0, 1]. The output is also in [0, 1].
    Higher is more recommended.

    Args:
        cosine_sim:       Cosine similarity from FAISS (candidate['score']).
        citation_overlap: Output of compute_citation_overlap().
        alpha:            Weight on the semantic component. (1 - alpha) goes
                          to the citation component.

    Returns:
        Float in [0, 1], the combined hybrid relevance score.
    """
    return alpha * cosine_sim + (1.0 - alpha) * citation_overlap


def rerank(
    candidates: list,
    query_refs: list,
    alpha: float = 0.7,
    top_k: int = 10,
) -> list:
    """
    Apply hybrid re-ranking to a list of FAISS candidates and return the top-k.

    Takes the raw candidate list from retrieve.search_index() (sorted by cosine
    similarity), scores each candidate with the hybrid formula, re-sorts by
    hybrid score, and returns the top_k results.

    Each candidate dict is augmented with two new fields:
      'citation_overlap' — the raw citation overlap score for this candidate
      'hybrid_score'     — the final combined score used for ranking

    The original 'score' field (cosine similarity from FAISS) is preserved so
    the Streamlit app can display both the semantic and hybrid scores if desired.

    Args:
        candidates: List of candidate dicts from retrieve.search_index().
                    Each must contain at least 'score' and 'references'.
        query_refs: List of paper_id strings cited by the query paper.
                    Pass an empty list if unknown (citation overlap will be 0).
        alpha:      Weight on cosine similarity. Range [0, 1]. Default 0.7.
                    At 0.7: semantic similarity is primary, citation overlap
                    provides a meaningful secondary boost (up to +0.30 on score).
        top_k:      Number of results to return after re-ranking.

    Returns:
        List of up to top_k candidate dicts, sorted by hybrid_score descending.
        Each dict has all original fields plus 'citation_overlap' and 'hybrid_score'.
    """
    if not candidates:
        log.warning("rerank() called with an empty candidate list.")
        return []

    # ── Score every candidate ─────────────────────────────────────────────────
    scored = []
    for candidate in candidates:
        cosine_sim = candidate["score"]

        citation_overlap = compute_citation_overlap(
            query_refs,
            candidate.get("references", []),
        )

        hybrid = _compute_hybrid_score(cosine_sim, citation_overlap, alpha)

        # Augment the candidate dict with scoring details.
        # We copy to avoid mutating the original list from retrieve.py.
        scored.append({
            **candidate,
            "citation_overlap": round(citation_overlap, 4),
            "hybrid_score":     round(hybrid, 4),
        })

    # ── Sort by hybrid score (highest first) and trim to top_k ───────────────
    scored.sort(key=lambda c: c["hybrid_score"], reverse=True)
    result = scored[:top_k]

    # ── Log a brief summary of score distribution ─────────────────────────────
    if result:
        top_hybrid  = result[0]["hybrid_score"]
        top_cosine  = result[0]["score"]
        top_overlap = result[0]["citation_overlap"]
        log.info(
            f"Re-ranked {len(candidates)} candidates → top {len(result)} results. "
            f"Best: hybrid={top_hybrid:.3f}  "
            f"(cosine={top_cosine:.3f}, cite_overlap={top_overlap:.3f}, alpha={alpha})"
        )

    return result
