"""
evaluate.py
-----------
Offline evaluation of the recommender system using a leave-one-out protocol
on real citation links, with a TF-IDF cosine baseline for comparison.

Evaluation protocol:
  For each test paper P in a random sample of N_TEST_PAPERS:
    1. Embed P's title and abstract with SPECTER2.
    2. Query the FAISS index for the top K candidates, excluding P itself.
    3. Compute the ground-truth relevant set: P's actual cited papers that also
       exist in our corpus (papers P cites but are not in the index cannot be
       retrieved, so they are excluded from ground truth — this is the standard
       'in-corpus' restriction used in IR evaluation).
    4. Compare the recommended paper IDs against the ground-truth set.
    5. Record Precision@K, Recall@K, NDCG@K, and MRR.
  Average metrics across all test papers and compare against the TF-IDF baseline.

Why this protocol is valid:
  If paper A cites paper B, it is strong evidence that a researcher reading A
  would benefit from reading B. A good recommender should therefore surface B
  when queried with A. Citation links are noisy labels (some citations are
  superficial), but they are the best proxy for "academic relevance" available
  at scale without user studies.

TF-IDF baseline:
  Fits a bag-of-words TF-IDF representation on all paper abstracts and ranks
  candidates by cosine similarity. This is the simplest non-trivial recommender;
  outperforming it demonstrates the value of SPECTER2's citation-aware training.
  Typical expectation: SPECTER2 NDCG@10 ~30-60% higher than TF-IDF.

Metrics:
  Precision@K   Fraction of top-K recommendations that are actual citations.
  Recall@K      Fraction of the paper's citations that appear in top-K.
  NDCG@K        Normalised Discounted Cumulative Gain — rewards correct
                results ranked higher. Range [0, 1], higher is better.
  MRR           Mean Reciprocal Rank — 1/rank of the first correct result,
                averaged across test papers. Measures how soon the user sees
                a relevant paper.

Output:
  evaluation/results.json — full metric report, saved for README reference.
  Terminal table           — side-by-side comparison printed to stdout.

Usage (from the project root):
  .\\venv\\Scripts\\python.exe evaluation\\evaluate.py
"""

import sys
import json
import random
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Path setup ─────────────────────────────────────────────────────────────────
# evaluate.py lives in evaluation/; the project root (one level up) must be on
# sys.path so that `from retrieval.retrieve import ...` resolves correctly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Dependency guard ───────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    from retrieval.retrieve import load_resources, embed_single_paper, search_index
except ImportError as exc:
    print(
        f"\n[ERROR] Missing package: {exc}\n"
        "Run with the project virtual environment:\n"
        r"  .\venv\Scripts\python.exe evaluation\evaluate.py",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

N_TEST_PAPERS      = 500   # number of papers to evaluate on
K                  = 10    # top-K recommendations to evaluate
MIN_REFS_IN_CORPUS = 3     # minimum in-corpus references a test paper must have
                           # (papers with fewer cannot produce meaningful recall scores)
RANDOM_SEED        = 42    # for reproducible test set sampling

OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "results.json"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Metric functions ───────────────────────────────────────────────────────────

def compute_metrics(ranked_ids: list, ground_truth: list, k: int) -> dict:
    """
    Compute Precision@K, Recall@K, NDCG@K, and MRR for a single test paper.

    All four metrics compare the ranked recommendation list against the ground
    truth set (the paper's cited papers that exist in our corpus).

    Precision@K:
        Fraction of the top-K recommendations that are relevant (i.e., in the
        ground truth set). A precision of 0.1 means 1 in 10 recommended papers
        is a paper that was actually cited — reasonable for a sparse corpus.

    Recall@K:
        Fraction of the ground truth citations that appear in the top-K. If a
        paper has 20 in-corpus citations and 3 appear in top-10, recall = 0.15.

    NDCG@K (Normalised Discounted Cumulative Gain):
        Rewards correct recommendations ranked higher. A correct result at
        position 1 scores more than one at position 10. Normalised by the
        ideal ranking (all correct results at the top), so range is [0, 1].

    MRR (Mean Reciprocal Rank):
        Reciprocal of the rank of the first correct recommendation. If the first
        relevant paper is at rank 3, MRR contribution is 1/3. Zero if none found.

    Args:
        ranked_ids:   Ordered list of recommended paper_id strings (index 0 = rank 1).
        ground_truth: List of paper_id strings that are genuinely relevant.
        k:            Evaluation cutoff — only positions 1..k are considered.

    Returns:
        Dict with keys: 'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'mrr'.
        All values are floats in [0, 1].
    """
    gt_set   = set(ground_truth)
    ranked_k = ranked_ids[:k]

    # ── Precision@K ──────────────────────────────────────────────────────────
    hits      = sum(1 for pid in ranked_k if pid in gt_set)
    precision = hits / k if k > 0 else 0.0

    # ── Recall@K ─────────────────────────────────────────────────────────────
    recall = hits / len(gt_set) if gt_set else 0.0

    # ── NDCG@K ───────────────────────────────────────────────────────────────
    # Discounted Cumulative Gain: sum of 1/log2(rank+1) for each hit.
    dcg = sum(
        1.0 / np.log2(i + 2)           # log2(2) = 1 at rank 1, log2(3) at rank 2, etc.
        for i, pid in enumerate(ranked_k)
        if pid in gt_set
    )
    # Ideal DCG: imagine the top min(|gt|, k) positions are all hits.
    n_ideal = min(len(gt_set), k)
    idcg    = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
    ndcg    = dcg / idcg if idcg > 0 else 0.0

    # ── MRR ──────────────────────────────────────────────────────────────────
    mrr = 0.0
    for i, pid in enumerate(ranked_k):
        if pid in gt_set:
            mrr = 1.0 / (i + 1)
            break

    return {
        "precision_at_k": precision,
        "recall_at_k":    recall,
        "ndcg_at_k":      ndcg,
        "mrr":            mrr,
    }


def aggregate_metrics(per_paper: list) -> dict:
    """
    Average per-paper metric dicts into a single macro-averaged result.

    Macro-averaging treats every test paper equally, regardless of how many
    ground-truth citations it has. This is standard in information retrieval
    evaluation; micro-averaging (pooling all hits before dividing) would bias
    results toward papers with many citations.

    Args:
        per_paper: List of metric dicts, one per evaluated test paper.

    Returns:
        Dict of averaged metrics, each rounded to 4 decimal places.
    """
    if not per_paper:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "mrr": 0.0}

    keys = per_paper[0].keys()
    return {
        key: round(float(np.mean([m[key] for m in per_paper])), 4)
        for key in keys
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_evaluation_data() -> tuple:
    """
    Load the FAISS index, metadata, and SPECTER2 model via retrieve.load_resources().

    Also derives two auxiliary data structures used throughout evaluation:
      - corpus_set: set of all paper_ids currently in the FAISS index.
        Used to filter each test paper's references to in-corpus citations only.
      - id_to_pos:  dict mapping paper_id → integer position in the metadata list.
        Used by the TF-IDF baseline to index into the TF-IDF matrix.

    Returns:
        Tuple of (resources, corpus_set, id_to_pos):
          resources   — dict from retrieve.load_resources()
          corpus_set  — set of all paper_id strings in the index
          id_to_pos   — dict {paper_id: int position in metadata list}
    """
    log.info("Loading retrieval resources (FAISS + metadata + SPECTER2 model) ...")
    resources = load_resources()

    metadata  = resources["metadata"]
    corpus_set = {m["paper_id"] for m in metadata}
    id_to_pos  = {m["paper_id"]: i for i, m in enumerate(metadata)}

    log.info(f"Corpus size: {len(corpus_set):,} papers in index.")
    return resources, corpus_set, id_to_pos


def sample_test_papers(
    metadata:       list,
    corpus_set:     set,
    n:              int = N_TEST_PAPERS,
    min_refs:       int = MIN_REFS_IN_CORPUS,
    seed:           int = RANDOM_SEED,
) -> list:
    """
    Select a reproducible sample of test papers for evaluation.

    A test paper is eligible only if it has at least `min_refs` of its cited
    papers present in the corpus. Papers with fewer in-corpus citations cannot
    produce meaningful Recall or NDCG scores (the ground truth would be too
    sparse), so they are excluded to avoid artificially depressing averages.

    Args:
        metadata:   Full metadata list from load_resources().
        corpus_set: Set of all paper_ids currently in the index.
        n:          Target number of test papers.
        min_refs:   Minimum number of in-corpus citations required.
        seed:       Random seed for reproducible sampling.

    Returns:
        List of metadata dicts for the selected test papers, each augmented
        with 'in_corpus_refs' — the subset of references present in our index.
    """
    eligible = []
    for paper in metadata:
        in_corpus = [ref for ref in paper.get("references", []) if ref in corpus_set]
        if len(in_corpus) >= min_refs:
            eligible.append({**paper, "in_corpus_refs": in_corpus})

    log.info(
        f"{len(eligible):,} papers have >= {min_refs} in-corpus references "
        f"(out of {len(metadata):,} total)."
    )

    if len(eligible) < n:
        log.warning(
            f"Only {len(eligible):,} eligible papers found — "
            f"using all of them instead of the target {n}."
        )
        return eligible

    random.seed(seed)
    sampled = random.sample(eligible, n)
    log.info(f"Sampled {len(sampled)} test papers (seed={seed}).")
    return sampled


# ── SPECTER2 evaluation ────────────────────────────────────────────────────────

def evaluate_specter(
    test_papers: list,
    resources:   dict,
    k:           int = K,
) -> list:
    """
    Evaluate the SPECTER2 + FAISS recommender on the sampled test papers.

    For each test paper:
      1. Embed with SPECTER2 using its title and abstract.
      2. Query the FAISS index (excluding the test paper itself).
      3. Compare the top-K paper IDs against the in-corpus citation ground truth.
      4. Compute and record the four metrics.

    A tqdm progress bar tracks papers evaluated. At 500 papers with CPU
    inference, expect approximately 5-10 minutes total.

    Args:
        test_papers: Sampled test paper dicts (with 'in_corpus_refs' field).
        resources:   Dict from load_resources(), used for embedding and search.
        k:           Evaluation cutoff (number of recommendations to consider).

    Returns:
        List of per-paper metric dicts (one per test paper).
    """
    log.info(f"Evaluating SPECTER2 on {len(test_papers)} test papers (K={k}) ...")
    per_paper_metrics = []

    for paper in tqdm(test_papers, desc="SPECTER2 eval", unit="paper"):
        # Embed the query paper using its title and abstract.
        query_vec = embed_single_paper(paper["title"], paper["abstract"], resources)

        # Search FAISS, excluding the test paper itself from candidates.
        candidates = search_index(
            query_vec,
            resources,
            k=k,
            exclude_ids=[paper["paper_id"]],
        )

        # Extract the ordered list of recommended paper IDs.
        ranked_ids    = [c["paper_id"] for c in candidates]
        ground_truth  = paper["in_corpus_refs"]

        per_paper_metrics.append(compute_metrics(ranked_ids, ground_truth, k))

    return per_paper_metrics


# ── TF-IDF baseline ────────────────────────────────────────────────────────────

def build_tfidf_data(metadata: list) -> dict:
    """
    Fit a TF-IDF vectorizer on all paper abstracts and build the full matrix.

    Uses unigrams only with English stopword removal. The resulting sparse
    matrix has shape [n_papers, vocab_size]. Cosine similarity between rows
    serves as the baseline retrieval score.

    This baseline is intentionally simple — it represents what a keyword-
    matching system achieves without any semantic understanding or citation
    graph information. Comparing against it quantifies SPECTER2's advantage.

    Args:
        metadata: Full metadata list from load_resources().

    Returns:
        Dict containing:
          'matrix'     — scipy sparse TF-IDF matrix, shape [n_papers, vocab]
          'paper_ids'  — list of paper_id strings, row-aligned with matrix
          'id_to_pos'  — {paper_id: int row index in matrix}
    """
    log.info(f"Fitting TF-IDF vectorizer on {len(metadata):,} abstracts ...")

    abstracts  = [m["abstract"] for m in metadata]
    paper_ids  = [m["paper_id"] for m in metadata]

    vectorizer = TfidfVectorizer(
        max_features=50_000,    # cap vocabulary to control memory
        ngram_range=(1, 1),     # unigrams only for speed
        stop_words="english",
        min_df=2,               # ignore terms appearing in only one document
        sublinear_tf=True,      # log(1 + tf) instead of raw tf, reduces outlier effect
    )
    matrix    = vectorizer.fit_transform(abstracts)
    id_to_pos = {pid: i for i, pid in enumerate(paper_ids)}

    log.info(f"TF-IDF matrix: {matrix.shape}, {matrix.nnz:,} non-zero entries.")
    return {"matrix": matrix, "paper_ids": paper_ids, "id_to_pos": id_to_pos}


def tfidf_recommend(
    paper_id:    str,
    tfidf_data:  dict,
    k:           int,
    exclude_ids: list = None,
) -> list:
    """
    Retrieve top-K papers by TF-IDF cosine similarity for a given paper.

    Computes the cosine similarity between the query paper's TF-IDF vector and
    every other paper's vector, then returns the top-K paper IDs (excluding
    the query paper and any specified exclude_ids).

    Args:
        paper_id:    The paper_id of the test (query) paper.
        tfidf_data:  Dict from build_tfidf_data().
        k:           Number of results to return.
        exclude_ids: Additional paper IDs to exclude from results.

    Returns:
        Ordered list of up to k paper_id strings, highest similarity first.
        Returns an empty list if the paper is not in the TF-IDF matrix.
    """
    pos = tfidf_data["id_to_pos"].get(paper_id)
    if pos is None:
        return []

    exclude_set = set(exclude_ids or [])
    exclude_set.add(paper_id)   # always exclude the query itself

    # Cosine similarity between this paper's TF-IDF row and all others.
    # sklearn returns shape [1, n_papers]; flatten to 1-D.
    query_vec = tfidf_data["matrix"][pos]
    sims      = sklearn_cosine(query_vec, tfidf_data["matrix"]).flatten()

    # Sort positions by descending similarity and filter excludes.
    sorted_pos = np.argsort(sims)[::-1]
    result_ids = []
    for p in sorted_pos:
        pid = tfidf_data["paper_ids"][p]
        if pid not in exclude_set:
            result_ids.append(pid)
        if len(result_ids) >= k:
            break

    return result_ids


def evaluate_tfidf(
    test_papers: list,
    tfidf_data:  dict,
    k:           int = K,
) -> list:
    """
    Evaluate the TF-IDF baseline on the same test papers as SPECTER2.

    Mirrors evaluate_specter() in structure so results are directly comparable.
    Each paper is evaluated by TF-IDF cosine similarity recommendation, and the
    same four metrics are computed against the identical in-corpus ground truth.

    Args:
        test_papers: Same list used for SPECTER2 evaluation.
        tfidf_data:  Dict from build_tfidf_data().
        k:           Evaluation cutoff.

    Returns:
        List of per-paper metric dicts, one per test paper.
    """
    log.info(f"Evaluating TF-IDF baseline on {len(test_papers)} test papers (K={k}) ...")
    per_paper_metrics = []

    for paper in tqdm(test_papers, desc="TF-IDF  eval", unit="paper"):
        ranked_ids   = tfidf_recommend(paper["paper_id"], tfidf_data, k)
        ground_truth = paper["in_corpus_refs"]
        per_paper_metrics.append(compute_metrics(ranked_ids, ground_truth, k))

    return per_paper_metrics


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_comparison_table(
    specter_metrics: dict,
    tfidf_metrics:   dict,
    k:               int,
    n_test:          int,
) -> None:
    """
    Print a formatted side-by-side comparison table to the terminal.

    Shows SPECTER2 and TF-IDF metrics alongside the percentage improvement.
    Designed to be copy-pasteable into a README or portfolio description.

    Args:
        specter_metrics: Averaged metrics dict for SPECTER2.
        tfidf_metrics:   Averaged metrics dict for TF-IDF baseline.
        k:               The K cutoff used.
        n_test:          Number of test papers evaluated.
    """
    def pct_delta(s, t):
        if t == 0:
            return "n/a"
        delta = (s - t) / t * 100
        sign  = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f}%"

    w = 16   # column width

    header  = f"{'Metric':<{w}}  {'SPECTER2':>{w}}  {'TF-IDF':>{w}}  {'vs baseline':>{w}}"
    divider = "-" * len(header)

    print()
    print(f"  Evaluation Results  (n={n_test} test papers, K={k})")
    print(divider)
    print(header)
    print(divider)

    metric_labels = {
        "precision_at_k": f"Precision@{k}",
        "recall_at_k":    f"Recall@{k}",
        "ndcg_at_k":      f"NDCG@{k}",
        "mrr":             "MRR",
    }

    for key, label in metric_labels.items():
        s = specter_metrics[key]
        t = tfidf_metrics[key]
        print(
            f"  {label:<{w}}  {s:>{w}.4f}  {t:>{w}.4f}  {pct_delta(s, t):>{w}}"
        )

    print(divider)
    print()


def save_results(
    specter_metrics: dict,
    tfidf_metrics:   dict,
    config:          dict,
    output_path:     Path,
) -> None:
    """
    Save the full evaluation report to a JSON file.

    The saved file contains the configuration, both sets of metrics, and the
    percentage improvement of SPECTER2 over TF-IDF. This is the number to
    quote in the project README and in interviews.

    Args:
        specter_metrics: Averaged SPECTER2 metrics.
        tfidf_metrics:   Averaged TF-IDF metrics.
        config:          Evaluation configuration parameters.
        output_path:     Destination JSON file path.
    """
    def safe_pct(s, t):
        if t == 0:
            return None
        return round((s - t) / t * 100, 1)

    report = {
        "config": config,
        "specter2": specter_metrics,
        "tfidf_baseline": tfidf_metrics,
        "improvement_over_tfidf_pct": {
            key: safe_pct(specter_metrics[key], tfidf_metrics[key])
            for key in specter_metrics
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info(f"Results saved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run the full offline evaluation pipeline.

    Steps:
      1. Load the FAISS index, metadata, and SPECTER2 model.
      2. Sample N_TEST_PAPERS eligible test papers.
      3. Evaluate the SPECTER2 recommender on all test papers.
      4. Fit TF-IDF and evaluate the baseline on the same test papers.
      5. Aggregate and compare metrics.
      6. Print the comparison table and save results to JSON.
    """
    log.info("=" * 60)
    log.info("  Offline evaluation starting")
    log.info(f"  Test papers : {N_TEST_PAPERS}  |  K={K}  |  Min refs={MIN_REFS_IN_CORPUS}")
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    resources, corpus_set, _ = load_evaluation_data()
    metadata = resources["metadata"]

    # ── Sample test papers ────────────────────────────────────────────────────
    test_papers = sample_test_papers(metadata, corpus_set)

    # ── SPECTER2 evaluation ───────────────────────────────────────────────────
    specter_per_paper = evaluate_specter(test_papers, resources, k=K)
    specter_metrics   = aggregate_metrics(specter_per_paper)
    log.info(f"SPECTER2 NDCG@{K}: {specter_metrics['ndcg_at_k']:.4f}")

    # ── TF-IDF baseline ───────────────────────────────────────────────────────
    tfidf_data       = build_tfidf_data(metadata)
    tfidf_per_paper  = evaluate_tfidf(test_papers, tfidf_data, k=K)
    tfidf_metrics    = aggregate_metrics(tfidf_per_paper)
    log.info(f"TF-IDF   NDCG@{K}: {tfidf_metrics['ndcg_at_k']:.4f}")

    # ── Report ────────────────────────────────────────────────────────────────
    config = {
        "n_test_papers":       len(test_papers),
        "k":                   K,
        "min_refs_in_corpus":  MIN_REFS_IN_CORPUS,
        "random_seed":         RANDOM_SEED,
    }

    print_comparison_table(specter_metrics, tfidf_metrics, K, len(test_papers))
    save_results(specter_metrics, tfidf_metrics, config, OUTPUT_FILE)

    log.info("=" * 60)
    log.info("  Evaluation complete")
    log.info(f"  Results saved : {OUTPUT_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
