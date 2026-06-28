"""
explain.py
----------
Generates a natural-language explanation for each recommendation by finding
the sentence pair — one from the query abstract, one from the candidate
abstract — with the highest semantic similarity.

Why sentence-level attribution?
  SPECTER2 produces paper-level embeddings by summarising the entire abstract
  into a single [CLS] vector. This is great for ranking but opaque for
  explanation: "these two papers are similar" is not useful to a user who wants
  to know *why*. Sentence-level matching surfaces the specific claim or concept
  that drove the recommendation, giving the user actionable context.

Model choice — all-MiniLM-L6-v2:
  A lightweight SentenceTransformer (384-dim, ~22 MB) optimised for semantic
  similarity at the sentence level. Using SPECTER2 for sentence embedding would
  be wrong: SPECTER2 was trained to embed *full papers*, not individual sentences,
  so its [CLS] representation for a single sentence carries little signal.
  MiniLM was trained specifically on sentence-pair similarity tasks (MNLI, SNLI,
  STS-B) and produces strong sentence representations in milliseconds.

Algorithm:
  1. Split query and candidate abstracts into sentences (NLTK punkt tokeniser,
     with a regex fallback if NLTK data is unavailable).
  2. Embed all sentences from both abstracts with all-MiniLM-L6-v2 in one batch.
  3. Compute the full cosine similarity matrix between query sentences and
     candidate sentences.
  4. Find the (i, j) pair with the highest similarity score.
  5. Format the explanation string from the two matched sentences.

Pipeline position:
    rerank.py  →  [explain.py]  →  app.py
"""

import re
import sys
import logging
import numpy as np

# ── Dependency guard ───────────────────────────────────────────────────────────
try:
    import nltk
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    print(
        f"\n[ERROR] Missing package: {exc}\n"
        "Run with the project virtual environment:\n"
        r"  .\venv\Scripts\python.exe retrieval\explain.py",
        file=sys.stderr,
    )
    sys.exit(1)

log = logging.getLogger(__name__)

# Module-level flag set by load_sentence_model(). Indicates whether NLTK's
# punkt tokeniser was successfully downloaded and is available for use.
# The regex fallback is used when punkt is unavailable.
_NLTK_AVAILABLE = False

# Minimum number of words a sentence must contain to be considered for matching.
# Very short sentences ("We conclude.", "Results follow.") carry little signal
# and tend to match spuriously across unrelated papers.
_MIN_SENTENCE_WORDS = 6

# Fallback explanation returned when both abstracts are too short to find a
# meaningful sentence pair (e.g. one-sentence abstracts, or after filtering
# all sentences as too short).
_FALLBACK_EXPLANATION = "Recommended based on overall research topic similarity."


# ── Functions ──────────────────────────────────────────────────────────────────

def load_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load the sentence embedding model and prepare NLTK sentence tokenisation.

    Downloads the NLTK punkt tokeniser data on first call if not already cached.
    The punkt data is small (~13 MB) and is stored in the NLTK data directory
    (~/.nltk_data on Unix, ~/AppData/Roaming/nltk_data on Windows).

    Tries the newer 'punkt_tab' resource first (NLTK >= 3.8), then falls back
    to the older 'punkt' resource, then falls back to regex splitting if
    neither is available. The fallback produces acceptable sentence splits for
    most academic abstracts.

    Args:
        model_name: SentenceTransformer model identifier. Defaults to
                    'all-MiniLM-L6-v2' — a fast, accurate sentence encoder.

    Returns:
        A loaded SentenceTransformer model ready for encode() calls.
    """
    global _NLTK_AVAILABLE

    # ── NLTK punkt download ───────────────────────────────────────────────────
    # Try punkt_tab (NLTK >= 3.8) first, then the older punkt package.
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.download(resource, quiet=True)
            # Verify the download actually worked by attempting a tokenisation.
            from nltk.tokenize import sent_tokenize
            sent_tokenize("Test sentence. Another one.")
            _NLTK_AVAILABLE = True
            log.info(f"NLTK '{resource}' tokeniser loaded successfully.")
            break
        except Exception:
            continue

    if not _NLTK_AVAILABLE:
        log.warning(
            "NLTK punkt tokeniser unavailable. "
            "Using regex sentence splitting as fallback."
        )

    # ── Sentence transformer model ────────────────────────────────────────────
    log.info(f"Loading sentence model '{model_name}' ...")
    model = SentenceTransformer(model_name)
    log.info(f"Sentence model ready (dim={model.get_sentence_embedding_dimension()}).")

    return model


def split_sentences(text: str) -> list:
    """
    Split an abstract into individual sentences.

    Uses NLTK's punkt tokeniser when available (more accurate, handles
    abbreviations like "et al." and "Fig." without splitting). Falls back to
    a regex split on sentence-ending punctuation when NLTK is unavailable.

    After splitting, sentences are filtered by minimum word count
    (_MIN_SENTENCE_WORDS) to discard boilerplate sentences ("We conclude.",
    "Results are shown in Table 1.") that match spuriously across papers.

    Args:
        text: The abstract text to split. May contain newlines and citations.

    Returns:
        List of sentence strings, each containing at least _MIN_SENTENCE_WORDS
        words. Returns an empty list if the text is blank.
    """
    text = (text or "").strip()
    if not text:
        return []

    if _NLTK_AVAILABLE:
        from nltk.tokenize import sent_tokenize
        raw_sentences = sent_tokenize(text)
    else:
        # Regex fallback: split after '.', '!', or '?' followed by whitespace.
        # This handles most cases but will incorrectly split "et al. (2020)".
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    # Filter out sentences that are too short to be meaningful for matching.
    filtered = [
        s.strip()
        for s in raw_sentences
        if len(s.strip().split()) >= _MIN_SENTENCE_WORDS
    ]

    return filtered


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between every pair of rows in two matrices.

    Used to find the most similar (query sentence, candidate sentence) pair
    without a nested Python loop — the full matrix computation is faster for
    the small number of sentences in an abstract (<50 per abstract).

    Args:
        a: Matrix of shape [n, dim], each row L2-normalised.
        b: Matrix of shape [m, dim], each row L2-normalised.

    Returns:
        np.ndarray of shape [n, m] containing cosine similarities.
        Entry [i, j] is the cosine similarity between a[i] and b[j].
    """
    # Both inputs are assumed to be L2-normalised by SentenceTransformer.encode()
    # (normalize_embeddings=True). Inner product then equals cosine similarity.
    return a @ b.T


def find_best_pair(
    query_abstract: str,
    candidate_abstract: str,
    model: SentenceTransformer,
) -> tuple:
    """
    Find the sentence pair with the highest semantic similarity across two abstracts.

    Embeds all sentences from both abstracts in a single batch call to the
    sentence model, then computes the full pairwise cosine similarity matrix
    to find the most similar (query_sentence, candidate_sentence) pair.

    Args:
        query_abstract:     The full abstract of the query paper.
        candidate_abstract: The full abstract of the candidate paper.
        model:              The loaded SentenceTransformer model.

    Returns:
        Tuple of (query_sentence, candidate_sentence, similarity_score).
        Returns (None, None, 0.0) if either abstract produces no valid sentences
        after filtering (e.g. the abstract is too short or blank).
    """
    query_sents     = split_sentences(query_abstract)
    candidate_sents = split_sentences(candidate_abstract)

    # Guard: need at least one sentence from each abstract.
    if not query_sents or not candidate_sents:
        return None, None, 0.0

    # Embed all sentences from both abstracts in a single batch.
    # normalize_embeddings=True produces unit-norm vectors so that dot product
    # equals cosine similarity — consistent with the similarity matrix below.
    all_texts = query_sents + candidate_sents
    all_vecs  = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)

    query_vecs     = all_vecs[:len(query_sents)]
    candidate_vecs = all_vecs[len(query_sents):]

    # Full pairwise similarity matrix: shape [n_query_sents, n_candidate_sents].
    sim_matrix = _cosine_similarity_matrix(query_vecs, candidate_vecs)

    # Find the (i, j) position with the highest similarity.
    best_i, best_j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    best_score     = float(sim_matrix[best_i, best_j])

    return query_sents[best_i], candidate_sents[best_j], best_score


def generate_explanation(
    query_abstract: str,
    candidate_abstract: str,
    model: SentenceTransformer,
) -> str:
    """
    Generate a human-readable explanation for why a paper was recommended.

    Calls find_best_pair() to identify the most semantically similar sentence
    pair across the two abstracts and formats the result as a short explanation
    string. If no valid pair can be found (both abstracts too short), a generic
    fallback explanation is returned.

    The explanation is intentionally phrased from the candidate's perspective:
    "This paper discusses X, which relates to your interest in Y." This framing
    anchors the explanation in what the user will find in the candidate — more
    actionable than abstract similarity scores.

    Args:
        query_abstract:     The query paper's abstract.
        candidate_abstract: The candidate paper's abstract.
        model:              The loaded SentenceTransformer model.

    Returns:
        A short string explaining the recommendation. Never raises — falls back
        to a generic message on any error so the app never crashes during display.
    """
    try:
        query_sent, candidate_sent, score = find_best_pair(
            query_abstract, candidate_abstract, model
        )

        if candidate_sent is None:
            return _FALLBACK_EXPLANATION

        # Truncate very long sentences in the display so the card stays readable.
        max_len = 120
        q_display = (query_sent[:max_len] + "…") if len(query_sent) > max_len else query_sent
        c_display = (candidate_sent[:max_len] + "…") if len(candidate_sent) > max_len else candidate_sent

        return (
            f'This paper discusses: "{c_display}" — '
            f'which relates to your interest in: "{q_display}"'
        )

    except Exception as exc:
        # Defensive catch: explanation failure should never crash the app.
        log.warning(f"Explanation generation failed: {exc}")
        return _FALLBACK_EXPLANATION


def explain_results(
    results: list,
    query_abstract: str,
    model: SentenceTransformer,
) -> list:
    """
    Add an 'explanation' field to every result in the ranked list.

    Iterates over the top-k results from rerank.py and generates an explanation
    for each by comparing it against the query abstract. The explanation is
    stored in-place in the result dict under the key 'explanation'.

    This is intentionally done one result at a time (not batched) because:
      - The top-k list is typically 10 items — per-call overhead is negligible.
      - Each call to find_best_pair() already batches the sentence embeddings
        for that single (query, candidate) pair efficiently.

    Args:
        results:        List of candidate dicts from rerank.rerank().
        query_abstract: The query paper's abstract text.
        model:          The loaded SentenceTransformer model.

    Returns:
        The same list with 'explanation' added to each dict (mutated in-place
        for efficiency, also returned for convenience in chained calls).
    """
    if not results:
        return results

    log.info(f"Generating explanations for {len(results)} results ...")

    for result in results:
        candidate_abstract = result.get("abstract", "")
        result["explanation"] = generate_explanation(
            query_abstract, candidate_abstract, model
        )

    log.info("Explanations generated.")
    return results
