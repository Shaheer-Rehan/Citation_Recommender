"""
retrieve.py
-----------
Loads the FAISS index, metadata store, and SPECTER2 model, then provides
functions for embedding query papers and searching for nearest neighbours.

Two query modes are supported:

  Single paper mode:
    The user pastes a title and abstract (or supplies an ArXiv ID).
    The text is embedded into a 768-dim SPECTER2 vector and the FAISS index
    is queried directly for the k most similar papers.

  Reading catalogue mode:
    The user supplies a list of papers they have already read.
    Each paper is embedded individually, then all embeddings are mean-pooled
    into a single "taste profile" vector. The pooled vector represents the
    centroid of the user's research interests in embedding space, which is
    then used as the FAISS query. Papers in the catalogue are excluded from
    the results.

Design note — load_resources():
    This function is the intended entry point for the Streamlit app. It loads
    the FAISS index, metadata JSON, and SPECTER2 model into a single dict that
    the app caches with @st.cache_resource. All other functions in this module
    are stateless given the resources dict, so they can be called repeatedly
    on different queries without any re-loading overhead.

Pipeline position:
  build_index.py  →  [retrieve.py]  →  rerank.py  →  explain.py  →  app.py

Usage (standalone test from the project root):
  .\\venv\\Scripts\\python.exe retrieval\\retrieve.py
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path

# ── Dependency guard ───────────────────────────────────────────────────────────
try:
    import torch
    import faiss
    from transformers import AutoTokenizer, AutoModel
except ImportError as exc:
    print(
        f"\n[ERROR] Missing package: {exc}\n"
        "Run with the project virtual environment:\n"
        r"  .\venv\Scripts\python.exe retrieval\retrieve.py",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

# Must match the model used in embed_papers.py. If you change the model
# there, update this constant too — mismatched models produce incompatible
# embedding spaces and will silently return nonsensical results.
MODEL_NAME = "allenai/specter2_base"

MAX_LENGTH  = 512   # tokenizer truncation limit — matches embed_papers.py
BATCH_SIZE  = 16    # papers per forward pass when embedding a catalogue
                    # (smaller than embed_papers.py since we're online)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
INDEX_PATH    = PROJECT_ROOT / "index" / "papers.index"
METADATA_PATH = PROJECT_ROOT / "index" / "metadata.json"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    """
    Select the best available compute device.

    Mirrors the logic in embed_papers.py. Called once inside load_resources()
    so the device is determined at startup and reused across all queries.

    Returns:
        torch.device: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_resources() -> dict:
    """
    Load all assets needed for retrieval into a single dict.

    Assets loaded:
      - FAISS index      (index/papers.index)
      - Metadata list    (index/metadata.json)
      - SPECTER2 tokenizer and model  (HuggingFace cache)
      - Compute device

    This function is deliberately slow (the model is ~440 MB and the metadata
    JSON is ~35 MB) because it is called only once at Streamlit startup and
    then cached with @st.cache_resource. Every subsequent query is fast because
    it operates on the already-loaded objects in memory.

    Returns:
        dict with keys:
          'index'     — faiss.IndexFlatIP, loaded and ready for search
          'metadata'  — list of dicts, position-aligned with the FAISS index
          'tokenizer' — AutoTokenizer for SPECTER2_base
          'model'     — AutoModel for SPECTER2_base, in eval mode on device
          'device'    — torch.device
          'sep_token' — tokenizer.sep_token string (cached to avoid repeated attr lookup)

    Raises:
        SystemExit if either index file is missing (build_index.py must run first).
    """
    # ── Device ────────────────────────────────────────────────────────────────
    device = _get_device()
    log.info(f"Using device: {device}")

    # ── FAISS index ───────────────────────────────────────────────────────────
    if not INDEX_PATH.exists():
        log.error(
            f"FAISS index not found: {INDEX_PATH}\n"
            "Run index/build_index.py first."
        )
        sys.exit(1)

    log.info(f"Loading FAISS index from {INDEX_PATH} ...")
    index = faiss.read_index(str(INDEX_PATH))
    log.info(f"FAISS index loaded: {index.ntotal:,} vectors, dim={index.d}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    if not METADATA_PATH.exists():
        log.error(
            f"Metadata file not found: {METADATA_PATH}\n"
            "Run index/build_index.py first."
        )
        sys.exit(1)

    log.info(f"Loading metadata from {METADATA_PATH} ...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    log.info(f"Metadata loaded: {len(metadata):,} records")

    # Sanity check: metadata length must equal the number of FAISS vectors.
    if len(metadata) != index.ntotal:
        log.error(
            f"Metadata length ({len(metadata):,}) does not match FAISS index "
            f"size ({index.ntotal:,}). Re-run build_index.py."
        )
        sys.exit(1)

    # ── SPECTER2 model ────────────────────────────────────────────────────────
    log.info(f"Loading SPECTER2 tokenizer and model ({MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # disable dropout for deterministic inference
    log.info("Model ready.")

    return {
        "index":     index,
        "metadata":  metadata,
        "tokenizer": tokenizer,
        "model":     model,
        "device":    device,
        "sep_token": tokenizer.sep_token,
    }


def _format_input(title: str, abstract: str, sep_token: str) -> str:
    """
    Format a title and abstract into SPECTER2's expected input string.

    SPECTER2 was trained with inputs of the form:
        "Title [SEP] Abstract"
    where [SEP] is the tokenizer's separator token. The tokenizer then adds
    [CLS] at the start and a final [SEP], producing:
        [CLS] Title [SEP] Abstract [SEP]

    The [CLS] token at position 0 is what we extract as the paper embedding.

    Args:
        title:     Paper title (may be empty — handled gracefully).
        abstract:  Paper abstract text.
        sep_token: The tokenizer's separator token (typically '[SEP]').

    Returns:
        Formatted string ready for tokenisation.
    """
    title    = (title    or "").strip()
    abstract = (abstract or "").strip()
    if not title:
        return abstract
    return title + sep_token + abstract


def _embed_texts(
    texts: list,
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Embed a list of formatted input strings using SPECTER2.

    Processes texts in batches of BATCH_SIZE to avoid out-of-memory errors
    when embedding a large reading catalogue. Returns the raw (un-normalised)
    CLS token embeddings for all inputs concatenated into a single matrix.

    This is a private helper — callers should use embed_single_paper() or
    embed_catalogue() which handle formatting and normalisation.

    Args:
        texts:     List of pre-formatted "Title [SEP] Abstract" strings.
        tokenizer: The SPECTER2 tokenizer.
        model:     The SPECTER2 model in eval mode.
        device:    The compute device.

    Returns:
        np.ndarray of shape [len(texts), 768], dtype float32, NOT normalised.
    """
    all_embeddings = []

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        # CLS token: position 0 of the last hidden state.
        cls_vecs = output.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_vecs.detach().cpu().to(torch.float32).numpy())

    return np.concatenate(all_embeddings, axis=0)


def _normalise(vec: np.ndarray) -> np.ndarray:
    """
    L2-normalise a 1-D or 2-D float32 array in-place and return it.

    After normalisation, inner product with other unit-norm vectors equals
    cosine similarity — consistent with the IndexFlatIP index.

    Args:
        vec: 1-D [dim] or 2-D [n, dim] float32 array.

    Returns:
        The same array rescaled to unit norm along axis=-1.
    """
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-10)

    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norms, 1e-10)


def embed_single_paper(
    title: str,
    abstract: str,
    resources: dict,
) -> np.ndarray:
    """
    Embed a single paper and return its L2-normalised SPECTER2 vector.

    This is the entry point for single-paper query mode. The returned vector
    is shaped [768] and can be passed directly to search_index().

    Args:
        title:     The query paper's title.
        abstract:  The query paper's abstract.
        resources: The dict returned by load_resources().

    Returns:
        np.ndarray of shape [768], dtype float32, L2-normalised.
    """
    text = _format_input(title, abstract, resources["sep_token"])
    raw  = _embed_texts([text], resources["tokenizer"], resources["model"], resources["device"])
    return _normalise(raw[0])  # shape [768]


def embed_catalogue(
    papers: list,
    resources: dict,
) -> np.ndarray:
    """
    Embed a list of papers and mean-pool them into a single query vector.

    This is the entry point for reading catalogue mode. Each paper in the
    list is embedded individually using SPECTER2, and all embeddings are
    averaged into one "taste profile" vector representing the centroid of the
    user's read papers in embedding space. The centroid is then re-normalised
    so it is compatible with the IndexFlatIP query interface.

    Why mean-pooling works:
      In a well-trained embedding space, the average of N paper vectors points
      towards the region occupied by papers on similar topics. FAISS then finds
      papers nearest to that region — i.e., papers the user has not yet read but
      whose content sits in the same neighbourhood as what they have read.

    Args:
        papers:    List of dicts, each with at least 'title' and 'abstract' keys.
                   May contain additional keys (e.g. 'paper_id') which are ignored.
        resources: The dict returned by load_resources().

    Returns:
        np.ndarray of shape [768], dtype float32, L2-normalised.
        Returns a zero vector if the input list is empty (caller should check).
    """
    if not papers:
        log.warning("embed_catalogue() called with an empty list.")
        return np.zeros(resources["index"].d, dtype=np.float32)

    texts = [
        _format_input(p.get("title", ""), p.get("abstract", ""), resources["sep_token"])
        for p in papers
    ]

    # Embed all catalogue papers — may be several batches if the list is long.
    embeddings = _embed_texts(
        texts,
        resources["tokenizer"],
        resources["model"],
        resources["device"],
    )

    # Mean-pool across papers to get the user taste profile vector.
    mean_vec = embeddings.mean(axis=0)

    # Re-normalise: the mean of unit vectors is not itself unit-norm, so we
    # must normalise again before querying IndexFlatIP.
    return _normalise(mean_vec).astype(np.float32)


def search_index(
    query_vec: np.ndarray,
    resources: dict,
    k: int = 50,
    exclude_ids: list = None,
) -> list:
    """
    Query the FAISS index and return the top-k most similar paper records.

    Searches for more candidates than k to compensate for papers that are
    filtered out by the exclude_ids set (used in catalogue mode to skip papers
    the user has already read). Each returned record is the full metadata dict
    from metadata.json with an additional 'score' field containing the cosine
    similarity (inner product between unit-norm vectors).

    FAISS returns index position -1 as a padding value when the index has fewer
    vectors than the requested k. These are filtered out silently.

    Args:
        query_vec:   L2-normalised query vector, shape [768], dtype float32.
        resources:   The dict returned by load_resources().
        k:           Number of results to return after filtering.
        exclude_ids: Optional list of paper_id strings to exclude from results
                     (used in catalogue mode to suppress already-read papers).

    Returns:
        List of up to k dicts, each containing the full paper metadata plus
        a 'score' key (float, cosine similarity, range approximately [0, 1]).
        Sorted by score descending (highest similarity first).
    """
    exclude_set = set(exclude_ids or [])

    # Search for more candidates than k to have a buffer after exclusion.
    # Cap at index.ntotal to avoid a FAISS assertion error.
    search_k = min(k + len(exclude_set) + 20, resources["index"].ntotal)

    # FAISS expects a 2-D array: shape [n_queries, dim]. We have one query.
    query_2d = query_vec.reshape(1, -1).astype(np.float32)

    distances, indices = resources["index"].search(query_2d, search_k)

    results = []
    for dist, pos in zip(distances[0], indices[0]):

        # FAISS pads with -1 when fewer vectors exist than search_k.
        if pos < 0:
            continue

        record = resources["metadata"][pos]

        # Skip papers the user has already read (catalogue mode).
        if record["paper_id"] in exclude_set:
            continue

        # Copy the metadata dict so we don't mutate the cached metadata list,
        # then add the similarity score from this specific query.
        candidate = {**record, "score": float(dist)}
        results.append(candidate)

        if len(results) >= k:
            break

    log.info(
        f"FAISS search returned {len(results)} candidates "
        f"(k={k}, excluded={len(exclude_set)}, searched={search_k})"
    )
    return results


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick end-to-end test: embed a sample paper and print the top-5 results.
    Run from the project root:
      .\\venv\\Scripts\\python.exe retrieval\\retrieve.py
    """
    log.info("Running standalone retrieval test ...")

    resources = load_resources()

    # Use the first paper in the metadata as a test query.
    sample = resources["metadata"][0]
    log.info(f"Query paper: '{sample['title'][:80]}'")

    query_vec = embed_single_paper(sample["title"], sample["abstract"], resources)
    candidates = search_index(query_vec, resources, k=5, exclude_ids=[sample["paper_id"]])

    print("\nTop-5 recommendations:")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. [{c['score']:.4f}] {c['title'][:70]}  ({c['year']})")
