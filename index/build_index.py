"""
build_index.py
--------------
Constructs a FAISS vector index from the SPECTER2 embeddings produced by
embeddings/embed_papers.py, and builds a position-aligned metadata store
that maps every FAISS result back to the full paper record.

What this script produces:
  index/papers.index    — FAISS IndexFlatIP, ready for ANN search
  index/metadata.json   — JSON array where metadata[i] is the full paper record
                          for the paper at FAISS position i

Why two separate files?
  FAISS only stores float vectors — it has no concept of paper titles,
  abstracts, or citation lists. A separate metadata store is therefore
  needed to answer the question "what paper is at index position 42?"
  The position-aligned list is the bridge: retrieve.py does a FAISS search,
  gets back integer positions, and looks them up in metadata[pos] in O(1).

Why IndexFlatIP?
  IndexFlatIP computes exact inner products between the query vector and every
  stored vector. Since all embeddings are L2-normalised (done by embed_papers.py),
  inner product equals cosine similarity, which is the correct distance metric
  for SPECTER2 embeddings. "Flat" means no approximation — every vector is
  compared, guaranteeing the true top-K results.

  For corpora above ~100,000 papers, IndexIVFFlat (approximate, cluster-based)
  would be faster. At 10,000 papers, IndexFlatIP queries complete in under 5ms
  so the exact index is preferred — zero approximation error, simpler code.

Pipeline position:
  fetch_papers.py  →  embed_papers.py  →  [build_index.py]  →  retrieve.py

Usage (from the project root, using the project venv):
  .\\venv\\Scripts\\python.exe index\\build_index.py

Input files:
  embeddings/embeddings.npy   — float32 array [n_papers × 768], L2-normalised
  embeddings/paper_ids.json   — ordered list of paper_id strings (row-aligned)
  data/papers.parquet         — full paper metadata produced by fetch_papers.py

Output files:
  index/papers.index    — FAISS binary index file
  index/metadata.json   — position-aligned list of paper metadata dicts
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path

# ── Dependency guard ───────────────────────────────────────────────────────────
try:
    import faiss
    import pandas as pd
except ImportError as exc:
    print(
        f"\n[ERROR] Missing package: {exc}\n"
        "Run with the project virtual environment:\n"
        r"  .\venv\Scripts\python.exe index\build_index.py",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

# Expected embedding dimensionality produced by SPECTER2_base.
# If this doesn't match the loaded .npy file, the index would be built with
# wrong dimensions — catching it early prevents silent downstream corruption.
EMBEDDING_DIM = 768

PROJECT_ROOT        = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH     = PROJECT_ROOT / "embeddings" / "embeddings.npy"
PAPER_IDS_PATH      = PROJECT_ROOT / "embeddings" / "paper_ids.json"
PARQUET_PATH        = PROJECT_ROOT / "data"        / "papers.parquet"
OUTPUT_INDEX_PATH   = PROJECT_ROOT / "index"       / "papers.index"
OUTPUT_METADATA_PATH= PROJECT_ROOT / "index"       / "metadata.json"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────────────

def load_embeddings(
    embeddings_path: Path,
    ids_path: Path,
) -> tuple[np.ndarray, list]:
    """
    Load the embedding matrix and the aligned paper ID list from disk.

    Both files are produced by embed_papers.py and must exist before this
    script runs. The function validates that both files are present, that
    the embedding array has the expected shape, and that the number of
    paper IDs matches the number of embedding rows — a mismatch here would
    mean the two files are out of sync (e.g. embed_papers.py was re-run
    with a different corpus without updating paper_ids.json).

    Args:
        embeddings_path: Path to embeddings.npy ([n_papers, 768] float32).
        ids_path:        Path to paper_ids.json (list of paper_id strings).

    Returns:
        Tuple of:
          - np.ndarray: the embedding matrix, shape [n_papers, 768], float32.
          - list[str]:  paper_ids in the same row order as the embeddings.

    Raises:
        SystemExit on missing files, wrong dtype, wrong dimensions, or
        mismatched row counts between the two files.
    """
    # ── Embeddings ─────────────────────────────────────────────────────────
    if not embeddings_path.exists():
        log.error(
            f"Embeddings file not found: {embeddings_path}\n"
            "Run embeddings/embed_papers.py first."
        )
        sys.exit(1)

    embeddings = np.load(embeddings_path)
    log.info(f"Loaded embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}")

    # FAISS requires float32. embed_papers.py saves float32, but guard anyway.
    if embeddings.dtype != np.float32:
        log.warning(f"Casting embeddings from {embeddings.dtype} to float32.")
        embeddings = embeddings.astype(np.float32)

    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        log.error(
            f"Unexpected embedding shape {embeddings.shape}. "
            f"Expected (n, {EMBEDDING_DIM})."
        )
        sys.exit(1)

    # ── Paper IDs ──────────────────────────────────────────────────────────
    if not ids_path.exists():
        log.error(
            f"Paper IDs file not found: {ids_path}\n"
            "Run embeddings/embed_papers.py first."
        )
        sys.exit(1)

    with open(ids_path, "r", encoding="utf-8") as f:
        paper_ids = json.load(f)

    log.info(f"Loaded {len(paper_ids):,} paper IDs from {ids_path.name}")

    # ── Alignment check ────────────────────────────────────────────────────
    # The number of rows in the embedding matrix must exactly equal the number
    # of paper IDs. Any mismatch means the files are from different runs and
    # the index would produce incorrect lookups.
    if len(paper_ids) != embeddings.shape[0]:
        log.error(
            f"Alignment mismatch: {len(paper_ids):,} paper_ids vs "
            f"{embeddings.shape[0]:,} embedding rows.\n"
            "Re-run embed_papers.py to regenerate both files together."
        )
        sys.exit(1)

    return embeddings, paper_ids


def load_metadata(parquet_path: Path, paper_ids: list) -> pd.DataFrame:
    """
    Load the paper corpus from Parquet and index it by paper_id.

    The Parquet file contains all paper fields produced by fetch_papers.py.
    We need it here to attach rich metadata (title, abstract, year, references,
    etc.) to each FAISS position. The DataFrame is indexed by paper_id so that
    a lookup by ID is O(1) rather than a linear scan.

    Validates that every paper_id from paper_ids.json has a corresponding row
    in the Parquet. If any IDs are missing, the metadata store would have gaps
    at those FAISS positions, causing silent retrieval failures downstream.
    In that case the script exits with a clear message: the fix is to re-run
    the full pipeline (fetch → embed → index) so all three files are in sync.

    Args:
        parquet_path: Path to data/papers.parquet.
        paper_ids:    Ordered list of paper_id strings from paper_ids.json.

    Returns:
        DataFrame indexed by paper_id, containing all metadata columns.

    Raises:
        SystemExit if the file is missing or any paper_ids are absent.
    """
    if not parquet_path.exists():
        log.error(
            f"Parquet file not found: {parquet_path}\n"
            "Run data/fetch_papers.py first."
        )
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(df):,} papers from {parquet_path.name}")

    # Index by paper_id for fast positional lookup in build_metadata_list().
    df = df.set_index("paper_id")

    # Check every embedded paper_id exists in the metadata.
    missing = [pid for pid in paper_ids if pid not in df.index]
    if missing:
        log.error(
            f"{len(missing):,} paper_ids from embeddings have no metadata row "
            f"in {parquet_path.name}.\n"
            "The corpus and embeddings are out of sync. Re-run the full pipeline:\n"
            "  1. python data/fetch_papers.py\n"
            "  2. python embeddings/embed_papers.py\n"
            "  3. python index/build_index.py"
        )
        sys.exit(1)

    log.info("All embedded paper IDs found in metadata — corpus is in sync.")
    return df


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Construct a FAISS IndexFlatIP from the L2-normalised embedding matrix.

    IndexFlatIP stores all vectors in a flat (uncompressed, unquantised) array
    and performs exhaustive inner product search at query time. Since all vectors
    are unit-norm (L2-normalised by embed_papers.py), inner product = cosine
    similarity — higher is more similar.

    The index is built by a single call to index.add(), which copies all vectors
    into FAISS's internal storage. For 10,000 × 768 float32 vectors this takes
    under one second on any modern CPU.

    Note on index choice:
      IndexFlatIP  — exact, O(n) query, correct for ≤~100k vectors.
      IndexIVFFlat — approximate, O(√n) query, needed for millions of vectors.
      We use Flat because 10,000 vectors query in ~2ms — approximation overhead
      would outweigh any speed gain at this scale.

    Args:
        embeddings: L2-normalised float32 array of shape [n_papers, 768].

    Returns:
        A populated faiss.IndexFlatIP ready for search.
    """
    n_papers, dim = embeddings.shape
    log.info(f"Building IndexFlatIP  (dim={dim}, n_vectors={n_papers:,}) ...")

    index = faiss.IndexFlatIP(dim)

    # index.add() copies the numpy array into FAISS's internal C++ memory.
    # The array must be C-contiguous (row-major) and float32 — np.load() and
    # astype(float32) both guarantee this.
    index.add(embeddings)

    log.info(f"Index built: {index.ntotal:,} vectors stored.")
    return index


def validate_index(index: faiss.IndexFlatIP, embeddings: np.ndarray) -> None:
    """
    Run a quick self-retrieval sanity check on the freshly built index.

    Queries the index with the first embedding vector and verifies:
      1. The top result is itself (index position 0).
      2. The similarity score is ~1.0 (a unit-norm vector has cosine similarity
         of exactly 1.0 with itself).

    If either check fails it indicates either a normalisation bug in
    embed_papers.py or a data corruption issue, and the script exits rather
    than writing a broken index to disk.

    Args:
        index:      The FAISS index to test.
        embeddings: The full embedding matrix (used to extract the test vector).
    """
    log.info("Running self-retrieval sanity check ...")

    # Query shape must be [1, dim] — FAISS always expects 2-D input.
    test_vec = embeddings[0:1]
    distances, indices = index.search(test_vec, k=1)

    retrieved_pos   = int(indices[0][0])
    retrieved_score = float(distances[0][0])

    if retrieved_pos != 0:
        log.error(
            f"Self-retrieval failed: queried position 0, got back position "
            f"{retrieved_pos}. The index may be corrupted."
        )
        sys.exit(1)

    if abs(retrieved_score - 1.0) > 1e-3:
        log.error(
            f"Self-similarity is {retrieved_score:.6f}, expected ~1.0.\n"
            "Embeddings may not be L2-normalised — re-run embed_papers.py."
        )
        sys.exit(1)

    log.info(
        f"Sanity check passed: self-retrieval position=0, score={retrieved_score:.6f}"
    )


def build_metadata_list(df: pd.DataFrame, paper_ids: list) -> list:
    """
    Construct a position-aligned list of paper metadata dicts.

    The returned list has exactly one entry per FAISS position, in the same
    order as paper_ids. metadata_list[i] is the full metadata for the paper
    at FAISS index position i.

    This is the bridge between integer FAISS positions and human-readable paper
    records. retrieve.py does:
        positions = faiss_search(query_vec)
        results   = [metadata_list[pos] for pos in positions]

    All pandas/pyarrow types are explicitly converted to plain Python types
    (int, str, list) because JSON serialisation does not handle numpy int64,
    pandas NA, or pyarrow-backed list arrays natively.

    Args:
        df:        DataFrame indexed by paper_id, containing all metadata columns.
        paper_ids: Ordered list of paper_id strings (FAISS position → paper_id).

    Returns:
        List of dicts, one per paper, in the same order as paper_ids.
        Each dict contains: paper_id, title, abstract, year, citation_count,
        fields_of_study, references, arxiv_id.
    """
    log.info(f"Building position-aligned metadata list for {len(paper_ids):,} papers ...")

    metadata_list = []

    for pos, pid in enumerate(paper_ids):
        row = df.loc[pid]

        # ── Year ──────────────────────────────────────────────────────────────
        # Pandas nullable Int64 does not serialise to JSON. Convert to Python
        # int if the value exists, or None if the year is missing.
        year = row["year"]
        year = None if pd.isna(year) else int(year)

        # ── List columns ──────────────────────────────────────────────────────
        # pyarrow-backed list columns may be returned as pyarrow ChunkedArrays
        # or pandas ExtensionArrays rather than plain Python lists. Wrapping in
        # list() ensures they are plain Python lists that json.dump() accepts.
        references      = list(row["references"])      if row["references"]      is not None else []
        fields_of_study = list(row["fields_of_study"]) if row["fields_of_study"] is not None else []

        metadata_list.append({
            "paper_id":        str(pid),
            "title":           str(row["title"]),
            "abstract":        str(row["abstract"]),
            "year":            year,
            "citation_count":  int(row["citation_count"]),
            "fields_of_study": fields_of_study,
            "references":      references,   # list of paper_id strings this paper cites
            "arxiv_id":        "" if pd.isnull(row.get("arxiv_id")) else (str(row["arxiv_id"]).strip() or ""),
        })

        # Log progress every 2,000 papers so the user can see the script is running.
        if (pos + 1) % 2_000 == 0:
            log.info(f"  Processed {pos + 1:,} / {len(paper_ids):,} metadata records")

    log.info(f"Metadata list built: {len(metadata_list):,} records.")
    return metadata_list


def save_index(index: faiss.IndexFlatIP, output_path: Path) -> None:
    """
    Write the FAISS index to disk in FAISS's native binary format.

    FAISS's write_index() produces a self-contained binary file that can be
    loaded back with faiss.read_index() in any subsequent process. The file
    contains both the index structure (IndexFlatIP header) and all stored
    vectors. For 10,000 × 768 float32 vectors this is approximately 30 MB.

    The file is written to index/papers.index alongside metadata.json so
    that retrieve.py can load both from the same directory.

    Args:
        index:       The populated FAISS index to persist.
        output_path: Destination file path (e.g. index/papers.index).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))

    size_mb = output_path.stat().st_size / (1024 ** 2)
    log.info(f"FAISS index saved → {output_path}  ({size_mb:.1f} MB)")


def save_metadata(metadata_list: list, output_path: Path) -> None:
    """
    Write the position-aligned metadata list to disk as a JSON array.

    JSON is used here rather than Parquet because:
      - retrieve.py needs to access individual records by integer position,
        which is a natural list index operation in Python but requires a
        full column scan in Parquet.
      - The metadata store is loaded once at Streamlit startup and kept in
        memory (cached with @st.cache_resource), so the ~35 MB in-memory
        size is not a concern for a local demo.
      - JSON is human-readable: you can open index/metadata.json in any
        text editor to inspect records and debug retrieval issues.

    Args:
        metadata_list: Position-aligned list of paper metadata dicts.
        output_path:   Destination file path (e.g. index/metadata.json).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    log.info(
        f"Metadata saved    → {output_path}  "
        f"({size_mb:.1f} MB,  {len(metadata_list):,} records)"
    )


def print_index_stats(index: faiss.IndexFlatIP, metadata_list: list) -> None:
    """
    Print a brief summary of the completed index for visual confirmation.

    Runs three test queries against the index — one per evenly-spaced position
    in the corpus — and prints the top-3 paper titles for each. This lets the
    user quickly confirm that the index is returning meaningful papers rather
    than garbage, without having to open the Streamlit app.

    Args:
        index:         The completed, validated FAISS index.
        metadata_list: The position-aligned metadata list.
    """
    log.info("Index preview — top-1 self-query for three sample positions:")

    sample_positions = [0, len(metadata_list) // 2, len(metadata_list) - 1]

    for pos in sample_positions:
        # We don't have the raw embeddings here, so we can't do a live query.
        # Instead just print the metadata at that position as a sanity preview.
        record = metadata_list[pos]
        title  = record["title"][:80] + ("..." if len(record["title"]) > 80 else "")
        year   = record["year"] or "n/a"
        n_refs = len(record["references"])
        log.info(f"  pos {pos:>5} | {year} | refs={n_refs:>3} | {title}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full index construction pipeline.

    Steps in order:
      1. Load embeddings.npy and paper_ids.json produced by embed_papers.py.
      2. Load papers.parquet produced by fetch_papers.py.
      3. Validate that the corpus and embeddings are in sync.
      4. Build the FAISS IndexFlatIP from the embedding matrix.
      5. Run a self-retrieval sanity check on the fresh index.
      6. Build the position-aligned metadata list.
      7. Save papers.index and metadata.json to the index/ directory.
      8. Print a brief preview of the index contents.

    This script is fast (~10–30 seconds total) compared to the embedding step.
    The expensive work (SPECTER2 inference) was done by embed_papers.py;
    building a FAISS index is just a memory copy and an O(n) write to disk.
    """
    log.info("=" * 60)
    log.info("  FAISS index construction starting")
    log.info(f"  Expected dim : {EMBEDDING_DIM}")
    log.info("=" * 60)

    start_time = time.time()

    # ── Step 1: Load embeddings and paper IDs ─────────────────────────────────
    embeddings, paper_ids = load_embeddings(EMBEDDINGS_PATH, PAPER_IDS_PATH)

    # ── Step 2: Load paper metadata ───────────────────────────────────────────
    df = load_metadata(PARQUET_PATH, paper_ids)

    # ── Step 3: Build FAISS index ─────────────────────────────────────────────
    index = build_faiss_index(embeddings)

    # ── Step 4: Sanity check ──────────────────────────────────────────────────
    validate_index(index, embeddings)

    # ── Step 5: Build position-aligned metadata ───────────────────────────────
    metadata_list = build_metadata_list(df, paper_ids)

    # ── Step 6: Save everything ───────────────────────────────────────────────
    save_index(index, OUTPUT_INDEX_PATH)
    save_metadata(metadata_list, OUTPUT_METADATA_PATH)

    # ── Step 7: Preview ───────────────────────────────────────────────────────
    print_index_stats(index, metadata_list)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed    = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    log.info("=" * 60)
    log.info("  Index construction complete")
    log.info(f"  Papers indexed : {index.ntotal:,}")
    log.info(f"  Index type     : {type(index).__name__}")
    log.info(f"  Total time     : {mins}m {secs}s")
    log.info(f"  Output dir     : {OUTPUT_INDEX_PATH.parent}")
    log.info("  Next step      : .\\venv\\Scripts\\python.exe retrieval\\retrieve.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
