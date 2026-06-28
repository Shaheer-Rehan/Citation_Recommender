"""
embed_papers.py
---------------
Generates citation-aware vector embeddings for every paper in the corpus
using the SPECTER2 model (allenai/specter2_base) from Allen AI.

What this script does:
  1. Loads the paper corpus produced by data/fetch_papers.py (papers.parquet)
  2. Downloads/loads the SPECTER2_base model from HuggingFace (~440 MB, cached
     after the first run in the HuggingFace cache directory)
  3. Formats each paper as "Title [SEP] Abstract" — the input format SPECTER2
     was trained on
  4. Runs all papers through the model in batches, extracting the [CLS] token
     embedding (position 0 of the last hidden state) as the paper vector
  5. L2-normalises every embedding so that FAISS inner product equals cosine
     similarity — a requirement for the IndexFlatIP index built in build_index.py
  6. Saves two output files:
       embeddings/embeddings.npy   — float32 array of shape [n_papers, 768]
       embeddings/paper_ids.json   — ordered list of paper_id strings whose
                                     row i matches embeddings[i]

Why [CLS] token pooling?
  SPECTER models are trained using a [CLS]-level classification objective: the
  [CLS] representation is directly optimised to be close to the representations
  of papers that are co-cited. Mean-pooling across all tokens was not used in
  training and typically produces slightly worse results for this model family.

Why L2-normalise?
  FAISS IndexFlatIP computes inner products. For unit-norm vectors, inner product
  equals cosine similarity, which is the correct distance metric for comparing
  SPECTER embeddings. Normalising here means build_index.py can use the faster
  IndexFlatIP rather than IndexFlatL2.

Runtime estimate on CPU (~CPU-only torch):
  10,000 papers at batch_size=32 → ~313 batches
  Each batch takes roughly 8–20 seconds depending on hardware.
  Total: approximately 60–100 minutes. Start this script and let it run.
  Increase BATCH_SIZE if you have more than 8 GB of RAM available.

Usage (from the project root, using the project venv):
  .\\venv\\Scripts\\python.exe embeddings\\embed_papers.py

Input:
  data/papers.parquet    — produced by data/fetch_papers.py

Output:
  embeddings/embeddings.npy    — [n_papers × 768] float32, L2-normalised
  embeddings/paper_ids.json    — list of paper_id strings (row-aligned with embeddings)
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path

# ── Dependency guard ───────────────────────────────────────────────────────────
# Catch missing packages early with a clear message pointing to the venv,
# rather than letting an unhelpful ImportError surface mid-script.
try:
    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
except ImportError as exc:
    print(
        f"\n[ERROR] Missing package: {exc}\n"
        "Make sure you are running with the project virtual environment:\n"
        r"  .\venv\Scripts\python.exe embeddings\embed_papers.py",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

# SPECTER2_base: the base encoder of SPECTER2, trained by Allen AI with a
# citation-aware triplet loss. Loads cleanly with standard AutoModel.
# To use the full SPECTER2 with proximity adapter (requires 'adapters' library),
# replace with 'allenai/specter2' and follow the adapter loading pattern.
# To fall back to SPECTER v1 (identical loading, slightly lower quality), use
# 'allenai/specter'.
MODEL_NAME = "allenai/specter2_base"

# Maximum token sequence length. SPECTER2 was trained with 512-token inputs.
# Truncation from the end is applied automatically by the tokenizer; since
# titles are short (< 50 tokens), the abstract always gets the majority.
MAX_LENGTH = 512

# Papers processed in a single forward pass. Increase if you have spare RAM
# (64 is comfortable on 16 GB). Lower to 16 if you hit out-of-memory errors.
BATCH_SIZE = 32

# Paths — resolved relative to this script's location so the script runs
# correctly regardless of which directory it is invoked from.
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
INPUT_PARQUET  = PROJECT_ROOT / "data"       / "papers.parquet"
OUTPUT_DIR     = PROJECT_ROOT / "embeddings"
OUTPUT_EMBEDDINGS = OUTPUT_DIR / "embeddings.npy"
OUTPUT_IDS        = OUTPUT_DIR / "paper_ids.json"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Select and log the compute device for model inference.

    Checks for CUDA (NVIDIA GPU) first, then MPS (Apple Silicon GPU), then
    falls back to CPU. Since this project installs torch+cpu, CUDA will not
    be available unless the user manually reinstalls torch with CUDA support.

    Returns:
        torch.device: The device object passed to the model and input tensors.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"GPU detected: {torch.cuda.get_device_name(0)} — using CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Apple Silicon GPU detected — using MPS.")
    else:
        device = torch.device("cpu")
        log.info("No GPU found — running on CPU. Embedding will take 60–100 minutes.")

    return device


def load_model(model_name: str, device: torch.device) -> tuple:
    """
    Download (first run) or load (cached) the SPECTER2_base tokenizer and model.

    HuggingFace caches downloaded model weights in ~/.cache/huggingface/ so
    subsequent runs skip the download entirely. The model is moved to the
    target device and switched to eval mode — this disables dropout layers,
    which must be off during inference to get deterministic embeddings.

    Args:
        model_name: HuggingFace model identifier (e.g. 'allenai/specter2_base').
        device:     Compute device to load the model onto.

    Returns:
        Tuple of (tokenizer, model) ready for inference.
    """
    log.info(f"Loading tokenizer from '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info(f"Loading model from '{model_name}' (downloads ~440 MB on first run) ...")
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()  # disables dropout — essential for deterministic inference

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model ready: {param_count:.0f}M parameters on {device}.")

    return tokenizer, model


def load_papers(parquet_path: Path) -> pd.DataFrame:
    """
    Load the paper corpus from Parquet and validate required columns are present.

    The Parquet file is produced by data/fetch_papers.py. We only need
    'paper_id', 'title', and 'abstract' for embedding — the other columns
    (references, citation_count, etc.) are passed through to build_index.py
    via the separate metadata store, not embedded here.

    Args:
        parquet_path: Path to the papers.parquet file.

    Returns:
        DataFrame containing at minimum: paper_id, title, abstract.

    Raises:
        SystemExit: If the file is missing or required columns are absent.
    """
    if not parquet_path.exists():
        log.error(
            f"Input file not found: {parquet_path}\n"
            "Run data/fetch_papers.py first to generate the corpus."
        )
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(df):,} papers from {parquet_path.name}.")

    required_cols = {"paper_id", "title", "abstract"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(f"Parquet file is missing required columns: {missing}")
        sys.exit(1)

    # Drop any rows that somehow slipped through fetch_papers.py without an
    # abstract. These would produce near-zero embeddings and corrupt the index.
    before = len(df)
    df = df[df["abstract"].str.strip().str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} papers with empty abstracts.")

    return df


def format_input(title: str, abstract: str, sep_token: str) -> str:
    """
    Format a paper's title and abstract into the SPECTER2 input string.

    SPECTER2 was trained with inputs structured as:
        "Title [SEP] Abstract"
    where [SEP] is the tokenizer's separator token. The tokenizer then wraps
    this into the full BERT format: [CLS] Title [SEP] Abstract [SEP].

    The [CLS] token's embedding (after the forward pass) is used as the paper
    representation, because SPECTER2's training objective directly optimises
    the [CLS] position to encode paper-level semantic proximity.

    Args:
        title:     Paper title string (may be empty — handled gracefully).
        abstract:  Paper abstract string.
        sep_token: The tokenizer's separator token (typically '[SEP]' for BERT).

    Returns:
        Formatted input string ready for tokenisation.
    """
    title    = (title or "").strip()
    abstract = (abstract or "").strip()

    # If the title is missing, use the abstract alone. The sep_token is still
    # included so the model sees the expected input structure.
    if not title:
        return abstract

    return title + sep_token + abstract


def embed_batch(
    texts: list,
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Tokenise a batch of input strings and run a single model forward pass.

    Returns the [CLS] token embedding for each input — position 0 of the
    last hidden state. The output is moved to CPU and converted to a float32
    numpy array before returning.

    Gradient computation is disabled via torch.no_grad(), which:
      - Prevents storing intermediate activations needed for backprop
      - Reduces memory usage by ~50%
      - Speeds up the forward pass by ~20%

    Args:
        texts:     List of pre-formatted "Title [SEP] Abstract" strings.
        tokenizer: The loaded SPECTER2 tokenizer.
        model:     The loaded SPECTER2 model (in eval mode, on the target device).
        device:    The device the model lives on.

    Returns:
        numpy array of shape [len(texts), 768], dtype float32.
        Contains raw (un-normalised) CLS embeddings for this batch.
    """
    # Tokenise: pad to the longest sequence in the batch, truncate at MAX_LENGTH.
    # 'return_tensors="pt"' gives us PyTorch tensors directly.
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Move token ID tensors to the same device as the model.
    encoded = {key: val.to(device) for key, val in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    # Extract the [CLS] token (index 0 along the sequence dimension).
    # Shape: [batch_size, 768]
    cls_embeddings = output.last_hidden_state[:, 0, :]

    # Detach from the computation graph, move to CPU, and cast to float32.
    # float32 is standard for FAISS; float16 would halve storage but reduce
    # indexing precision.
    return cls_embeddings.detach().cpu().to(torch.float32).numpy()


def embed_all_papers(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, list]:
    """
    Iterate over the full paper corpus in batches and collect all embeddings.

    Papers are processed in order of their row index in the DataFrame. The
    returned paper_ids list is aligned with the embedding rows: paper_ids[i]
    corresponds to the embedding at embeddings[i]. This alignment is the
    critical contract between this script and build_index.py.

    A tqdm progress bar displays batches completed, estimated time remaining,
    and processing speed (batches/second). At 10,000 papers with batch_size=32,
    expect ~313 bar steps.

    Args:
        df:         DataFrame with at minimum 'paper_id', 'title', 'abstract'.
        tokenizer:  The loaded SPECTER2 tokenizer.
        model:      The loaded SPECTER2 model.
        device:     The compute device.
        batch_size: Number of papers per forward pass.

    Returns:
        Tuple of:
          - np.ndarray: shape [n_papers, 768], dtype float32, un-normalised.
          - list[str]:  paper_id for each row, in the same order.
    """
    n_papers     = len(df)
    n_batches    = (n_papers + batch_size - 1) // batch_size  # ceiling division
    all_embeddings = []
    paper_ids      = []

    log.info(
        f"Embedding {n_papers:,} papers in {n_batches} batches "
        f"(batch_size={batch_size}, device={device})."
    )

    with tqdm(total=n_batches, desc="Embedding", unit="batch") as pbar:
        for batch_start in range(0, n_papers, batch_size):
            batch_df = df.iloc[batch_start : batch_start + batch_size]

            # Build the formatted input strings for this batch.
            texts = [
                format_input(row["title"], row["abstract"], tokenizer.sep_token)
                for _, row in batch_df.iterrows()
            ]

            # Collect the paper IDs in the same order as the texts.
            paper_ids.extend(batch_df["paper_id"].tolist())

            # Run the forward pass and accumulate the raw embeddings.
            batch_embeddings = embed_batch(texts, tokenizer, model, device)
            all_embeddings.append(batch_embeddings)

            pbar.update(1)

    # Concatenate all batch arrays into a single matrix along axis 0.
    # Shape: [n_papers, 768]
    embeddings = np.concatenate(all_embeddings, axis=0)
    log.info(f"Raw embeddings shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    return embeddings, paper_ids


def normalise_l2(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalise every row of the embedding matrix in-place.

    After normalisation, every embedding vector has unit norm (magnitude = 1).
    This ensures that the inner product between any two vectors equals their
    cosine similarity — a requirement for FAISS IndexFlatIP, which computes
    inner products rather than cosine similarity directly.

    A small epsilon (1e-10) is added to norms before dividing to prevent
    division by zero for any degenerate all-zero vectors (extremely rare but
    possible if a paper's text failed to tokenise meaningfully).

    Args:
        embeddings: Float32 array of shape [n_papers, 768]. Modified in-place.

    Returns:
        The same array with each row rescaled to unit norm.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-10)

    # Sanity check: all norms should now be very close to 1.0.
    post_norms = np.linalg.norm(embeddings, axis=1)
    log.info(
        f"L2 normalisation done. "
        f"Post-norm stats — min: {post_norms.min():.6f}, "
        f"max: {post_norms.max():.6f}, "
        f"mean: {post_norms.mean():.6f}  (should all be ~1.0)"
    )

    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    paper_ids: list,
    output_dir: Path,
) -> None:
    """
    Persist the embedding matrix and the aligned paper ID list to disk.

    Two separate files are written:
      embeddings.npy   — the float32 embedding matrix, loadable with np.load().
      paper_ids.json   — the ordered list of paper_id strings. Row i in the
                         embedding matrix corresponds to paper_ids[i]. This
                         alignment is how build_index.py maps FAISS result
                         indices back to paper metadata.

    Numpy's .npy format is chosen over CSV or Parquet because it:
      - Preserves the exact float32 dtype without precision loss
      - Loads back into a numpy array in one call with zero parsing overhead
      - Is a fixed-size binary format (~30 MB for 10k × 768 float32 vectors)

    Args:
        embeddings:  L2-normalised float32 array of shape [n_papers, 768].
        paper_ids:   List of paper_id strings, row-aligned with embeddings.
        output_dir:  Directory to write both output files into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the embedding matrix.
    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    emb_size_mb = emb_path.stat().st_size / (1024 ** 2)
    log.info(f"Saved embeddings → {emb_path}  ({emb_size_mb:.1f} MB)")

    # Save the paper IDs as a JSON list — plain text, human-inspectable.
    ids_path = output_dir / "paper_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(paper_ids, f)
    log.info(f"Saved paper IDs  → {ids_path}  ({len(paper_ids):,} entries)")

    # Final alignment check: the number of IDs must exactly match the number
    # of embedding rows. A mismatch here would silently corrupt all downstream
    # lookups in build_index.py and retrieval/retrieve.py.
    assert len(paper_ids) == embeddings.shape[0], (
        f"Alignment mismatch: {len(paper_ids)} paper_ids vs "
        f"{embeddings.shape[0]} embedding rows."
    )
    log.info("Alignment check passed: paper_ids and embedding rows match.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full embedding pipeline end-to-end.

    Steps in order:
      1. Select compute device (GPU if available, else CPU).
      2. Load SPECTER2_base tokenizer and model from HuggingFace.
      3. Load the paper corpus from data/papers.parquet.
      4. Embed all papers in batches, collecting CLS token representations.
      5. L2-normalise the full embedding matrix.
      6. Save embeddings.npy and paper_ids.json to the embeddings/ directory.

    Wall-clock timing is measured and reported at the end to give the user
    a reference for future runs or hardware comparisons.
    """
    log.info("=" * 60)
    log.info("  SPECTER2 embedding pipeline starting")
    log.info(f"  Model      : {MODEL_NAME}")
    log.info(f"  Batch size : {BATCH_SIZE}")
    log.info(f"  Max tokens : {MAX_LENGTH}")
    log.info("=" * 60)

    start_time = time.time()

    # ── Step 1: Device ────────────────────────────────────────────────────────
    device = get_device()

    # ── Step 2: Model ─────────────────────────────────────────────────────────
    tokenizer, model = load_model(MODEL_NAME, device)

    # ── Step 3: Load corpus ───────────────────────────────────────────────────
    df = load_papers(INPUT_PARQUET)

    # ── Step 4: Embed ─────────────────────────────────────────────────────────
    embeddings, paper_ids = embed_all_papers(df, tokenizer, model, device)

    # ── Step 5: Normalise ─────────────────────────────────────────────────────
    embeddings = normalise_l2(embeddings)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    save_embeddings(embeddings, paper_ids, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed     = time.time() - start_time
    mins, secs  = divmod(int(elapsed), 60)
    per_paper   = elapsed / len(paper_ids) if paper_ids else 0

    log.info("=" * 60)
    log.info("  Embedding pipeline complete")
    log.info(f"  Papers embedded : {len(paper_ids):,}")
    log.info(f"  Embedding shape : {embeddings.shape}")
    log.info(f"  Total time      : {mins}m {secs}s  ({per_paper:.2f}s/paper)")
    log.info(f"  Output dir      : {OUTPUT_DIR}")
    log.info("  Next step       : .\\venv\\Scripts\\python.exe index\\build_index.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
