"""
fetch_papers.py
---------------
Fetches academic paper metadata from the Semantic Scholar Academic Graph API
and saves the result as a Parquet file for downstream embedding.

Pipeline (in order):
  1. Search the API with multiple diverse queries to build a broad ML/NLP/AI corpus
  2. Deduplicate papers that appeared across multiple queries
  3. Filter out papers with missing or very short abstracts
  4. Normalise the raw API response into a clean, flat schema
  5. Save to data/papers.parquet

Output schema (papers.parquet):
  paper_id         str          Semantic Scholar unique paper ID
  title            str          Paper title
  abstract         str          Full abstract text
  year             Int64        Publication year (nullable)
  citation_count   int          Number of times this paper has been cited
  fields_of_study  list[str]    E.g. ["Computer Science", "Mathematics"]
  references       list[str]    paper_ids of papers this paper directly cites
                                (used for citation-overlap re-ranking in rerank.py)
  arxiv_id         str          ArXiv ID if available, else empty string

Usage:
  1. Get a free API key at https://www.semanticscholar.org/product/api
  2. Set the environment variable:  S2_API_KEY=<your-key>
  3. Run from the project root:     python data/fetch_papers.py

  Without an API key the script still works but is subject to tighter
  rate limits (~1 req/s). The REQUEST_DELAY constant is set conservatively
  to stay within unauthenticated limits; with a key you can lower it.
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields requested from the API for each paper.
# 'references' returns the papers this paper cites — each reference is a dict
# with at least {"paperId": str}. These IDs are later used to compute
# citation overlap between query and candidate papers during re-ranking.
# 'externalIds' gives us the ArXiv ID for direct linking in the Streamlit demo.
PAPER_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "citationCount",
    "fieldsOfStudy",
    "references",
    "externalIds",
]

# Diverse queries to cover a broad ML/NLP/AI corpus.
# A single query clusters tightly around one sub-topic; multiple queries
# give us a more varied index, which produces more interesting recommendations.
SEARCH_QUERIES = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "transformer neural network",
    "graph neural network",
    "generative adversarial network",
    "representation learning",
    "neural network optimization",
]

TARGET_PAPERS = 10_000   # desired unique papers in the final Parquet file
MAX_PER_QUERY = 1_500    # maximum papers to fetch per query (15 pages of 100)
PAGE_SIZE     = 100      # Semantic Scholar API hard maximum per request
REQUEST_DELAY = 0.12     # seconds between requests — safe for unauthenticated tier

OUTPUT_FILE = Path(__file__).parent / "papers.parquet"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Functions ──────────────────────────────────────────────────────────────────

def get_headers() -> dict:
    """
    Build HTTP headers for Semantic Scholar API requests.

    Checks for the S2_API_KEY environment variable. An API key increases the
    rate limit from roughly 1 request/second to 100 requests/second. Without
    a key the script still functions but fetching will be slower. A warning is
    logged so the user knows what to do if they want faster collection.

    Returns:
        dict: Headers dict, optionally containing 'x-api-key'.
    """
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("S2_API_KEY")

    if api_key:
        headers["x-api-key"] = api_key
        log.info("S2_API_KEY found — using authenticated rate limits (100 req/s).")
    else:
        log.warning(
            "S2_API_KEY not set. Running unauthenticated (~1 req/s). "
            "Set this environment variable for much faster fetching."
        )

    return headers


def make_request(
    url: str,
    params: dict,
    headers: dict,
    max_retries: int = 6,
) -> Optional[dict]:
    """
    Execute an HTTP GET request with exponential backoff on transient errors.

    Semantic Scholar returns HTTP 429 when requests arrive too fast, and
    occasionally 504 (gateway timeout) under load. Rather than crashing,
    this function waits an exponentially increasing amount of time and retries.
    Any other non-200 status is treated as a non-recoverable error and returns
    None so the caller can skip the page gracefully.

    Args:
        url:         Full API endpoint URL.
        params:      Query-string parameters (e.g. {"query": "...", "limit": 100}).
        headers:     HTTP headers, including the optional API key.
        max_retries: How many times to retry before giving up entirely.

    Returns:
        Parsed JSON response as a dict, or None on unrecoverable failure.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                # Rate limited: back off exponentially — 1s, 2s, 4s, 8s, ...
                wait = 2 ** attempt
                log.warning(
                    f"Rate limited (429). Backing off {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})."
                )
                time.sleep(wait)

            elif response.status_code == 504:
                # Gateway timeout: API overloaded, wait a fixed interval then retry.
                wait = 5 * (attempt + 1)
                log.warning(f"Gateway timeout (504). Waiting {wait}s before retry.")
                time.sleep(wait)

            else:
                log.error(
                    f"Unrecoverable HTTP {response.status_code} for {url}. "
                    f"Response: {response.text[:300]}"
                )
                return None

        except requests.exceptions.Timeout:
            log.warning(f"Request timed out (attempt {attempt + 1}/{max_retries}).")
            time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as exc:
            # Covers connection errors, DNS failures, etc.
            log.error(f"Request exception: {exc}")
            return None

    log.error(f"Gave up after {max_retries} retries. URL: {url}")
    return None


def fetch_page(
    query: str,
    offset: int,
    headers: dict,
) -> tuple[list[dict], int]:
    """
    Fetch a single paginated page from the Semantic Scholar paper search endpoint.

    The search endpoint returns papers matching a free-text query. Results are
    paginated: the first page uses offset=0, the second offset=100, and so on.
    Along with the papers, the API also returns the total number of results
    available for the query, which the caller uses to know when to stop.

    Args:
        query:   Free-text search string sent to the API.
        offset:  Number of results to skip — incremented by PAGE_SIZE each call.
        headers: HTTP headers for the API request.

    Returns:
        Tuple of:
          - list[dict]: Paper records returned on this page (may be empty).
          - int: Total results the API reports for this query.
    """
    params = {
        "query":  query,
        "fields": ",".join(PAPER_FIELDS),
        "limit":  PAGE_SIZE,
        "offset": offset,
    }

    data = make_request(f"{BASE_URL}/paper/search", params, headers)

    if data is None:
        # Request failed after retries — return empty so the caller skips this page.
        return [], 0

    papers = data.get("data", [])
    total  = data.get("total", 0)

    return papers, total


def search_papers_for_query(
    query: str,
    headers: dict,
    max_results: int = MAX_PER_QUERY,
) -> list[dict]:
    """
    Collect papers for a single search query by paginating through all results.

    The Semantic Scholar API caps the accessible offset at 10,000, so at most
    10,000 results per query are reachable regardless of the total count. We
    stop early once we hit max_results or exhaust what the API can return.

    Progress is printed to the terminal as pages are fetched, showing how many
    papers have been collected so the user knows the script is running.

    Args:
        query:       The search string to send to the API.
        headers:     HTTP headers for API authentication.
        max_results: Stop collecting after this many papers for this query.

    Returns:
        List of raw paper dicts. Papers missing a paperId are excluded here
        since paperId is the primary key used throughout the rest of the pipeline.
    """
    collected = []
    offset    = 0

    log.info(f"  Query: '{query}'")

    while len(collected) < max_results:
        papers, total = fetch_page(query, offset, headers)

        if not papers:
            # End of available results or unrecoverable error on this page.
            break

        # Exclude any records without a paperId — they cannot be indexed or
        # looked up, so including them would corrupt the downstream mapping.
        valid = [p for p in papers if p.get("paperId")]
        collected.extend(valid)

        offset += PAGE_SIZE

        # The API's maximum accessible offset is 10,000; stop before exceeding it.
        if offset >= min(total, 10_000):
            break

        time.sleep(REQUEST_DELAY)

    log.info(f"  → {len(collected)} raw results for '{query}'")
    return collected


def deduplicate_papers(raw_papers: list[dict]) -> list[dict]:
    """
    Remove papers that appeared in more than one search query.

    Semantic Scholar paper IDs are globally unique, so two records with the
    same paperId are identical regardless of which query surfaced them. We
    keep only the first occurrence (insertion order) and discard subsequent
    duplicates. This is a simple seen-set pass, O(n) time.

    Args:
        raw_papers: Flat list of all paper dicts collected across all queries,
                    potentially containing many duplicates.

    Returns:
        List of paper dicts with each paperId appearing exactly once.
    """
    seen   = set()
    unique = []

    for paper in raw_papers:
        pid = paper["paperId"]
        if pid not in seen:
            seen.add(pid)
            unique.append(paper)

    log.info(
        f"Deduplication: {len(raw_papers)} raw records → {len(unique)} unique papers "
        f"({len(raw_papers) - len(unique)} duplicates removed)"
    )
    return unique


def filter_papers(papers: list[dict], min_abstract_words: int = 20) -> list[dict]:
    """
    Remove papers that are unsuitable for embedding with SPECTER2.

    SPECTER2 encodes the concatenated "Title [SEP] Abstract" string. A paper
    with a missing or very short abstract produces an embedding that carries
    almost no domain signal — essentially noise in the vector index. We
    therefore require a minimum word count in the abstract before accepting
    a paper into the corpus.

    Args:
        papers:             List of raw paper dicts to filter.
        min_abstract_words: Papers whose abstract has fewer words than this
                            threshold are discarded. Default is 20 words.

    Returns:
        Filtered list containing only papers with usable abstracts.
    """
    filtered            = []
    skipped_no_abstract = 0
    skipped_too_short   = 0

    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()

        if not abstract:
            skipped_no_abstract += 1
            continue

        if len(abstract.split()) < min_abstract_words:
            # Abstract exists but is too short to carry meaningful signal.
            skipped_too_short += 1
            continue

        filtered.append(paper)

    log.info(
        f"Filtering: kept {len(filtered)} papers  "
        f"(dropped {skipped_no_abstract} missing abstract, "
        f"{skipped_too_short} abstract < {min_abstract_words} words)"
    )
    return filtered


def normalise_paper(paper: dict) -> dict:
    """
    Convert a raw Semantic Scholar API dict into a flat, typed record.

    The raw API response has nested structures that do not serialise cleanly
    to Parquet without extra handling. This function flattens everything:

      - references: extracted to a plain list of paperId strings. The reference
        title and other metadata are dropped — only the IDs are needed for the
        citation-overlap score in rerank.py.

      - fieldsOfStudy: the API can return either a list of plain strings or
        a list of {"category": str, "source": str} dicts depending on the
        API version. Both cases are handled and normalised to list[str].

      - externalIds: the ArXiv ID is unpacked from the nested dict so the
        Streamlit demo can build a direct arxiv.org link for each result.

      - year: kept as-is here; coerced to nullable Int64 in build_dataframe().

    Args:
        paper: Single raw paper dict as returned by the Semantic Scholar API.

    Returns:
        Clean dict with consistent, flat types ready to become a DataFrame row.
    """
    # ── References ────────────────────────────────────────────────────────────
    # Each reference object looks like {"paperId": "abc123", "title": "..."}.
    # We only need the IDs; filtering out entries where paperId is None or ""
    # avoids corrupting the reference set with unresolvable links.
    raw_refs = paper.get("references") or []
    ref_ids  = [r["paperId"] for r in raw_refs if r.get("paperId")]

    # ── Fields of study ───────────────────────────────────────────────────────
    # Newer API versions return [{"category": "Computer Science", "source": "..."}];
    # older versions return plain strings. Handle both so the code is robust
    # to API version changes.
    raw_fields = paper.get("fieldsOfStudy") or []
    if raw_fields and isinstance(raw_fields[0], dict):
        fields_of_study = [f.get("category", "") for f in raw_fields if f.get("category")]
    else:
        fields_of_study = [str(f) for f in raw_fields if f]

    # ── ArXiv ID ──────────────────────────────────────────────────────────────
    # externalIds may contain keys like "ArXiv", "DOI", "PubMed", etc.
    # We only extract ArXiv for now; the rest are not needed.
    external_ids = paper.get("externalIds") or {}
    arxiv_id     = external_ids.get("ArXiv") or ""

    return {
        "paper_id":        paper["paperId"],
        "title":           (paper.get("title") or "").strip(),
        "abstract":        (paper.get("abstract") or "").strip(),
        "year":            paper.get("year"),         # coerced to Int64 later
        "citation_count":  int(paper.get("citationCount") or 0),
        "fields_of_study": fields_of_study,           # list[str]
        "references":      ref_ids,                   # list[str]  — paper IDs only
        "arxiv_id":        arxiv_id,
    }


def build_dataframe(papers: list[dict]) -> pd.DataFrame:
    """
    Normalise all paper dicts and assemble them into a pandas DataFrame.

    List-typed columns ('references', 'fields_of_study') are stored as Python
    lists directly in the DataFrame. PyArrow — which pandas uses under the hood
    for Parquet I/O — handles list columns natively, so no JSON encoding is
    needed. They round-trip cleanly through save → load.

    The 'year' column is cast to pandas nullable integer (Int64) rather than
    float64 so that missing values display as <NA> instead of NaN, which is
    cleaner for downstream filtering in the Streamlit demo.

    Args:
        papers: List of raw paper dicts from the API (not yet normalised).

    Returns:
        DataFrame with one row per paper and columns matching the output schema
        described at the top of this file.
    """
    records = [normalise_paper(p) for p in papers]
    df      = pd.DataFrame(records)

    # Cast year to nullable integer so missing years show as <NA>, not NaN.
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Log a quick summary so the user can sanity-check the corpus.
    year_min = df["year"].min()
    year_max = df["year"].max()
    avg_refs = df["references"].apply(len).mean()
    avg_cits = df["citation_count"].mean()

    log.info(
        f"DataFrame ready: {len(df)} papers | "
        f"years {year_min}–{year_max} | "
        f"avg {avg_refs:.1f} refs/paper | "
        f"avg {avg_cits:.0f} citations/paper"
    )

    return df


def save_papers(df: pd.DataFrame, output_path: Path) -> None:
    """
    Persist the paper DataFrame to a Parquet file on disk.

    Parquet is chosen over CSV for three reasons:
      1. Native support for list-typed columns (references, fields_of_study)
         without any JSON encoding workarounds.
      2. Snappy compression reduces the file to roughly one-fifth of equivalent
         CSV size, which speeds up loading during the embedding step.
      3. Columnar storage means pandas reads only the columns it needs, which
         is faster when embed_papers.py reads only title + abstract.

    Args:
        df:          The completed paper DataFrame.
        output_path: Destination file path (e.g. data/papers.parquet).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")

    size_mb = output_path.stat().st_size / (1024 ** 2)
    log.info(f"Saved → {output_path}  ({size_mb:.1f} MB,  {len(df):,} papers)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full data collection pipeline end-to-end.

    Runs all steps in sequence:
      1. Load API headers (with or without API key).
      2. Iterate over SEARCH_QUERIES, fetching papers for each query.
         Stops issuing new queries once enough raw results have been collected
         to expect TARGET_PAPERS papers after deduplication and filtering.
      3. Deduplicate all collected papers by paperId.
      4. Apply a light trim so the filtering step doesn't over-sample.
      5. Filter out papers with missing or too-short abstracts.
      6. Build the DataFrame, coerce types, and save to Parquet.

    Note: the script overwrites the output file on every run. There is no
    resumption / checkpointing — if interrupted, restart from scratch.
    """
    log.info("=" * 60)
    log.info("  Semantic Scholar fetch pipeline starting")
    log.info(f"  Target: {TARGET_PAPERS:,} papers  |  Queries: {len(SEARCH_QUERIES)}")
    log.info("=" * 60)

    headers = get_headers()
    all_raw = []

    # ── Step 1: Search ────────────────────────────────────────────────────────
    # We need more raw results than TARGET_PAPERS to account for the papers that
    # will be removed by deduplication and filtering. The 0.55 factor assumes
    # roughly 45% of raw results are either duplicates or missing abstracts —
    # a conservative estimate based on typical Semantic Scholar data quality.
    raw_target = int(TARGET_PAPERS / 0.55)

    with tqdm(total=raw_target, desc="Fetching papers", unit="paper",
              bar_format="{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}]") as pbar:

        for i, query in enumerate(SEARCH_QUERIES, start=1):
            if len(all_raw) >= raw_target:
                pbar.set_postfix_str("target reached — stopping early")
                break

            pbar.set_postfix_str(f"query {i}/{len(SEARCH_QUERIES)}: {query}")
            papers = search_papers_for_query(query, headers)
            all_raw.extend(papers)
            pbar.update(min(len(papers), raw_target - (pbar.n or 0)))

            # Brief pause between queries to be respectful to the API.
            time.sleep(REQUEST_DELAY)

    # ── Step 2: Deduplicate ───────────────────────────────────────────────────
    unique_papers = deduplicate_papers(all_raw)

    # ── Step 3: Trim before filtering ────────────────────────────────────────
    # Keep 20% more than the target so we still hit TARGET_PAPERS after the
    # filtering step removes papers with short or missing abstracts.
    trim_limit = int(TARGET_PAPERS * 1.20)
    if len(unique_papers) > trim_limit:
        unique_papers = unique_papers[:trim_limit]
        log.info(f"Pre-filter trim: keeping {len(unique_papers):,} papers for filtering")

    # ── Step 4: Filter ────────────────────────────────────────────────────────
    filtered_papers = filter_papers(unique_papers)

    # Hard-trim to exactly TARGET_PAPERS if we still have a surplus.
    if len(filtered_papers) > TARGET_PAPERS:
        filtered_papers = filtered_papers[:TARGET_PAPERS]
        log.info(f"Final trim to exactly {TARGET_PAPERS:,} papers.")

    # Guard against a completely empty result (network issue, bad API key, etc.)
    if not filtered_papers:
        log.error(
            "No papers survived filtering. "
            "Check your network connection and S2_API_KEY environment variable."
        )
        sys.exit(1)

    if len(filtered_papers) < TARGET_PAPERS:
        log.warning(
            f"Collected {len(filtered_papers):,} papers — below the target of "
            f"{TARGET_PAPERS:,}. Add more entries to SEARCH_QUERIES to increase coverage."
        )

    # ── Step 5: Build DataFrame and save ─────────────────────────────────────
    df = build_dataframe(filtered_papers)
    save_papers(df, OUTPUT_FILE)

    # ── Done ──────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Fetch complete")
    log.info(f"  Papers collected : {len(df):,}")
    log.info(f"  Output file      : {OUTPUT_FILE}")
    log.info(f"  Columns          : {list(df.columns)}")
    log.info("  Next step        : python embeddings/embed_papers.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
