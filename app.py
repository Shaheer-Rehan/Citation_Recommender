"""
app.py
------
Streamlit demo for the Citation Recommender.

Run from the project root with:
  .\\venv\\Scripts\\python.exe -m streamlit run app.py

Two query modes:

  Single Paper (Tab 1)
    Paste a title and abstract, or enter an ArXiv ID to auto-fetch them.
    The app embeds the paper with SPECTER2, queries the FAISS index, re-ranks
    using citation overlap, and displays the top results with explanations.

  Reading Catalogue (Tab 2)
    Enter paper titles you have already read (one per line). The app looks
    each title up in the corpus, embeds all found papers, mean-pools their
    embeddings into a taste profile vector, and recommends papers in the same
    research neighbourhood — excluding what you have already read.

Sidebar controls let the user tune:
  - Number of results (5–20)
  - Alpha (semantic vs citation-overlap weight)
  - Minimum publication year
  - Minimum citation count

All heavy assets (FAISS index, metadata, SPECTER2 model, sentence model) are
loaded once at startup and cached by @st.cache_resource so repeated queries
are fast (< 1 second for search + rerank; a few seconds for explanation).
"""

import re
import sys
import logging
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
# Ensures `from retrieval.xxx import ...` resolves when Streamlit runs this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from retrieval.retrieve import (
    load_resources,
    embed_single_paper,
    embed_catalogue,
    search_index,
)
from retrieval.rerank   import rerank
from retrieval.explain  import load_sentence_model, explain_results

log = logging.getLogger(__name__)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Citation Recommender",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Resource loading ───────────────────────────────────────────────────────────
# Both functions are wrapped in @st.cache_resource so the model weights, FAISS
# index, and metadata are loaded exactly once per session and held in memory.

@st.cache_resource(show_spinner="Loading FAISS index and SPECTER2 model…")
def get_resources() -> dict:
    """Load FAISS index, metadata, and SPECTER2 model. Cached for the session."""
    return load_resources()


@st.cache_resource(show_spinner="Loading sentence model…")
def get_sentence_model():
    """Load all-MiniLM-L6-v2 for explanation generation. Cached for the session."""
    return load_sentence_model()


@st.cache_data(show_spinner=False)
def get_title_lookup(_resources: dict) -> dict:
    """
    Build a case-insensitive title → metadata dict for catalogue lookup.

    The leading underscore on _resources tells Streamlit not to hash the dict
    (it contains non-serialisable FAISS objects). The lookup is derived purely
    from the metadata list which is stable for the session.
    """
    return {m["title"].lower().strip(): m for m in _resources["metadata"]}


# ── Utility helpers ────────────────────────────────────────────────────────────

def clean_arxiv_id(raw: str) -> str:
    """
    Normalise an ArXiv ID or URL to the bare ID string.

    Accepts:
      - Plain IDs:          "2310.06825"
      - Abstract URLs:      "https://arxiv.org/abs/2310.06825"
      - PDF URLs:           "https://arxiv.org/pdf/2310.06825"
      - IDs with version:   "2310.06825v2"

    Returns the bare ID (e.g. "2310.06825v2") ready for the arxiv library.
    """
    raw = raw.strip()
    # Strip URL prefixes such as https://arxiv.org/abs/ or https://arxiv.org/pdf/
    raw = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", raw)
    # Remove trailing .pdf
    raw = raw.replace(".pdf", "")
    return raw.strip()


def fetch_arxiv_paper(arxiv_id: str) -> dict | None:
    """
    Fetch paper title and abstract from the ArXiv API.

    Uses the `arxiv` Python library (already in requirements.txt). Returns a
    dict with 'title' and 'abstract' on success, or None on failure.

    Args:
        arxiv_id: Cleaned ArXiv paper ID (e.g. "2310.06825").

    Returns:
        Dict with 'title' and 'abstract', or None if the paper was not found.
    """
    try:
        import arxiv
        client  = arxiv.Client()
        results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
        if not results:
            return None
        paper = results[0]
        return {"title": paper.title, "abstract": paper.summary}
    except Exception as exc:
        log.warning(f"ArXiv fetch failed for id={arxiv_id}: {exc}")
        return None


def apply_filters(candidates: list, min_year: int, min_citations: int) -> list:
    """
    Filter candidate results by publication year and citation count.

    Papers without a known publication year are kept (year=None is not
    treated as a disqualifier — common for recent preprints).

    Args:
        candidates:    List of candidate dicts from search_index().
        min_year:      Discard papers published before this year.
        min_citations: Discard papers with fewer than this many citations.

    Returns:
        Filtered list preserving original ordering.
    """
    filtered = []
    for c in candidates:
        year  = c.get("year")
        cites = c.get("citation_count", 0)
        if year is not None and year < min_year:
            continue
        if cites < min_citations:
            continue
        filtered.append(c)
    return filtered


def make_paper_url(result: dict) -> str:
    """
    Build the best available URL for a result card.

    Prefers an ArXiv link (more accessible) over Semantic Scholar when the
    paper has an ArXiv ID. Falls back to the Semantic Scholar paper page.

    Args:
        result: A result dict containing 'paper_id' and optionally 'arxiv_id'.

    Returns:
        URL string.
    """
    arxiv_id = (result.get("arxiv_id") or "").strip()
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return f"https://www.semanticscholar.org/paper/{result['paper_id']}"


# ── Display components ─────────────────────────────────────────────────────────

def display_result_card(result: dict, rank: int) -> None:
    """
    Render a single recommendation as a bordered Streamlit card.

    Shows:
      - Rank and hybrid score
      - Title linked to ArXiv or Semantic Scholar
      - Year, citation count, and fields of study
      - Explanation in an expandable section
      - Score breakdown (semantic / citation overlap / hybrid)

    Args:
        result: A fully enriched result dict (after rerank + explain).
        rank:   1-based position in the ranked list.
    """
    with st.container(border=True):
        top_col, score_col = st.columns([0.82, 0.18])

        with top_col:
            url   = make_paper_url(result)
            title = result.get("title", "Untitled")
            st.markdown(f"**{rank}.** &nbsp; [{title}]({url})")

        with score_col:
            # Highlight the hybrid score prominently.
            st.metric(
                label="Score",
                value=f"{result.get('hybrid_score', result.get('score', 0)):.3f}",
            )

        # Metadata row: year, citations, fields.
        year   = result.get("year") or "n/a"
        cites  = f"{result.get('citation_count', 0):,}"
        fields = result.get("fields_of_study", [])
        field_str = ", ".join(fields[:2]) if fields else "n/a"

        st.caption(f"📅 {year}  ·  📚 {cites} citations  ·  🏷️ {field_str}")

        # Explanation — in an expander to keep cards compact by default.
        explanation = result.get("explanation", "")
        if explanation:
            with st.expander("Why recommended?"):
                st.info(explanation)

        # Fine-grained score breakdown for transparency.
        sem   = result.get("score", 0)
        cite  = result.get("citation_overlap", 0)
        hyb   = result.get("hybrid_score", sem)
        st.caption(
            f"Semantic similarity: **{sem:.3f}**  ·  "
            f"Citation overlap: **{cite:.3f}**  ·  "
            f"Hybrid score: **{hyb:.3f}**"
        )


def display_results(results: list) -> None:
    """
    Render the full ranked list of result cards.

    Args:
        results: List of enriched result dicts, ordered by hybrid_score desc.
    """
    if not results:
        st.warning("No results found. Try relaxing the year or citation filters.")
        return

    st.markdown(f"**{len(results)} recommendation{'s' if len(results) != 1 else ''}**")
    for rank, result in enumerate(results, start=1):
        display_result_card(result, rank)


# ── Core search pipeline ───────────────────────────────────────────────────────

def run_search(
    query_vec:    object,         # np.ndarray [768]
    query_refs:   list,
    query_abstract: str,
    resources:    dict,
    sent_model:   object,
    n_results:    int,
    alpha:        float,
    min_year:     int,
    min_citations: int,
    exclude_ids:  list = None,
) -> list:
    """
    Execute the full retrieval → filter → rerank → explain pipeline.

    This function is called by both query modes (single paper and catalogue)
    with the appropriate pre-computed query vector and query references.

    Args:
        query_vec:      L2-normalised SPECTER2 query embedding, shape [768].
        query_refs:     Reference paper IDs from the query paper (for citation
                        overlap scoring). Empty list if unknown.
        query_abstract: Abstract of the query paper (for explanation generation).
        resources:      Dict from load_resources().
        sent_model:     SentenceTransformer for explanation generation.
        n_results:      Number of results to return after all filtering.
        alpha:          Semantic vs citation weight for hybrid scoring.
        min_year:       Filter: discard papers before this year.
        min_citations:  Filter: discard papers below this citation count.
        exclude_ids:    Paper IDs to exclude from results (e.g. already-read papers).

    Returns:
        List of enriched result dicts with 'explanation' and 'hybrid_score' fields.
    """
    # Retrieve top-100 candidates from FAISS — search broadly before filtering.
    candidates = search_index(query_vec, resources, k=100, exclude_ids=exclude_ids)

    # Apply sidebar filters before reranking so we don't waste the rerank pass
    # on papers the user has already asked to exclude by year or citation count.
    candidates = apply_filters(candidates, min_year, min_citations)

    if not candidates:
        return []

    # Rerank with hybrid scoring and trim to n_results.
    ranked = rerank(candidates, query_refs=query_refs, alpha=alpha, top_k=n_results)

    # Generate natural-language explanations for each result.
    explain_results(ranked, query_abstract, sent_model)

    return ranked


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render the settings sidebar and return the user's chosen parameter values.

    Returns:
        Dict with keys: n_results, alpha, min_year, min_citations.
    """
    with st.sidebar:
        st.header("Settings")

        n_results = st.slider(
            "Number of results",
            min_value=5, max_value=20, value=10, step=1,
        )

        alpha = st.slider(
            "Semantic ↔ Citation weight (α)",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help=(
                "Higher α → more weight on SPECTER2 semantic similarity.\n"
                "Lower α → more weight on citation graph overlap.\n"
                "At α=1.0, citation overlap is ignored entirely."
            ),
        )

        st.divider()
        st.subheader("Filters")

        min_year = st.slider(
            "Minimum publication year",
            min_value=2000, max_value=2025, value=2010, step=1,
        )

        min_citations = st.number_input(
            "Minimum citations",
            min_value=0, max_value=10_000, value=0, step=10,
        )

        st.divider()
        st.caption(
            "**Model:** allenai/specter2_base  \n"
            "**Index:** FAISS IndexFlatIP  \n"
            "**Explanations:** all-MiniLM-L6-v2"
        )

    return {
        "n_results":     n_results,
        "alpha":         alpha,
        "min_year":      min_year,
        "min_citations": min_citations,
    }


# ── Tab 1: Single paper ────────────────────────────────────────────────────────

def render_single_paper_tab(resources: dict, sent_model, settings: dict) -> None:
    """
    Render the Single Paper query tab.

    Allows the user to:
      - Enter an ArXiv ID to auto-fetch the title and abstract.
      - Manually paste a title and abstract.
      - Click "Find Similar Papers" to run the pipeline.

    Args:
        resources:  Dict from load_resources().
        sent_model: SentenceTransformer for explanation generation.
        settings:   Sidebar settings dict.
    """
    st.subheader("Find papers similar to a single paper")
    st.caption(
        "Paste a title and abstract below, or enter an ArXiv ID to auto-fetch them."
    )

    # ── ArXiv fetch ───────────────────────────────────────────────────────────
    arxiv_col, btn_col = st.columns([0.75, 0.25])
    with arxiv_col:
        arxiv_input = st.text_input(
            "ArXiv ID or URL (optional)",
            placeholder="e.g. 2310.06825 or https://arxiv.org/abs/2310.06825",
            key="arxiv_input",
        )
    with btn_col:
        st.write("")   # vertical alignment spacer
        st.write("")
        fetch_clicked = st.button("Fetch from ArXiv", key="fetch_btn")

    # Store fetched content in session state so it persists across reruns.
    if "prefill_title" not in st.session_state:
        st.session_state.prefill_title    = ""
        st.session_state.prefill_abstract = ""

    if fetch_clicked and arxiv_input.strip():
        with st.spinner("Fetching from ArXiv…"):
            clean_id = clean_arxiv_id(arxiv_input)
            paper    = fetch_arxiv_paper(clean_id)
        if paper:
            st.session_state.prefill_title    = paper["title"]
            st.session_state.prefill_abstract = paper["abstract"]
            st.success(f"Fetched: {paper['title'][:80]}")
        else:
            st.error(f"Could not fetch ArXiv paper '{clean_id}'. Check the ID and try again.")

    # ── Title and abstract inputs ─────────────────────────────────────────────
    title = st.text_input(
        "Paper title",
        value=st.session_state.prefill_title,
        placeholder="e.g. Attention Is All You Need",
    )
    abstract = st.text_area(
        "Abstract",
        value=st.session_state.prefill_abstract,
        height=200,
        placeholder="Paste the paper's abstract here…",
    )

    # ── Search ────────────────────────────────────────────────────────────────
    if st.button("Find Similar Papers", type="primary", key="single_search_btn"):
        if not abstract.strip():
            st.error("Please provide an abstract before searching.")
            return

        with st.spinner("Embedding and searching…"):
            query_vec = embed_single_paper(title, abstract, resources)
            results   = run_search(
                query_vec       = query_vec,
                query_refs      = [],        # no known references for pasted text
                query_abstract  = abstract,
                resources       = resources,
                sent_model      = sent_model,
                n_results       = settings["n_results"],
                alpha           = settings["alpha"],
                min_year        = settings["min_year"],
                min_citations   = settings["min_citations"],
            )

        st.divider()
        display_results(results)


# ── Tab 2: Reading catalogue ───────────────────────────────────────────────────

def render_catalogue_tab(resources: dict, sent_model, settings: dict) -> None:
    """
    Render the Reading Catalogue query tab.

    The user pastes paper titles they have already read (one per line). The app
    looks up each title in the corpus metadata, embeds all found (and unfound)
    papers, mean-pools them into a taste profile vector, and recommends new
    papers in that region of embedding space.

    Papers found in the corpus also contribute their reference lists to the
    citation-overlap scoring, making the recommendations more structurally
    precise for catalogues of well-known papers.

    Args:
        resources:  Dict from load_resources().
        sent_model: SentenceTransformer for explanation generation.
        settings:   Sidebar settings dict.
    """
    st.subheader("Get recommendations from your reading history")
    st.caption(
        "Enter the titles of papers you have already read — one per line. "
        "The app builds a taste profile from your reading history and finds new papers in the same research neighbourhood."
    )

    catalogue_input = st.text_area(
        "Your reading list (one title per line)",
        height=200,
        placeholder=(
            "Attention Is All You Need\n"
            "BERT: Pre-training of Deep Bidirectional Transformers\n"
            "GPT-3: Language Models are Few-Shot Learners"
        ),
        key="catalogue_input",
    )

    if not st.button("Get Recommendations", type="primary", key="catalogue_search_btn"):
        return

    raw_titles = [t.strip() for t in catalogue_input.strip().splitlines() if t.strip()]
    if not raw_titles:
        st.error("Please enter at least one paper title.")
        return

    # ── Title lookup ──────────────────────────────────────────────────────────
    title_lookup = get_title_lookup(resources)

    found_papers   = []   # papers matched in our corpus (full metadata)
    unknown_papers = []   # titles not in corpus (embed title only)
    found_titles   = []
    unknown_titles = []

    for raw_title in raw_titles:
        match = title_lookup.get(raw_title.lower())
        if match:
            found_papers.append(match)
            found_titles.append(raw_title)
        else:
            # Not in corpus — we can still embed the title and include it in
            # the mean pool, but it won't contribute references for reranking.
            unknown_papers.append({"title": raw_title, "abstract": ""})
            unknown_titles.append(raw_title)

    # Report match status before running the search.
    if found_titles:
        st.success(f"Found {len(found_titles)} paper(s) in corpus: {', '.join(f'*{t}*' for t in found_titles[:3])}{'…' if len(found_titles) > 3 else ''}")
    if unknown_titles:
        st.warning(
            f"{len(unknown_titles)} title(s) not found in corpus (will embed title text only, "
            f"no citation signals): {', '.join(unknown_titles[:3])}{'…' if len(unknown_titles) > 3 else ''}"
        )

    # ── Build query ───────────────────────────────────────────────────────────
    all_papers = found_papers + unknown_papers
    if not all_papers:
        st.error("No papers to embed. Please check your titles.")
        return

    # Collect all references from found papers for citation overlap scoring.
    query_refs = list({
        ref
        for p in found_papers
        for ref in p.get("references", [])
    })

    # For the explanation, use the abstract of the first found paper as a
    # representative "query abstract". Falls back to empty string if no
    # found papers (explanation will use the generic fallback).
    query_abstract = found_papers[0]["abstract"] if found_papers else ""

    # IDs of papers already read — excluded from recommendations.
    exclude_ids = [p["paper_id"] for p in found_papers]

    with st.spinner(f"Embedding {len(all_papers)} paper(s) and searching…"):
        query_vec = embed_catalogue(all_papers, resources)
        results   = run_search(
            query_vec       = query_vec,
            query_refs      = query_refs,
            query_abstract  = query_abstract,
            resources       = resources,
            sent_model      = sent_model,
            n_results       = settings["n_results"],
            alpha           = settings["alpha"],
            min_year        = settings["min_year"],
            min_citations   = settings["min_citations"],
            exclude_ids     = exclude_ids,
        )

    st.divider()
    display_results(results)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point — renders the full Streamlit application.

    Layout:
      - Sidebar: settings controls
      - Header: title and description
      - Two tabs: Single Paper | Reading Catalogue
    """
    # Render sidebar first (controls are needed before building tabs).
    settings = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("Citation Recommender")
    st.markdown(
        "Semantic paper recommendation powered by **SPECTER2** citation-aware "
        "embeddings, **FAISS** vector search, and hybrid citation-overlap "
        "re-ranking."
    )
    st.divider()

    # ── Load assets ───────────────────────────────────────────────────────────
    # These calls hit the cache on every rerun — actual loading happens once.
    resources  = get_resources()
    sent_model = get_sentence_model()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_single, tab_catalogue = st.tabs(["Single Paper", "Reading Catalogue"])

    with tab_single:
        render_single_paper_tab(resources, sent_model, settings)

    with tab_catalogue:
        render_catalogue_tab(resources, sent_model, settings)


if __name__ == "__main__":
    main()
