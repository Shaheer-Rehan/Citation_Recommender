"""
conftest.py
-----------
Shared pytest fixtures used across all test modules.

All heavy objects (FAISS index, metadata, embeddings) are session-scoped so
they are created once and reused across every test that requests them.
The model and tokenizer are always MagicMocks — no real weights are loaded
during testing.
"""

import sys
import numpy as np
import pytest
import faiss
import torch
from pathlib import Path
from unittest.mock import MagicMock

# Ensure the project root is on sys.path regardless of where pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

N_PAPERS = 10
DIM      = 768   # SPECTER2 output dimension


# ── Raw paper fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_raw_papers():
    """
    10 raw paper dicts mimicking Semantic Scholar API responses.
    Paper i cites up to 3 of the papers before it so the reference graph
    is non-trivial enough to test citation-overlap scoring.
    """
    papers = []
    for i in range(N_PAPERS):
        papers.append({
            "paperId": f"pid_{i:03d}",
            "title": f"Deep Learning Approach {i} for Natural Language Processing",
            "abstract": (
                f"In this work we present novel method {i} for natural language "
                f"processing tasks using deep neural networks. Our approach achieves "
                f"state of the art results on multiple benchmark datasets. "
                f"Extensive experiments validate the effectiveness of our technique."
            ),
            "year": 2018 + (i % 6),
            "citationCount": i * 15,
            "fieldsOfStudy": [
                {"category": "Computer Science", "source": "external"}
            ],
            "references": [
                {"paperId": f"pid_{j:03d}", "title": f"Prior Work {j}"}
                for j in range(max(0, i - 3), i)
            ],
            "externalIds": {"ArXiv": f"2301.{i:05d}"} if i % 2 == 0 else {},
        })
    return papers


@pytest.fixture(scope="session")
def sample_paper_ids():
    return [f"pid_{i:03d}" for i in range(N_PAPERS)]


# ── Embedding fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_embeddings():
    """10 L2-normalised float32 vectors of dim 768 (unit-norm, as stored in index)."""
    np.random.seed(42)
    vecs  = np.random.randn(N_PAPERS, DIM).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ── Metadata fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_metadata(sample_raw_papers):
    """
    Normalised metadata list as produced by build_index.build_metadata_list().
    Position i corresponds to sample_paper_ids[i] and sample_embeddings[i].
    """
    result = []
    for i, p in enumerate(sample_raw_papers):
        result.append({
            "paper_id":        p["paperId"],
            "title":           p["title"],
            "abstract": (
                f"Abstract {i}: This work proposes deep learning method {i} "
                f"applied to natural language processing. We compare against "
                f"competitive baselines and achieve new performance records."
            ),
            "year":            p["year"],
            "citation_count":  p["citationCount"],
            "fields_of_study": ["Computer Science"],
            "references":      [f"pid_{j:03d}" for j in range(max(0, i - 3), i)],
            "arxiv_id":        f"2301.{i:05d}" if i % 2 == 0 else "",
        })
    return result


# ── FAISS fixture ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_faiss_index(sample_embeddings):
    """Real FAISS IndexFlatIP populated with sample_embeddings (10 unit-norm vectors)."""
    index = faiss.IndexFlatIP(DIM)
    index.add(sample_embeddings)
    return index


# ── Resources fixture ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_resources(small_faiss_index, sample_metadata):
    """
    Minimal resources dict mimicking retrieve.load_resources() output.
    Model and tokenizer are MagicMocks — no real weights are loaded.
    """
    mock_tokenizer           = MagicMock()
    mock_tokenizer.sep_token = "[SEP]"
    mock_model               = MagicMock()

    return {
        "index":     small_faiss_index,
        "metadata":  sample_metadata,
        "tokenizer": mock_tokenizer,
        "model":     mock_model,
        "device":    torch.device("cpu"),
        "sep_token": "[SEP]",
    }


# ── Sentence model fixture ─────────────────────────────────────────────────────

@pytest.fixture
def mock_sentence_model():
    """
    MagicMock that mimics SentenceTransformer.encode().
    Returns L2-normalised random 384-dim float32 vectors — deterministic via seed.
    """
    model = MagicMock()

    def fake_encode(texts, normalize_embeddings=True, show_progress_bar=False):
        np.random.seed(0)
        vecs  = np.random.randn(len(texts), 384).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs  = vecs / np.maximum(norms, 1e-10)
        return vecs

    model.encode.side_effect = fake_encode
    return model


# ── Candidate list fixture ─────────────────────────────────────────────────────

@pytest.fixture
def sample_candidates(sample_metadata):
    """5 candidate dicts as returned by retrieve.search_index() with a 'score' field."""
    return [
        {**sample_metadata[i], "score": round(0.9 - i * 0.1, 2)}
        for i in range(5)
    ]
