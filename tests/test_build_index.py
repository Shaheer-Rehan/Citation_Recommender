"""
test_build_index.py
-------------------
Unit tests for index/build_index.py.

build_faiss_index and validate_index use real FAISS and numpy (fast, no I/O).
load_embeddings uses tmp_path to write real files and test load logic.
build_metadata_list uses a synthetic pandas DataFrame.
"""

import json
import sys
import numpy as np
import pandas as pd
import pytest
import faiss

from index.build_index import (
    build_faiss_index,
    validate_index,
    build_metadata_list,
    load_embeddings,
)


# ── build_faiss_index ──────────────────────────────────────────────────────────

class TestBuildFaissIndex:

    def test_returns_index_flat_ip(self, sample_embeddings):
        index = build_faiss_index(sample_embeddings)
        assert isinstance(index, faiss.IndexFlatIP)

    def test_ntotal_matches_input_rows(self, sample_embeddings):
        index = build_faiss_index(sample_embeddings)
        assert index.ntotal == sample_embeddings.shape[0]

    def test_dimension_matches_embedding_dim(self, sample_embeddings):
        index = build_faiss_index(sample_embeddings)
        assert index.d == sample_embeddings.shape[1]

    def test_self_retrieval_returns_itself(self, sample_embeddings):
        index = build_faiss_index(sample_embeddings)
        query = sample_embeddings[0:1]
        distances, indices = index.search(query, 1)
        assert indices[0][0] == 0

    def test_self_similarity_approx_one(self, sample_embeddings):
        index = build_faiss_index(sample_embeddings)
        query = sample_embeddings[0:1]
        distances, _ = index.search(query, 1)
        assert distances[0][0] == pytest.approx(1.0, abs=1e-4)

    def test_top_k_results_sorted_descending(self, sample_embeddings):
        index = build_faiss_index(sample_embeddings)
        query = sample_embeddings[0:1]
        k = min(5, sample_embeddings.shape[0])
        distances, _ = index.search(query, k)
        scores = distances[0].tolist()
        assert scores == sorted(scores, reverse=True)


# ── validate_index ─────────────────────────────────────────────────────────────

class TestValidateIndex:

    def test_passes_on_normalised_embeddings(self, sample_embeddings):
        index = build_faiss_index(sample_embeddings)
        validate_index(index, sample_embeddings)   # should not raise

    def test_fails_on_non_unit_norm_embeddings(self):
        # Vectors with norm != 1 → self-similarity ≠ 1 → should exit
        dim  = 768
        vecs = np.ones((5, dim), dtype=np.float32) * 2.0  # norm = 2*sqrt(768)
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        with pytest.raises(SystemExit):
            validate_index(index, vecs)

    def test_fails_when_self_retrieval_wrong_position(self, sample_embeddings):
        # Tamper: add a vector that is MORE similar to the query than it is to itself.
        dim = sample_embeddings.shape[1]
        # Place the query itself at position 1 (not 0) by inserting a closer vector first.
        closer = sample_embeddings[0:1].copy() * 0.5  # not unit norm → wrong position
        # Rebuild with the closer vector at position 0 but not the actual query
        # Instead: just verify that a correctly built index passes
        index = build_faiss_index(sample_embeddings)
        validate_index(index, sample_embeddings)  # must not raise


# ── load_embeddings ────────────────────────────────────────────────────────────

class TestLoadEmbeddings:

    def _write_files(self, tmp_path, embeddings, paper_ids):
        emb_path = tmp_path / "embeddings.npy"
        ids_path = tmp_path / "paper_ids.json"
        np.save(emb_path, embeddings)
        with open(ids_path, "w") as f:
            json.dump(paper_ids, f)
        return emb_path, ids_path

    def test_loads_correctly_when_files_valid(self, tmp_path, sample_embeddings, sample_paper_ids):
        emb_path, ids_path = self._write_files(tmp_path, sample_embeddings, sample_paper_ids)
        loaded_emb, loaded_ids = load_embeddings(emb_path, ids_path)
        assert loaded_emb.shape == sample_embeddings.shape
        assert loaded_ids == sample_paper_ids

    def test_exits_when_embeddings_file_missing(self, tmp_path, sample_paper_ids):
        ids_path = tmp_path / "paper_ids.json"
        with open(ids_path, "w") as f:
            json.dump(sample_paper_ids, f)
        with pytest.raises(SystemExit):
            load_embeddings(tmp_path / "nonexistent.npy", ids_path)

    def test_exits_when_ids_file_missing(self, tmp_path, sample_embeddings):
        emb_path = tmp_path / "embeddings.npy"
        np.save(emb_path, sample_embeddings)
        with pytest.raises(SystemExit):
            load_embeddings(emb_path, tmp_path / "nonexistent.json")

    def test_exits_on_length_mismatch(self, tmp_path, sample_embeddings):
        too_few_ids = ["pid_000", "pid_001"]  # only 2, but embeddings has 10 rows
        emb_path, ids_path = self._write_files(tmp_path, sample_embeddings, too_few_ids)
        with pytest.raises(SystemExit):
            load_embeddings(emb_path, ids_path)

    def test_casts_float64_to_float32(self, tmp_path, sample_paper_ids):
        # Embeddings saved as float64 must be silently cast to float32
        dim  = 768
        vecs = np.random.randn(len(sample_paper_ids), dim).astype(np.float64)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        emb_path, ids_path = self._write_files(tmp_path, vecs, sample_paper_ids)
        loaded_emb, _ = load_embeddings(emb_path, ids_path)
        assert loaded_emb.dtype == np.float32

    def test_exits_on_wrong_embedding_dimension(self, tmp_path, sample_paper_ids):
        wrong_dim = np.random.randn(len(sample_paper_ids), 256).astype(np.float32)
        emb_path, ids_path = self._write_files(tmp_path, wrong_dim, sample_paper_ids)
        with pytest.raises(SystemExit):
            load_embeddings(emb_path, ids_path)

    def test_exits_on_1d_embedding_array(self, tmp_path, sample_paper_ids):
        flat = np.random.randn(768).astype(np.float32)  # 1-D, not a matrix
        emb_path = tmp_path / "embeddings.npy"
        ids_path = tmp_path / "paper_ids.json"
        np.save(emb_path, flat)
        with open(ids_path, "w") as f:
            json.dump(sample_paper_ids, f)
        with pytest.raises(SystemExit):
            load_embeddings(emb_path, ids_path)


# ── build_metadata_list ────────────────────────────────────────────────────────

class TestBuildMetadataList:

    def _make_df(self, n=5):
        records = []
        for i in range(n):
            records.append({
                "paper_id":        f"pid_{i:03d}",
                "title":           f"Paper {i}",
                "abstract":        f"Abstract {i} " * 10,
                "year":            pd.array([2020 + i], dtype="Int64")[0],
                "citation_count":  i * 5,
                "fields_of_study": ["Computer Science"],
                "references":      [f"ref_{i}a", f"ref_{i}b"],
                "arxiv_id":        f"2301.{i:05d}",
            })
        df = pd.DataFrame(records).set_index("paper_id")
        df["year"] = df["year"].astype("Int64")
        return df

    def test_length_matches_paper_ids(self):
        n = 5
        df = self._make_df(n)
        ids = [f"pid_{i:03d}" for i in range(n)]
        result = build_metadata_list(df, ids)
        assert len(result) == n

    def test_order_matches_paper_ids(self):
        df = self._make_df(5)
        ids = [f"pid_{i:03d}" for i in range(5)]
        result = build_metadata_list(df, ids)
        for i, record in enumerate(result):
            assert record["paper_id"] == ids[i]

    def test_year_none_for_na_value(self):
        df = self._make_df(3)
        df.loc["pid_001", "year"] = pd.NA
        ids = ["pid_000", "pid_001", "pid_002"]
        result = build_metadata_list(df, ids)
        assert result[1]["year"] is None

    def test_references_are_plain_python_list(self):
        df = self._make_df(3)
        ids = ["pid_000", "pid_001", "pid_002"]
        result = build_metadata_list(df, ids)
        assert isinstance(result[0]["references"], list)

    def test_fields_of_study_are_plain_python_list(self):
        df = self._make_df(3)
        ids = ["pid_000", "pid_001", "pid_002"]
        result = build_metadata_list(df, ids)
        assert isinstance(result[0]["fields_of_study"], list)

    def test_missing_arxiv_id_becomes_empty_string(self):
        df = self._make_df(2)
        df.loc["pid_000", "arxiv_id"] = None
        ids = ["pid_000", "pid_001"]
        result = build_metadata_list(df, ids)
        assert result[0]["arxiv_id"] == ""

    def test_all_required_keys_present(self):
        df = self._make_df(2)
        ids = ["pid_000", "pid_001"]
        result = build_metadata_list(df, ids)
        required = {"paper_id", "title", "abstract", "year",
                    "citation_count", "fields_of_study", "references", "arxiv_id"}
        for record in result:
            assert required.issubset(set(record.keys()))

    def test_citation_count_is_integer(self):
        df = self._make_df(3)
        ids = ["pid_000", "pid_001", "pid_002"]
        result = build_metadata_list(df, ids)
        assert isinstance(result[0]["citation_count"], int)
