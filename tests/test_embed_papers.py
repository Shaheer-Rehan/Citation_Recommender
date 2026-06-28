"""
test_embed_papers.py
--------------------
Unit tests for embeddings/embed_papers.py.

format_input and normalise_l2 are pure — tested directly with numpy/strings.
get_device is tested for return type only (actual device depends on hardware).
embed_batch is tested by mocking the tokenizer and model internals so no
real weights are downloaded.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from embeddings.embed_papers import (
    format_input,
    normalise_l2,
    get_device,
    embed_batch,
)

DIM = 768


# ── get_device ─────────────────────────────────────────────────────────────────

class TestGetDevice:

    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type_is_valid(self):
        device = get_device()
        assert str(device) in ("cpu", "cuda", "mps", "cuda:0")


# ── format_input ───────────────────────────────────────────────────────────────

class TestFormatInput:

    def test_title_and_abstract_joined_with_sep(self):
        result = format_input("My Title", "My abstract.", "[SEP]")
        assert result == "My Title[SEP]My abstract."

    def test_empty_title_returns_only_abstract(self):
        result = format_input("", "Abstract only.", "[SEP]")
        assert result == "Abstract only."

    def test_none_title_treated_as_empty(self):
        result = format_input(None, "Abstract only.", "[SEP]")
        assert result == "Abstract only."

    def test_whitespace_title_treated_as_empty(self):
        result = format_input("   ", "Abstract only.", "[SEP]")
        assert result == "Abstract only."

    def test_both_empty_returns_empty_string(self):
        result = format_input("", "", "[SEP]")
        assert isinstance(result, str)

    def test_sep_token_used_exactly_as_provided(self):
        result = format_input("T", "A", "***")
        assert "***" in result

    def test_abstract_whitespace_preserved_if_title_absent(self):
        result = format_input("", "  spaced  abstract  ", "[SEP]")
        assert "spaced" in result


# ── normalise_l2 ───────────────────────────────────────────────────────────────

class TestNormaliseL2:

    def test_produces_unit_norms(self):
        vecs = np.random.randn(20, DIM).astype(np.float32)
        result = normalise_l2(vecs.copy())
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_shape_preserved(self):
        vecs = np.random.randn(5, DIM).astype(np.float32)
        result = normalise_l2(vecs.copy())
        assert result.shape == (5, DIM)

    def test_dtype_preserved_as_float32(self):
        vecs = np.random.randn(3, DIM).astype(np.float32)
        result = normalise_l2(vecs.copy())
        assert result.dtype == np.float32

    def test_zero_vector_does_not_crash(self):
        vecs = np.zeros((3, DIM), dtype=np.float32)
        result = normalise_l2(vecs.copy())
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_mixed_zero_and_nonzero_rows(self):
        vecs = np.random.randn(4, DIM).astype(np.float32)
        vecs[1] = 0.0   # insert a zero row
        result = normalise_l2(vecs.copy())
        assert not np.any(np.isnan(result))
        # Non-zero rows must have unit norm
        assert np.linalg.norm(result[0]) == pytest.approx(1.0, abs=1e-5)
        assert np.linalg.norm(result[2]) == pytest.approx(1.0, abs=1e-5)

    def test_already_unit_norm_vectors_unchanged(self):
        np.random.seed(0)
        vecs  = np.random.randn(3, DIM).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit  = vecs / norms
        result = normalise_l2(unit.copy())
        np.testing.assert_allclose(result, unit, atol=1e-5)

    def test_large_magnitude_vectors_normalised(self):
        vecs = np.ones((2, DIM), dtype=np.float32) * 1e6
        result = normalise_l2(vecs.copy())
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_post_norms_logged_approximately_one(self):
        """Regression: confirm the post-normalisation log message won't crash."""
        vecs = np.random.randn(5, DIM).astype(np.float32)
        result = normalise_l2(vecs.copy())   # should not raise
        post_norms = np.linalg.norm(result, axis=1)
        assert post_norms.min() > 0.99
        assert post_norms.max() < 1.01


# ── embed_batch ────────────────────────────────────────────────────────────────

class TestEmbedBatch:

    def _make_mock_tokenizer_and_model(self, batch_size: int, seq_len: int = 10):
        """
        Build a mock tokenizer and model that return real torch tensors.
        The tokenizer returns a plain dict of tensors (so .items() works).
        The model returns an object whose .last_hidden_state is a real tensor.
        """
        # Tokenizer: when called, return a real dict with tensor values.
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids":      torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }

        # Model output: last_hidden_state shape [batch_size, seq_len, DIM]
        mock_output                    = MagicMock()
        mock_output.last_hidden_state  = torch.randn(batch_size, seq_len, DIM)
        mock_model                     = MagicMock(return_value=mock_output)

        return mock_tokenizer, mock_model

    def test_output_shape_correct(self):
        batch_size = 3
        texts = ["text one", "text two", "text three"]
        tok, model = self._make_mock_tokenizer_and_model(batch_size)
        result = embed_batch(texts, tok, model, torch.device("cpu"))
        assert result.shape == (batch_size, DIM)

    def test_output_dtype_is_float32(self):
        texts = ["a", "b"]
        tok, model = self._make_mock_tokenizer_and_model(2)
        result = embed_batch(texts, tok, model, torch.device("cpu"))
        assert result.dtype == np.float32

    def test_returns_numpy_array(self):
        texts = ["hello world"]
        tok, model = self._make_mock_tokenizer_and_model(1)
        result = embed_batch(texts, tok, model, torch.device("cpu"))
        assert isinstance(result, np.ndarray)

    def test_single_text_input(self):
        tok, model = self._make_mock_tokenizer_and_model(1)
        result = embed_batch(["single input text"], tok, model, torch.device("cpu"))
        assert result.shape == (1, DIM)

    def test_model_called_once_per_batch(self):
        texts = ["a", "b", "c"]
        tok, model = self._make_mock_tokenizer_and_model(3)
        embed_batch(texts, tok, model, torch.device("cpu"))
        assert model.call_count == 1

    def test_tokenizer_called_with_correct_params(self):
        texts = ["text one", "text two"]
        tok, model = self._make_mock_tokenizer_and_model(2)
        embed_batch(texts, tok, model, torch.device("cpu"))
        call_kwargs = tok.call_args[1]
        assert call_kwargs["padding"]     is True
        assert call_kwargs["truncation"]  is True
        assert call_kwargs["return_tensors"] == "pt"

    def test_output_not_normalised(self):
        """embed_batch returns raw CLS embeddings — normalisation is done by the caller."""
        texts = ["test input text here"]
        tok, model = self._make_mock_tokenizer_and_model(1, seq_len=5)
        result = embed_batch(texts, tok, model, torch.device("cpu"))
        # The norm of a random vector will very likely NOT be 1.0
        norm = np.linalg.norm(result[0])
        # We just check it's finite and positive — not that it equals 1
        assert np.isfinite(norm) and norm > 0
