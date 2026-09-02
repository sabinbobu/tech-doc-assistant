"""
Tests for cross-encoder reranking.

KEY CONCEPT — MOCKING THE MODEL:
Loading a real cross-encoder downloads a model file and runs actual neural
network inference — slow, non-deterministic in timing, and unnecessary to
prove the reranking LOGIC is correct. We mock _get_model() to return a fake
scorer instead, same pattern as tests/test_embedding/test_vector_store.py
mocking the ChromaDB collection.
"""

from unittest.mock import MagicMock, patch

from src.retrieval import rerank as rerank_module
from src.retrieval.rerank import rerank


def _candidate(text: str, source_filename: str = "doc.pdf") -> dict:
    return {
        "text": text,
        "metadata": {"source_filename": source_filename, "citation": f"{source_filename}, page 1"},
        "distance": 0.5,
    }


class TestRerank:
    def setup_method(self):
        """Reset the module-level model cache between tests — otherwise a
        model loaded (or a load failure) in one test leaks into the next."""
        rerank_module._model = None
        rerank_module._model_load_failed = False

    def test_reorders_by_score_descending(self):
        candidates = [_candidate("low relevance"), _candidate("high relevance")]
        fake_model = MagicMock()
        # Scores intentionally in the OPPOSITE order of the input list, so a
        # passing test proves reordering actually happened.
        fake_model.predict.return_value = [0.1, 0.9]

        with patch.object(rerank_module, "_get_model", return_value=fake_model):
            result = rerank("query", candidates, top_k=2)

        assert result[0]["text"] == "high relevance"
        assert result[1]["text"] == "low relevance"

    def test_trims_to_top_k(self):
        candidates = [_candidate(f"passage {i}") for i in range(5)]
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.1, 0.2, 0.9, 0.3, 0.5]

        with patch.object(rerank_module, "_get_model", return_value=fake_model):
            result = rerank("query", candidates, top_k=2)

        assert len(result) == 2
        assert result[0]["text"] == "passage 2"  # score 0.9
        assert result[1]["text"] == "passage 4"  # score 0.5

    def test_attaches_rerank_score(self):
        candidates = [_candidate("only passage")]
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.77]

        with patch.object(rerank_module, "_get_model", return_value=fake_model):
            result = rerank("query", candidates, top_k=1)

        assert result[0]["rerank_score"] == 0.77

    def test_empty_candidates_returns_empty(self):
        result = rerank("query", [], top_k=5)
        assert result == []

    def test_model_load_failure_falls_back_to_input_order(self):
        """If the cross-encoder can't load, retrieval must still work —
        return the hybrid-search order unchanged, don't raise."""
        candidates = [_candidate("first"), _candidate("second"), _candidate("third")]

        with patch.object(rerank_module, "_get_model", return_value=None):
            result = rerank("query", candidates, top_k=2)

        assert result == candidates[:2]

    def test_inference_failure_falls_back_to_input_order(self):
        candidates = [_candidate("first"), _candidate("second")]
        fake_model = MagicMock()
        fake_model.predict.side_effect = RuntimeError("out of memory")

        with patch.object(rerank_module, "_get_model", return_value=fake_model):
            result = rerank("query", candidates, top_k=2)

        assert result == candidates[:2]

    def test_reranking_disabled_returns_input_order(self):
        candidates = [_candidate("first"), _candidate("second"), _candidate("third")]
        fake_model = MagicMock()
        fake_model.predict.return_value = [0.9, 0.1, 0.5]  # would reorder if called

        with (
            patch.object(rerank_module.settings, "reranking_enabled", False),
            patch.object(rerank_module, "_get_model", return_value=fake_model),
        ):
            result = rerank("query", candidates, top_k=2)

        assert result == candidates[:2]
        fake_model.predict.assert_not_called()


class TestGetModel:
    def setup_method(self):
        rerank_module._model = None
        rerank_module._model_load_failed = False

    def test_caches_model_across_calls(self):
        """The model should only be constructed once per process."""
        fake_model = MagicMock()
        with patch(
            "sentence_transformers.CrossEncoder", return_value=fake_model
        ) as mock_cross_encoder:
            first = rerank_module._get_model()
            second = rerank_module._get_model()

        assert first is second is fake_model
        mock_cross_encoder.assert_called_once()

    def test_load_failure_is_cached_not_retried(self):
        """A failed load shouldn't retry on every query — that would mean
        every single query pays the failed-import cost."""
        with patch(
            "sentence_transformers.CrossEncoder", side_effect=OSError("no network")
        ) as mock_cross_encoder:
            first = rerank_module._get_model()
            second = rerank_module._get_model()

        assert first is None
        assert second is None
        mock_cross_encoder.assert_called_once()


class TestWarmUp:
    """
    warm_up() moves the ~12.4s model load to process startup. Its whole value
    is that a long-lived process (Streamlit, MCP server) calls it before
    serving anyone — which means it runs at the least convenient possible
    moment to raise, and must not.
    """

    def setup_method(self):
        rerank_module._model = None
        rerank_module._model_load_failed = False

    def test_returns_true_when_model_loads(self):
        with patch.object(rerank_module, "_get_model", return_value=MagicMock()):
            assert rerank_module.warm_up() is True

    def test_returns_false_without_raising_when_load_fails(self):
        """A failed warm-up degrades to hybrid search order — it must never
        take down the app that called it at startup."""
        with patch.object(rerank_module, "_get_model", return_value=None):
            assert rerank_module.warm_up() is False

    def test_returns_false_when_reranking_disabled(self):
        """Nothing to warm if the kill switch is off — and warming anyway
        would pay the load cost for a model that's never used."""
        with patch.object(rerank_module.settings, "reranking_enabled", False):
            with patch.object(rerank_module, "_get_model") as mock_get_model:
                assert rerank_module.warm_up() is False
            mock_get_model.assert_not_called()

    def test_actually_loads_the_model_once(self):
        """warm_up() must do the loading itself — if it deferred, the first
        query would still pay for it and the fix would be a no-op."""
        fake_model = MagicMock()
        with patch(
            "sentence_transformers.CrossEncoder", return_value=fake_model
        ) as mock_cross_encoder:
            assert rerank_module.warm_up() is True
            assert rerank_module.warm_up() is True

        assert rerank_module._model is fake_model
        mock_cross_encoder.assert_called_once()
