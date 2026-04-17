"""
Tests for the embedding module.

KEY CONCEPT — MOCKING:
We do NOT call the real OpenAI API in unit tests. Reasons:
  1. Costs money on every test run
  2. Slow (network latency)
  3. Tests should be deterministic — network failures shouldn't fail your tests
  4. BMW will expect you to know this pattern

Instead we use unittest.mock to replace the real ChromaDB collection
with a fake one that behaves predictably.

ANALOGY: In embedded HIL testing, you don't connect a real engine
to test your ECU software — you use a simulator that produces
known signals. Mocking is the software equivalent.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.chunking.models import Chunk
from src.embedding.vector_store import embed_chunks, query_collection


# ── Fixtures ──

@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """A small set of realistic chunks for testing."""
    return [
        Chunk(
            text="Rule 8.4: A compatible declaration shall be visible when an object is defined.",
            source_filename="misra.pdf",
            source_filepath="/tmp/misra.pdf",
            page_number=23,
            chunk_index=0,
            start_char=0,
            end_char=80,
        ),
        Chunk(
            text="Rule 8.5: An external object or function shall be declared in one and only one file.",
            source_filename="misra.pdf",
            source_filepath="/tmp/misra.pdf",
            page_number=24,
            chunk_index=1,
            start_char=0,
            end_char=84,
        ),
    ]


# ── Tests for embed_chunks ──

class TestEmbedChunks:

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_upserts_correct_number_of_chunks(
        self, mock_client, mock_collection_fn, sample_chunks
    ):
        """embed_chunks should upsert all provided chunks."""
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        embed_chunks(sample_chunks)

        # upsert should have been called once (both chunks fit in one batch)
        assert mock_collection.upsert.called

        # Check the ids passed to upsert
        call_args = mock_collection.upsert.call_args
        ids = call_args.kwargs["ids"]
        assert len(ids) == 2

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_empty_chunks_skips_upsert(self, mock_client, mock_collection_fn):
        """embed_chunks should not call upsert when given empty list."""
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        embed_chunks([])

        mock_collection.upsert.assert_not_called()

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_metadata_contains_citation(self, mock_client, mock_collection_fn, sample_chunks):
        """Each chunk's metadata should include a citation field."""
        mock_collection = MagicMock()
        mock_collection_fn.return_value = mock_collection

        embed_chunks(sample_chunks)

        call_args = mock_collection.upsert.call_args
        metadatas = call_args.kwargs["metadatas"]

        for meta in metadatas:
            assert "citation" in meta
            assert "misra.pdf" in meta["citation"]
            assert "page" in meta["citation"]


# ── Tests for query_collection ──

class TestQueryCollection:

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_returns_results_with_expected_keys(self, mock_client, mock_collection_fn):
        """Query results should always have text, metadata, and distance keys."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["Some relevant chunk text"]],
            "metadatas": [[{"citation": "misra.pdf, page 23", "page_number": 23}]],
            "distances": [[0.12]],
        }
        mock_collection_fn.return_value = mock_collection

        results = query_collection("What is Rule 8.4?", n_results=1)

        assert len(results) == 1
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_empty_collection_returns_empty_list(self, mock_client, mock_collection_fn):
        """Querying an empty collection should return empty list, not crash."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection_fn.return_value = mock_collection

        results = query_collection("any question")

        assert results == []
        mock_collection.query.assert_not_called()
