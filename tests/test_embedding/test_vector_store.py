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
from src.embedding.vector_store import embed_chunks, list_indexed_documents, query_collection


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


# ── Tests for _vector_search (cosine similarity via ChromaDB) ──

class TestVectorSearch:

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_no_source_filenames_omits_where_clause(self, mock_client, mock_collection_fn):
        """Without source_filenames, the query should search the whole collection."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{"citation": "a.pdf, page 1"}]],
            "distances": [[0.1]],
        }
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        mock_collection_fn.return_value = mock_collection

        query_collection("a question")

        assert "where" not in mock_collection.query.call_args.kwargs

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_single_source_filename_scopes_query(self, mock_client, mock_collection_fn):
        """A single source_filename should be passed as a plain equality filter."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{"citation": "bts7200.pdf, page 1"}]],
            "distances": [[0.1]],
        }
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        mock_collection_fn.return_value = mock_collection

        query_collection("a question", source_filenames=["bts7200.pdf"])

        assert mock_collection.query.call_args.kwargs["where"] == {
            "source_filename": "bts7200.pdf"
        }

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_multiple_source_filenames_use_in_filter(self, mock_client, mock_collection_fn):
        """Multiple source_filenames should be combined with a $in filter."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{"citation": "a.pdf, page 1"}]],
            "distances": [[0.1]],
        }
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        mock_collection_fn.return_value = mock_collection

        query_collection("a question", source_filenames=["a.pdf", "b.pdf"])

        assert mock_collection.query.call_args.kwargs["where"] == {
            "source_filename": {"$in": ["a.pdf", "b.pdf"]}
        }


# ── Tests for _keyword_search (BM25) ──

class TestKeywordSearch:

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_ranks_exact_term_match_first(self, mock_client, mock_collection_fn):
        """A chunk containing the query's exact keyword should outrank one that doesn't."""
        # Three documents, not two — with only two docs, a term that appears
        # in exactly one of them has a classic-BM25 idf of exactly zero
        # (log((N-freq+0.5)/(freq+0.5)) == log(1) when N=2, freq=1), which
        # would make this test pass or fail by degenerate accident rather
        # than by testing real ranking behavior.
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.get.return_value = {
            "documents": [
                "The BTS7200-2EPA is a two-channel high-side switch for automotive use.",
                "General guidance on writing safe embedded C code.",
                "MISRA Rule 8.4 requires a compatible declaration to be visible.",
            ],
            "metadatas": [
                {"source_filename": "bts7200.pdf", "chunk_index": 0},
                {"source_filename": "misra.pdf", "chunk_index": 0},
                {"source_filename": "misra.pdf", "chunk_index": 1},
            ],
        }
        # No vector-search hits — isolates this test to BM25 ranking behavior.
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_collection_fn.return_value = mock_collection

        results = query_collection("BTS7200-2EPA high-side switch", n_results=1)

        assert results[0]["metadata"]["source_filename"] == "bts7200.pdf"

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_empty_corpus_returns_empty_list(self, mock_client, mock_collection_fn):
        """An empty (post-filter) corpus shouldn't crash BM25 indexing."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_collection_fn.return_value = mock_collection

        results = query_collection("anything")

        assert results == []


# ── Tests for _reciprocal_rank_fusion ──

class TestReciprocalRankFusion:

    def test_chunk_present_in_both_lists_outranks_single_list_hits(self):
        from src.embedding.vector_store import _reciprocal_rank_fusion

        shared = {
            "text": "shared chunk",
            "metadata": {"source_filename": "a.pdf", "chunk_index": 0},
            "distance": 0.2,
        }
        vector_only = {
            "text": "vector-only chunk",
            "metadata": {"source_filename": "a.pdf", "chunk_index": 1},
            "distance": 0.1,
        }
        keyword_only = {
            "text": "keyword-only chunk",
            "metadata": {"source_filename": "a.pdf", "chunk_index": 2},
            "distance": None,
        }

        fused = _reciprocal_rank_fusion(
            vector_results=[vector_only, shared],
            keyword_results=[shared, keyword_only],
            n_results=3,
        )

        assert fused[0]["text"] == "shared chunk"

    def test_keyword_only_hit_gets_neutral_distance_placeholder(self):
        from src.embedding.vector_store import _reciprocal_rank_fusion

        keyword_only = {
            "text": "keyword-only chunk",
            "metadata": {"source_filename": "a.pdf", "chunk_index": 2},
            "distance": None,
        }

        fused = _reciprocal_rank_fusion(
            vector_results=[], keyword_results=[keyword_only], n_results=1
        )

        assert fused[0]["distance"] == 0.5

    def test_respects_n_results_limit(self):
        from src.embedding.vector_store import _reciprocal_rank_fusion

        chunks = [
            {
                "text": f"chunk {i}",
                "metadata": {"source_filename": "a.pdf", "chunk_index": i},
                "distance": 0.1 * i,
            }
            for i in range(5)
        ]

        fused = _reciprocal_rank_fusion(vector_results=chunks, keyword_results=[], n_results=2)

        assert len(fused) == 2


# ── Tests for query_collection (composition) ──

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
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
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


# ── Tests for list_indexed_documents ──

class TestListIndexedDocuments:

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_empty_collection_returns_empty_list(self, mock_client, mock_collection_fn):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection_fn.return_value = mock_collection

        assert list_indexed_documents() == []

    @patch("src.embedding.vector_store.get_or_create_collection")
    @patch("src.embedding.vector_store.get_chroma_client")
    def test_groups_and_counts_by_source_filename(self, mock_client, mock_collection_fn):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_filename": "misra.pdf"},
                {"source_filename": "misra.pdf"},
                {"source_filename": "bts7200.pdf"},
            ]
        }
        mock_collection_fn.return_value = mock_collection

        result = list_indexed_documents()

        assert result == [
            {"filename": "bts7200.pdf", "chunk_count": 1},
            {"filename": "misra.pdf", "chunk_count": 2},
        ]
