"""
Tests for the docs_search MCP tool's reranking wiring.

WHY asyncio.run() INSTEAD OF pytest-asyncio:
docs_search is an `async def` (FastMCP tools are async), but this project
doesn't have an async pytest plugin configured (only anyio, pulled in
transitively by mcp/httpx — not set up with the marker pytest-anyio needs).
Driving the coroutine with asyncio.run() inside an ordinary sync test avoids
adding a new test-framework dependency for one file's worth of tests.

We only test docs_search's reranking wiring here — the tool's core behavior
(JSON shape, empty-collection message, error handling) predates this
session and isn't the subject of this change; testing it wasn't previously
required and adding full coverage is a larger, separate undertaking than
this session's scope.
"""

import asyncio
import json
from unittest.mock import patch

from src.mcp_server.server import SearchInput, docs_search


def _candidate(text: str, page: int, source: str = "misra.pdf") -> dict:
    return {
        "text": text,
        "metadata": {
            "source_filename": source,
            "page_number": page,
            "citation": f"{source}, page {page}",
        },
        "distance": 0.3,
    }


class TestDocsSearchReranking:
    def test_fetches_a_candidate_pool_wider_than_n_results(self):
        """docs_search must ask query_collection for more than n_results —
        candidate_k, per the same pattern as generate_answer() — so
        reranking has something to actually re-sort."""
        with (
            patch("src.mcp_server.server.query_collection", return_value=[]) as mock_query,
            patch("src.mcp_server.server.rerank", return_value=[]),
        ):
            asyncio.run(docs_search(SearchInput(query="what is a deviation", n_results=3)))

        called_n_results = (
            mock_query.call_args.kwargs.get("n_results") or mock_query.call_args[0][1]
        )
        assert called_n_results > 3

    def test_result_order_reflects_reranking_not_raw_retrieval_order(self):
        """The whole point of wiring rerank() in: the returned JSON order
        must be rerank's order, not query_collection's raw order."""
        candidates = [_candidate("low relevance", 1), _candidate("high relevance", 2)]
        reranked = [
            {**candidates[1], "rerank_score": 0.9},
            {**candidates[0], "rerank_score": 0.1},
        ]
        with (
            patch("src.mcp_server.server.query_collection", return_value=candidates),
            patch("src.mcp_server.server.rerank", return_value=reranked) as mock_rerank,
        ):
            result = asyncio.run(
                docs_search(SearchInput(query="what is a deviation", n_results=2))
            )

        mock_rerank.assert_called_once()
        parsed = json.loads(result)
        assert parsed["results"][0]["text"] == "high relevance"
        assert parsed["results"][0]["rerank_score"] == 0.9
        assert parsed["results"][1]["text"] == "low relevance"

    def test_missing_rerank_score_serializes_as_null(self):
        """rerank() falls back to returning candidates unchanged (no
        rerank_score key added) when reranking is disabled or the model
        fails to load — docs_search must not crash serializing that."""
        candidates = [_candidate("some passage", 1)]
        with (
            patch("src.mcp_server.server.query_collection", return_value=candidates),
            patch("src.mcp_server.server.rerank", return_value=candidates),
        ):
            result = asyncio.run(
                docs_search(SearchInput(query="what is a deviation", n_results=1))
            )

        parsed = json.loads(result)
        assert parsed["results"][0]["rerank_score"] is None

    def test_empty_collection_returns_graceful_message(self):
        with (
            patch("src.mcp_server.server.query_collection", return_value=[]),
            patch("src.mcp_server.server.rerank", return_value=[]),
        ):
            result = asyncio.run(
                docs_search(SearchInput(query="what is a deviation", n_results=5))
            )

        parsed = json.loads(result)
        assert parsed["results"] == []
        assert "ingestion" in parsed["message"].lower()
