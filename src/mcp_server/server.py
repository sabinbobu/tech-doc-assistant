"""
MCP Server for the Technical Documentation Assistant.

This wraps your RAG pipeline in the Model Context Protocol —
exposing it as a standardized interface any MCP-compatible client
can call: Claude Desktop, VS Code extensions, other AI agents.

ANALOGY:
Your RAG pipeline is an ECU with custom firmware.
This file is the OBD-II adapter — it translates the standard protocol
into calls your ECU understands, without modifying the ECU itself.

TRANSPORT: stdio (local tool, runs as a subprocess)
This is the correct choice for a local portfolio project.
Claude Desktop spawns this script as a child process and communicates
over stdin/stdout — no HTTP server, no ports, no authentication needed.

RUN STANDALONE (for testing):
    uv run python src/mcp_server/server.py

CONNECT TO CLAUDE DESKTOP:
    See mcp_server/README.md for configuration instructions.
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.embedding.vector_store import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
)
from src.generation.generator import generate_answer
from src.retrieval.rerank import rerank
from src.retrieval.rerank import warm_up as warm_up_reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan management ──
# Initialize ChromaDB connection once at server startup,
# reuse it across all tool calls.
# Like initializing your peripheral drivers once in main() —
# not on every function call.


@asynccontextmanager
async def app_lifespan(app) -> AsyncIterator[dict]:
    """Initialize shared resources for the server's lifetime."""
    logger.info("MCP server starting — connecting to ChromaDB...")
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    doc_count = collection.count()
    logger.info(f"ChromaDB ready — {doc_count} chunks indexed")

    # The cross-encoder is lazy-loaded, and that load is ~12.4s on this
    # machine — long enough that an MCP client's first docs_ask/docs_search
    # would look hung while it happened. Same reasoning as the ChromaDB
    # connection above: pay for the peripheral once at init, not per call.
    logger.info("Loading reranker model...")
    if warm_up_reranker():
        logger.info(f"Reranker ready — {settings.reranker_model}")
    else:
        logger.warning("Reranker unavailable — falling back to hybrid search order")

    yield {"collection": collection, "doc_count": doc_count}
    logger.info("MCP server shutting down")


# ── Server initialization ──
mcp = FastMCP(
    "tech_docs_mcp",
    lifespan=app_lifespan,
)


# ── Input models ──
# Pydantic models validate inputs before they reach your pipeline.
# If an agent passes an empty query, it's rejected here — not buried
# in a stack trace inside ChromaDB.


class SearchInput(BaseModel):
    """Input for the search_documentation tool."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )
    query: str = Field(
        ...,
        description="Search query to find relevant documentation passages. "
        "Example: 'external linkage rules', 'deviation approval process'",
        min_length=3,
        max_length=500,
    )
    n_results: int = Field(
        default=5,
        description="Number of relevant passages to return (1–10)",
        ge=1,
        le=10,
    )


class AskInput(BaseModel):
    """Input for the ask_documentation tool."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )
    question: str = Field(
        ...,
        description="Natural language question to answer using the documentation. "
        "Example: 'What is a deviation in MISRA compliance?'",
        min_length=10,
        max_length=500,
    )
    n_results: int = Field(
        default=5,
        description="Number of documentation passages to use as context (1–10)",
        ge=1,
        le=10,
    )


# ── Tools ──


@mcp.tool(
    name="docs_search",
    annotations=ToolAnnotations(
        title="Search Technical Documentation",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def docs_search(params: SearchInput) -> str:
    """Search indexed technical documentation and return relevant passages.

    Use this tool when you need raw source passages without LLM synthesis —
    for example, when you want to inspect what the documentation says directly,
    or when you need to feed passages into your own reasoning chain.

    This tool performs RETRIEVAL ONLY — no LLM call, no answer generation.
    Results ARE reranked by a local cross-encoder (no LLM involved — see
    src/retrieval/rerank.py) for the same reason docs_ask's answers are:
    hybrid search's fused ranking is a fast approximation, and a model that
    actually reads query+passage together orders results more accurately.
    For a complete Q&A answer with citations, use docs_ask instead.

    Args:
        params (SearchInput): Validated input containing:
            - query (str): Search query (3–500 chars)
            - n_results (int): Number of passages to return (1–10, default 5)

    Returns:
        str: JSON array of passages, each containing:
            {
                "text": "passage content",
                "citation": "filename, page N",
                "page_number": N,
                "source_filename": "filename.pdf",
                "relevance_score": 0.0–1.0  (higher = more relevant; from
                    hybrid search's distance, kept for backward compatibility)
                "rerank_score": raw cross-encoder score (higher = more
                    relevant; this is what results are actually ORDERED by —
                    prefer this over relevance_score when comparing results)
            }
    """
    try:
        # Same candidate-pool-then-rerank pattern as generate_answer() in
        # src/generation/generator.py: fetch more than n_results so the
        # reranker has something to actually re-sort.
        candidate_k = max(settings.retrieval_candidate_k, params.n_results)
        candidates = query_collection(params.query, n_results=candidate_k)
        raw_results = rerank(params.query, candidates, top_k=params.n_results)

        if not raw_results:
            return json.dumps(
                {
                    "results": [],
                    "message": "No documents indexed. Run the ingestion pipeline first.",
                }
            )

        # relevance_score: convert distance to a 0-1 score for backward
        # compatibility (Distance is 0-2 for cosine, lower = better;
        # keyword-only hits get a neutral placeholder — see
        # _reciprocal_rank_fusion in vector_store.py). rerank_score is the
        # score results are actually sorted by — see the docstring above.
        formatted = [
            {
                "text": r["text"],
                "citation": r["metadata"].get("citation", "Unknown"),
                "page_number": r["metadata"].get("page_number", 0),
                "source_filename": r["metadata"].get("source_filename", "Unknown"),
                "relevance_score": round(1 - (r["distance"] / 2), 3),
                "rerank_score": round(r["rerank_score"], 3) if "rerank_score" in r else None,
            }
            for r in raw_results
        ]

        return json.dumps(
            {
                "query": params.query,
                "total_results": len(formatted),
                "results": formatted,
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"docs_search failed: {e}")
        return json.dumps({"error": f"Search failed: {str(e)}"})


@mcp.tool(
    name="docs_ask",
    annotations=ToolAnnotations(
        title="Ask Technical Documentation",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=False,  # LLM responses may vary slightly
        openWorldHint=False,
    ),
)
async def docs_ask(params: AskInput) -> str:
    """Answer a question using indexed technical documentation with citations.

    This tool runs the full RAG pipeline:
    1. Retrieves relevant documentation passages
    2. Sends them to an LLM with a structured prompt
    3. Returns a cited answer grounded exclusively in the documentation

    The answer will NEVER use information outside the indexed documents.
    If the documentation doesn't cover the topic, has_answer will be false.

    Use this tool for natural language Q&A over technical documents.
    For raw passage retrieval without synthesis, use docs_search instead.

    Args:
        params (AskInput): Validated input containing:
            - question (str): Natural language question (10–500 chars)
            - n_results (int): Context passages to use (1–10, default 5)

    Returns:
        str: JSON object containing:
            {
                "question": "original question",
                "answer": "LLM-generated answer with inline citations",
                "has_answer": true/false,
                "sources": [
                    {
                        "citation": "filename, page N",
                        "page_number": N,
                        "source_filename": "filename.pdf",
                        "relevance_score": 0.0–1.0
                    }
                ]
            }
    """
    try:
        answer = generate_answer(params.question, n_results=params.n_results)

        return json.dumps(
            {
                "question": answer.question,
                "answer": answer.answer,
                "has_answer": answer.has_answer,
                "sources": [
                    {
                        "citation": s.citation,
                        "page_number": s.page_number,
                        "source_filename": s.source_filename,
                        "relevance_score": round(1 - (s.distance / 2), 3),
                    }
                    for s in answer.sources
                ],
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"docs_ask failed: {e}")
        return json.dumps({"error": f"Answer generation failed: {str(e)}"})


@mcp.tool(
    name="docs_status",
    annotations=ToolAnnotations(
        title="Get Documentation Index Status",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def docs_status() -> str:
    """Check the status of the documentation index.

    Call this first to verify documents are loaded before searching or asking.
    Returns the number of indexed chunks and whether the system is ready.

    Returns:
        str: JSON object containing:
            {
                "ready": true/false,
                "indexed_chunks": N,
                "message": "human-readable status"
            }
    """
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        count = collection.count()

        return json.dumps(
            {
                "ready": count > 0,
                "indexed_chunks": count,
                "message": (
                    f"{count} chunks indexed and ready for queries."
                    if count > 0
                    else "No documents indexed. Run the ingestion pipeline first."
                ),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "ready": False,
                "indexed_chunks": 0,
                "message": f"Could not connect to vector store: {str(e)}",
            }
        )


# ── Entry point ──
if __name__ == "__main__":
    mcp.run()  # stdio transport by default — correct for Claude Desktop
