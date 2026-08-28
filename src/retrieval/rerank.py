"""
Cross-encoder reranking — re-scores hybrid search's candidate pool.

WHY THIS EXISTS:
Hybrid search (vector + BM25, fused via RRF — see embedding/vector_store.py)
is a BI-ENCODER approach: the query and every chunk are embedded separately,
then compared by distance. That's fast enough to search thousands of chunks,
but it's an approximation — the model never actually reads the query and a
passage side by side.

A CROSS-ENCODER does read them together: (query, passage) goes in as one
input, and the model outputs a single relevance score for that specific pair.
Much more accurate, much slower — too slow to run over an entire collection,
which is why it only runs on the small candidate pool hybrid search already
narrowed things down to.

ANALOGY:
Hybrid search is triage — a fast first pass that gets you from "everything"
to "these 20 look plausible." Reranking is the specialist actually reading
those 20 charts side by side with the patient's symptoms before deciding
which ones are actually relevant.

FAILURE MODE:
The model is downloaded on first use (~500MB) and runs on CPU here. If it's
unavailable for any reason — not installed, no network for the first
download, out of memory — retrieval must still work. Every failure path
below falls back to returning the input candidates unchanged (already
ranked by hybrid search), never raises.
"""

import logging

from src.config import settings

logger = logging.getLogger(__name__)

_model = None
_model_load_failed = False


def _get_model():
    """
    Lazily load the cross-encoder, once per process.

    Lazy because importing sentence_transformers pulls in torch — that cost
    (several seconds) should only be paid the first time reranking is
    actually used, not on every `import src.retrieval.rerank`.
    """
    global _model, _model_load_failed

    if _model is not None:
        return _model
    if _model_load_failed:
        return None

    try:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {settings.reranker_model}")
        _model = CrossEncoder(settings.reranker_model)
        return _model
    except Exception:
        logger.exception(
            f"Failed to load reranker model '{settings.reranker_model}' — "
            "reranking disabled for this process, falling back to hybrid "
            "search's own ranking."
        )
        _model_load_failed = True
        return None


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Re-score and re-sort candidates by actual relevance to the query.

    Args:
        query: The user's question (or rewritten search query).
        candidates: Chunks from query_collection() — dicts with 'text',
            'metadata', 'distance'. Already ranked by hybrid search, but that
            ranking is what we're improving on here.
        top_k: How many to return after reranking. Must be <= len(candidates)
            to have any effect — reranking can only reorder what it's given.

    Returns:
        The top_k candidates, reordered by cross-encoder relevance score
        (highest first). On any failure, or if reranking is disabled via
        settings.reranking_enabled, returns candidates[:top_k] unchanged —
        this function never raises.
    """
    if not settings.reranking_enabled or not candidates:
        return candidates[:top_k]

    model = _get_model()
    if model is None:
        return candidates[:top_k]

    try:
        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = model.predict(pairs)
    except Exception:
        logger.exception("Reranker inference failed — falling back to hybrid search order.")
        return candidates[:top_k]

    scored = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)

    reranked = []
    for candidate, score in scored[:top_k]:
        reranked_candidate = dict(candidate)
        reranked_candidate["rerank_score"] = float(score)
        reranked.append(reranked_candidate)
    return reranked
