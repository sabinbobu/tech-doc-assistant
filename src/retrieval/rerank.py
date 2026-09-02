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

import opik

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


def warm_up() -> bool:
    """
    Load the cross-encoder now, so the first real query doesn't pay for it.

    WHY THIS EXISTS — measured, not assumed:
    _get_model() is lazy, and "lazy" means the ~12.4s SentenceTransformer load
    (torch import + weights off disk) happens inside whichever request happens
    to be first. On this machine that made the first Opik-traced answer 16.7s
    end to end, with a 9.0s `rerank` span — while the SECOND rerank in the same
    process took 296ms. Same model, same 20 candidates, 30x apart.

    That gap is a trap for anyone reading a single trace: it looks exactly like
    a slow reranker, and the obvious "fix" (swap the model, shrink the candidate
    pool, call a reranking API) targets the 296ms rather than the 12.4s. See the
    benchmark comment on settings.reranker_model in src/config.py.

    Long-lived processes (the Streamlit app, the MCP server) call this at
    startup, moving that one-time cost to boot where a user isn't waiting on it.

    WHAT THIS DOES NOT FIX — measured, so nobody re-derives it from a trace:
    inference latency decays with process idle time, independently of this
    warm-up. Same model, same 20 candidates, same process:
        immediately after warm_up ...  431 ms
        after  5s idle ..............  300 ms
        after 12s idle .............. 1386 ms   (next call back to 302 ms)
    That's torch's thread pool parking (or the OS reclaiming pages) during the
    wait, and it lands squarely on rerank because rerank always runs right
    after plan_query's LLM call has blocked for several seconds. So a traced
    `rerank` span of ~1.3s is expected and healthy in the full pipeline; ~300ms
    only happens when a rerank closely follows another one. Keeping the model
    hot through idle would need a background keep-alive thread burning CPU
    forever to save ~1s — not worth it. Deliberately not built.

    Returns:
        True if the model is loaded and reranking will actually run, False if
        it failed to load (in which case rerank() degrades to hybrid search's
        own ordering, exactly as it does without this call). Never raises —
        warming up is an optimization, and an optimization must not be able to
        take the app down.
    """
    if not settings.reranking_enabled:
        return False

    model = _get_model()
    if model is None:
        return False

    # Constructing the CrossEncoder is not the whole cost — the FIRST predict()
    # pays again for torch's lazy first forward pass. Measured: with the model
    # loaded but never run, the first real rerank took 1,171ms; with one
    # throwaway pair run first, 424ms; steady state 300ms. Warm-up batch size
    # makes no difference (1 pair and 20 pairs measured identical), so one pair.
    try:
        model.predict([("warm up", "warm up")])
    except Exception:
        # A failed dummy inference says nothing about real queries (and rerank()
        # has its own try/except around predict anyway). Log and carry on — the
        # model loaded, which is the part that mattered.
        logger.exception("Reranker warm-up inference failed — model loaded, continuing.")

    return True


@opik.track(type="tool")
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
