"""
Deterministic retrieval metrics — recall@k, MRR, nDCG.

WHY THIS EXISTS, SEPARATE FROM RAGAS:
RAGAS's context_precision is LLM-judged: it costs an API call per question,
takes ~20s for 8 questions, and (as this project's own eval history shows —
see CLAUDE.md-equivalent notes and eval_results_*.json) its scores can move
a few points between identical runs just from judge-model variance. That's
fine as a slow, occasional end-to-end gate. It's useless as an inner loop —
you can't run it 50 times while tuning a chunk_size or a fusion constant.

These metrics answer a narrower, cheaper, deterministic question: did
retrieval surface the RIGHT PAGE at all, and how high did it rank? That only
needs the gold_pages already hand-labeled in evaluation/dataset.py and the
metadata retrieval already returns — no LLM, no network call, milliseconds
per question. Use RAGAS to ask "is the final answer good"; use these to ask
"is retrieval finding the right needle in the haystack" while iterating.
"""

import math


def _is_relevant(chunk_metadata: dict, gold_doc: str, gold_pages: list[int]) -> bool:
    """A retrieved chunk counts as relevant if it's from the gold document
    and its page is one of the gold pages — an exact document+page match,
    not a fuzzy "close enough" heuristic. gold_pages is usually 1-3 pages
    (see dataset.py), so this is a deliberately strict bar."""
    return (
        chunk_metadata.get("source_filename") == gold_doc
        and chunk_metadata.get("page_number") in gold_pages
    )


def recall_at_k(retrieved: list[dict], gold_doc: str, gold_pages: list[int], k: int) -> float:
    """
    Fraction of gold pages found somewhere in the top-k retrieved chunks.

    Args:
        retrieved: Chunks from query_collection() / retrieve() — dicts with
            a 'metadata' key containing 'source_filename' and 'page_number'.
        gold_doc: The filename that actually answers the question.
        gold_pages: The page number(s) within gold_doc that answer it.
        k: How many of the top retrieved chunks to consider.

    Returns:
        1.0 if at least one of the top-k chunks matches a gold page, else
        0.0. (With 1-3 gold pages and chunks that don't map 1:1 to pages,
        "found the right page at all" is the meaningful signal here — not
        "found every gold page", which chunk boundaries make unreliable.)
    """
    top_k = retrieved[:k]
    found = any(_is_relevant(c["metadata"], gold_doc, gold_pages) for c in top_k)
    return 1.0 if found else 0.0


def mrr(retrieved: list[dict], gold_doc: str, gold_pages: list[int]) -> float:
    """
    Reciprocal rank of the first relevant chunk (1-indexed): 1/rank.

    0.0 if no relevant chunk appears anywhere in `retrieved`. Rewards
    ranking the right chunk near the top, not just including it somewhere
    in a large candidate pool — a chunk at rank 1 scores 1.0, at rank 5
    scores 0.2, unlike recall@k which treats every position within k equally.
    """
    for rank, chunk in enumerate(retrieved, start=1):
        if _is_relevant(chunk["metadata"], gold_doc, gold_pages):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[dict], gold_doc: str, gold_pages: list[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain over the top-k, with binary
    relevance (a chunk is either on a gold page or it isn't — this dataset
    doesn't grade "how relevant", just "relevant or not").

    DCG = sum(relevance_i / log2(rank_i + 1)) for the top-k.
    IDCG = the DCG of the best possible ordering (all relevant chunks
    ranked first) — the normalizer, so a question with 3 relevant chunks
    in the pool isn't penalized against a question with only 1.
    nDCG = DCG / IDCG, in [0, 1]. 0.0 when IDCG is 0 (no relevant chunk
    anywhere in `retrieved`, so there's nothing to rank correctly).
    """
    top_k = retrieved[:k]
    relevances = [
        1.0 if _is_relevant(c["metadata"], gold_doc, gold_pages) else 0.0 for c in top_k
    ]

    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))

    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))

    return dcg / idcg if idcg > 0 else 0.0


def score_dataset(
    results: list[tuple[list[dict], str, list[int]]], k: int = 5
) -> dict[str, float]:
    """
    Aggregate recall@k / MRR / nDCG@k across a full evaluation run.

    Args:
        results: One (retrieved_chunks, gold_doc, gold_pages) tuple per
            question — retrieved_chunks is whatever query_collection() or
            retrieve() returned for that question.
        k: Cutoff for recall@k and nDCG@k.

    Returns:
        {"recall_at_k": ..., "mrr": ..., "ndcg_at_k": ...} — means across
        all questions in `results`. Empty `results` returns all zeros
        rather than raising a ZeroDivisionError.
    """
    if not results:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    recalls = [recall_at_k(r, doc, pages, k) for r, doc, pages in results]
    mrrs = [mrr(r, doc, pages) for r, doc, pages in results]
    ndcgs = [ndcg_at_k(r, doc, pages, k) for r, doc, pages in results]

    n = len(results)
    return {
        "recall_at_k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "ndcg_at_k": sum(ndcgs) / n,
    }
