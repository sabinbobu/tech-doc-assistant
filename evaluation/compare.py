"""
Compare retrieval configurations — the fast, free inner loop.

WHY THIS IS SEPARATE FROM evaluate.py:
evaluate.py runs the full pipeline (retrieval + LLM answer generation +
RAGAS's own LLM-judge scoring) — accurate, but slow (~20s) and costs API
credits on every run. That's the right tool for a pre-commit gate, wrong
tool for iterating on a fusion constant or deciding whether a reranker
earns its latency.

This script skips answer generation and RAGAS entirely. It only asks: did
retrieval surface the right page, and how high did it rank? That's answered
deterministically from the hand-labeled gold_pages in evaluation/dataset.py
via evaluation/retrieval_metrics.py — no LLM judge, no generation call,
results in well under a second per question (except the rewrite stage,
which still needs one LLM call per question to produce the rewritten
query — much cheaper than a full RAGAS pass, which also generates AND
judges an answer per question).

RUN WITH:
    uv run python -m evaluation.compare

Requires OPENAI_API_KEY (embedding calls) and a populated vector store
(see scripts/reingest.py).
"""

import logging

from evaluation.dataset import EVALUATION_DATASET
from evaluation.retrieval_metrics import score_dataset
from src.config import settings
from src.embedding.vector_store import query_collection
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.rerank import rerank

# Quiet by design — this script's own summary table is the output, not logs
# from the pipeline it's driving.
logging.basicConfig(level=logging.WARNING)


def _retrieve_hybrid(question: str, candidate_k: int, top_k: int) -> list[dict]:
    """Current main: hybrid (BM25 + vector, RRF-fused) search, no reranking."""
    candidates = query_collection(question, n_results=candidate_k)
    return candidates[:top_k]


def _retrieve_hybrid_rerank(question: str, candidate_k: int, top_k: int) -> list[dict]:
    """Shipped default: hybrid search, cross-encoder reranked down to top_k."""
    candidates = query_collection(question, n_results=candidate_k)
    return rerank(question, candidates, top_k=top_k)


def _retrieve_hybrid_rerank_rewrite(question: str, candidate_k: int, top_k: int) -> list[dict]:
    """Everything on: query rewriting feeds retrieval, then reranking.
    Shipped with query_rewrite_enabled=False by default — see src/config.py
    for why (measured regression). Still useful here to re-check that
    finding whenever the rewrite prompt changes."""
    search_query, _identifiers = rewrite_query(question)
    candidates = query_collection(search_query, n_results=candidate_k)
    return rerank(question, candidates, top_k=top_k)


# Cumulative stages, matching the sequence this project's own RAGAS history
# was measured in (see src/config.py's query_rewrite_enabled comment) — each
# stage adds one capability on top of the last, so a regression is
# attributable to the stage that introduced it.
STAGES = {
    "hybrid": _retrieve_hybrid,
    "hybrid+rerank": _retrieve_hybrid_rerank,
    "hybrid+rerank+rewrite": _retrieve_hybrid_rerank_rewrite,
}


def run_stage(
    stage_fn, dataset: list[dict], candidate_k: int, top_k: int
) -> dict[str, float]:
    """Run one retrieval stage over every question in `dataset` and score it."""
    results = []
    for item in dataset:
        retrieved = stage_fn(item["question"], candidate_k, top_k)
        results.append((retrieved, item["gold_doc"], item["gold_pages"]))
    return score_dataset(results, k=top_k)


def main() -> None:
    top_k = settings.retrieval_top_k
    candidate_k = max(settings.retrieval_candidate_k, top_k)

    print(
        f"\nComparing retrieval configs over {len(EVALUATION_DATASET)} questions "
        f"(top_k={top_k}, candidate_k={candidate_k})\n"
    )
    print(f"{'stage':<24}{'recall@k':>10}{'mrr':>10}{'ndcg@k':>10}")
    print("-" * 54)

    for stage_name, stage_fn in STAGES.items():
        scores = run_stage(stage_fn, EVALUATION_DATASET, candidate_k, top_k)
        print(
            f"{stage_name:<24}{scores['recall_at_k']:>10.3f}"
            f"{scores['mrr']:>10.3f}{scores['ndcg_at_k']:>10.3f}"
        )

    # Does BM25's exact-term matching actually help identifier-heavy
    # questions (rule numbers, defined category names, spec values) more
    # than general ones, as the theory predicts? Measured on the shipped
    # default (hybrid+rerank), not the experimental rewrite stage.
    identifier_heavy = [d for d in EVALUATION_DATASET if d["identifier_heavy"]]
    general = [d for d in EVALUATION_DATASET if not d["identifier_heavy"]]

    print("\nBy question type (hybrid+rerank, the shipped default):")
    for label, subset in [("identifier-heavy", identifier_heavy), ("general", general)]:
        scores = run_stage(_retrieve_hybrid_rerank, subset, candidate_k, top_k)
        print(
            f"  {label:<20}n={len(subset):<4}"
            f"recall@k={scores['recall_at_k']:.3f}  "
            f"mrr={scores['mrr']:.3f}  "
            f"ndcg@k={scores['ndcg_at_k']:.3f}"
        )


if __name__ == "__main__":
    main()
