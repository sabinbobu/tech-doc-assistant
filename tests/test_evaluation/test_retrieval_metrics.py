"""
Tests for deterministic retrieval metrics (recall@k, MRR, nDCG).

No mocking needed here — these are pure functions over plain dicts, the
same style of unit test as tests/test_chunking (no external calls at all,
so no I/O to fake).
"""

from evaluation.retrieval_metrics import mrr, ndcg_at_k, recall_at_k, score_dataset


def _chunk(source_filename: str, page_number: int) -> dict:
    return {"metadata": {"source_filename": source_filename, "page_number": page_number}}


GOLD_DOC = "misra.pdf"


class TestRecallAtK:
    def test_finds_gold_page_in_top_k(self):
        retrieved = [_chunk("other.pdf", 1), _chunk(GOLD_DOC, 23), _chunk("other.pdf", 2)]
        assert recall_at_k(retrieved, GOLD_DOC, [23], k=5) == 1.0

    def test_gold_page_outside_k_scores_zero(self):
        retrieved = [_chunk("other.pdf", 1), _chunk("other.pdf", 2), _chunk(GOLD_DOC, 23)]
        assert recall_at_k(retrieved, GOLD_DOC, [23], k=2) == 0.0

    def test_no_relevant_chunk_scores_zero(self):
        retrieved = [_chunk("other.pdf", 1), _chunk("other.pdf", 2)]
        assert recall_at_k(retrieved, GOLD_DOC, [23], k=5) == 0.0

    def test_right_page_wrong_document_does_not_count(self):
        """A chunk from a different document that happens to share a page
        number must not count as relevant — this is document+page matching,
        not page-number matching alone."""
        retrieved = [_chunk("other.pdf", 23)]
        assert recall_at_k(retrieved, GOLD_DOC, [23], k=5) == 0.0

    def test_matches_any_of_multiple_gold_pages(self):
        retrieved = [_chunk(GOLD_DOC, 21)]
        assert recall_at_k(retrieved, GOLD_DOC, [20, 21, 38], k=5) == 1.0

    def test_empty_retrieved_list_scores_zero(self):
        assert recall_at_k([], GOLD_DOC, [23], k=5) == 0.0


class TestMrr:
    def test_relevant_chunk_at_rank_one(self):
        retrieved = [_chunk(GOLD_DOC, 23), _chunk("other.pdf", 1)]
        assert mrr(retrieved, GOLD_DOC, [23]) == 1.0

    def test_relevant_chunk_at_rank_four(self):
        retrieved = [
            _chunk("other.pdf", 1),
            _chunk("other.pdf", 2),
            _chunk("other.pdf", 3),
            _chunk(GOLD_DOC, 23),
        ]
        assert mrr(retrieved, GOLD_DOC, [23]) == 0.25

    def test_no_relevant_chunk_scores_zero(self):
        retrieved = [_chunk("other.pdf", 1)]
        assert mrr(retrieved, GOLD_DOC, [23]) == 0.0

    def test_only_first_relevant_hit_counts(self):
        """If the gold page appears twice, MRR should reward the earliest
        occurrence, not average or double-count."""
        retrieved = [_chunk("other.pdf", 1), _chunk(GOLD_DOC, 23), _chunk(GOLD_DOC, 23)]
        assert mrr(retrieved, GOLD_DOC, [23]) == 0.5


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self):
        """A single relevant chunk, ranked first, is by definition the best
        possible ordering — nDCG must be exactly 1.0."""
        retrieved = [_chunk(GOLD_DOC, 23), _chunk("other.pdf", 1)]
        assert ndcg_at_k(retrieved, GOLD_DOC, [23], k=5) == 1.0

    def test_relevant_chunk_ranked_lower_scores_less_than_one(self):
        retrieved = [_chunk("other.pdf", 1), _chunk(GOLD_DOC, 23)]
        score = ndcg_at_k(retrieved, GOLD_DOC, [23], k=5)
        assert 0.0 < score < 1.0

    def test_no_relevant_chunk_scores_zero(self):
        retrieved = [_chunk("other.pdf", 1), _chunk("other.pdf", 2)]
        assert ndcg_at_k(retrieved, GOLD_DOC, [23], k=5) == 0.0

    def test_multiple_relevant_chunks_both_ranked_first_scores_one(self):
        """Two relevant chunks in the pool, both placed at the top (in
        either order) — this is the ideal ordering for this pool, so nDCG
        should be 1.0, not penalized for 'only 2 of some larger gold set'."""
        retrieved = [_chunk(GOLD_DOC, 20), _chunk(GOLD_DOC, 21), _chunk("other.pdf", 1)]
        assert ndcg_at_k(retrieved, GOLD_DOC, [20, 21], k=5) == 1.0

    def test_respects_k_cutoff(self):
        retrieved = [_chunk("other.pdf", 1), _chunk("other.pdf", 2), _chunk(GOLD_DOC, 23)]
        assert ndcg_at_k(retrieved, GOLD_DOC, [23], k=2) == 0.0


class TestScoreDataset:
    def test_aggregates_across_questions(self):
        results = [
            ([_chunk(GOLD_DOC, 23)], GOLD_DOC, [23]),  # perfect: rank 1
            ([_chunk("other.pdf", 1)], GOLD_DOC, [23]),  # miss entirely
        ]
        scores = score_dataset(results, k=5)

        assert scores["recall_at_k"] == 0.5  # 1 hit out of 2
        assert scores["mrr"] == 0.5  # (1.0 + 0.0) / 2
        assert scores["ndcg_at_k"] == 0.5  # (1.0 + 0.0) / 2

    def test_empty_results_returns_zeros_not_a_crash(self):
        scores = score_dataset([], k=5)
        assert scores == {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
