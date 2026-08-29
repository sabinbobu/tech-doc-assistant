"""
Tests for the generation module.

We mock both the retrieval step and the LLM call.
This lets us test our logic (prompt building, answer packaging,
has_answer detection) without network calls or API costs.
"""

from unittest.mock import patch

import pytest

from src.config import settings
from src.generation.generator import _is_refusal, generate_answer
from src.generation.models import Answer
from src.generation.prompts import build_user_prompt
from src.retrieval.query_plan import QueryPlan

# ── Fixtures ──


@pytest.fixture
def sample_raw_chunks() -> list[dict]:
    """Simulates what query_collection() returns."""
    return [
        {
            "text": (
                "A deviation is a process by which a project may use a guideline "
                "differently from how it is specified."
            ),
            "metadata": {
                "citation": "MISRA-Compliance-2020.pdf, page 14",
                "page_number": 14,
                "source_filename": "MISRA-Compliance-2020.pdf",
            },
            "distance": 0.08,
        },
        {
            "text": "Deviations shall be documented and approved before the software is released.",
            "metadata": {
                "citation": "MISRA-Compliance-2020.pdf, page 15",
                "page_number": 15,
                "source_filename": "MISRA-Compliance-2020.pdf",
            },
            "distance": 0.14,
        },
    ]


# ── Tests for prompt building ──


class TestBuildUserPrompt:
    def test_includes_question(self, sample_raw_chunks):
        """The question must appear in the prompt."""
        prompt = build_user_prompt("What is a deviation?", sample_raw_chunks)
        assert "What is a deviation?" in prompt

    def test_includes_chunk_text(self, sample_raw_chunks):
        """Each chunk's text must appear in the prompt."""
        prompt = build_user_prompt("What is a deviation?", sample_raw_chunks)
        assert "process by which a project" in prompt

    def test_includes_citations(self, sample_raw_chunks):
        """Source citations must appear alongside each chunk."""
        prompt = build_user_prompt("What is a deviation?", sample_raw_chunks)
        assert "MISRA-Compliance-2020.pdf, page 14" in prompt
        assert "MISRA-Compliance-2020.pdf, page 15" in prompt

    def test_includes_section_when_chunk_has_one(self, sample_raw_chunks):
        """A chunk from a document with an outline should surface its
        section in the passage header — the whole point of tagging chunks
        with document structure (src/chunking/chunker.py's
        document.section_for_page call)."""
        sample_raw_chunks[0]["metadata"]["section"] = "4 Deviations"
        prompt = build_user_prompt("What is a deviation?", sample_raw_chunks)
        assert "Section: 4 Deviations" in prompt

    def test_omits_section_label_when_chunk_has_none(self, sample_raw_chunks):
        """Documents with no embedded outline (section is None/absent) must
        not show a bogus "Section: None" label."""
        prompt = build_user_prompt("What is a deviation?", sample_raw_chunks)
        assert "Section:" not in prompt

    def test_context_appears_before_question(self, sample_raw_chunks):
        """Context must come before the question in the prompt."""
        prompt = build_user_prompt("What is a deviation?", sample_raw_chunks)
        context_pos = prompt.find("Context passage")
        question_pos = prompt.find("Question:")
        assert context_pos < question_pos


# ── Tests for generate_answer ──


class TestGenerateAnswer:
    @pytest.fixture(autouse=True)
    def mock_rerank(self):
        """
        Reranking sits between retrieval and generation now, and these tests
        exist to check generator logic — not reranker behavior, which has its
        own tests in tests/test_retrieval/test_rerank.py. Without this mock,
        every test here would load a real cross-encoder model (slow, and a
        cold CI cache means a network call) and its real output order isn't
        guaranteed stable across model versions — exactly the kind of
        non-determinism this file's own docstring says to avoid.
        """
        with patch(
            "src.generation.generator.rerank",
            side_effect=lambda query, candidates, top_k: candidates[:top_k],
        ) as mock:
            yield mock

    @pytest.fixture(autouse=True)
    def mock_rewrite_query(self):
        """
        Query rewriting makes a real LLM call now. Same reasoning as
        mock_rerank above: these tests check generator logic, not the
        rewriter's own behavior (tests/test_retrieval/test_query_rewrite.py
        covers that), and a real call would cost money and hit the network
        on every test run.
        """
        with patch(
            "src.generation.generator.rewrite_query",
            side_effect=lambda question: (question, []),
        ) as mock:
            yield mock

    @pytest.fixture(autouse=True)
    def mock_plan_query(self):
        """
        generate_answer's first step now classifies the question via a real
        LLM call (src/retrieval/query_plan.py). These tests exist to check
        generator logic for the (pre-existing) lookup path — routing itself
        has its own tests in tests/test_retrieval/test_query_plan.py — so
        default every test here to "lookup" mode, matching what these tests
        were written against before query planning existed. Tests that
        specifically exercise procedural/structural routing override this.
        """
        with patch(
            "src.generation.generator.plan_query",
            return_value=QueryPlan(mode="lookup", sub_queries=[]),
        ) as mock:
            yield mock

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_returns_answer_object(self, mock_retrieval, mock_llm, sample_raw_chunks):
        """generate_answer should always return an Answer object."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = (
            "A deviation is a formal process. [Source: MISRA-Compliance-2020.pdf, page 14]"
        )

        result = generate_answer("What is a deviation?")

        assert isinstance(result, Answer)
        assert result.question == "What is a deviation?"
        assert result.mode == "lookup"

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_has_answer_true_when_answered(self, mock_retrieval, mock_llm, sample_raw_chunks):
        """has_answer should be True when LLM provides a real answer."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is a formal approval process."

        result = generate_answer("What is a deviation?")

        assert result.has_answer is True

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_has_answer_false_when_not_in_context(
        self, mock_retrieval, mock_llm, sample_raw_chunks
    ):
        """has_answer should be False when LLM signals missing information."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = (
            "The provided documentation does not contain information about this topic."
        )

        result = generate_answer("What is the meaning of life?")

        assert result.has_answer is False

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_has_answer_true_for_partial_answer_mentioning_the_refusal_phrase(
        self, mock_retrieval, mock_llm, sample_raw_chunks
    ):
        """A real, useful partial answer that happens to mention what ISN'T
        covered must not be flagged as a total failure -- this is the bug
        _is_refusal exists to fix. The old substring check treated this
        exact shape of answer as a refusal, which then hid its own Sources
        panel in the UI (src/ui/app.py) behind a "no information" warning."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = (
            "The sensor reports Overtemperature and Short-Circuit faults "
            "[Source: manual.pdf, page 12]. The documentation does not "
            "contain information about this topic in Python form -- it "
            "only describes the hardware CAL pin procedure "
            "[Source: manual.pdf, page 15]."
        )

        result = generate_answer("What faults does it report, and how do I calibrate it in Python?")

        assert result.has_answer is True

    @patch("src.generation.generator.query_collection")
    def test_empty_retrieval_returns_graceful_answer(self, mock_retrieval):
        """Should return a helpful message when no chunks are retrieved."""
        mock_retrieval.return_value = []

        result = generate_answer("What is a deviation?")

        assert result.has_answer is False
        assert "ingestion" in result.answer.lower()

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_reranks_candidates_before_prompting(
        self, mock_retrieval, mock_llm, mock_rerank, sample_raw_chunks
    ):
        """generate_answer must call rerank() on the retrieved candidates,
        and build the prompt from rerank's output — not straight off
        query_collection's raw ordering."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is..."
        # Override the autouse passthrough to prove the reranked (reversed)
        # order — not the original retrieval order — is what reaches the LLM.
        reversed_chunks = list(reversed(sample_raw_chunks))
        mock_rerank.side_effect = None
        mock_rerank.return_value = reversed_chunks

        result = generate_answer("What is a deviation?")

        mock_rerank.assert_called_once_with(
            "What is a deviation?", sample_raw_chunks, top_k=settings.retrieval_top_k
        )
        assert result.sources[0].page_number == reversed_chunks[0]["metadata"]["page_number"]

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_retrieves_using_rewritten_search_query(
        self, mock_retrieval, mock_llm, mock_rewrite_query, sample_raw_chunks
    ):
        """generate_answer must search with the REWRITTEN query, not the raw
        question — that's the entire point of query rewriting. The original
        question must still reach the LLM prompt unchanged (checked via
        mock_llm's call args) since citations and prompt context should
        reflect what the user actually asked, not the search-engine version."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is..."
        mock_rewrite_query.side_effect = None
        mock_rewrite_query.return_value = ("deviation definition MISRA", ["Deviation"])

        generate_answer("What, exactly, is a deviation in MISRA compliance?")

        retrieval_query = mock_retrieval.call_args[0][0]
        assert retrieval_query == "deviation definition MISRA"
        prompt_sent_to_llm = mock_llm.call_args[0][0]
        assert "What, exactly, is a deviation in MISRA compliance?" in prompt_sent_to_llm

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_rewrite_failure_falls_back_to_original_question(
        self, mock_retrieval, mock_llm, mock_rewrite_query, sample_raw_chunks
    ):
        """If rewrite_query itself raised, generate_answer must not crash —
        but rewrite_query's own contract (see test_query_rewrite.py) is to
        catch its own failures and return the original question. This test
        exercises that generator.py trusts that contract rather than adding
        a redundant try/except of its own."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is..."
        mock_rewrite_query.side_effect = None
        mock_rewrite_query.return_value = ("What is a deviation?", [])

        result = generate_answer("What is a deviation?")

        assert isinstance(result, Answer)
        retrieval_query = mock_retrieval.call_args[0][0]
        assert retrieval_query == "What is a deviation?"

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_forwards_source_filenames_to_retrieval(
        self, mock_retrieval, mock_llm, sample_raw_chunks
    ):
        """source_filenames should be passed through to query_collection to scope the search."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is..."

        generate_answer("What is a deviation?", source_filenames=["MISRA-Compliance-2020.pdf"])

        # Retrieval now fetches a candidate pool wider than the final
        # n_results — see generate_answer's Step 1 comment — so reranking
        # has more than n_results items to actually re-sort.
        mock_retrieval.assert_called_once_with(
            "What is a deviation?",
            n_results=max(settings.retrieval_candidate_k, settings.retrieval_top_k),
            source_filenames=["MISRA-Compliance-2020.pdf"],
        )

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_llm_failure_raises_clear_runtime_error(
        self, mock_retrieval, mock_llm, sample_raw_chunks
    ):
        """A raw LLM SDK exception should surface as a clear RuntimeError, not crash."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.side_effect = ValueError("model not found: gpt-5.6-luna")

        with pytest.raises(RuntimeError, match="LLM call failed"):
            generate_answer("What is a deviation?")

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_sources_populated_from_chunks(self, mock_retrieval, mock_llm, sample_raw_chunks):
        """Answer sources should map correctly from retrieved chunks."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is..."

        result = generate_answer("What is a deviation?")

        assert len(result.sources) == 2
        assert result.sources[0].page_number == 14
        assert result.sources[1].page_number == 15

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_rerank_score_threaded_into_sources(
        self, mock_retrieval, mock_llm, mock_rerank, sample_raw_chunks
    ):
        """When rerank() attaches a rerank_score to a chunk (the real
        behavior — see rerank()'s docstring), that score must reach the
        final Answer.sources, not just distance. This is what the UI and
        MCP docs_search tool both surface."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is..."
        scored_chunks = [{**c, "rerank_score": 0.87} for c in sample_raw_chunks]
        mock_rerank.side_effect = None
        mock_rerank.return_value = scored_chunks

        result = generate_answer("What is a deviation?")

        assert result.sources[0].rerank_score == 0.87

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_missing_rerank_score_defaults_to_none(
        self, mock_retrieval, mock_llm, sample_raw_chunks
    ):
        """Reranking disabled or unavailable (rerank() falls back to
        returning candidates unchanged, per rerank.py) must not crash
        Answer construction — rerank_score is optional."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is..."

        result = generate_answer("What is a deviation?")

        assert result.sources[0].rerank_score is None

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_procedural_mode_retrieves_once_per_sub_query(
        self, mock_retrieval, mock_llm, mock_plan_query, sample_raw_chunks
    ):
        """Procedural mode must search once per sub-query (the whole point:
        evidence for a configuration procedure is scattered across sections
        a single query's top-k won't span) rather than falling through to
        the single rewrite_query() lookup path."""
        mock_plan_query.return_value = QueryPlan(
            mode="procedural",
            sub_queries=["pin configuration", "protection features", "diagnosis timing"],
        )
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "1. Set DEN high [Source: ds.pdf, page 6]."

        generate_answer("Guide me step by step on how to configure X")

        assert mock_retrieval.call_count == 3
        called_queries = [call.args[0] for call in mock_retrieval.call_args_list]
        assert called_queries == ["pin configuration", "protection features", "diagnosis timing"]

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_procedural_mode_dedupes_across_sub_queries(
        self, mock_retrieval, mock_llm, mock_plan_query, mock_rerank
    ):
        """The same chunk can legitimately match more than one sub-query
        (e.g. a diagnosis-timing passage matching both a "diagnosis" and a
        "timing" sub-query) -- it must reach the reranker once, not once
        per sub-query it happened to match."""
        mock_plan_query.return_value = QueryPlan(
            mode="procedural", sub_queries=["diagnosis", "timing"]
        )
        shared_chunk = {
            "text": "shared passage",
            "metadata": {
                "citation": "ds.pdf, page 45",
                "page_number": 45,
                "source_filename": "ds.pdf",
                "chunk_index": 7,
            },
            "distance": 0.1,
        }
        mock_retrieval.return_value = [shared_chunk]  # same chunk both calls
        mock_llm.return_value = "An answer."

        generate_answer("Guide me step by step to configure X")

        # rerank() receives the deduped candidate pool -- exactly one copy
        # of the shared chunk, not two.
        reranked_input = mock_rerank.call_args[0][1]
        assert len(reranked_input) == 1

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_procedural_mode_uses_synthesis_prompt_and_larger_budget(
        self, mock_retrieval, mock_llm, mock_plan_query, sample_raw_chunks
    ):
        """The synthesis prompt (which permits assembling a procedure and
        requires marking inference) and a larger token budget must be used
        for procedural mode -- SYSTEM_PROMPT's "use ONLY the provided
        context" framing is exactly what caused the original refusal this
        feature exists to fix."""
        from src.config import settings
        from src.generation.prompts import SYNTHESIS_SYSTEM_PROMPT

        mock_plan_query.return_value = QueryPlan(mode="procedural", sub_queries=["pin config"])
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "1. Set DEN high [Source: ds.pdf, page 6]."

        result = generate_answer("Guide me step by step to configure X")

        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        assert kwargs["system_prompt"] == SYNTHESIS_SYSTEM_PROMPT
        assert kwargs["max_tokens"] == settings.synthesis_answer_tokens
        assert result.mode == "procedural"

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_procedural_mode_with_no_sub_queries_falls_back_to_the_question(
        self, mock_retrieval, mock_llm, mock_plan_query, sample_raw_chunks
    ):
        """A classifier that says "procedural" but forgets to decompose
        must still retrieve something, not silently return zero results --
        routing must degrade, never break retrieval."""
        mock_plan_query.return_value = QueryPlan(mode="procedural", sub_queries=[])
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "An answer."

        generate_answer("Guide me step by step to configure X")

        mock_retrieval.assert_called_once()
        assert mock_retrieval.call_args.args[0] == "Guide me step by step to configure X"

    def test_structural_mode_renders_full_toc_without_calling_llm(self, mock_plan_query):
        """Structural mode must never call the LLM -- an LLM reformatting an
        already-correct outline can only make it less accurate, never more.
        See src/retrieval/structure.py's module docstring."""
        mock_plan_query.return_value = QueryPlan(mode="structural", sub_queries=[])

        with (
            patch(
                "src.generation.generator.list_indexed_documents",
                return_value=[{"filename": "ds.pdf"}],
            ),
            patch(
                "src.generation.generator.load_document_outline",
                return_value=[{"level": 1, "title": "1 Overview", "page": 1}],
            ),
            patch(
                "src.generation.generator.render_toc_for_all_documents",
                return_value="ds.pdf:\n  - 1 Overview (page 1)",
            ) as mock_render,
            patch("src.generation.generator._call_llm") as mock_llm,
        ):
            result = generate_answer("List the table of contents")

        mock_llm.assert_not_called()
        mock_render.assert_called_once()
        assert result.has_answer is True
        assert result.sources == []
        assert result.mode == "structural"
        assert "1 Overview" in result.answer

    def test_structural_mode_with_keyword_searches_sections(self, mock_plan_query):
        """A targeted structural question ("which sections cover
        diagnostics") must search section titles, not render the whole
        TOC."""
        mock_plan_query.return_value = QueryPlan(
            mode="structural", sub_queries=[], structural_keyword="diagnostics"
        )
        match = type("Entry", (), {"title": "9 Diagnosis", "page": 41})()

        with (
            patch(
                "src.generation.generator.list_indexed_documents",
                return_value=[{"filename": "ds.pdf"}],
            ),
            patch("src.generation.generator.find_sections", return_value=[match]) as mock_find,
            patch("src.generation.generator._call_llm") as mock_llm,
        ):
            result = generate_answer("Which sections cover diagnostics?")

        mock_llm.assert_not_called()
        mock_find.assert_called_once_with("ds.pdf", "diagnostics")
        assert result.has_answer is True
        assert "9 Diagnosis" in result.answer

    def test_structural_mode_no_matches_returns_has_answer_false(self, mock_plan_query):
        mock_plan_query.return_value = QueryPlan(
            mode="structural", sub_queries=[], structural_keyword="nonexistent"
        )

        with (
            patch(
                "src.generation.generator.list_indexed_documents",
                return_value=[{"filename": "ds.pdf"}],
            ),
            patch("src.generation.generator.find_sections", return_value=[]),
            patch("src.generation.generator._call_llm") as mock_llm,
        ):
            result = generate_answer("Which sections cover blorp?")

        mock_llm.assert_not_called()
        assert result.has_answer is False


class TestRetrieveForProcedural:
    """Direct tests for the dedup/union helper, separate from the routing
    tests above which check it's wired in correctly."""

    def test_unions_results_across_sub_queries(self):
        from src.generation.generator import _retrieve_for_procedural

        def fake_query_collection(query, n_results, source_filenames):
            return [
                {
                    "text": query,
                    "metadata": {"source_filename": "ds.pdf", "chunk_index": hash(query) % 1000},
                }
            ]

        with patch("src.generation.generator.query_collection", side_effect=fake_query_collection):
            result = _retrieve_for_procedural(["a", "b", "c"], None, candidate_k=10)

        assert len(result) == 3

    def test_dedupes_by_source_filename_and_chunk_index(self):
        from src.generation.generator import _retrieve_for_procedural

        shared = {"text": "x", "metadata": {"source_filename": "ds.pdf", "chunk_index": 5}}

        with patch("src.generation.generator.query_collection", return_value=[shared]):
            result = _retrieve_for_procedural(["a", "b"], None, candidate_k=10)

        assert len(result) == 1

    def test_same_chunk_index_different_document_is_not_deduped(self):
        """chunk_index restarts per document (src/embedding/vector_store.py)
        -- (source_filename, chunk_index) is the real identity, not
        chunk_index alone."""
        from src.generation.generator import _retrieve_for_procedural

        def fake_query_collection(query, n_results, source_filenames):
            filename = "a.pdf" if query == "first" else "b.pdf"
            return [{"text": query, "metadata": {"source_filename": filename, "chunk_index": 0}}]

        with patch("src.generation.generator.query_collection", side_effect=fake_query_collection):
            result = _retrieve_for_procedural(["first", "second"], None, candidate_k=10)

        assert len(result) == 2


# ── Tests for Answer model ──


class TestAnswerModel:
    def test_formatted_sources_deduplicates(self):
        """Same citation appearing twice should appear once in formatted output."""
        from src.generation.models import RetrievedContext

        sources = [
            RetrievedContext(
                text="chunk 1",
                citation="misra.pdf, page 14",
                page_number=14,
                source_filename="misra.pdf",
                distance=0.1,
            ),
            RetrievedContext(
                text="chunk 2",
                citation="misra.pdf, page 14",  # same citation
                page_number=14,
                source_filename="misra.pdf",
                distance=0.2,
            ),
        ]
        answer = Answer(question="test", answer="test answer", sources=sources, has_answer=True)
        formatted = answer.formatted_sources
        assert formatted.count("misra.pdf, page 14") == 1


# ── Tests for _is_refusal ──


class TestIsRefusal:
    def test_exact_refusal_phrase(self):
        assert (
            _is_refusal("The provided documentation does not contain information about this topic.")
            is True
        )

    def test_refusal_with_surrounding_whitespace(self):
        assert (
            _is_refusal(
                "  The provided documentation does not contain information about this topic.  "
            )
            is True
        )

    def test_partial_answer_mentioning_the_phrase_is_not_a_refusal(self):
        assert (
            _is_refusal(
                "The sensor reports Overtemperature and Short-Circuit faults "
                "[Source: manual.pdf, page 12]. The documentation does not "
                "contain information about this topic in Python form -- it "
                "only describes the hardware CAL pin procedure "
                "[Source: manual.pdf, page 15]."
            )
            is False
        )

    def test_normal_answer_is_not_a_refusal(self):
        assert (
            _is_refusal(
                "The BTS7200-2EPA is a Smart High-Side Power Switch [Source: ds.pdf, page 2]."
            )
            is False
        )
