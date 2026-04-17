"""
Tests for the generation module.

We mock both the retrieval step and the LLM call.
This lets us test our logic (prompt building, answer packaging,
has_answer detection) without network calls or API costs.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.generation.generator import generate_answer
from src.generation.models import Answer
from src.generation.prompts import build_user_prompt, SYSTEM_PROMPT


# ── Fixtures ──

@pytest.fixture
def sample_raw_chunks() -> list[dict]:
    """Simulates what query_collection() returns."""
    return [
        {
            "text": "A deviation is a process by which a project may use a guideline differently from how it is specified.",
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

    def test_context_appears_before_question(self, sample_raw_chunks):
        """Context must come before the question in the prompt."""
        prompt = build_user_prompt("What is a deviation?", sample_raw_chunks)
        context_pos = prompt.find("Context passage")
        question_pos = prompt.find("Question:")
        assert context_pos < question_pos


# ── Tests for generate_answer ──

class TestGenerateAnswer:

    @patch("src.generation.generator._call_llm")
    @patch("src.generation.generator.query_collection")
    def test_returns_answer_object(self, mock_retrieval, mock_llm, sample_raw_chunks):
        """generate_answer should always return an Answer object."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "A deviation is a formal process. [Source: MISRA-Compliance-2020.pdf, page 14]"

        result = generate_answer("What is a deviation?")

        assert isinstance(result, Answer)
        assert result.question == "What is a deviation?"

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
    def test_has_answer_false_when_not_in_context(self, mock_retrieval, mock_llm, sample_raw_chunks):
        """has_answer should be False when LLM signals missing information."""
        mock_retrieval.return_value = sample_raw_chunks
        mock_llm.return_value = "The provided documentation does not contain information about this topic."

        result = generate_answer("What is the meaning of life?")

        assert result.has_answer is False

    @patch("src.generation.generator.query_collection")
    def test_empty_retrieval_returns_graceful_answer(self, mock_retrieval):
        """Should return a helpful message when no chunks are retrieved."""
        mock_retrieval.return_value = []

        result = generate_answer("What is a deviation?")

        assert result.has_answer is False
        assert "ingestion" in result.answer.lower()

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


# ── Tests for Answer model ──

class TestAnswerModel:

    def test_formatted_sources_deduplicates(self):
        """Same citation appearing twice should appear once in formatted output."""
        from src.generation.models import RetrievedContext

        sources = [
            RetrievedContext(
                text="chunk 1", citation="misra.pdf, page 14",
                page_number=14, source_filename="misra.pdf", distance=0.1
            ),
            RetrievedContext(
                text="chunk 2", citation="misra.pdf, page 14",  # same citation
                page_number=14, source_filename="misra.pdf", distance=0.2
            ),
        ]
        answer = Answer(
            question="test", answer="test answer",
            sources=sources, has_answer=True
        )
        formatted = answer.formatted_sources
        assert formatted.count("misra.pdf, page 14") == 1
