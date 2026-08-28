"""
Tests for query rewriting.

We mock the LLM call, same pattern as tests/test_generation/test_generator.py
mocking _call_llm — this tests the rewrite/parse/fallback LOGIC, not whether
a real model produces good search queries (that's a prompt-quality question,
not a unit-test question).
"""

from unittest.mock import patch

import pytest

from src.retrieval import query_rewrite as query_rewrite_module
from src.retrieval.query_rewrite import rewrite_query


class TestRewriteQuery:
    @pytest.fixture(autouse=True)
    def enabled(self):
        """
        query_rewrite_enabled defaults to False (see src/config.py — measured
        to regress faithfulness on this project's own eval). This class
        tests rewrite_query()'s own logic when the feature IS on; the one
        test that specifically covers the disabled path overrides this with
        its own patch.
        """
        with patch.object(query_rewrite_module.settings, "query_rewrite_enabled", True):
            yield

    def test_returns_search_query_and_identifiers_on_success(self):
        raw_response = (
            '{"search_query": "mandatory guideline deviation permitted", '
            '"identifiers": ["Mandatory"]}'
        )
        with patch.object(query_rewrite_module, "_call_llm", return_value=raw_response):
            search_query, identifiers = rewrite_query(
                "Can a mandatory MISRA guideline be deviated from?"
            )

        assert search_query == "mandatory guideline deviation permitted"
        assert identifiers == ["Mandatory"]

    def test_missing_identifiers_key_defaults_to_empty_list(self):
        raw_response = '{"search_query": "deviation process"}'
        with patch.object(query_rewrite_module, "_call_llm", return_value=raw_response):
            search_query, identifiers = rewrite_query("What is a deviation?")

        assert search_query == "deviation process"
        assert identifiers == []

    def test_malformed_json_falls_back_to_original_question(self):
        with patch.object(
            query_rewrite_module, "_call_llm", return_value="not valid json at all"
        ):
            search_query, identifiers = rewrite_query("What is a deviation?")

        assert search_query == "What is a deviation?"
        assert identifiers == []

    def test_empty_search_query_falls_back_to_original_question(self):
        """An LLM that returns a technically-valid but empty search_query
        (e.g. it misunderstood the task) must not send an empty string to
        the index — that would match nothing, or everything, depending on
        the search backend. Fall back rather than trust it blindly."""
        raw_response = '{"search_query": "", "identifiers": []}'
        with patch.object(query_rewrite_module, "_call_llm", return_value=raw_response):
            search_query, identifiers = rewrite_query("What is a deviation?")

        assert search_query == "What is a deviation?"
        assert identifiers == []

    def test_llm_exception_falls_back_to_original_question(self):
        with patch.object(
            query_rewrite_module, "_call_llm", side_effect=RuntimeError("API down")
        ):
            search_query, identifiers = rewrite_query("What is a deviation?")

        assert search_query == "What is a deviation?"
        assert identifiers == []

    def test_disabled_returns_original_question_without_calling_llm(self):
        with (
            patch.object(query_rewrite_module.settings, "query_rewrite_enabled", False),
            patch.object(query_rewrite_module, "_call_llm") as mock_call_llm,
        ):
            search_query, identifiers = rewrite_query("What is a deviation?")

        assert search_query == "What is a deviation?"
        assert identifiers == []
        mock_call_llm.assert_not_called()


class TestCallLlmProviderDispatch:
    def test_uses_openai_by_default(self):
        with (
            patch.object(query_rewrite_module.settings, "llm_provider", "openai"),
            patch.object(query_rewrite_module, "_call_openai", return_value="{}") as mock_openai,
            patch.object(query_rewrite_module, "_call_anthropic") as mock_anthropic,
        ):
            query_rewrite_module._call_llm("a question")

        mock_openai.assert_called_once()
        mock_anthropic.assert_not_called()

    def test_uses_anthropic_when_configured(self):
        with (
            patch.object(query_rewrite_module.settings, "llm_provider", "anthropic"),
            patch.object(query_rewrite_module, "_call_openai") as mock_openai,
            patch.object(
                query_rewrite_module, "_call_anthropic", return_value="{}"
            ) as mock_anthropic,
        ):
            query_rewrite_module._call_llm("a question")

        mock_anthropic.assert_called_once()
        mock_openai.assert_not_called()

    def test_rewrite_model_override_takes_precedence_over_llm_model(self):
        with (
            patch.object(query_rewrite_module.settings, "llm_provider", "openai"),
            patch.object(query_rewrite_module.settings, "llm_model", "gpt-4o"),
            patch.object(query_rewrite_module.settings, "rewrite_model", "gpt-4o-mini"),
            patch.object(query_rewrite_module, "_call_openai", return_value="{}") as mock_openai,
        ):
            query_rewrite_module._call_llm("a question")

        mock_openai.assert_called_once_with("a question", "gpt-4o-mini")

    def test_none_rewrite_model_falls_back_to_llm_model(self):
        with (
            patch.object(query_rewrite_module.settings, "llm_provider", "openai"),
            patch.object(query_rewrite_module.settings, "llm_model", "gpt-4o-mini"),
            patch.object(query_rewrite_module.settings, "rewrite_model", None),
            patch.object(query_rewrite_module, "_call_openai", return_value="{}") as mock_openai,
        ):
            query_rewrite_module._call_llm("a question")

        mock_openai.assert_called_once_with("a question", "gpt-4o-mini")
