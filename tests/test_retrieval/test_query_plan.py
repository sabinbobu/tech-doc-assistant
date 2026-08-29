"""
Tests for question routing (classification + decomposition).

We mock the LLM call, same pattern as test_query_rewrite.py — this tests
the plan/parse/fallback LOGIC, not whether a real model classifies well
(that's a prompt-quality question, exercised manually against the real
Infineon/MISRA/Epson corpus during development, not a unit test).
"""

from unittest.mock import patch

from src.retrieval import query_plan as query_plan_module
from src.retrieval.query_plan import plan_query


class TestPlanQuery:
    def test_lookup_question_returns_lookup_mode_with_no_sub_queries(self):
        raw_response = '{"mode": "lookup", "sub_queries": []}'
        with patch.object(query_plan_module, "_call_llm", return_value=raw_response):
            plan = plan_query("What is the BTS7200?")

        assert plan.mode == "lookup"
        assert plan.sub_queries == []

    def test_procedural_question_returns_decomposed_sub_queries(self):
        raw_response = (
            '{"mode": "procedural", "sub_queries": '
            '["short circuit to ground diagnosis", "pin configuration", "protection features"]}'
        )
        with patch.object(query_plan_module, "_call_llm", return_value=raw_response):
            plan = plan_query(
                "Guide me step by step on how to configure the Short to Ground error."
            )

        assert plan.mode == "procedural"
        assert len(plan.sub_queries) == 3
        assert "short circuit to ground diagnosis" in plan.sub_queries

    def test_structural_question_returns_structural_mode(self):
        raw_response = '{"mode": "structural", "sub_queries": [], "structural_keyword": null}'
        with patch.object(query_plan_module, "_call_llm", return_value=raw_response):
            plan = plan_query("List the table of contents")

        assert plan.mode == "structural"
        assert plan.structural_keyword is None

    def test_structural_question_with_topic_extracts_keyword(self):
        raw_response = (
            '{"mode": "structural", "sub_queries": [], "structural_keyword": "diagnostics"}'
        )
        with patch.object(query_plan_module, "_call_llm", return_value=raw_response):
            plan = plan_query("Which sections cover diagnostics?")

        assert plan.mode == "structural"
        assert plan.structural_keyword == "diagnostics"

    def test_missing_structural_keyword_key_defaults_to_none(self):
        raw_response = '{"mode": "structural", "sub_queries": []}'
        with patch.object(query_plan_module, "_call_llm", return_value=raw_response):
            plan = plan_query("List the table of contents")

        assert plan.structural_keyword is None

    def test_missing_sub_queries_key_defaults_to_empty_list(self):
        raw_response = '{"mode": "lookup"}'
        with patch.object(query_plan_module, "_call_llm", return_value=raw_response):
            plan = plan_query("What is a deviation?")

        assert plan.sub_queries == []

    def test_invalid_mode_falls_back_to_lookup(self):
        """An LLM that invents a mode outside the three we handle must not
        propagate garbage downstream — pydantic's Literal validation catches
        this and plan_query's except-Exception falls back safely."""
        raw_response = '{"mode": "summary", "sub_queries": []}'
        with patch.object(query_plan_module, "_call_llm", return_value=raw_response):
            plan = plan_query("Summarize this document")

        assert plan.mode == "lookup"
        assert plan.sub_queries == []

    def test_malformed_json_falls_back_to_lookup(self):
        with patch.object(query_plan_module, "_call_llm", return_value="not valid json"):
            plan = plan_query("What is a deviation?")

        assert plan.mode == "lookup"
        assert plan.sub_queries == []

    def test_llm_exception_falls_back_to_lookup(self):
        with patch.object(query_plan_module, "_call_llm", side_effect=RuntimeError("API down")):
            plan = plan_query("What is a deviation?")

        assert plan.mode == "lookup"
        assert plan.sub_queries == []

    def test_disabled_returns_lookup_without_calling_llm(self):
        with (
            patch.object(query_plan_module.settings, "query_planning_enabled", False),
            patch.object(query_plan_module, "_call_llm") as mock_call_llm,
        ):
            plan = plan_query("Guide me step by step on how to configure X")

        assert plan.mode == "lookup"
        assert plan.sub_queries == []
        mock_call_llm.assert_not_called()


class TestCallLlmDelegatesToLlmClient:
    """
    Provider dispatch itself (openai/anthropic/openrouter) is now
    src/generation/llm_client.py's job and tested there
    (tests/test_generation/test_llm_client.py) -- these tests only check
    that query_plan._call_llm resolves its model override correctly and
    hands off to that shared gateway with the right arguments.
    """

    def test_query_plan_model_override_takes_precedence(self):
        with (
            patch.object(query_plan_module.settings, "llm_model", "gpt-4o"),
            patch.object(query_plan_module.settings, "query_plan_model", "gpt-4o-mini"),
            patch.object(query_plan_module.llm_client, "call_llm", return_value="{}") as mock,
        ):
            query_plan_module._call_llm("a question")

        mock.assert_called_once_with(
            query_plan_module.SYSTEM_PROMPT, "a question", max_tokens=256, model="gpt-4o-mini"
        )

    def test_none_query_plan_model_falls_back_to_llm_model(self):
        with (
            patch.object(query_plan_module.settings, "llm_model", "gpt-4o-mini"),
            patch.object(query_plan_module.settings, "query_plan_model", None),
            patch.object(query_plan_module.llm_client, "call_llm", return_value="{}") as mock,
        ):
            query_plan_module._call_llm("a question")

        mock.assert_called_once_with(
            query_plan_module.SYSTEM_PROMPT, "a question", max_tokens=256, model="gpt-4o-mini"
        )
