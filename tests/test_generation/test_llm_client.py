"""
Tests for the shared LLM gateway (src/generation/llm_client.py).

This is where provider dispatch (openai/anthropic/openrouter) is actually
tested — generator.py, query_rewrite.py, and query_plan.py all delegate here,
so their own tests only need to check that they call this module correctly,
not re-test dispatch themselves.

We mock the SDK client classes directly (openai.OpenAI, anthropic.Anthropic) —
no real network calls, no API keys needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.generation import llm_client


def _openai_style_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    return response


def _anthropic_style_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


class TestCallLlmProviderDispatch:
    def test_defaults_to_openai(self):
        with (
            patch.object(llm_client.settings, "llm_provider", "openai"),
            patch.object(llm_client, "_call_openai", return_value="hi") as mock_openai,
            patch.object(llm_client, "_call_anthropic") as mock_anthropic,
            patch.object(llm_client, "_call_openrouter") as mock_openrouter,
        ):
            result = llm_client.call_llm("system", "user", max_tokens=100)

        assert result == "hi"
        mock_openai.assert_called_once_with("system", "user", llm_client.settings.llm_model, 100)
        mock_anthropic.assert_not_called()
        mock_openrouter.assert_not_called()

    def test_uses_anthropic_when_configured(self):
        with (
            patch.object(llm_client.settings, "llm_provider", "anthropic"),
            patch.object(llm_client, "_call_openai") as mock_openai,
            patch.object(llm_client, "_call_anthropic", return_value="hi") as mock_anthropic,
            patch.object(llm_client, "_call_openrouter") as mock_openrouter,
        ):
            result = llm_client.call_llm("system", "user", max_tokens=100)

        assert result == "hi"
        mock_anthropic.assert_called_once()
        mock_openai.assert_not_called()
        mock_openrouter.assert_not_called()

    def test_uses_openrouter_when_configured(self):
        with (
            patch.object(llm_client.settings, "llm_provider", "openrouter"),
            patch.object(llm_client, "_call_openai") as mock_openai,
            patch.object(llm_client, "_call_anthropic") as mock_anthropic,
            patch.object(llm_client, "_call_openrouter", return_value="hi") as mock_openrouter,
        ):
            result = llm_client.call_llm("system", "user", max_tokens=100)

        assert result == "hi"
        mock_openrouter.assert_called_once()
        mock_openai.assert_not_called()
        mock_anthropic.assert_not_called()

    def test_provider_param_overrides_settings(self):
        with (
            patch.object(llm_client.settings, "llm_provider", "openai"),
            patch.object(llm_client, "_call_openrouter", return_value="hi") as mock_openrouter,
        ):
            llm_client.call_llm("system", "user", max_tokens=100, provider="openrouter")

        mock_openrouter.assert_called_once()

    def test_model_param_overrides_settings_llm_model(self):
        with (
            patch.object(llm_client.settings, "llm_model", "gpt-4o-mini"),
            patch.object(llm_client, "_call_openai", return_value="hi") as mock_openai,
        ):
            llm_client.call_llm("system", "user", max_tokens=100, model="gpt-4o")

        mock_openai.assert_called_once_with("system", "user", "gpt-4o", 100)

    def test_no_model_param_falls_back_to_settings_llm_model(self):
        with (
            patch.object(llm_client.settings, "llm_model", "gpt-4o-mini"),
            patch.object(llm_client, "_call_openai", return_value="hi") as mock_openai,
        ):
            llm_client.call_llm("system", "user", max_tokens=100)

        mock_openai.assert_called_once_with("system", "user", "gpt-4o-mini", 100)

    def test_provider_failure_propagates_without_fallback(self):
        """
        Generation is not an optional optimization like reranking/rewriting/
        routing -- a failed call must raise, never silently retry a
        different provider. See this module's docstring.
        """
        with (
            patch.object(llm_client.settings, "llm_provider", "openrouter"),
            patch.object(llm_client, "_call_openrouter", side_effect=RuntimeError("rate limited")),
            patch.object(llm_client, "_call_openai") as mock_openai,
            pytest.raises(RuntimeError, match="rate limited"),
        ):
            llm_client.call_llm("system", "user", max_tokens=100)

        mock_openai.assert_not_called()


class TestCallOpenAI:
    def test_builds_client_and_returns_content(self):
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _openai_style_response("hello")
            mock_openai_cls.return_value = mock_client

            result = llm_client._call_openai("sys", "user", "gpt-4o-mini", 256)

        assert result == "hello"
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0


class TestCallAnthropic:
    def test_builds_client_and_returns_content(self):
        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _anthropic_style_response("hello")
            mock_anthropic_cls.return_value = mock_client

            result = llm_client._call_anthropic("sys", "user", "claude-sonnet-5", 256)

        assert result == "hello"
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["system"] == "sys"


class TestCallOpenRouter:
    def test_uses_openrouter_base_url_and_api_key(self):
        with (
            patch.object(llm_client.settings, "openrouter_api_key", "or-key-123"),
            patch("openai.OpenAI") as mock_openai_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _openai_style_response("hello")
            mock_openai_cls.return_value = mock_client

            result = llm_client._call_openrouter(
                "sys", "user", "deepseek/deepseek-chat-v3.1:free", 256
            )

        assert result == "hello"
        mock_openai_cls.assert_called_once_with(
            api_key="or-key-123", base_url="https://openrouter.ai/api/v1"
        )
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "deepseek/deepseek-chat-v3.1:free"
        assert kwargs["max_tokens"] == 256
