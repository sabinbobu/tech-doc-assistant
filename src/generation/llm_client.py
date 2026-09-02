"""
Shared LLM-calling gateway — one place that knows how to reach OpenAI, Anthropic,
or OpenRouter, used by every module that needs a raw completion call: answer
generation (generator.py), query rewriting (query_rewrite.py), and query
classification (query_plan.py). Before this module existed, all three had their
own copy of the same "if anthropic call X else call Y" dispatch — a third
provider meant writing it a third time. Now it's written once.

ANALOGY: This is the HAL layer underneath the app's three LLM call sites — they
each still decide WHAT to ask (system prompt, user prompt, token budget) and
WHICH model override applies to them, but none of them know how to actually
reach OpenAI vs. Anthropic vs. OpenRouter. That knowledge lives here only.

OPENROUTER: proxies many providers (including several genuinely free models,
tagged with a ":free" slug suffix, e.g. "deepseek/deepseek-chat-v3.1:free")
behind a single OpenAI-compatible Chat Completions API. That means no new SDK
dependency — the `openai` package already used for the "openai" provider works
unchanged against OpenRouter, just pointed at a different base_url with a
different API key.

A single pinned ":free" model shares a rate-limited pool with every other
OpenRouter user of that model — this project hit that directly (a 30-question
eval run 429'd upstream on "z-ai/glm-5.2:free"). model="openrouter/free" is a
different thing: an OpenRouter-side router that randomly picks a capable model
from ~24 free ones per call, so load isn't concentrated on one model. No code
change needed here to use it — it's just a model string, same as any other —
but see .env.example for the tradeoff (model identity, and therefore answer
phrasing/quality, is no longer deterministic per call).

FAILURE MODE — deliberately NOT the same as reranking/rewriting/routing:
Those three degrade gracefully because they're optional optimizations sitting
in front of retrieval. This function IS the actual generation step; there's no
simpler fallback behind it. It raises on failure and never silently switches
providers — callers (generator.py) wrap it and turn the exception into a
clear, user-facing error instead.
"""

from src.config import settings


def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    """
    Call the configured LLM provider and return the raw response text.

    Args:
        system_prompt: The system/instruction prompt.
        user_prompt: The user-turn content.
        max_tokens: Output token budget.
        model: Defaults to settings.llm_model. Callers with their own model
            override (settings.rewrite_model, settings.query_plan_model) pass
            it explicitly here.
        provider: Defaults to settings.llm_provider. Exists mainly for tests;
            callers in this codebase don't override it today.

    Returns:
        Raw response text.

    Raises:
        Whatever the underlying SDK raises on failure (network error, auth
        error, rate limit, ...) — this function never swallows an error or
        falls back to a different provider.
    """
    provider = provider or settings.llm_provider
    model = model or settings.llm_model

    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, model, max_tokens)
    elif provider == "openrouter":
        return _call_openrouter(system_prompt, user_prompt, model, max_tokens)
    else:
        return _call_openai(system_prompt, user_prompt, model, max_tokens)


def _call_openai(system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    """Call OpenAI API and return response text."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  # 0 = deterministic — we want consistent, factual answers
        # not creative variation. Same query should give same answer.
        # Like disabling dithering on your ADC for stable readings.
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or ""


def _call_anthropic(system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    """Call Anthropic API and return response text."""
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.content[0].text


def _call_openrouter(system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    """
    Call OpenRouter's API and return response text.

    OpenRouter speaks the OpenAI Chat Completions wire format, so this reuses
    the `openai` SDK client pointed at OpenRouter's base_url instead of a
    dedicated OpenRouter SDK.
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or ""
