"""
Query rewriting — clean up the user's question before it hits the index.

WHY THIS EXISTS:
A user asks a full natural-language question: "Can a mandatory MISRA
guideline be deviated from?" But the index doesn't care about grammar — it
cares about terms. BM25 rewards exact term overlap, and this technical
corpus is dense with identifiers (rule numbers, "Mandatory"/"Advisory",
part numbers) whose exact wording matters more to a keyword search than
"can", "a", "be", "from" do. A short, keyword-dense search query built from
the same question can match better than the question's own prose does.

This module makes ONE LLM call to turn a question into:
  - search_query: a cleaner, keyword-forward version of the question
  - identifiers: any exact identifiers mentioned (rule numbers, part
    numbers, defined terms) — extracted for visibility today; not yet wired
    into a dedicated filter (see CLAUDE.md's Phase 1e notes for why that's
    deliberately out of scope for this pass).

FAILURE MODE:
Same philosophy as reranking: this is an optimization, not a dependency.
Any failure (LLM error, malformed response, timeout) falls back to using
the original question unchanged, with no identifiers. Retrieval must never
break because the rewriter broke.
"""

import json
import logging

from src.config import settings
from src.generation import llm_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You rewrite user questions into search queries for a hybrid \
(keyword + semantic) search engine over technical documentation.

Given a question, respond with ONLY a JSON object (no markdown fences, no \
explanation) with exactly these keys:
  "search_query": a short, keyword-forward version of the question — drop \
filler words ("can", "what is", "how do I"), keep every technical term, \
identifier, and named concept from the original question.
  "identifiers": a list of exact identifiers mentioned in the question \
(rule/section numbers, part numbers, defined terms like "Mandatory" or \
"Advisory"). Empty list if none.

Example:
Question: "Can a mandatory MISRA guideline be deviated from?"
{"search_query": "mandatory guideline deviation permitted", "identifiers": ["Mandatory"]}"""


def rewrite_query(question: str) -> tuple[str, list[str]]:
    """
    Rewrite a question into a search query, plus any extracted identifiers.

    Args:
        question: The user's original natural-language question.

    Returns:
        (search_query, identifiers). On any failure, returns
        (question, []) unchanged — never raises.
    """
    if not settings.query_rewrite_enabled:
        return question, []

    try:
        raw_response = _call_llm(question)
        parsed = json.loads(raw_response)
        search_query = parsed["search_query"]
        identifiers = parsed.get("identifiers", [])
        if not isinstance(search_query, str) or not search_query.strip():
            raise ValueError(f"Empty or non-string search_query in response: {parsed!r}")
        return search_query, identifiers
    except Exception:
        logger.exception("Query rewrite failed — falling back to the original question.")
        return question, []


def _call_llm(question: str) -> str:
    """
    Call the configured LLM provider and return the raw response text.

    Provider dispatch (OpenAI / Anthropic / OpenRouter) lives in
    src/generation/llm_client.py, shared with generator.py and query_plan.py.
    This wrapper's job is just resolving the rewrite-specific model override
    before delegating — kept as a named function so tests can keep patching
    "src.retrieval.query_rewrite._call_llm".
    """
    model = settings.rewrite_model or settings.llm_model
    return llm_client.call_llm(SYSTEM_PROMPT, question, max_tokens=256, model=model)
