"""
LLM answer generation — the final step in the RAG pipeline.

This module combines:
  1. Retrieved context chunks (from embedding/vector_store.py)
  2. A well-structured prompt (from generation/prompts.py)
  3. An LLM call (OpenAI or Anthropic)

Into a single Answer object with citations.

ANALOGY:
If retrieval is fetching the relevant pages from your service manual,
generation is the experienced engineer who reads those pages and
writes a clear, cited diagnosis report.
The engineer (LLM) is only as good as the pages you hand them (retrieval).
This is why Steps 3 and 4 matter so much — garbage retrieval = garbage answers.
"""

import logging

from anthropic import Anthropic
from openai import OpenAI

from src.config import settings
from src.embedding.vector_store import query_collection
from src.generation.models import Answer, RetrievedContext
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.rerank import rerank

logger = logging.getLogger(__name__)

# Phrase the LLM uses when context is insufficient — we detect this
# to set has_answer=False on the Answer object
NO_ANSWER_PHRASE = "does not contain information about this topic"


def _is_refusal(answer_text: str) -> bool:
    """
    True only when the model refused outright — not when a real answer
    happens to mention what the docs don't cover.

    WHY NOT A SUBSTRING TEST:
    This used to be `NO_ANSWER_PHRASE not in answer_text.lower()`, which
    misfires on exactly the answers we most want to keep. The system prompt
    tells the model to answer the supported parts of a multi-part question
    and separately note the unsupported ones — so a *good* partial answer
    routinely contains a sentence like "...the documentation does not contain
    information about this topic in code form, but Section 8 describes...".
    The substring test flagged that whole answer as a failure, and the UI
    (src/ui/app.py) then hid the Sources panel and showed a "no information"
    warning over a perfectly good answer.

    The prompt instructs the model to emit the refusal as its *entire*
    response ("respond with exactly: ..."), so that's what we detect: the
    phrase carrying essentially the whole message, not appearing somewhere
    inside it.
    """
    normalized = answer_text.strip().lower()
    if NO_ANSWER_PHRASE not in normalized:
        return False
    # A refusal is the phrase and little else (allowing for the leading
    # "The provided documentation " and trailing punctuation the prompt
    # specifies). A real answer that merely mentions it is much longer.
    return len(normalized) <= len(NO_ANSWER_PHRASE) + 40


def generate_answer(
    question: str,
    n_results: int | None = None,
    source_filenames: list[str] | None = None,
) -> Answer:
    """
    Full RAG pipeline: retrieve relevant chunks → generate cited answer.

    This is the single public function of this module.
    Everything else (prompts, LLM clients) is an implementation detail.

    Args:
        question: The user's natural language question.
        n_results: Number of chunks to retrieve and use as context.
            Defaults to settings.retrieval_top_k.
        source_filenames: If given, restrict retrieval to these documents
            only. None searches every indexed document (previous behavior).

    Returns:
        Answer object with generated text and source citations.
    """
    if n_results is None:
        n_results = settings.retrieval_top_k

    # ── Step 1: Rewrite the question into a search query ──
    # The index doesn't care about grammar — BM25 rewards term overlap, and
    # this corpus is dense with identifiers (rule numbers, "Mandatory" vs
    # "Advisory") that a short keyword query matches better than a full
    # question's prose does. `question` itself is untouched — the LLM prompt
    # and citations always use what the user actually asked.
    search_query, _identifiers = rewrite_query(question)

    # ── Step 2: Retrieve a candidate pool ──
    # Fetch more than we need (settings.retrieval_candidate_k) so reranking
    # has something to actually re-sort — hybrid search's rank-60 fusion order
    # isn't necessarily the cross-encoder's relevance order, so a chunk that
    # placed #15 by RRF might belong in the final top n_results after a model
    # actually reads it against the question.
    candidate_k = max(settings.retrieval_candidate_k, n_results)
    logger.info(f"Retrieving context for: '{question}' (search query: '{search_query}')")
    raw_chunks = query_collection(
        search_query, n_results=candidate_k, source_filenames=source_filenames
    )

    if not raw_chunks:
        return Answer(
            question=question,
            answer="No documents have been ingested yet. Please run the ingestion pipeline first.",
            sources=[],
            has_answer=False,
        )

    # ── Step 3: Rerank down to the final top-k ──
    # Reranked against the original `question`, not `search_query` — the
    # cross-encoder reads natural language well (it's what it's trained on),
    # unlike BM25/embeddings where a terse keyword query helps.
    raw_chunks = rerank(question, raw_chunks, top_k=n_results)

    # ── Step 4: Build prompt ──
    user_prompt = build_user_prompt(question, raw_chunks)

    # ── Step 5: Call LLM ──
    logger.info(f"Generating answer with {settings.llm_provider}/{settings.llm_model}")
    try:
        answer_text = _call_llm(user_prompt)
    except Exception as exc:
        raise RuntimeError(
            f"LLM call failed ({settings.llm_provider}/{settings.llm_model}): {exc}"
        ) from exc

    # ── Step 6: Package into Answer with sources ──
    sources = [
        RetrievedContext(
            text=chunk["text"],
            citation=chunk["metadata"].get("citation", "Unknown"),
            page_number=chunk["metadata"].get("page_number", 0),
            source_filename=chunk["metadata"].get("source_filename", "Unknown"),
            distance=chunk["distance"],
            rerank_score=chunk.get("rerank_score"),
        )
        for chunk in raw_chunks
    ]

    has_answer = not _is_refusal(answer_text)

    return Answer(
        question=question,
        answer=answer_text,
        sources=sources,
        has_answer=has_answer,
    )


def _call_llm(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int | None = None,
) -> str:
    """
    Call the configured LLM provider and return the response text.

    Supports both OpenAI and Anthropic — the job description asks for both.
    The provider is controlled by settings.llm_provider in your .env.

    This is a private function (underscore prefix) — callers use
    generate_answer() and don't need to know which LLM is being used.
    Like a HAL function in embedded — the caller doesn't care if it's
    SPI or I2C underneath.

    Args:
        user_prompt: The fully constructed prompt with context + question.
        system_prompt: Defaults to the lookup-mode SYSTEM_PROMPT. Synthesis
            mode (query_plan.py) passes SYNTHESIS_SYSTEM_PROMPT instead.
        max_tokens: Defaults to settings.max_answer_tokens. Synthesis mode
            passes settings.synthesis_answer_tokens — a step-by-step
            procedure doesn't fit in a lookup-sized budget.

    Returns:
        Raw response string from the LLM.
    """
    if max_tokens is None:
        max_tokens = settings.max_answer_tokens
    if settings.llm_provider == "anthropic":
        return _call_anthropic(user_prompt, system_prompt, max_tokens)
    else:
        return _call_openai(user_prompt, system_prompt, max_tokens)


def _call_openai(user_prompt: str, system_prompt: str, max_tokens: int) -> str:
    """Call OpenAI API and return response text."""
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.llm_model,
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


def _call_anthropic(user_prompt: str, system_prompt: str, max_tokens: int) -> str:
    """Call Anthropic API and return response text."""
    client = Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        temperature=0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.content[0].text
