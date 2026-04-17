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

logger = logging.getLogger(__name__)

# Phrase the LLM uses when context is insufficient — we detect this
# to set has_answer=False on the Answer object
NO_ANSWER_PHRASE = "does not contain information about this topic"


def generate_answer(question: str, n_results: int = 5) -> Answer:
    """
    Full RAG pipeline: retrieve relevant chunks → generate cited answer.

    This is the single public function of this module.
    Everything else (prompts, LLM clients) is an implementation detail.

    Args:
        question: The user's natural language question.
        n_results: Number of chunks to retrieve and use as context.

    Returns:
        Answer object with generated text and source citations.
    """
    # ── Step 1: Retrieve relevant chunks ──
    logger.info(f"Retrieving context for: '{question}'")
    raw_chunks = query_collection(question, n_results=n_results)

    if not raw_chunks:
        return Answer(
            question=question,
            answer="No documents have been ingested yet. Please run the ingestion pipeline first.",
            sources=[],
            has_answer=False,
        )

    # ── Step 2: Build prompt ──
    user_prompt = build_user_prompt(question, raw_chunks)

    # ── Step 3: Call LLM ──
    logger.info(f"Generating answer with {settings.llm_provider}/{settings.llm_model}")
    answer_text = _call_llm(user_prompt)

    # ── Step 4: Package into Answer with sources ──
    sources = [
        RetrievedContext(
            text=chunk["text"],
            citation=chunk["metadata"].get("citation", "Unknown"),
            page_number=chunk["metadata"].get("page_number", 0),
            source_filename=chunk["metadata"].get("source_filename", "Unknown"),
            distance=chunk["distance"],
        )
        for chunk in raw_chunks
    ]

    has_answer = NO_ANSWER_PHRASE not in answer_text.lower()

    return Answer(
        question=question,
        answer=answer_text,
        sources=sources,
        has_answer=has_answer,
    )


def _call_llm(user_prompt: str) -> str:
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

    Returns:
        Raw response string from the LLM.
    """
    if settings.llm_provider == "anthropic":
        return _call_anthropic(user_prompt)
    else:
        return _call_openai(user_prompt)


def _call_openai(user_prompt: str) -> str:
    """Call OpenAI API and return response text."""
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  # 0 = deterministic — we want consistent, factual answers
                        # not creative variation. Same query should give same answer.
                        # Like disabling dithering on your ADC for stable readings.
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""


def _call_anthropic(user_prompt: str) -> str:
    """Call Anthropic API and return response text."""
    client = Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.content[0].text
