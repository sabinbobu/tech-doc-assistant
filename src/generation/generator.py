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

import opik

from src.config import settings
from src.embedding.vector_store import (
    list_indexed_documents,
    load_document_outline,
    query_collection,
)
from src.generation.llm_client import call_llm
from src.generation.models import Answer, RetrievedContext
from src.generation.prompts import SYNTHESIS_SYSTEM_PROMPT, SYSTEM_PROMPT, build_user_prompt
from src.retrieval.query_plan import QueryPlan, plan_query
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.rerank import rerank
from src.retrieval.structure import find_sections, render_toc_for_all_documents

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


@opik.track(name="rag_answer", project_name="Tech-Doc-Assistant")
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

    # ── Step 1: Classify and route ──
    # Three question shapes need three different strategies — see
    # src/retrieval/query_plan.py's module docstring for why one strategy
    # for all three was exactly what was broken (verified: a "guide me step
    # by step" question got refused even though retrieval had the right
    # page, and "list the table of contents" has no chunk-retrieval answer
    # at all). Degrades to "lookup" (today's original behavior) on any
    # classification failure.
    plan = plan_query(question)

    if plan.mode == "structural":
        return _generate_structural_answer(question, plan, source_filenames)

    candidate_k = max(settings.retrieval_candidate_k, n_results)

    if plan.mode == "procedural":
        # ── Step 2 (procedural): multi-query retrieval ──
        # A single query only searches one neighborhood of the document.
        # "Guide me step by step to configure X" needs evidence from pin
        # definitions, protection features, and timing diagrams that don't
        # sit near each other — so we search once per sub-query and union
        # the results, instead of hoping one query's top_k spans all of it.
        sub_queries = plan.sub_queries or [question]
        raw_chunks = _retrieve_for_procedural(sub_queries, source_filenames, candidate_k)
    else:
        # ── Step 2 (lookup): unchanged from before query planning existed ──
        # The index doesn't care about grammar — BM25 rewards term overlap,
        # and this corpus is dense with identifiers (rule numbers, "Mandatory"
        # vs "Advisory") that a short keyword query matches better than a
        # full question's prose does. `question` itself is untouched — the
        # LLM prompt and citations always use what the user actually asked.
        search_query, _identifiers = rewrite_query(question)
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
            mode=plan.mode,
        )

    # ── Step 3: Rerank down to the final top-k ──
    # Reranked against the original `question` in every mode, including
    # procedural (not any one sub-query) — the cross-encoder reads natural
    # language well (it's what it's trained on), and the question is what
    # the answer actually needs to satisfy.
    raw_chunks = rerank(question, raw_chunks, top_k=n_results)

    # ── Step 4: Build prompt ──
    # Procedural mode gets the synthesis prompt (permits assembling a
    # procedure from documented facts, requires marking inference — see
    # SYNTHESIS_SYSTEM_PROMPT's docstring) and a larger output budget; a
    # multi-step procedure doesn't fit in a lookup-sized response.
    user_prompt = build_user_prompt(question, raw_chunks)
    if plan.mode == "procedural":
        system_prompt = SYNTHESIS_SYSTEM_PROMPT
        max_tokens = settings.synthesis_answer_tokens
    else:
        system_prompt = SYSTEM_PROMPT
        max_tokens = None  # _call_llm defaults to settings.max_answer_tokens

    # ── Step 5: Call LLM ──
    logger.info(f"Generating answer with {settings.llm_provider}/{settings.llm_model}")
    try:
        answer_text = _call_llm(user_prompt, system_prompt=system_prompt, max_tokens=max_tokens)
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
        mode=plan.mode,
    )


def _retrieve_for_procedural(
    sub_queries: list[str],
    source_filenames: list[str] | None,
    candidate_k: int,
) -> list[dict]:
    """
    Run hybrid search once per sub-query and union the results.

    Deduped by (source_filename, chunk_index) — the same chunk legitimately
    turns up under more than one sub-query (e.g. a diagnosis-timing passage
    matching both "short circuit to ground diagnosis" and "diagnosis retry
    timing"), and it should only reach the reranker once.
    """
    seen: set[tuple[str, int]] = set()
    union: list[dict] = []
    for sub_query in sub_queries:
        results = query_collection(
            sub_query, n_results=candidate_k, source_filenames=source_filenames
        )
        for chunk in results:
            key = (
                chunk["metadata"].get("source_filename", ""),
                chunk["metadata"].get("chunk_index", -1),
            )
            if key not in seen:
                seen.add(key)
                union.append(chunk)
    return union


def _generate_structural_answer(
    question: str,
    plan: QueryPlan,
    source_filenames: list[str] | None,
) -> Answer:
    """
    Answer a structural question directly from the document outline — no
    chunk retrieval, no LLM call. See src/retrieval/structure.py's module
    docstring for why: there's no chunk boundary that reconstructs "the
    whole table of contents", and an LLM reformatting an already-correct
    outline can only make it less accurate, never more.
    """
    available = source_filenames or [d["filename"] for d in list_indexed_documents()]

    if plan.structural_keyword:
        matches = [
            (filename, entry)
            for filename in available
            for entry in find_sections(filename, plan.structural_keyword)
        ]
        if matches:
            lines = [
                f"- {filename}: {entry.title} (page {entry.page})" for filename, entry in matches
            ]
            answer_text = f"Sections matching '{plan.structural_keyword}':\n" + "\n".join(lines)
        else:
            answer_text = (
                f"No sections matching '{plan.structural_keyword}' were found in the "
                "loaded documents' tables of contents."
            )
        has_answer = bool(matches)
    else:
        has_answer = any(load_document_outline(filename) for filename in available)
        answer_text = render_toc_for_all_documents(source_filenames)

    return Answer(
        question=question,
        answer=answer_text,
        sources=[],
        has_answer=has_answer,
        mode="structural",
    )


def _call_llm(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int | None = None,
) -> str:
    """
    Call the configured LLM provider and return the response text.

    The actual provider dispatch (OpenAI / Anthropic / OpenRouter) lives in
    src/generation/llm_client.py, shared with query_rewrite.py and
    query_plan.py. This wrapper's job is just resolving generator.py's own
    default token budget before delegating — kept as a named function (rather
    than calling call_llm() directly from generate_answer()) so every existing
    test can keep patching "src.generation.generator._call_llm".

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
    return call_llm(system_prompt, user_prompt, max_tokens)
