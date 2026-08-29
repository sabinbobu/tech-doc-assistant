"""
Question routing — classify a question and, for procedural ones, decompose it.

WHY THIS EXISTS:
Three different question shapes need three different answering strategies,
and using one strategy for all three is exactly what was broken (verified
against the live index this session):

  "What is the BTS7200?"                          -> lookup:      one topic,
                                                       answerable from one or
                                                       two nearby passages.
  "Guide me step by step on how to configure X"    -> procedural: the answer
                                                       is scattered across
                                                       pin tables, protection
                                                       features, and timing
                                                       diagrams that don't sit
                                                       near each other. A
                                                       single top-8 retrieval
                                                       from ONE query finds
                                                       one neighborhood, not
                                                       all of them.
  "List the table of contents"                     -> structural: not a
                                                       content question at
                                                       all. No chunk boundary
                                                       reconstructs "the whole
                                                       TOC" — see
                                                       src/retrieval/structure.py.

Retrieval quality was never the bottleneck for the procedural case — for the
literal "Short to Ground" question this module was built to fix, retrieval
already returned the page containing "Short circuit to ground" among its
top-8. The problem was that ONE query only searches one neighborhood.
Decomposing into several sub-queries (pin config, protection features,
diagnosis timing, ...) and unioning their results is what gathers evidence
from across the whole document instead of one corner of it.

FAILURE MODE:
Same philosophy as reranking and query rewriting: this is routing, not a
dependency. Any failure falls back to "lookup" mode with no decomposition —
today's existing retrieval path — so a broken classifier degrades to
current behavior, never to a worse one.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel

from src.config import settings
from src.generation import llm_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You classify a user's question about technical documentation \
(datasheets, standards, user guides) into exactly one mode, and for procedural \
questions, break it into search queries.

Modes:
- "lookup": a single fact, answerable from one or two nearby passages. \
Examples: "What is the BTS7200?", "Does it support open load detection?", \
"What is a deviation in MISRA compliance?"
- "procedural": the user wants a step-by-step process, configuration, or \
worked example. Evidence for this is typically scattered across several \
unrelated sections (pin definitions, protection features, timing diagrams, \
electrical characteristics) — not one passage. Examples: "Guide me step by \
step on how to configure X", "Give me an example of Y", "How do I set up Z".
- "structural": the question is about THIS LOADED DOCUMENT'S OWN NAVIGATIONAL \
STRUCTURE as a physical artifact — its table of contents, what a chapter \
covers, which sections discuss a topic. It is asking "where in this PDF is \
X", not "what is X". Examples: "List the table of contents", "What's in \
chapter 8?", "Which sections cover diagnostics?"
NOT structural: a question that merely contains the word "document" or \
"section" while actually asking about technical content or a general \
concept — e.g. "What should a MISRA compliance summary document contain?" \
is asking what belongs in that KIND of document per the standard (a \
content/definition question -> lookup), not asking to see this PDF's \
own table of contents. If the question would be answered by quoting a \
chapter/section list, it's structural; if it would be answered by \
explaining a concept, fact, or requirement, it's lookup or procedural \
even if the words "document" or "section" appear in it.

Respond with ONLY a JSON object (no markdown fences, no explanation):
  "mode": one of "lookup", "procedural", "structural"
  "sub_queries": for "procedural" mode ONLY, 3-5 short keyword-forward search \
queries covering the distinct pieces of evidence needed — the specific \
pins/registers/features involved, the relevant protection or diagnostic \
mechanism, timing or threshold values, and a general query for the overall \
procedure. Empty list for "lookup" and "structural" — they don't need \
decomposition.
  "structural_keyword": for "structural" mode ONLY, a single short topic \
word/phrase if the question asks about a SPECIFIC topic or chapter (e.g. \
"which sections cover diagnostics" -> "diagnostics", "what's in the pin \
configuration chapter" -> "pin configuration"). null if the question asks \
for the WHOLE table of contents with no specific topic — e.g. "list the \
table of contents". Omit or null for "lookup"/"procedural" modes.

Examples:

Question: "What is this BTS7200?"
{"mode": "lookup", "sub_queries": [], "structural_keyword": null}

Question: "Guide me step by step on how to configure this integrate in \
order to implement the Short to Ground error. Give me an example of configuration."
{"mode": "procedural", "sub_queries": ["short circuit to ground diagnosis", \
"pin configuration DEN DSEL", "overcurrent protection features", \
"diagnosis retry timing", "configuration example"], "structural_keyword": null}

Question: "List the Table of contents for this document"
{"mode": "structural", "sub_queries": [], "structural_keyword": null}

Question: "Which sections cover diagnostics?"
{"mode": "structural", "sub_queries": [], "structural_keyword": "diagnostics"}

Question: "What should a MISRA compliance summary document contain?"
{"mode": "lookup", "sub_queries": [], "structural_keyword": null}"""


class QueryPlan(BaseModel):
    mode: Literal["lookup", "procedural", "structural"]
    sub_queries: list[str] = []
    structural_keyword: str | None = None


def plan_query(question: str) -> QueryPlan:
    """
    Classify a question and decompose it if procedural.

    Returns:
        QueryPlan(mode="lookup", sub_queries=[]) on any failure, or when
        settings.query_planning_enabled is False — never raises.
    """
    if not settings.query_planning_enabled:
        return QueryPlan(mode="lookup", sub_queries=[])

    try:
        raw_response = _call_llm(question)
        parsed = json.loads(raw_response)
        return QueryPlan(
            mode=parsed["mode"],
            sub_queries=parsed.get("sub_queries", []),
            structural_keyword=parsed.get("structural_keyword"),
        )
    except Exception:
        logger.exception("Query planning failed — falling back to lookup mode.")
        return QueryPlan(mode="lookup", sub_queries=[])


def _call_llm(question: str) -> str:
    """
    Call the configured LLM provider and return the raw response text.

    Provider dispatch (OpenAI / Anthropic / OpenRouter) lives in
    src/generation/llm_client.py, shared with generator.py and
    query_rewrite.py. This wrapper's job is just resolving the
    classification-specific model override before delegating — kept as a
    named function so tests can keep patching
    "src.retrieval.query_plan._call_llm".
    """
    model = settings.query_plan_model or settings.llm_model
    return llm_client.call_llm(SYSTEM_PROMPT, question, max_tokens=256, model=model)
