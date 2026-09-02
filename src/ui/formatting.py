"""
Formatting utilities for the Streamlit UI.

Pure functions (no import streamlit) that transform data into display strings.
This module is isolated so it's readable and trivially verifiable with pytest.
"""

from typing import Literal

from src.generation.models import RetrievedContext


def style_inference_markers(text: str) -> str:
    """
    Replace inference markers with Streamlit color-markdown syntax.

    `(Inferred)` indicates a step assembled from documented facts rather than
    stated outright — cite the facts but mark the step itself as reasoned-from
    rather than quoted. `(Not specified — verify against your design)` marks a
    concrete value the docs don't provide rather than an invented one.

    For hardware/safety-relevant docs, the distinction is the load-bearing part
    of answering at all: refuse nothing, invent nothing, explain everything.
    """
    text = text.replace("(Inferred)", ":orange[(Inferred)]")
    text = text.replace(
        "(Not specified — verify against your design)",
        ":red[(Not specified — verify against your design)]",
    )
    return text


def relevance_from_distance(distance: float) -> float:
    """
    Convert Chroma's cosine distance to a display-friendly relevance [0, 1].

    Chroma's `{"hnsw:space": "cosine"}` (vector_store.py:81) returns distance
    in [0, 2]: 0 = identical, 1 = orthogonal, 2 = opposite. The UI displays
    `1 - distance` as "relevance", which naturally goes negative for distance > 1
    (low-similarity or anti-correlated vectors). This clamp prevents that display
    glitch without inventing fake scores — the clamp is transparent in the
    returned value itself, not a silent correction.
    """
    return max(0.0, min(1.0, 1.0 - distance))


def format_scores(source: RetrievedContext) -> str:
    """
    Format retrieval scores for display.

    `relevance` is the hybrid-search distance (0-1 after clamping).
    `rerank_score` is an unbounded cross-encoder logit (observed values: ~-7.8
    to ~+4.91) and is never rendered as a 0-1 bar — displaying it as-is
    preserves the model's magnitude signal (higher = better ranked). When
    `rerank_score` is None (reranking disabled or failed), omit the line entirely.
    """
    relevance = relevance_from_distance(source.distance)
    line = f"p.{source.page_number} • relevance: {relevance:.2f}"
    if source.rerank_score is not None:
        line += f" • rerank: {source.rerank_score:.2f}"
    return line


MODE_LABELS: dict[str, str] = {
    "lookup": "🔍 Lookup",
    "procedural": "🛠️ Procedural (synthesized)",
    "structural": "📑 Structural (document outline)",
}

# Typed as the literal set st.badge() accepts, not plain str: these values are
# passed straight to Streamlit, which types `color` as a Literal. dict[str, str]
# would widen them back to str at the call site and fail type checking even
# though every value here is valid.
BadgeColor = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"
]

MODE_COLORS: dict[str, BadgeColor] = {
    "lookup": "blue",
    "procedural": "orange",
    "structural": "violet",
}


def unique_citations(answer) -> list[str]:
    """
    Extract unique citations from sources, preserving order.

    The `Answer.formatted_sources` property does this internally, but as a
    pre-formatted string ("Sources:" + list). This function extracts the list
    so the UI can render each citation as a badge instead of a single block.
    """
    if not answer.sources:
        return []
    seen = set()
    result = []
    for source in answer.sources:
        if source.citation not in seen:
            seen.add(source.citation)
            result.append(source.citation)
    return result
