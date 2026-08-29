"""
Document structure — answer questions from the outline, not from chunks.

WHY THIS EXISTS:
"List the table of contents" or "what's in chapter 8" cannot be answered by
retrieval, no matter how good hybrid search or reranking get. The TOC page
in a PDF's raw text is prose with dot leaders — extracted verbatim it
chunks into a dozen fragments across a couple of pages, and retrieval
returns at most top_k of them. There's no chunk boundary that reconstructs
"the whole table of contents" or "everything under chapter 8", because
those are properties of the DOCUMENT, not of any single passage in it.

The fix isn't better retrieval — it's a different data source. PDFs that
have an embedded outline (src/ingestion/pdf_reader.py's _extract_outline,
persisted by src/embedding/vector_store.py's save/load_document_outline)
already have this structure recorded by the document's own author. This
module reads it back and renders it directly into a prompt, bypassing
chunk retrieval entirely for structural questions.

ANALOGY:
Retrieval is asking "which pages mention X" — a search over content.
This is reading the book's own table of contents — a lookup over
structure. Different questions need different tools; conflating them is
why "list the TOC" used to retrieve 8 fragments of the TOC page itself
and still couldn't answer the question.
"""

import logging

from src.embedding.vector_store import list_indexed_documents, load_document_outline
from src.ingestion.models import OutlineEntry

logger = logging.getLogger(__name__)


def render_toc(filename: str) -> str | None:
    """
    Render one document's outline as an indented, readable table of contents.

    Returns None if this document has no saved outline (never ingested with
    one, or genuinely has none) — the caller decides how to handle that
    (fall back to a lookup-mode answer, or say structure isn't available).
    """
    outline = load_document_outline(filename)
    if not outline:
        return None
    return _format_outline(filename, outline)


def render_toc_for_all_documents(source_filenames: list[str] | None = None) -> str:
    """
    Render the outline for every currently-indexed document, or a specific
    subset — used when a structural question doesn't name one document (the
    common case: the UI's document picker may have several selected, or the
    user just asked "list the table of contents" with one document loaded).

    Documents with no outline are skipped, not reported as an error — a
    mixed corpus (some PDFs have bookmarks, some don't) is normal.
    """
    available = {d["filename"] for d in list_indexed_documents()}
    targets = sorted(available & set(source_filenames)) if source_filenames else sorted(available)

    sections = [render_toc(name) for name in targets]
    rendered = [s for s in sections if s is not None]

    if not rendered:
        return "None of the loaded documents have a table of contents available."
    return "\n\n".join(rendered)


def find_sections(filename: str, keyword: str) -> list[OutlineEntry]:
    """
    Sections whose title mentions `keyword` — for "which sections cover
    diagnostics" style questions. Plain case-insensitive substring match,
    deliberately simple: outline titles are short, author-written phrases
    (a handful to a couple hundred per document), not free text needing
    semantic search. Empty list (not None) when the document has no
    outline or nothing matches — both are "no results", not an error.
    """
    outline = load_document_outline(filename) or []
    needle = keyword.lower()
    return [entry for entry in outline if needle in entry.title.lower()]


def _format_outline(filename: str, outline: list[OutlineEntry]) -> str:
    """2 spaces of indent per nesting level below the shallowest entry."""
    min_level = min(entry.level for entry in outline)
    lines = [f"{filename}:"]
    for entry in outline:
        indent = "  " * (entry.level - min_level + 1)
        lines.append(f"{indent}- {entry.title} (page {entry.page})")
    return "\n".join(lines)
