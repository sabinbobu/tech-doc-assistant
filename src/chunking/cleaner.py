"""
Header/footer detection and removal.

WHY THIS IS ITS OWN MODULE:
Separation of concerns. The PDF reader's job is extraction.
The cleaner's job is data quality. Mixing them would make both harder to test
and harder to swap out later (e.g., if you want a different cleaning strategy).

Same principle as in embedded: your UART driver doesn't do protocol parsing.
Your protocol layer doesn't touch hardware registers. Each layer has one job.

STRATEGY — Content-based repetition detection:
We scan all pages, find lines that appear on more than N% of pages,
and flag them as headers/footers. This is more robust than y-coordinate
filtering because:
  - It works on any PDF regardless of layout
  - It's self-adapting — it learns what's repeated FROM the document itself
  - It handles multi-line headers/footers naturally

The tradeoff: it requires seeing all pages before cleaning any page.
This is fine for our pipeline — we process whole documents at once.
"""

import logging
from collections import Counter

from src.ingestion.models import DocumentContent, PageContent

logger = logging.getLogger(__name__)


def detect_repeating_lines(document: DocumentContent, threshold: float = 0.4) -> set[str]:
    """
    Detect lines that repeat across many pages — these are headers/footers.

    Args:
        document: The full document with all pages.
        threshold: If a line appears on more than this fraction of pages,
                   it's considered a header/footer. 0.4 = 40% of pages.
                   Lower = more aggressive cleaning.
                   Higher = more conservative (keeps more content).

    Returns:
        Set of strings that should be removed from all pages.

    HOW IT WORKS:
    Count how many pages each unique line appears on.
    If "MISRA Compliance 2020" appears on 45 out of 50 pages → it's a footer.
    If "Rule 8.4 External linkage" appears on 1 page → it's content, keep it.

    This is like a frequency filter on a signal — you're suppressing the
    DC component (constant noise across pages) while preserving the
    AC component (unique content per page).
    """
    total_pages = len(document.pages)
    if total_pages == 0:
        return set()

    # Count how many pages each line appears on
    # Counter is like a hash map of {line: count}
    line_page_count: Counter = Counter()

    for page in document.pages:
        # Use a set per page — we count pages, not total occurrences
        # "MISRA" appearing 3 times on one page still counts as 1 page
        unique_lines_on_page = {
            line.strip()
            for line in page.text.split("\n")
            if line.strip()  # ignore blank lines
        }
        line_page_count.update(unique_lines_on_page)

    # A line is a header/footer if it appears on more than threshold% of pages
    min_pages = total_pages * threshold
    repeating = {
        line
        for line, count in line_page_count.items()
        if count >= min_pages
        # Don't remove very long lines — headers/footers are short
        # A line with 200+ chars is almost certainly content, not a footer
        and len(line) < 200
    }

    if repeating:
        logger.info(f"Detected {len(repeating)} repeating header/footer lines")
        for line in sorted(repeating)[:10]:  # log first 10
            logger.debug(f"  Repeating line: '{line[:80]}'")

    return repeating


def clean_page(page: PageContent, lines_to_remove: set[str]) -> PageContent:
    """
    Remove header/footer lines from a single page.

    Returns a NEW PageContent with cleaned text.
    We never mutate the original — immutability makes pipelines
    easier to reason about and test. Same principle as avoiding
    global state in embedded firmware.

    Args:
        page: Original page content.
        lines_to_remove: Set of line strings to strip out.

    Returns:
        New PageContent with headers/footers removed.
    """
    lines = page.text.split("\n")
    cleaned_lines = [
        line for line in lines
        if line.strip() not in lines_to_remove
    ]
    cleaned_text = "\n".join(cleaned_lines).strip()

    return PageContent(
        page_number=page.page_number,
        text=cleaned_text,
    )


def clean_document(document: DocumentContent, threshold: float = 0.4) -> DocumentContent:
    """
    Full document cleaning pipeline — detect and remove headers/footers.

    This is the single entry point for this module.
    Call this before chunking to ensure clean input to the chunker.

    Args:
        document: Raw DocumentContent from the PDF reader.
        threshold: Repetition threshold for header/footer detection.

    Returns:
        New DocumentContent with cleaned pages.
    """
    logger.info(f"Cleaning document: {document.filename}")

    # Step 1: Detect what's repeated
    repeating_lines = detect_repeating_lines(document, threshold)

    if not repeating_lines:
        logger.info("No repeating headers/footers detected — document is already clean")
        return document

    # Step 2: Clean each page
    cleaned_pages = [
        clean_page(page, repeating_lines)
        for page in document.pages
    ]

    # Step 3: Return a new document with cleaned pages
    # total_pages stays the same — we didn't remove pages, just noise within them
    cleaned_doc = DocumentContent(
        filename=document.filename,
        filepath=document.filepath,
        total_pages=document.total_pages,
        pages=cleaned_pages,
    )

    original_chars = sum(len(p.text) for p in document.pages)
    cleaned_chars = sum(len(p.text) for p in cleaned_pages)
    removed_chars = original_chars - cleaned_chars
    logger.info(
        f"Cleaning complete. Removed ~{removed_chars:,} characters "
        f"({removed_chars / original_chars * 100:.1f}% of total)"
    )

    return cleaned_doc
