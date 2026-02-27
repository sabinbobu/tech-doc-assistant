"""
PDF text extraction using PyMuPDF (fitz).

WHY PyMuPDF AND NOT PyPDF2 OR pdfplumber?
- PyMuPDF is written in C/C++ (you'll appreciate this!) — it's 10-50x faster
- Better text extraction quality, especially for complex layouts
- Handles tables and columns better than alternatives
- The library is imported as 'fitz' (named after the rendering engine)

ANALOGY TO YOUR EMBEDDED WORLD:
This module is like a UART/SPI driver — it's the lowest-level interface
that reads raw data (PDF bytes) and converts it into something our
application can work with (structured text). Just like you wouldn't
parse CAN frames in your application layer, we isolate PDF parsing here.

The rest of the pipeline never needs to know how PDFs work internally.
"""

import logging
from pathlib import Path

import pymupdf as fitz

from src.ingestion.models import DocumentContent, PageContent

logger = logging.getLogger(__name__)


def extract_text_from_pdf(filepath: str | Path) -> DocumentContent:
    """
    Extract text from a PDF file, page by page.

    Args:
        filepath: Path to the PDF file.

    Returns:
        DocumentContent with all pages extracted.

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        ValueError: If the file isn't a valid PDF.

    Why we return a structured object instead of just a string:
    - We preserve page boundaries (needed for citations later)
    - We can add metadata per page (tables, images) in the future
    - Downstream code gets a clean contract to work with
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    if filepath.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {filepath.suffix}")

    logger.info(f"Extracting text from: {filepath.name}")

    pages: list[PageContent] = []

    # fitz.open() is like fopen() — opens the PDF for reading
    # We use a context manager (with) to ensure it's always closed,
    # like wrapping fopen/fclose in an RAII-style pattern.
    with fitz.open(filepath) as doc:
        for page_num, page in enumerate(doc, start=1):
            # get_text() extracts all text from the page
            # The "text" option gives us plain text with layout preserved
            raw_text = page.get_text("text")

            # Clean the extracted text
            cleaned_text = _clean_page_text(raw_text)

            if cleaned_text.strip():  # Skip empty pages
                pages.append(
                    PageContent(
                        page_number=page_num,
                        text=cleaned_text,
                    )
                )
            else:
                logger.debug(f"Skipping empty page {page_num}")

        total_pages = len(doc)

    logger.info(
        f"Extracted {len(pages)} non-empty pages from {total_pages} total pages"
    )

    return DocumentContent(
        filename=filepath.name,
        filepath=str(filepath),
        total_pages=total_pages,
        pages=pages,
    )


def _clean_page_text(text: str) -> str:
    """
    Clean raw extracted text from a PDF page.

    PDF text extraction is messy — you get:
    - Extra whitespace from column layouts
    - Hyphenated words split across lines
    - Headers/footers repeated on every page
    - Random line breaks in the middle of sentences

    This is like cleaning up raw ADC readings —
    you need to filter noise before the data is useful.

    We keep this simple for now and can enhance later.
    """
    # Remove excessive blank lines (keep max 2 consecutive)
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    blank_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(stripped)

    # Rejoin and handle hyphenation at line breaks
    # "auto-\nmatic" → "automatic"
    result = "\n".join(cleaned_lines)
    result = result.replace("-\n", "")

    return result.strip()


def extract_all_pdfs(directory: str | Path) -> list[DocumentContent]:
    """
    Extract text from all PDFs in a directory.

    This is the batch processing entry point — like running
    your test suite across all .c files in a directory.

    Args:
        directory: Path to directory containing PDFs.

    Returns:
        List of DocumentContent objects, one per PDF.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in: {directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    documents: list[DocumentContent] = []
    for pdf_path in pdf_files:
        try:
            doc = extract_text_from_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            # Don't let one bad PDF crash the whole pipeline
            # Log and continue — like a watchdog timer recovering from a fault
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.info(f"Successfully processed {len(documents)}/{len(pdf_files)} PDFs")
    return documents
