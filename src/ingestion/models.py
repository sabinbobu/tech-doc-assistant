"""
Data models for document ingestion.

WHY WE DEFINE MODELS FIRST:
In embedded C, before writing any logic, you define your data structures:
    typedef struct {
        uint8_t channel;
        uint16_t fault_code;
        char description[64];
    } FaultRecord;

In Python with Pydantic, we do the same thing — but with runtime validation.
If someone passes a string where we expect an int, Pydantic catches it immediately.
This is like having compile-time AND runtime type checking.

These models define what a "document" and a "page" look like as they flow
through our pipeline. Every function will speak in terms of these types.
"""

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """
    Represents a single page extracted from a PDF.

    Why track page numbers? For citations!
    When the LLM answers a question, we want to say
    "Source: document.pdf, page 42" — users need to verify answers.
    In automotive, traceability is everything (think ISO 26262).
    """

    page_number: int = Field(description="1-indexed page number")
    text: str = Field(description="Extracted text content of the page")
    # We'll add metadata like tables, images later as we enhance the system


class OutlineEntry(BaseModel):
    """
    One entry from a PDF's embedded outline (bookmarks / table of contents).

    PDF readers show this as the collapsible sidebar tree. Most
    professionally-produced technical PDFs embed one — datasheets, standards,
    user guides. It's structured data the *author* already built; we just
    have to read it rather than reconstruct it from chunk text (which is
    unreliable — see DocumentContent.full_text's TOC page, which is prose
    with dot leaders, not structure).
    """

    level: int = Field(description="Nesting depth, 1-indexed (1 = top-level chapter)")
    title: str = Field(description="Section/chapter title as it appears in the outline")
    page: int = Field(description="1-indexed page number where this section begins")


class DocumentContent(BaseModel):
    """
    Represents a fully ingested document with all its pages.

    This is the output of our ingestion pipeline —
    a clean, structured representation of a PDF that downstream
    modules (chunking, embedding) can consume.
    """

    filename: str = Field(description="Original filename of the PDF")
    filepath: str = Field(description="Full path to the source PDF")
    total_pages: int = Field(description="Total number of pages in the document")
    pages: list[PageContent] = Field(description="List of extracted pages")
    outline: list[OutlineEntry] = Field(
        default_factory=list,
        description=(
            "The PDF's embedded outline/bookmarks, in document order. "
            "Empty for PDFs that don't have one — never required."
        ),
    )

    def section_for_page(self, page_number: int) -> str | None:
        """
        The most specific outline entry that applies to a given page.

        WHY "LAST ENTRY AT OR BEFORE THIS PAGE, IN DOCUMENT ORDER":
        Outline entries come in traversal order — a chapter's subsections are
        listed immediately after it, before the next chapter. So walking
        forward and taking the last entry with page <= page_number naturally
        lands on the deepest applicable section, not just the nearest
        chapter. Example: "3 Pin Configuration" (page 5), "3.1 Pin
        Configuration" (page 5), "4 Electrical..." (page 8) — page 6 resolves
        to "3.1", the specific subsection, not just "3".

        Returns None if the document has no outline, or the page precedes
        every outline entry (e.g. a cover page before "1 Overview").
        """
        applicable = [entry for entry in self.outline if entry.page <= page_number]
        return applicable[-1].title if applicable else None

    @property
    def full_text(self) -> str:
        """Concatenate all pages into a single string.

        Useful when you need the whole document as one blob,
        e.g., for document-level summarization.
        """
        return "\n\n".join(page.text for page in self.pages)

    def get_page_text(self, page_number: int) -> str | None:
        """Get text for a specific page (1-indexed).

        Returns None if page doesn't exist — defensive programming,
        just like checking a pointer isn't NULL before dereferencing.
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        return None
