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
